"""
BC-ResNet-8 Training Script for Ship Bridge Voice Command Classification.

Architecture reference:
    "Broadcasted Residual Learning for Efficient Keyword Spotting"
    Byeonggeun Kim et al., Interspeech 2021
    https://arxiv.org/abs/2106.04140

Usage:
    # Training
    python scripts/train_bcresnet8.py \
        --csv data/dataset.csv \
        --wav_dir data/wavs \
        --output artifacts/bcresnet8

    # Latency benchmark only (skip training)
    python scripts/train_bcresnet8.py \
        --csv data/dataset.csv \
        --wav_dir data/wavs \
        --output artifacts/bcresnet8 \
        --benchmark_only \
        --checkpoint artifacts/bcresnet8/best_model.pt

Classes (4):
    0 - "Машина"
    1 - "Самый малый вперёд"
    2 - "Самый малый назад"
    3 - "Приготовить машину"
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching BC-ResNet paper for 1-second / 16 kHz input)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # Hz
AUDIO_DURATION_SEC = 1.0      # seconds → 16000 samples
N_MFCC = 40                   # number of MFCC coefficients
HOP_LENGTH = 160              # 10 ms hop → 100 frames per second
N_FFT = 400                   # 25 ms window
NUM_CLASSES = 4
SEED = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VoiceCommandDataset(Dataset):
    """
    Loads WAV files listed in a CSV and returns MFCC feature tensors.

    CSV schema:
        filepath  - relative or absolute path to WAV file
        label     - integer class index (0..NUM_CLASSES-1)
        split     - "train" | "val" | "test"
    """

    def __init__(
        self,
        csv_path: str,
        wav_dir: str,
        split: str,
        sample_rate: int = SAMPLE_RATE,
        duration_sec: float = AUDIO_DURATION_SEC,
    ) -> None:
        super().__init__()
        self.wav_dir = Path(wav_dir)
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_sec)
        self.split = split

        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        logger.info(f"[{split}] {len(self.df)} samples loaded")

        # MFCC transform — fixed parameters for BC-ResNet input
        # Output shape: (N_MFCC, T) where T = ceil(target_len / hop_length)
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=N_MFCC,
            melkwargs={
                "hop_length": HOP_LENGTH,
                "n_fft": N_FFT,
                "n_mels": 80,          # intermediate mel bins before DCT
                "f_min": 0.0,
                "f_max": sample_rate / 2,
            },
        )

    def _load_wav(self, path: Path) -> torch.Tensor:
        """Load WAV, resample if needed, pad/trim to exactly target_len samples."""
        waveform, sr = torchaudio.load(str(path))
        # Convert stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if source rate differs
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        # Pad or trim to exactly target_len
        length = waveform.shape[-1]
        if length < self.target_len:
            pad = self.target_len - length
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, : self.target_len]
        return waveform  # shape: (1, target_len)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        wav_path = self.wav_dir / row["filepath"]
        waveform = self._load_wav(wav_path)

        # MFCC: (1, N_MFCC, T)
        mfcc = self.mfcc_transform(waveform)

        # Normalize per-sample to zero mean, unit variance (standard practice)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        label = int(row["label"])
        return mfcc, label


# ---------------------------------------------------------------------------
# BC-ResNet-8 Architecture
# ---------------------------------------------------------------------------
#
# BC-ResNet (Broadcasted Residual Network) design principles:
#
#   1. "Broadcasted" residual connections: the shortcut path is a 1D
#      (frequency-wise) vector that gets broadcast across the time axis.
#      This forces the model to learn time-invariant frequency patterns —
#      a strong inductive bias for speech commands.
#
#   2. Separable convolutions: depthwise (DW) + pointwise (PW) reduce
#      parameter count vs. standard convolutions.
#
#   3. Scaling factor "s": BC-ResNet-s has s stacked BC-blocks.
#      BC-ResNet-8 uses s=8 → 8 BC-ResNet blocks after the stem.
#
#   4. Frequency sub-sampling: strides in the frequency dimension only,
#      preserving full time resolution for temporal modeling.
# ---------------------------------------------------------------------------


class SubSpectralNorm(nn.Module):
    """
    Sub-Spectral Normalization (SSN) from the BC-ResNet paper.

    Divides the frequency axis into S sub-bands and applies
    GroupNorm independently within each sub-band. This allows
    the model to normalize statistics per frequency region,
    which is more appropriate than global BN for speech features.
    """

    def __init__(self, channels: int, sub_bands: int = 5) -> None:
        super().__init__()
        # GroupNorm with num_groups = sub_bands operates on the frequency axis
        # (channels here represents the flattened freq * ch dimension)
        self.norm = nn.GroupNorm(num_groups=sub_bands, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        return self.norm(x)


class BroadcastedResidualBlock(nn.Module):
    """
    Core BC-ResNet block.

    Two-branch structure:
        Branch 1 (full 2D path): captures local time-frequency patterns
            → depthwise conv 3×3 (frequency × time) + pointwise 1×1
        Branch 2 (broadcast path): captures global frequency patterns
            → average pool over time → 1D conv on frequency → broadcast back

    The broadcasted shortcut ensures that low-frequency global structure
    (pitch, formants) is preserved even as the 2D path learns local details.

    Args:
        in_channels:  input feature map channels
        out_channels: output channels
        stride_f:     stride along frequency axis (1 or 2 for sub-sampling)
        dropout:      dropout probability applied before residual add
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_f: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- Branch 1: 2D depthwise-separable conv ---
        # Depthwise: operates on each channel independently, kernel 3×3
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(stride_f, 1),   # sub-sample frequency, preserve time
            padding=(1, 1),
            groups=in_channels,     # depthwise
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(in_channels)

        # Pointwise: 1×1 conv mixes channels
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # --- Branch 2: Broadcasted 1D path ---
        # Pool time → shape becomes (B, C, F, 1) → apply 1D conv along F
        self.bc_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 1),
            stride=(stride_f, 1),
            padding=(1, 0),
            bias=False,
        )
        self.norm_bc = nn.BatchNorm2d(out_channels)

        # --- Shortcut projection (if dimensions change) ---
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride_f != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride_f, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.dropout = nn.Dropout2d(p=dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, F, T)

        # Branch 1: 2D path
        out_2d = self.act(self.norm1(self.dw_conv(x)))
        out_2d = self.norm2(self.pw_conv(out_2d))  # (B, C_out, F', T)

        # Branch 2: broadcast path
        # Average over time to get frequency profile (B, C_in, F, 1)
        bc = x.mean(dim=-1, keepdim=True)
        bc = self.act(self.norm_bc(self.bc_conv(bc)))  # (B, C_out, F', 1)
        # Broadcast back to full time dimension
        bc = bc.expand_as(out_2d)                      # (B, C_out, F', T)

        # Combine branches
        out = out_2d + bc
        out = self.dropout(out)
        out = out + self.shortcut(x)
        return self.act(out)


class BCResNet8(nn.Module):
    """
    BC-ResNet-8 for 4-class keyword spotting.

    Architecture (following Table 1 from the paper):
        Stem    : Conv2d 3×3, 16 channels
        Stage 1 : 2 × BC-block, 16 ch, no freq sub-sampling
        Stage 2 : 2 × BC-block, 32 ch, stride_f=2 on first block
        Stage 3 : 2 × BC-block, 48 ch, stride_f=2 on first block
        Stage 4 : 2 × BC-block, 64 ch, stride_f=2 on first block
        Head    : Global average pool → FC → softmax

    Total: 8 BC-ResNet blocks → "BC-ResNet-8"

    Input tensor shape: (B, 1, N_MFCC=40, T=101)
    Output tensor shape: (B, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.1) -> None:
        super().__init__()

        # Stem: initial feature extraction
        # Converts (B, 1, 40, T) → (B, 16, 40, T)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 2 blocks, 16 channels, no sub-sampling
        # (B, 16, 40, T) → (B, 16, 40, T)
        self.stage1 = nn.Sequential(
            BroadcastedResidualBlock(16, 16, stride_f=1, dropout=dropout),
            BroadcastedResidualBlock(16, 16, stride_f=1, dropout=dropout),
        )

        # Stage 2: 2 blocks, 32 channels, stride_f=2 on first
        # (B, 16, 40, T) → (B, 32, 20, T)
        self.stage2 = nn.Sequential(
            BroadcastedResidualBlock(16, 32, stride_f=2, dropout=dropout),
            BroadcastedResidualBlock(32, 32, stride_f=1, dropout=dropout),
        )

        # Stage 3: 2 blocks, 48 channels, stride_f=2 on first
        # (B, 32, 20, T) → (B, 48, 10, T)
        self.stage3 = nn.Sequential(
            BroadcastedResidualBlock(32, 48, stride_f=2, dropout=dropout),
            BroadcastedResidualBlock(48, 48, stride_f=1, dropout=dropout),
        )

        # Stage 4: 2 blocks, 64 channels, stride_f=2 on first
        # (B, 48, 10, T) → (B, 64, 5, T)
        self.stage4 = nn.Sequential(
            BroadcastedResidualBlock(48, 64, stride_f=2, dropout=dropout),
            BroadcastedResidualBlock(64, 64, stride_f=1, dropout=dropout),
        )

        # Classification head: global average pool → linear
        # (B, 64, 5, T) → (B, 64) → (B, num_classes)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        x = self.stem(x)    # (B, 16, F, T)
        x = self.stage1(x)  # (B, 16, F, T)
        x = self.stage2(x)  # (B, 32, F/2, T)
        x = self.stage3(x)  # (B, 48, F/4, T)
        x = self.stage4(x)  # (B, 64, F/8, T)
        x = self.head(x)    # (B, num_classes)
        return x

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, macro_f1)."""
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for mfcc, labels in loader:
        mfcc = mfcc.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(mfcc)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataloader. Returns (avg_loss, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for mfcc, labels in loader:
        mfcc = mfcc.to(device)
        labels = labels.to(device)

        logits = model(mfcc)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "Машина",
    "Самый малый вперёд",
    "Самый малый назад",
    "Приготовить машину",
]


@torch.no_grad()
def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    """
    Run full evaluation on test split:
    - macro F1 and weighted F1
    - per-class classification report
    - confusion matrix (logged as text)
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for mfcc, labels in loader:
        mfcc = mfcc.to(device)
        logits = model(mfcc)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    logger.info("=" * 60)
    logger.info(f"TEST RESULTS")
    logger.info(f"  Macro F1    : {macro_f1:.4f}")
    logger.info(f"  Weighted F1 : {weighted_f1:.4f}")
    logger.info("=" * 60)

    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0
    )
    logger.info("\nClassification Report:\n" + report)

    cm = confusion_matrix(all_labels, all_preds)
    logger.info("Confusion Matrix (rows=true, cols=pred):")
    header = "           " + "  ".join(f"{n[:6]:>6}" for n in CLASS_NAMES)
    logger.info(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        logger.info(f"  {CLASS_NAMES[i][:10]:>10} {row_str}")

    # Save metrics to file
    metrics_path = output_dir / "test_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    logger.info(f"Metrics saved → {metrics_path}")


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(
    model: nn.Module,
    wav_path: str,
    n_runs: int = 200,
) -> None:
    """
    Measure inference latency on CPU for a single 1-second WAV file.

    Protocol:
        - Model forced to CPU, eval mode
        - 20 warm-up runs (excluded from stats) to stabilize CPU frequency
        - 200 timed runs, each measuring end-to-end:
            load WAV → MFCC → model.forward → argmax
        - Reports: median, p95, p99 in milliseconds
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Build MFCC transform (same params as training)
    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "n_mels": 80,
            "f_min": 0.0,
            "f_max": SAMPLE_RATE / 2,
        },
    )

    # Pre-load WAV (file I/O excluded from latency measurement)
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    target_len = SAMPLE_RATE
    if waveform.shape[-1] < target_len:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
    else:
        waveform = waveform[:, :target_len]

    latencies_ms: list[float] = []
    total_runs = 20 + n_runs  # 20 warm-up + n_runs measured

    with torch.no_grad():
        for i in range(total_runs):
            t0 = time.perf_counter()

            # Full pipeline: MFCC + normalize + forward
            mfcc = mfcc_transform(waveform)
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
            mfcc = mfcc.unsqueeze(0)        # add batch dim: (1, 1, F, T)
            logits = model(mfcc)
            _ = logits.argmax(dim=1).item()

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            if i >= 20:  # skip warm-up
                latencies_ms.append(elapsed_ms)

    arr = np.array(latencies_ms)
    logger.info("=" * 60)
    logger.info(f"LATENCY BENCHMARK (n={n_runs}, CPU only)")
    logger.info(f"  Median : {np.median(arr):.2f} ms")
    logger.info(f"  Mean   : {np.mean(arr):.2f} ms")
    logger.info(f"  Std    : {np.std(arr):.2f} ms")
    logger.info(f"  P95    : {np.percentile(arr, 95):.2f} ms")
    logger.info(f"  P99    : {np.percentile(arr, 99):.2f} ms")
    logger.info(f"  Min    : {np.min(arr):.2f} ms")
    logger.info(f"  Max    : {np.max(arr):.2f} ms")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BC-ResNet-8 for ship bridge voice command classification"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to dataset CSV (columns: filepath, label, split)",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Root directory containing WAV files (filepaths in CSV are relative to this)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/bcresnet8",
        help="Output directory for checkpoints and metrics",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience (epochs without val F1 improvement)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability in BC-ResNet blocks",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--benchmark_wav",
        type=str,
        default=None,
        help="Path to a single WAV file for latency benchmarking (optional)",
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=200,
        help="Number of inference runs for latency benchmark",
    )
    parser.add_argument(
        "--benchmark_only",
        action="store_true",
        help="Skip training, only run latency benchmark (requires --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to load (for benchmark_only or resume)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build model
    model = BCResNet8(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)
    logger.info(f"BC-ResNet-8 parameters: {model.count_parameters():,}")

    # Load checkpoint if requested
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Benchmark-only mode
    if args.benchmark_only:
        if args.benchmark_wav is None:
            raise ValueError("--benchmark_wav is required when using --benchmark_only")
        benchmark_latency(model, args.benchmark_wav, n_runs=args.benchmark_runs)
        return

    # ----- Data -----
    train_ds = VoiceCommandDataset(args.csv, args.wav_dir, split="train")
    val_ds = VoiceCommandDataset(args.csv, args.wav_dir, split="val")
    test_ds = VoiceCommandDataset(args.csv, args.wav_dir, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ----- Optimizer & scheduler -----
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Cosine annealing: smoothly reduces LR to near-zero over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # ----- Training loop -----
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    best_ckpt_path = output_dir / "best_model.pt"

    logger.info(f"Starting training for up to {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            logger.info(f"  ✓ New best val F1={val_f1:.4f} → checkpoint saved")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement for {epochs_without_improvement}/{args.patience} epochs"
            )

        # Early stopping
        if epochs_without_improvement >= args.patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {args.patience} consecutive epochs)"
            )
            break

    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")

    # ----- Test evaluation (load best checkpoint) -----
    logger.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    full_evaluation(model, test_loader, device, output_dir)

    # ----- Optional latency benchmark -----
    if args.benchmark_wav:
        benchmark_latency(model, args.benchmark_wav, n_runs=args.benchmark_runs)


if __name__ == "__main__":
    main()
