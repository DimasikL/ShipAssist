"""
train_bcresnet8.py — BC-ResNet-8 baseline for ship bridge voice command classification.

Architecture reference:
    "Broadcasted Residual Learning for Efficient Keyword Spotting"
    Byeonggeun Kim et al., Interspeech 2021  https://arxiv.org/abs/2106.04140

Usage
-----
    # Training (from project root)
    python scripts/train/train_bcresnet8.py --data_csv dset_meta_only_2026-06-04_08-20-24.csv

    # Training with specific groups (Windows CMD — use double quotes for names with spaces)
    python scripts/train/train_bcresnet8.py ^
        --data_csv dset_meta_only_2026-06-04_08-20-24.csv ^
        --train_groups gtts gtts-aug gtts-drug gtts-drug-aug ^
            "new user 10" "new user 10-aug" ^
            "new user 11" "new user 11-aug" ^
            "new user 12" "new user 12-aug" ^
            "new user 13" "new user 13-aug" ^
            "new user 14" "new user 14-aug" ^
            "new user 15" "new user 15-aug" ^
            silero silero-aug silero-drug silero-drug-aug

    # Latency benchmark only (requires existing checkpoint)
    python scripts/train/train_bcresnet8.py ^
        --data_csv dset_meta_only_2026-06-04_08-20-24.csv ^
        --benchmark_only ^
        --checkpoint artifacts/benchmarks/bcresnet8/best_model.pt ^
        --benchmark_wav path/to/sample.wav

Output
------
    artifacts/benchmarks/bcresnet8/best_model.pt  -- best checkpoint by val macro F1
    artifacts/benchmarks/bcresnet8_results.json   -- clean test metrics
    artifacts/benchmarks/bcresnet8_noisy_results.json  -- noisy test metrics (if --noisy_test)

Classes (4):
    Determined dynamically from sorted(df["class"].unique()).
    Typical order: "Машина", "Приготовить машину", "Самый малый вперёд", "Самый малый назад"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root — scripts/train/ is 2 levels below project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.trainer_utils import EMA, FocalLoss  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # Hz — all WAVs are already 16 kHz
AUDIO_DURATION_SEC = 1.0      # seconds → 16 000 samples
WIN_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION_SEC)

# MFCC parameters for BC-ResNet (match paper: 40 coefficients, 10 ms hop, 25 ms window)
N_MFCC = 40
HOP_LENGTH = 160              # 10 ms hop at 16 kHz
N_FFT = 400                   # 25 ms window at 16 kHz

SEED = 42

# ---------------------------------------------------------------------------
# Split definitions (mirrors core/config.py SplitsConfig)
# ---------------------------------------------------------------------------

TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
    "drug slova4",
]

VAL_GROUPS = [
    "train user 3",
    "drug slova3",
]


# ---------------------------------------------------------------------------
# Path helper (handles absolute Windows paths in CSV when running on any OS)
# ---------------------------------------------------------------------------

def _fix_path(p: str) -> Path:
    """Normalise a path stored in CSV to the current filesystem layout.

    The CSV may contain absolute Windows paths like
    ``C:/Users/Dmitriy/PycharmProjects/ShipAssistant/artifacts/...``.
    This function strips the known Windows prefix and re-anchors the
    relative tail to the actual project root, so the script works both
    on the Windows dev machine and on a Linux training server.
    """
    p = p.replace("\\", "/")
    win_root = "C:/Users/Dmitriy/PycharmProjects/ShipAssistant"
    if win_root in p:
        rel = p.split(win_root, 1)[1].lstrip("/")
        return _PROJECT_ROOT / rel
    return Path(p)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VoiceCommandDataset(Dataset):
    """
    Loads WAV files from a metadata CSV and returns MFCC feature tensors.

    CSV schema (subset used):
        audio_path   - absolute or relative path to WAV file
        class        - string class label (e.g. "Машина")
        audio_group  - speaker/session group used to define train/val/test splits

    Split logic:
        test  → audio_group in TEST_GROUPS
        val   → audio_group in VAL_GROUPS
        train → everything else
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label2id: dict[str, int],
        training: bool = False,
        snr_db: float | None = None,
    ) -> None:
        """
        Args:
            df:        DataFrame already filtered to the desired split.
            label2id:  Mapping from class string to integer index.
            training:  True for the train split — enables maritime noise aug.
            snr_db:    If set, always add Gaussian noise at this SNR (for noisy eval).
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.training = training
        self.snr_db = snr_db

        # MFCC transform — runs on CPU, output shape (1, N_MFCC, T)
        # T = ceil(WIN_SAMPLES / HOP_LENGTH) ≈ 101 frames for 1 s audio
        self.mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                "hop_length": HOP_LENGTH,
                "n_fft": N_FFT,
                "n_mels": 80,       # intermediate mel bins before DCT
                "f_min": 0.0,
                "f_max": SAMPLE_RATE / 2,
            },
        )

    def _load_wav(self, path: str) -> np.ndarray:
        """Load WAV via librosa, pad/trim to WIN_SAMPLES."""
        fixed = _fix_path(path)
        try:
            wav, _ = librosa.load(str(fixed), sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            logger.warning("Could not load %s: %s — using silence.", fixed, exc)
            return np.zeros(WIN_SAMPLES, dtype=np.float32)
        # Pad zeros or trim to exactly 1 second
        if len(wav) < WIN_SAMPLES:
            wav = np.pad(wav, (0, WIN_SAMPLES - len(wav)))
        else:
            wav = wav[:WIN_SAMPLES]
        return wav.astype(np.float32)

    @staticmethod
    def _add_gaussian_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
        """Add white Gaussian noise at the requested SNR level."""
        signal_power = float(np.mean(wav ** 2)) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(wav)).astype(np.float32) * float(np.sqrt(noise_power))
        return wav + noise

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        wav = self._load_wav(row["audio_path"])

        # Deterministic SNR noise for noisy-eval benchmark
        if self.snr_db is not None:
            wav = self._add_gaussian_noise(wav, self.snr_db)

        # Convert numpy → tensor, add channel dim: (1, WIN_SAMPLES)
        wav_t = torch.from_numpy(wav).unsqueeze(0)

        # MFCC: (1, N_MFCC, T)
        mfcc = self.mfcc_transform(wav_t)

        # Per-sample normalisation (zero mean, unit variance)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        label = self.label2id[row["class"]]
        return mfcc, label


# ---------------------------------------------------------------------------
# BC-ResNet-8 Architecture
# ---------------------------------------------------------------------------
#
# Design principles:
#
#   1. "Broadcasted" shortcuts: the residual path is a 1D (frequency-only)
#      vector broadcast across the time axis, encoding a global frequency
#      profile. This imposes a time-invariance inductive bias — useful for
#      isolated speech commands whose identity lives in spectral shape.
#
#   2. Depthwise-separable 2D path: captures local time-frequency patterns
#      at a fraction of the parameters of a standard conv.
#
#   3. Frequency sub-sampling via stride along F only — time resolution is
#      preserved throughout, which keeps temporal boundaries sharp.
#
#   4. BC-ResNet-s uses s stacked BC-blocks; s=8 is this variant.
# ---------------------------------------------------------------------------


class BroadcastedResidualBlock(nn.Module):
    """
    BC-ResNet building block.

    Two parallel branches:
        • 2D branch  : DW-conv 3×3 → PW-conv 1×1   (local time-freq patterns)
        • BC branch  : avg-pool over time → 1D conv → broadcast back to full T
                       (global frequency profile, time-invariant)

    The time-averaged shortcut forces the model to disentangle "what command"
    (frequency structure, BC path) from "when" (local 2D path).

    Args:
        in_channels:  input channels C_in
        out_channels: output channels C_out
        stride_f:     stride along frequency axis (1 = keep, 2 = halve)
        dropout:      Dropout2d probability applied before the residual add
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_f: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # 2D path: depthwise conv (per-channel) + pointwise mix
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(3, 3),
            stride=(stride_f, 1),   # sub-sample freq only
            padding=(1, 1),
            groups=in_channels,     # depthwise = one filter per channel
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # BC path: global avg over time → 1D conv along freq → broadcast
        self.bc_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(3, 1),
            stride=(stride_f, 1),
            padding=(1, 0),
            bias=False,
        )
        self.norm_bc = nn.BatchNorm2d(out_channels)

        # Shortcut projection when shape changes
        self.shortcut: nn.Module
        if in_channels != out_channels or stride_f != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride_f, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout2d(p=dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, F, T)

        # 2D branch: local time-frequency features
        out_2d = self.act(self.norm1(self.dw_conv(x)))
        out_2d = self.norm2(self.pw_conv(out_2d))        # (B, C_out, F', T)

        # BC branch: time-averaged frequency profile, broadcast back to T
        bc = x.mean(dim=-1, keepdim=True)                # (B, C_in, F, 1)
        bc = self.act(self.norm_bc(self.bc_conv(bc)))    # (B, C_out, F', 1)
        bc = bc.expand_as(out_2d)                        # (B, C_out, F', T)

        out = self.dropout(out_2d + bc)
        return self.act(out + self.shortcut(x))


class BCResNet8(nn.Module):
    """
    BC-ResNet-8: 8 BC-blocks in 4 stages for 4-class keyword spotting.

    Stage layout (Table 1 from the paper):
        Stem    : Conv2d 3×3 → 16 ch
        Stage 1 : 2 × BC-block, 16 ch,  stride_f=1  — (B, 16, 40, T)
        Stage 2 : 2 × BC-block, 32 ch,  stride_f=2  — (B, 32, 20, T)
        Stage 3 : 2 × BC-block, 48 ch,  stride_f=2  — (B, 48, 10, T)
        Stage 4 : 2 × BC-block, 64 ch,  stride_f=2  — (B, 64,  5, T)
        Head    : Global avg pool → Dropout → Linear

    Input : (B, 1, N_MFCC=40, T≈101)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 8 BC-ResNet blocks = BC-ResNet-8
        self.stage1 = nn.Sequential(
            BroadcastedResidualBlock(16, 16, stride_f=1, dropout=dropout),
            BroadcastedResidualBlock(16, 16, stride_f=1, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            BroadcastedResidualBlock(16, 32, stride_f=2, dropout=dropout),
            BroadcastedResidualBlock(32, 32, stride_f=1, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            BroadcastedResidualBlock(32, 48, stride_f=2, dropout=dropout),
            BroadcastedResidualBlock(48, 48, stride_f=1, dropout=dropout),
        )
        self.stage4 = nn.Sequential(
            BroadcastedResidualBlock(48, 64, stride_f=2, dropout=dropout),
            BroadcastedResidualBlock(64, 64, stride_f=1, dropout=dropout),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

    def count_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training helpers
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
    ema: EMA | None,
) -> Tuple[float, float]:
    """One training epoch with optional EMA weight update.

    Returns:
        Tuple of (avg_loss, macro_f1).
    """
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for mfcc, labels in tqdm(loader, desc="  train", leave=False):
        mfcc = mfcc.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(mfcc)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents instability on small batches
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if ema is not None:
            ema.update()

        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataloader.

    Returns:
        Tuple of (avg_loss, macro_f1).
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for mfcc, labels in tqdm(loader, desc="  eval ", leave=False):
        mfcc = mfcc.to(device)
        labels = labels.to(device)
        logits = model(mfcc)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: list[str],
    out_dir: Path,
    tag: str = "clean",
    snr_db: float | None = None,
) -> dict:
    """Full test-split evaluation: F1, classification report, confusion matrix.

    Results are saved as JSON to out_dir.

    Returns:
        Dict with macro_f1, weighted_f1, and classification_report string.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for mfcc, labels in tqdm(loader, desc=f"  test ({tag})"):
        mfcc = mfcc.to(device)
        all_preds.extend(model(mfcc).argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())

    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report_str  = classification_report(all_labels, all_preds,
                                        target_names=label_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info("=" * 60)
    logger.info(f"TEST RESULTS ({tag})")
    logger.info(f"  Macro F1    : {macro_f1:.4f}")
    logger.info(f"  Weighted F1 : {weighted_f1:.4f}")
    logger.info("=" * 60)
    logger.info("\nClassification Report:\n%s", report_str)
    logger.info("Confusion Matrix (rows=true, cols=pred):\n%s", str(cm))

    results = {
        "method": "BC-ResNet-8",
        "test_type": tag if snr_db is None else f"noisy_snr{snr_db:.0f}dB",
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_samples": len(all_labels),
        "n_classes": len(label_names),
        "label_names": label_names,
        "classification_report": report_str,
        "confusion_matrix": cm.tolist(),
    }
    if snr_db is not None:
        results["snr_db"] = snr_db

    suffix = "" if snr_db is None else f"_noisy"
    out_path = out_dir / f"bcresnet8_results{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Results saved → %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(
    model: nn.Module,
    wav_path: str,
    n_runs: int = 200,
) -> None:
    """Measure end-to-end CPU inference latency for a single 1-second WAV.

    Protocol:
        - Model on CPU, eval mode
        - 20 warm-up runs (excluded from stats) to stabilise CPU clock
        - n_runs timed runs: librosa load → MFCC → forward → argmax
        - Reports: median, mean, std, p95, p99 in milliseconds

    This exactly mirrors the inference path used at runtime.
    """
    device = torch.device("cpu")
    model = model.to(device).eval()

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

    latencies_ms: list[float] = []

    with torch.no_grad():
        for i in range(20 + n_runs):
            t0 = time.perf_counter()

            # Full pipeline including audio loading (as in production)
            wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            if len(wav) < WIN_SAMPLES:
                wav = np.pad(wav, (0, WIN_SAMPLES - len(wav)))
            else:
                wav = wav[:WIN_SAMPLES]
            wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)

            mfcc = mfcc_transform(wav_t)
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
            mfcc = mfcc.unsqueeze(0)   # (1, 1, F, T)

            _ = model(mfcc).argmax(dim=1).item()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if i >= 20:   # skip warm-up
                latencies_ms.append(elapsed_ms)

    arr = np.array(latencies_ms)
    logger.info("=" * 60)
    logger.info(f"LATENCY BENCHMARK  (n={n_runs}, CPU only)")
    logger.info(f"  Median : {np.median(arr):.2f} ms")
    logger.info(f"  Mean   : {np.mean(arr):.2f} ms")
    logger.info(f"  Std    : {np.std(arr):.2f} ms")
    logger.info(f"  P95    : {np.percentile(arr, 95):.2f} ms")
    logger.info(f"  P99    : {np.percentile(arr, 99):.2f} ms")
    logger.info(f"  Min    : {np.min(arr):.2f} ms")
    logger.info(f"  Max    : {np.max(arr):.2f} ms")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BC-ResNet-8 on ship bridge voice commands."
    )
    p.add_argument(
        "--data_csv",
        required=True,
        help="Path to dataset CSV (audio_path, class, audio_group columns required). "
             "Relative paths are resolved from project root.",
    )
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10,
                   help="Early stopping patience (epochs without val F1 improvement).")
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument(
        "--no_balance",
        action="store_true",
        help="Disable WeightedRandomSampler (class balancing on by default).",
    )
    p.add_argument(
        "--min_val_samples",
        type=int,
        default=200,
        help=(
            "If val split has fewer than this many samples, supplement it by "
            "randomly holding out --val_fraction of the train set. Default: 200."
        ),
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of train to use as supplemental val when val set is too small. Default: 0.1.",
    )
    p.add_argument("--use_ema",      action="store_true", default=True,
                   help="Use EMA weights during validation (default: on).")
    p.add_argument("--no_ema",       dest="use_ema", action="store_false",
                   help="Disable EMA.")
    p.add_argument(
        "--noisy_test",
        action="store_true",
        help="Also run test evaluation with Gaussian noise at --snr_db.",
    )
    p.add_argument("--snr_db",       type=float, default=12.0,
                   help="SNR for noisy test evaluation (dB).")
    p.add_argument(
        "--benchmark_wav",
        default=None,
        help="Path to a WAV file for latency benchmarking (optional).",
    )
    p.add_argument("--benchmark_runs", type=int, default=200)
    p.add_argument(
        "--benchmark_only",
        action="store_true",
        help="Skip training; only run latency benchmark. Requires --checkpoint.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .pt checkpoint to load (for --benchmark_only or to resume).",
    )
    p.add_argument(
        "--train_groups",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Whitelist of audio_group values to include in training. "
            "When omitted, all non-test/non-val groups are used. "
            "Example: --train_groups gtts gtts-aug silero 'new user 10'"
        ),
    )
    p.add_argument(
        "--val_groups",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Whitelist of audio_group values to use as the validation set. "
            "These groups are excluded from training. Overrides the hardcoded "
            "VAL_GROUPS constant. Useful for routing real-user groups to val "
            "so that early stopping reflects out-of-domain performance. "
            "Example: --val_groups 'train user 3' 'drug slova3' 'new user 9'"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(SEED)

    # Resolve CSV path (support both absolute and project-root-relative)
    csv_path = Path(args.data_csv)
    if not csv_path.is_absolute():
        csv_path = _PROJECT_ROOT / csv_path
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    out_dir = _PROJECT_ROOT / "artifacts" / "benchmarks"
    ckpt_dir = out_dir / "bcresnet8"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --------------- Load CSV and build label mapping -----------------------
    df = pd.read_csv(csv_path)
    for col in ("audio_path", "class", "audio_group"):
        if col not in df.columns:
            logger.error("CSV is missing required column: '%s'", col)
            sys.exit(1)

    from sklearn.model_selection import train_test_split

    # Determine splits by audio_group (mirrors eval_onnx_model.py convention)
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)

    # --val_groups overrides the hardcoded VAL_GROUPS constant.
    # Use it to route real-user groups into val so that early stopping
    # reflects out-of-domain (real-speech) performance rather than TTS.
    effective_val_groups: list[str] = args.val_groups if args.val_groups else VAL_GROUPS
    if args.val_groups:
        unknown_val = set(args.val_groups) - set(df["audio_group"].unique())
        if unknown_val:
            logger.warning("--val_groups: these groups not found in CSV: %s", sorted(unknown_val))
        logger.info("--val_groups applied: %s", args.val_groups)

    val_df   = df[df["audio_group"].isin(effective_val_groups)].reset_index(drop=True)
    train_df = df[~df["audio_group"].isin(TEST_GROUPS + effective_val_groups)].reset_index(drop=True)

    # Optional whitelist: restrict training to specific audio_groups.
    # Groups excluded here but not in test/val become a "held-out pool" of
    # real-user data — used below to supplement val before falling back to
    # splitting from TTS train data.
    excluded_pool_df = pd.DataFrame()
    if args.train_groups:
        before = len(train_df)
        mask = train_df["audio_group"].isin(args.train_groups)
        excluded_pool_df = train_df[~mask].reset_index(drop=True)  # real users not in train
        train_df = train_df[mask].reset_index(drop=True)
        unknown = set(args.train_groups) - set(df["audio_group"].unique())
        if unknown:
            logger.warning("--train_groups: these groups not found in CSV: %s", sorted(unknown))
        logger.info(
            "train_groups filter applied: %d → %d samples  (groups: %s)",
            before, len(train_df), args.train_groups,
        )
        if len(excluded_pool_df) > 0:
            logger.info(
                "Excluded-from-train pool: %d samples across groups %s "
                "(used to supplement val before falling back to TTS train split).",
                len(excluded_pool_df),
                sorted(excluded_pool_df["audio_group"].unique().tolist()),
            )

    label_list: list[str] = sorted(df["class"].unique().tolist())
    label2id: dict[str, int] = {lbl: i for i, lbl in enumerate(label_list)}
    num_classes = len(label_list)

    # ---- Supplement val when it is too small --------------------------------
    # Priority 1: use excluded real-user groups (groups removed from train via
    #             --train_groups but not assigned to test or val).  These are
    #             closer in distribution to the test set than TTS-generated data.
    # Priority 2: fall back to a stratified fraction of train (TTS/aug data).
    # Both paths are additive: we keep going until val >= min_val_samples.
    if len(val_df) < args.min_val_samples:
        logger.warning(
            "Val set has only %d samples (< min_val_samples=%d).",
            len(val_df), args.min_val_samples,
        )

        # Priority 1 — excluded real-user pool
        if len(excluded_pool_df) > 0:
            val_df = pd.concat([val_df, excluded_pool_df], ignore_index=True)
            logger.info(
                "Supplemented val with %d real-user samples from excluded pool → %d total.",
                len(excluded_pool_df), len(val_df),
            )

        # Priority 2 — split from TTS train if still insufficient
        if len(val_df) < args.min_val_samples:
            logger.warning(
                "Val still has only %d samples after adding excluded pool. "
                "Holding out %.0f%% of train (TTS data) as supplemental val.",
                len(val_df), args.val_fraction * 100,
            )
            try:
                train_df, extra_val_df = train_test_split(
                    train_df,
                    test_size=args.val_fraction,
                    stratify=train_df["class"],
                    random_state=SEED,
                )
            except ValueError:
                train_df, extra_val_df = train_test_split(
                    train_df, test_size=args.val_fraction, random_state=SEED,
                )
            val_df = pd.concat([val_df, extra_val_df], ignore_index=True)
            logger.info(
                "Val set expanded with TTS split: %d samples total.", len(val_df)
            )
    logger.info("Val set: %d samples", len(val_df))

    logger.info("Labels (%d): %s", num_classes, label_list)
    logger.info(
        "Splits — train: %d  val: %d  test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Val groups: %s",
        sorted(val_df["audio_group"].unique().tolist()) if len(val_df) > 0 else "[]",
    )

    # --------------- Build model --------------------------------------------
    model = BCResNet8(num_classes=num_classes, dropout=args.dropout).to(device)
    logger.info("BC-ResNet-8 parameters: %s", f"{model.count_parameters():,}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    # --------------- Benchmark-only mode ------------------------------------
    if args.benchmark_only:
        if args.benchmark_wav is None:
            logger.error("--benchmark_wav is required with --benchmark_only")
            sys.exit(1)
        benchmark_latency(model, args.benchmark_wav, n_runs=args.benchmark_runs)
        return

    # --------------- Datasets & loaders ------------------------------------
    train_ds = VoiceCommandDataset(train_df, label2id, training=True)
    val_ds   = VoiceCommandDataset(val_df,   label2id, training=False)
    test_ds  = VoiceCommandDataset(test_df,  label2id, training=False)

    _loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # WeightedRandomSampler: each class gets equal expected frequency per batch.
    # Prevents collapse to majority class (e.g. "другие слова" dominating).
    if not args.no_balance:
        train_labels = train_df["class"].map(label2id).tolist()
        class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        class_weight = 1.0 / np.maximum(class_counts, 1)
        sample_weights = torch.tensor([class_weight[l] for l in train_labels], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info(
            "WeightedRandomSampler enabled. Class counts: %s",
            {lbl: int(class_counts[i]) for lbl, i in label2id.items()},
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **_loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **_loader_kwargs)

    val_loader   = DataLoader(val_ds,   shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **_loader_kwargs)

    # --------------- Optimizer & loss ---------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine annealing: gradually reduces LR to near-zero, avoids sharp LR drops
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    # Per-class weights (inverse frequency, normalised so mean weight = 1.0).
    # Passed as FocalLoss alpha — penalises errors on rare classes more heavily.
    # This is especially important for "машина" which is the smallest command class.
    train_class_ids = train_df["class"].map(label2id).values
    class_counts = np.bincount(train_class_ids, minlength=num_classes).astype(float)
    raw_weights = 1.0 / np.maximum(class_counts, 1)
    focal_alpha = torch.tensor(
        raw_weights / raw_weights.mean(), dtype=torch.float
    ).to(device)
    logger.info(
        "FocalLoss alpha (class weights): %s",
        {lbl: f"{focal_alpha[i].item():.3f}" for lbl, i in label2id.items()},
    )
    criterion = FocalLoss(gamma=2.0, alpha=focal_alpha)

    ema = EMA(model, decay=0.999) if args.use_ema else None

    # --------------- Training loop -----------------------------------------
    best_val_f1 = 0.0
    patience_counter = 0
    best_ckpt_path = ckpt_dir / "best_model.pt"

    logger.info("Starting training (max %d epochs, patience=%d) ...", args.epochs, args.patience)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, ema
        )

        # Evaluate with EMA weights if enabled
        if ema is not None:
            ema.apply_shadow()
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        if ema is not None:
            ema.restore()

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %02d/%d | train_loss=%.4f f1=%.4f | val_loss=%.4f f1=%.4f | lr=%.2e",
            epoch, args.epochs, train_loss, train_f1, val_loss, val_f1, lr_now,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save model in eval state (EMA weights if available)
            if ema is not None:
                ema.apply_shadow()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "label2id": label2id,
                    "num_classes": num_classes,
                },
                best_ckpt_path,
            )
            if ema is not None:
                ema.restore()
            logger.info("  ✓ Best val F1=%.4f — checkpoint saved", val_f1)
        else:
            patience_counter += 1
            logger.info("  No improvement for %d/%d epochs", patience_counter, args.patience)

        if patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs).",
                epoch, args.patience,
            )
            break

    logger.info("Training complete. Best val macro F1: %.4f", best_val_f1)

    # --------------- Test evaluation (load best checkpoint) -----------------
    logger.info("Loading best checkpoint for test evaluation ...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    full_evaluation(model, test_loader, device, label_list, out_dir, tag="clean")

    if args.noisy_test:
        test_noisy_ds = VoiceCommandDataset(
            test_df, label2id, training=False, snr_db=args.snr_db
        )
        test_noisy_loader = DataLoader(test_noisy_ds, shuffle=False, **_loader_kwargs)
        full_evaluation(
            model, test_noisy_loader, device, label_list, out_dir,
            tag=f"noisy_snr{args.snr_db:.0f}dB", snr_db=args.snr_db,
        )

    # --------------- Optional latency benchmark ----------------------------
    if args.benchmark_wav:
        benchmark_latency(model, args.benchmark_wav, n_runs=args.benchmark_runs)


if __name__ == "__main__":
    main()
