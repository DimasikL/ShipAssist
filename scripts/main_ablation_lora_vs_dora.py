#!/usr/bin/env python3
"""scripts/ablation_lora_vs_dora.py

Ablation study: LoRA vs DoRA fine-tuning of Wav2Vec2-XLSR-53
for 4-class voice command classification.

Trains each method for 20 epochs across 3 random seeds, records:
  - val macro-F1 per epoch (convergence curve)
  - final macro-F1 / weighted-F1 / accuracy on test split
  - mean epoch wall-clock time (GPU)
  - number of trainable parameters

Outputs:
  artifacts/plots/lora_vs_dora.pdf  (convergence + bar chart with error bars)
  Console: Wilcoxon test p-value + summary table

Usage:
  python scripts/ablation_lora_vs_dora.py
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # headless rendering — must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification

# ---------------------------------------------------------------------------
# Paths — all relative to project root, never hardcoded
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts" / "plots"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation.lora_vs_dora")

# ---------------------------------------------------------------------------
# Hyperparameter config (single source of truth for both methods)
# ---------------------------------------------------------------------------
@dataclass
class AblationConfig:
    """Shared hyperparameters for the LoRA vs DoRA ablation.

    Both PEFT methods use identical rank / alpha / dropout / target_modules
    so that any performance difference is attributable solely to the
    weight-decomposed reparameterisation introduced by DoRA.

    Attributes:
        model_name: HuggingFace model identifier.
        num_labels: Number of voice-command classes.
        num_epochs: Training epochs per run.
        batch_size: DataLoader batch size.
        learning_rate: AdamW peak learning rate.
        weight_decay: AdamW L2 penalty.
        lora_r: LoRA / DoRA rank.
        lora_alpha: LoRA / DoRA scaling factor (alpha / r = 2).
        lora_dropout: Dropout probability inside adapter layers.
        target_modules: Transformer sub-modules to apply adapters to.
        seeds: Random seeds used across the three-run ablation.
        device: Torch device (auto-detected from CUDA availability).
    """
    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    num_labels: int = 4
    num_epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 2024])
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


CFG = AblationConfig()

# Type alias for the dataloader factory used by run_experiment
DataloaderFactory = Callable[[int, int], tuple[DataLoader, DataLoader, DataLoader]]


# ---------------------------------------------------------------------------
# ── 1. PEFT configuration factory ──────────────────────────────────────────
# ---------------------------------------------------------------------------
def get_peft_config(method: str) -> LoraConfig:
    """Return a LoraConfig for the requested PEFT method.

    Both methods share identical rank / alpha / dropout / target_modules
    from :data:`CFG`. DoRA is enabled by setting ``use_dora=True``
    (requires ``peft >= 0.9.0``).

    Args:
        method: ``'lora'`` for standard LoRA or ``'dora'`` for
            Weight-Decomposed Low-Rank Adaptation.

    Returns:
        Fully configured :class:`peft.LoraConfig` instance.

    Raises:
        ValueError: If ``method`` is not ``'lora'`` or ``'dora'``.
    """
    if method not in {"lora", "dora"}:
        raise ValueError(f"method must be 'lora' or 'dora', got '{method!r}'")

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=CFG.target_modules,
        # classifier + projector are not in target_modules but must be unfrozen
        # so their weights update during seq-cls fine-tuning (mirrors Trainer._init_model)
        modules_to_save=["classifier", "projector"],
        bias="none",
        use_dora=(method == "dora"),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility.

    Covers Python, NumPy, PyTorch CPU/GPU, and cuDNN determinism.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count parameters that will receive gradient updates.

    Args:
        model: Any PyTorch module.

    Returns:
        Number of trainable scalar parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loading — uses existing CommandDataset / parse_metadata infrastructure
# ---------------------------------------------------------------------------
def make_dataloaders(seed: int, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders built from the project's CommandDataset.

    Splits the full CSV 70 / 15 / 15 (train / val / test) with ``random_state=seed``
    so each seed produces an independent but deterministic partition.
    Batch format is a dict with keys ``"input_values"``, ``"attention_mask"``,
    ``"labels"`` — identical to what :class:`src.data_utils.CommandDataset` yields.

    Args:
        seed: Random seed for the DataFrame sample split.
        batch_size: Samples per batch.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from src.data_utils import CommandDataset, make_balanced_sampler, parse_metadata
    from transformers import Wav2Vec2FeatureExtractor
    from core.config import settings

    df, label2id, _ = parse_metadata(str(settings.paths.dataset_csv))
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(CFG.model_name)

    # 70 / 15 / 15 deterministic split
    train_df = df.sample(frac=0.70, random_state=seed)
    remain   = df.drop(train_df.index)
    val_df   = remain.sample(frac=0.50, random_state=seed)
    test_df  = remain.drop(val_df.index)

    train_ds = CommandDataset(train_df.reset_index(drop=True), feature_extractor, label2id)
    val_ds   = CommandDataset(val_df.reset_index(drop=True),   feature_extractor, label2id)
    test_ds  = CommandDataset(test_df.reset_index(drop=True),  feature_extractor, label2id)

    sampler = make_balanced_sampler(train_df, label2id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,    num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
) -> float:
    """Run one full training pass and return the mean cross-entropy loss.

    Uses AMP (``autocast`` + ``GradScaler``) to match the existing Trainer,
    passes ``attention_mask`` to every forward call, and clips gradients at
    max_norm=1.0 for stable adapter fine-tuning.

    Batch format expected from DataLoader:
        ``{"input_values": FloatTensor(B,T), "attention_mask": LongTensor(B,T),
           "labels": LongTensor(B)}``
    — identical to what :class:`src.data_utils.CommandDataset` produces.

    Args:
        model: PEFT-wrapped :class:`Wav2Vec2ForSequenceClassification`.
        loader: Training DataLoader.
        optimizer: Configured optimizer (AdamW).
        scaler: AMP gradient scaler (no-op on CPU).
        device: Torch device string.

    Returns:
        Mean batch loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = device.startswith("cuda")

    for batch in loader:
        input_values: torch.Tensor    = batch["input_values"].to(device)
        attention_mask: torch.Tensor  = batch["attention_mask"].to(device)
        labels: torch.Tensor          = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate classification metrics over one DataLoader pass.

    Passes ``attention_mask`` to the forward call and handles the dict
    batch format from :class:`src.data_utils.CommandDataset`.

    Args:
        model: PEFT-wrapped model (set to eval mode internally).
        loader: Val or test DataLoader.
        device: Torch device string.

    Returns:
        Dict with keys ``'macro_f1'``, ``'weighted_f1'``, ``'accuracy'``.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_values: torch.Tensor   = batch["input_values"].to(device)
            attention_mask: torch.Tensor = batch["attention_mask"].to(device)
            labels: torch.Tensor         = batch["labels"]

            logits = model(input_values=input_values, attention_mask=attention_mask).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.tolist())

    return {
        "macro_f1":    float(f1_score(all_labels, all_preds, average="macro",    zero_division=0)),
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "accuracy":    float(accuracy_score(all_labels, all_preds)),
    }


# ---------------------------------------------------------------------------
# Experiment result container
# ---------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    """Metrics collected during a single (method, seed) run.

    Attributes:
        method: ``'lora'`` or ``'dora'``.
        seed: Random seed used.
        trainable_params: Count of adapter parameters.
        val_f1_per_epoch: Val macro-F1 after each training epoch.
        epoch_times_sec: GPU wall-clock training time per epoch.
        final_macro_f1: Macro-F1 on held-out test split.
        final_weighted_f1: Weighted-F1 on test split.
        final_accuracy: Classification accuracy on test split.
    """
    method: str
    seed: int
    trainable_params: int
    val_f1_per_epoch: list[float]
    epoch_times_sec: list[float]
    final_macro_f1: float
    final_weighted_f1: float
    final_accuracy: float

    @property
    def mean_epoch_time(self) -> float:
        """Mean wall-clock training time per epoch (seconds)."""
        return float(np.mean(self.epoch_times_sec))


# ---------------------------------------------------------------------------
# ── 2. Training loop ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def run_experiment(
    peft_config: LoraConfig,
    method: str,
    seed: int,
    dataloader_factory: DataloaderFactory = make_dataloaders,
) -> ExperimentResult:
    """Train a PEFT model for :data:`CFG.num_epochs` and collect ablation metrics.

    Seeds are reset at the start of every call so that runs with different
    seeds are independent yet deterministic.

    Args:
        peft_config: Config produced by :func:`get_peft_config`.
        method: Human-readable method tag (``'lora'`` or ``'dora'``).
        seed: Random seed for this run.
        dataloader_factory: Callable returning ``(train, val, test)`` loaders.

    Returns:
        Populated :class:`ExperimentResult` instance.
    """
    set_seed(seed)
    device = CFG.device
    logger.info("▶ %s | seed=%d | device=%s", method.upper(), seed, device)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = dataloader_factory(seed, CFG.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────
    base = Wav2Vec2ForSequenceClassification.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_labels,
        ignore_mismatched_sizes=True,
    )
    model = get_peft_model(base, peft_config)
    model.to(device)

    trainable_params = count_trainable_params(model)
    logger.info("  trainable params: %d", trainable_params)

    # ── Optimiser + AMP scaler + schedule ────────────────────────────────
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )
    # GradScaler is a no-op on CPU — same pattern as Trainer in src/train.py
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.num_epochs)

    # ── 20-epoch loop ──────────────────────────────────────────────────────
    val_f1_per_epoch: list[float] = []
    epoch_times: list[float] = []

    for epoch in range(1, CFG.num_epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        epoch_sec = time.perf_counter() - t0
        epoch_times.append(epoch_sec)

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        val_f1_per_epoch.append(val_f1)

        logger.info(
            "  epoch %02d/%02d | loss=%.4f | val_macro_F1=%.4f | %.1fs",
            epoch, CFG.num_epochs, train_loss, val_f1, epoch_sec,
        )

    # ── Test split evaluation ─────────────────────────────────────────────
    test_metrics = evaluate(model, test_loader, device)
    logger.info(
        "  TEST | macro_F1=%.4f | weighted_F1=%.4f | acc=%.4f",
        test_metrics["macro_f1"], test_metrics["weighted_f1"], test_metrics["accuracy"],
    )

    return ExperimentResult(
        method=method,
        seed=seed,
        trainable_params=trainable_params,
        val_f1_per_epoch=val_f1_per_epoch,
        epoch_times_sec=epoch_times,
        final_macro_f1=test_metrics["macro_f1"],
        final_weighted_f1=test_metrics["weighted_f1"],
        final_accuracy=test_metrics["accuracy"],
    )


# ---------------------------------------------------------------------------
# ── 6. Wilcoxon signed-rank test ────────────────────────────────────────────
# ---------------------------------------------------------------------------
def run_wilcoxon_test(
    lora_results: list[ExperimentResult],
    dora_results: list[ExperimentResult],
    last_n_epochs: int = 5,
) -> float:
    """Wilcoxon signed-rank test on val F1 of the final N epochs.

    Flattens ``last_n_epochs × n_seeds`` values from each method and tests
    whether the paired differences are symmetrically distributed around zero.
    A Bonferroni-style zero-difference guard avoids a degenerate input.

    Args:
        lora_results: ExperimentResult list for LoRA (one per seed).
        dora_results: ExperimentResult list for DoRA (one per seed).
        last_n_epochs: Number of trailing epochs to include per seed.

    Returns:
        Two-sided p-value from :func:`scipy.stats.wilcoxon`.
    """
    scores_a = np.array([r.val_f1_per_epoch[-last_n_epochs:] for r in lora_results]).flatten()
    scores_b = np.array([r.val_f1_per_epoch[-last_n_epochs:] for r in dora_results]).flatten()

    diff = scores_a - scores_b
    if np.all(diff == 0):
        logger.warning("All paired differences are zero — Wilcoxon test undefined (p=1.0)")
        return 1.0

    stat, p_value = wilcoxon(scores_a, scores_b, alternative="two-sided")
    logger.info(
        "Wilcoxon (last %d epochs × %d seeds) | W=%.4f | p=%.6f",
        last_n_epochs, len(lora_results), stat, p_value,
    )
    return float(p_value)


# ---------------------------------------------------------------------------
# ── 5a. Convergence curve ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def plot_convergence(
    lora_results: list[ExperimentResult],
    dora_results: list[ExperimentResult],
    ax: plt.Axes,
) -> None:
    """Draw val macro-F1 convergence curves with shaded ±std bands.

    Args:
        lora_results: Results for LoRA across all seeds.
        dora_results: Results for DoRA across all seeds.
        ax: Matplotlib :class:`~matplotlib.axes.Axes` to draw on.
    """
    epochs = np.arange(1, CFG.num_epochs + 1)

    for results, color, label in (
        (lora_results, "#1f77b4", "LoRA"),
        (dora_results, "#ff7f0e", "DoRA"),
    ):
        matrix = np.array([r.val_f1_per_epoch for r in results])  # (n_seeds, n_epochs)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)

        ax.plot(epochs, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.20)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val macro-F1", fontsize=12)
    ax.set_title(f"Convergence (mean ± std, {len(lora_results)} seeds)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(1, CFG.num_epochs)
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle="--", alpha=0.45)


# ---------------------------------------------------------------------------
# ── 5b. Bar chart of final metrics ──────────────────────────────────────────
# ---------------------------------------------------------------------------
def plot_bar_chart(
    lora_results: list[ExperimentResult],
    dora_results: list[ExperimentResult],
    ax: plt.Axes,
) -> None:
    """Bar chart of final test metrics with error bars (std across seeds).

    Args:
        lora_results: Results for LoRA across all seeds.
        dora_results: Results for DoRA across all seeds.
        ax: Matplotlib :class:`~matplotlib.axes.Axes` to draw on.
    """
    metric_keys = ["final_macro_f1", "final_weighted_f1"]
    metric_labels = ["Macro F1", "Weighted F1"]
    x = np.arange(len(metric_keys))
    width = 0.35

    for offset, results, color, tag in (
        (-width / 2, lora_results, "#1f77b4", "LoRA"),
        (+width / 2, dora_results, "#ff7f0e", "DoRA"),
    ):
        means = [np.mean([getattr(r, k) for r in results]) for k in metric_keys]
        stds  = [np.std ([getattr(r, k) for r in results]) for k in metric_keys]

        ax.bar(
            x + offset, means, width,
            yerr=stds, label=tag, color=color, alpha=0.85,
            capsize=5, error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Final Test Metrics (mean ± std, {len(lora_results)} seeds)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.45)


# ---------------------------------------------------------------------------
# ── 7. Summary table ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def print_summary_table(
    lora_results: list[ExperimentResult],
    dora_results: list[ExperimentResult],
    p_value: float,
) -> None:
    """Print the ablation summary table and Wilcoxon p-value to stdout.

    Columns: method | parameters | F1_macro | F1_weighted | train_time/epoch

    Args:
        lora_results: Results for LoRA.
        dora_results: Results for DoRA.
        p_value: Wilcoxon test result between last-5-epoch val F1 distributions.
    """
    rows: list[dict[str, str]] = []

    for results in (lora_results, dora_results):
        method_tag = results[0].method.upper()
        params = results[0].trainable_params  # fixed across seeds

        macro_vals  = [r.final_macro_f1    for r in results]
        weight_vals = [r.final_weighted_f1 for r in results]
        time_vals   = [r.mean_epoch_time   for r in results]

        rows.append({
            "Method"               : method_tag,
            "Params"               : f"{params:,}",
            "F1_macro"             : f"{np.mean(macro_vals):.4f} ± {np.std(macro_vals):.4f}",
            "F1_weighted"          : f"{np.mean(weight_vals):.4f} ± {np.std(weight_vals):.4f}",
            "Train time/epoch (s)" : f"{np.mean(time_vals):.1f}",
        })

    sep = "=" * 80
    print(f"\n{sep}")
    print("ABLATION  LoRA vs DoRA — Wav2Vec2-XLSR-53 (4-class command recognition)")
    print(sep)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWilcoxon signed-rank test (last 5 epochs val F1, {len(lora_results)} seeds): p = {p_value:.6f}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# ── 3. Orchestration ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full LoRA vs DoRA ablation study.

    Iterates over :data:`CFG.seeds` × methods, collecting
    :class:`ExperimentResult` objects, then produces plots, the Wilcoxon
    test, and the summary table.
    """
    logger.info("AblationConfig: %s", CFG)

    lora_results: list[ExperimentResult] = []
    dora_results: list[ExperimentResult] = []

    # ── 3. Sequential runs — same seed order for both methods ─────────────
    for seed in CFG.seeds:
        for method in ("lora", "dora"):
            peft_cfg = get_peft_config(method)
            result = run_experiment(peft_cfg, method, seed)
            (lora_results if method == "lora" else dora_results).append(result)

    # ── 6. Wilcoxon test ──────────────────────────────────────────────────
    p_value = run_wilcoxon_test(lora_results, dora_results, last_n_epochs=5)

    # ── 7. Summary table ──────────────────────────────────────────────────
    print_summary_table(lora_results, dora_results, p_value)

    # ── 5. Plots → PDF ────────────────────────────────────────────────────
    output_path = ARTIFACTS_DIR / "lora_vs_dora.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "LoRA vs DoRA Ablation — Wav2Vec2-XLSR-53 (4 classes)",
        fontsize=14,
        fontweight="bold",
    )

    plot_convergence(lora_results, dora_results, axes[0])
    plot_bar_chart  (lora_results, dora_results, axes[1])
    plt.tight_layout()

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
