"""
scripts/hybrid/verify_ood_detector.py — Full verification of EnsembleOutlierGate.

Runs the gate on in-distribution (test-split commands, label=0) and OOD samples
(ESC-50 / AudioSet non-target signals, label=1), builds an ROC curve, computes
AUROC, finds the 95th-percentile working point, and saves a two-panel PDF report.

Embedding sources (mutually exclusive, checked in order):
  1. Pre-computed .npz with keys ``embeddings`` (N, D) — fastest.
  2. Pre-computed .npy array of shape (N, D).
  3. Audio directory (.wav / .flac files) — embeddings extracted via ONNX.
  4. Default fallback: ``artifacts/hybrid/embeddings_cache.npz`` split by label
     (non-"другие слова" → in-dist; "другие слова" → OOD proxy).

Usage
-----
    # Use pre-computed embeddings:
    python scripts/hybrid/verify_ood_detector.py \\
        --indist  artifacts/data/test_commands_emb.npz \\
        --ood     artifacts/data/esc50_emb.npz

    # Use audio directories (slow — runs ONNX per file):
    python scripts/hybrid/verify_ood_detector.py \\
        --indist  artifacts/data/test_commands/ \\
        --ood     artifacts/data/esc50_audio/ \\
        --onnx_dir onnx_model/run_2026-04-30

    # Use default cache fallback (no extra arguments required):
    python scripts/hybrid/verify_ood_detector.py

    # Override output path:
    python scripts/hybrid/verify_ood_detector.py --out artifacts/plots/roc_ood_detector.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.hybrid.outlier_gate import EnsembleOutlierGate
from core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_GATE_PATH = _PROJECT_ROOT / "artifacts" / "hybrid" / "outlier_gate.pkl"
_DEFAULT_CACHE_PATH = _PROJECT_ROOT / "artifacts" / "hybrid" / "embeddings_cache.npz"
_DEFAULT_OUT_PATH = _PROJECT_ROOT / "artifacts" / "plots" / "roc_ood_detector.pdf"
_OOD_LABEL_IN_CACHE = "другие слова"

_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

_N_INDIST_DEFAULT = 300
_N_OOD_DEFAULT = 500

# Matplotlib font for Cyrillic labels
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12


# ── Embedding extraction ───────────────────────────────────────────────────────


def _load_embeddings_from_npz(path: Path) -> np.ndarray:
    """Load embedding matrix from a .npz file (key 'embeddings' or first key).

    Args:
        path: Path to the .npz file.

    Returns:
        Float32 array of shape (N, D).

    Raises:
        KeyError: If no suitable key is found.
        ValueError: If the array is not 2-D.
    """
    data = np.load(path, allow_pickle=False)
    # Priority: embeddings → ood_embeddings → indist_embeddings → first key
    for key in ("embeddings", "ood_embeddings", "indist_embeddings"):
        if key in data:
            arr = data[key]
            break
    else:
        arr = data[list(data.keys())[0]]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D embeddings array from {path}, got shape {arr.shape}"
        )
    return arr


def _load_embeddings_from_npy(path: Path) -> np.ndarray:
    """Load embedding matrix from a .npy file.

    Args:
        path: Path to the .npy file.

    Returns:
        Float32 array of shape (N, D).

    Raises:
        ValueError: If the array is not 2-D.
    """
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D embeddings array from {path}, got shape {arr.shape}"
        )
    return arr


def _extract_embeddings_from_audio_dir(
    audio_dir: Path,
    onnx_dir: Path,
) -> np.ndarray:
    """Extract embeddings for all audio files in *audio_dir* via OnnxEngine.

    Args:
        audio_dir: Directory containing audio files (.wav / .flac / ...).
        onnx_dir:  ONNX bundle directory (must contain onnx_config.json).

    Returns:
        Float32 array of shape (N, D) where N is number of audio files.

    Raises:
        FileNotFoundError: If *audio_dir* or *onnx_dir* does not exist.
        RuntimeError:      If no audio files are found or ONNX is unavailable.
    """
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not onnx_dir.is_dir():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    audio_files: List[Path] = sorted(
        p for p in audio_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise RuntimeError(f"No audio files found in {audio_dir}")

    from core.audio_utils import load_wav
    from core.onnx_engine import OnnxEngine

    engine = OnnxEngine(str(onnx_dir), precision="int8")
    embeddings: List[np.ndarray] = []
    errors = 0

    for i, fpath in enumerate(audio_files):
        try:
            wav, _ = load_wav(str(fpath), target_sr=engine.sr)
            _, emb = engine.predict(wav)
            if emb is not None:
                embeddings.append(emb.astype(np.float32))
            else:
                logger.warning("No embedding returned for %s — skipping", fpath.name)
                errors += 1
        except Exception as exc:
            logger.warning("Error on %s: %s", fpath.name, exc)
            errors += 1

        if (i + 1) % 50 == 0:
            logger.info("  Extracted %d / %d embeddings", i + 1 - errors, len(audio_files))

    if not embeddings:
        raise RuntimeError(f"All {len(audio_files)} files in {audio_dir} failed.")

    logger.info(
        "Extracted %d embeddings from %s (%d skipped)",
        len(embeddings), audio_dir, errors,
    )
    return np.stack(embeddings, axis=0)


def _load_embeddings(
    source: Optional[str],
    onnx_dir: Optional[Path],
    label: str,
) -> np.ndarray:
    """Resolve embedding source and load a (N, D) float32 array.

    Accepts .npz, .npy, or an audio directory path. If *source* is ``None``,
    raises RuntimeError (caller must handle the None/default case separately).

    Args:
        source:   String path to embeddings file or audio directory.
        onnx_dir: Required when *source* is an audio directory.
        label:    Human-readable label for log messages (e.g., "in-dist").

    Returns:
        Float32 array of shape (N, D).

    Raises:
        RuntimeError: If *source* is None or the path type is unrecognised.
        FileNotFoundError: If the path does not exist.
    """
    if source is None:
        raise RuntimeError("source is None — use default fallback instead.")

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"{label} path not found: {p}")

    if p.is_dir():
        if onnx_dir is None:
            raise ValueError(
                "--onnx_dir must be provided when embedding source is an audio directory."
            )
        logger.info("Extracting %s embeddings from audio dir: %s", label, p)
        return _extract_embeddings_from_audio_dir(p, onnx_dir)

    if p.suffix == ".npz":
        logger.info("Loading %s embeddings from npz: %s", label, p)
        return _load_embeddings_from_npz(p)

    if p.suffix == ".npy":
        logger.info("Loading %s embeddings from npy: %s", label, p)
        return _load_embeddings_from_npy(p)

    raise RuntimeError(
        f"Unrecognised embedding source type for {label}: {p}  "
        f"(expected .npz, .npy, or a directory)"
    )


def _load_default_fallback(
    cache_path: Path,
    n_indist: int,
    n_ood: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load in-dist and OOD embeddings from the training cache.

    Uses ``_OOD_LABEL_IN_CACHE`` as the OOD proxy class and all other labels
    as in-distribution.  Subsamples to *n_indist* and *n_ood* respectively
    (with replacement if needed).

    Args:
        cache_path: Path to embeddings_cache.npz.
        n_indist:   Target in-distribution sample count.
        n_ood:      Target OOD sample count.
        rng:        Seeded numpy Generator for reproducible subsampling.

    Returns:
        Tuple (indist_embeddings, ood_embeddings), both float32 (N, D).

    Raises:
        FileNotFoundError: If *cache_path* does not exist.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Default embedding cache not found: {cache_path}. "
            "Provide --indist and --ood explicitly."
        )

    data = np.load(cache_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"]

    ood_mask = labels == _OOD_LABEL_IN_CACHE
    indist_mask = ~ood_mask

    indist_emb = embeddings[indist_mask]
    ood_emb = embeddings[ood_mask]

    def _subsample(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) >= n:
            idx = rng.choice(len(arr), size=n, replace=False)
        else:
            logger.warning(
                "Only %d samples available, requested %d — sampling with replacement.",
                len(arr), n,
            )
            idx = rng.choice(len(arr), size=n, replace=True)
        return arr[idx]

    indist_emb = _subsample(indist_emb, n_indist)
    ood_emb = _subsample(ood_emb, n_ood)

    logger.info(
        "Default fallback: %d in-dist samples (target classes), "
        "%d OOD samples ('%s')",
        len(indist_emb), len(ood_emb), _OOD_LABEL_IN_CACHE,
    )
    return indist_emb, ood_emb


# ── Score computation ──────────────────────────────────────────────────────────


def _compute_scores(
    gate: EnsembleOutlierGate,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Run gate.score() on every row of *embeddings*.

    Args:
        gate:       Fitted EnsembleOutlierGate instance.
        embeddings: Float32 array of shape (N, D).

    Returns:
        1-D float64 array of ensemble OOD scores (higher = more OOD).
    """
    n = len(embeddings)
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        s, _ = gate.score(embeddings[i])
        scores[i] = s
        if (i + 1) % 100 == 0:
            logger.info("  Scored %d / %d", i + 1, n)
    return scores


# ── Metrics ───────────────────────────────────────────────────────────────────


def _working_point_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float, float, float]:
    """Compute FPR, FNR, Precision, Recall, F1 at a fixed threshold.

    Positive class = OOD (label=1); negative = in-dist (label=0).
    A sample is predicted OOD when score > threshold.

    Args:
        scores:    1-D float array of ensemble scores.
        labels:    1-D int array {0, 1} (1 = OOD).
        threshold: Decision boundary.

    Returns:
        (fpr, fnr, precision, recall, f1)
    """
    preds = (scores > threshold).astype(int)

    # Guard: all zeros or all ones → sklearn metrics are ill-defined
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return fpr, fnr, precision, recall, f1


# ── Plotting ───────────────────────────────────────────────────────────────────


def _plot_roc(
    ax: plt.Axes,
    fpr_curve: np.ndarray,
    tpr_curve: np.ndarray,
    auroc: float,
    wp_fpr: float,
    wp_tpr: float,
    wp_threshold: float,
) -> None:
    """Draw an ROC curve with the calibrated working point highlighted.

    Args:
        ax:           Matplotlib axes to draw on.
        fpr_curve:    FPR values from roc_curve().
        tpr_curve:    TPR values from roc_curve().
        auroc:        Area under the ROC curve.
        wp_fpr:       FPR at the working point.
        wp_tpr:       TPR (= 1 − FNR) at the working point.
        wp_threshold: Threshold value at the working point (for annotation).
    """
    ax.plot(
        fpr_curve, tpr_curve,
        color="steelblue", linewidth=2,
        label=f"ROC (AUROC = {auroc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Случайный классификатор")
    ax.scatter(
        [wp_fpr], [wp_tpr],
        color="crimson", s=100, zorder=5,
        label=f"Рабочая точка\nFPR={wp_fpr:.3f}, TPR={wp_tpr:.3f}",
    )
    ax.annotate(
        f"τ = {wp_threshold:.3f}",
        xy=(wp_fpr, wp_tpr),
        xytext=(wp_fpr + 0.05, wp_tpr - 0.08),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="crimson"),
        color="crimson",
    )

    ax.set_xlabel("FPR (Ложная тревога, доля)")
    ax.set_ylabel("TPR (Чувствительность, доля)")
    ax.set_title("ROC-кривая OOD-детектора")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)


def _plot_score_histogram(
    ax: plt.Axes,
    indist_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float,
) -> None:
    """Draw overlapping score histograms for in-dist and OOD sets.

    Args:
        ax:            Matplotlib axes to draw on.
        indist_scores: Ensemble scores for in-distribution samples.
        ood_scores:    Ensemble scores for OOD samples.
        threshold:     95th-percentile working point threshold.
    """
    all_scores = np.concatenate([indist_scores, ood_scores])
    bins = np.linspace(all_scores.min(), all_scores.max(), 60)

    ax.hist(
        indist_scores, bins=bins,
        color="steelblue", alpha=0.6, label="In-distribution (команды)",
        density=True,
    )
    ax.hist(
        ood_scores, bins=bins,
        color="darkorange", alpha=0.6, label="OOD (нецелевые сигналы)",
        density=True,
    )
    ax.axvline(
        threshold, color="crimson", linewidth=2, linestyle="--",
        label=f"Порог τ = {threshold:.3f} (95-й перцентиль in-dist)",
    )

    ax.set_xlabel("Ансамблевый OOD-скор")
    ax.set_ylabel("Нормализованная плотность")
    ax.set_title("Распределение OOD-скора")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)


# ── Main ───────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    """Run full OOD verification pipeline and save PDF report.

    Args:
        args: Parsed command-line arguments from :func:`_build_parser`.
    """
    rng = np.random.default_rng(seed=42)

    # ── 1. Load gate ───────────────────────────────────────────────────────────
    gate_path = Path(args.gate)
    logger.info("Loading EnsembleOutlierGate from %s", gate_path)
    gate: EnsembleOutlierGate = EnsembleOutlierGate.load(gate_path)  # type: ignore[assignment]
    logger.info("Gate loaded. Global threshold: %.4f", gate._global_threshold or 0.0)

    # ── 2. Load embeddings ─────────────────────────────────────────────────────
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else None

    cache_path = Path(args.cache)

    # Load cache once if either source falls back to it
    _cache_indist: Optional[np.ndarray] = None
    _cache_ood: Optional[np.ndarray] = None
    if args.indist is None or args.ood is None:
        _cache_indist, _cache_ood = _load_default_fallback(
            cache_path, n_indist=args.n_indist, n_ood=args.n_ood, rng=rng
        )

    if args.indist is not None:
        indist_emb = _load_embeddings(args.indist, onnx_dir, "in-dist")
    else:
        logger.info("--indist not specified — using cache fallback for in-dist.")
        assert _cache_indist is not None
        indist_emb = _cache_indist

    if args.ood is not None:
        ood_emb = _load_embeddings(args.ood, onnx_dir, "OOD")
    else:
        logger.info("--ood not specified — using cache fallback for OOD.")
        assert _cache_ood is not None
        ood_emb = _cache_ood

    logger.info(
        "Embeddings loaded: in-dist=%d, OOD=%d, dim=%d",
        len(indist_emb), len(ood_emb), indist_emb.shape[1],
    )

    # ── 3. Score both sets ─────────────────────────────────────────────────────
    logger.info("Scoring in-distribution set (%d samples)…", len(indist_emb))
    indist_scores = _compute_scores(gate, indist_emb)

    logger.info("Scoring OOD set (%d samples)…", len(ood_emb))
    ood_scores = _compute_scores(gate, ood_emb)

    # Build (score, label) pairs: 0 = in-dist, 1 = OOD
    all_scores = np.concatenate([indist_scores, ood_scores])
    all_labels = np.concatenate([
        np.zeros(len(indist_scores), dtype=int),
        np.ones(len(ood_scores), dtype=int),
    ])

    # ── 4. ROC curve and AUROC ─────────────────────────────────────────────────
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(all_labels, all_scores)
    auroc = float(roc_auc_score(all_labels, all_scores))

    # ── 5. Working point: 95th percentile of in-dist scores ───────────────────
    threshold = float(np.percentile(indist_scores, 95.0))
    logger.info("Working point threshold (95th pct of in-dist): %.6f", threshold)

    # TPR at working point (= 1 - FNR) for ROC annotation
    wp_fpr, wp_fnr, wp_precision, wp_recall, wp_f1 = _working_point_metrics(
        all_scores, all_labels, threshold
    )
    wp_tpr = 1.0 - wp_fnr

    # ── 6. Build PDF with two plots ────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Верификация OOD-детектора (EnsembleOutlierGate)",
        fontsize=14, fontweight="bold",
    )

    _plot_roc(
        axes[0],
        fpr_curve=fpr_curve,
        tpr_curve=tpr_curve,
        auroc=auroc,
        wp_fpr=wp_fpr,
        wp_tpr=wp_tpr,
        wp_threshold=threshold,
    )
    _plot_score_histogram(
        axes[1],
        indist_scores=indist_scores,
        ood_scores=ood_scores,
        threshold=threshold,
    )

    plt.tight_layout()

    with PdfPages(str(out_path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        pdf.infodict()["Title"] = "OOD Detector Verification Report"
        pdf.infodict()["Author"] = "ShipAssistant / verify_ood_detector.py"

    plt.close(fig)
    logger.info("PDF saved to %s", out_path)

    # ── 7. Console summary ─────────────────────────────────────────────────────
    separator = "─" * 70
    print(f"\n{separator}")
    print("  Верификация EnsembleOutlierGate — итоги")
    print(separator)
    print(
        f"  {'Метрика':<40} {'Значение':>12}"
    )
    print(f"  {'─' * 52}")
    print(f"  {'AUROC':<40} {auroc:>12.4f}")
    print(f"  {'Порог τ (95-й перцентиль in-dist)':<40} {threshold:>12.4f}")
    print(f"  {'FPR @ τ  (команды → OOD, ошибка I рода)':<40} {wp_fpr:>12.4f}  ({wp_fpr * 100:.1f}%)")
    print(f"  {'FNR @ τ  (OOD → команда, ошибка II рода)':<40} {wp_fnr:>12.4f}  ({wp_fnr * 100:.1f}%)")
    print(f"  {'Precision @ τ':<40} {wp_precision:>12.4f}")
    print(f"  {'Recall @ τ':<40} {wp_recall:>12.4f}")
    print(f"  {'F1 @ τ':<40} {wp_f1:>12.4f}")
    print(separator)
    print(f"  In-dist samples:  {len(indist_scores)}")
    print(f"  OOD samples:      {len(ood_scores)}")
    print(f"  PDF report:       {out_path}")
    print(separator)

    # Persist raw scores AND embeddings for downstream analysis / reuse
    scores_out = out_path.with_suffix(".npz")
    np.savez_compressed(
        str(scores_out),
        scores=all_scores,
        labels=all_labels,
        indist_scores=indist_scores,
        ood_scores=ood_scores,
        # Save raw embeddings so --ood / --indist can point to this file next run
        embeddings=np.concatenate([indist_emb, ood_emb], axis=0),
        indist_embeddings=indist_emb,
        ood_embeddings=ood_emb,
    )
    logger.info("Raw scores + embeddings saved to %s", scores_out)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the verification script.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        description=(
            "Full verification of EnsembleOutlierGate: ROC curve, "
            "AUROC, working-point metrics, and PDF report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gate",
        default=str(_DEFAULT_GATE_PATH),
        help="Path to fitted EnsembleOutlierGate .pkl file.",
    )
    p.add_argument(
        "--indist",
        default=None,
        help=(
            "In-distribution embeddings: .npz (key 'embeddings'), "
            ".npy, or audio directory. If omitted, uses --cache fallback."
        ),
    )
    p.add_argument(
        "--ood",
        default=None,
        help=(
            "OOD embeddings: .npz (key 'embeddings'), .npy, or audio directory. "
            "If omitted, uses --cache fallback."
        ),
    )
    p.add_argument(
        "--onnx_dir",
        default=str(_PROJECT_ROOT / "onnx_model" / "run_2026-04-30"),
        help="ONNX bundle directory. Required when embedding source is an audio directory.",
    )
    p.add_argument(
        "--cache",
        default=str(_DEFAULT_CACHE_PATH),
        help="Fallback embedding cache .npz (used when --indist/--ood are not given).",
    )
    p.add_argument(
        "--n_indist",
        type=int,
        default=_N_INDIST_DEFAULT,
        help="Number of in-dist samples to draw from the cache fallback.",
    )
    p.add_argument(
        "--n_ood",
        type=int,
        default=_N_OOD_DEFAULT,
        help="Number of OOD samples to draw from the cache fallback.",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_OUT_PATH),
        help="Output PDF path.",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main(_build_parser().parse_args())
