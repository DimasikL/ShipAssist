"""
scripts/eval_confusion_matrix.py — Confusion matrix over the clf_dset/test split.

Walks clf_dset/test, infers every .wav with the ONNX model, and produces:
  - artifacts/plots/confusion_matrix_<tag>.png   — normalised heat-map
  - artifacts/plots/confusion_matrix_<tag>_raw.png — raw counts heat-map
  - artifacts/benchmarks/confusion_matrix_<tag>.json — full metrics + per-file results

Directory layout understood
---------------------------
  test/group=*/samples/{class xN}/*.wav   (augmented samples)
  test/group=*/commands/{class}/*.wav     (command recordings)
  test/group=*/negatives/*.wav            (negative recordings)

Files under scr/ and src/ are source originals — skipped deliberately.

Usage
-----
    # Evaluate the latest INT8 model (default):
    python scripts/eval_confusion_matrix.py

    # Explicit paths:
    python scripts/eval_confusion_matrix.py \\
        --onnx_path  onnx_model/models/run_2026-05-22_09-50-17/model_int8.onnx \\
        --onnx_cfg   onnx_model/models/run_2026-05-22_09-50-17/onnx_config.json \\
        --test_dir   clf_dset/test \\
        --tag        int8_may22

    # FP32 variant:
    python scripts/eval_confusion_matrix.py \\
        --onnx_path  onnx_model/models/run_2026-05-22_09-50-17/model_fp32.onnx \\
        --tag        fp32_may22
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root  →  absolute imports work when called from repo root or directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SR: int = 16_000

# Canonical class order (must match id2label in the model config)
CANONICAL_LABELS: List[str] = [
    "другие слова",
    "машина",
    "приготовить машину",
    "самый малый вперед",
]

# Normalise label variants found in directory names to canonical labels
_LABEL_NORM: Dict[str, str] = {
    "другие слова":       "другие слова",
    "negatives":          "другие слова",
    "машина":             "машина",
    "приготовить машину": "приготовить машину",
    "приготовить_машину": "приготовить машину",
    "самый малый вперед": "самый малый вперед",
    "самый_малый_вперед": "самый малый вперед",
    "самый_малый_вперёд": "самый малый вперед",   # ё → е handled separately
    "самый малый вперёд": "самый малый вперед",
}


def _normalise_label(raw: str) -> Optional[str]:
    """Map a directory-derived raw label to a canonical model label.

    Handles underscore/space variants and the ё/е Cyrillic difference.

    Args:
        raw: Label string extracted from the directory path.

    Returns:
        Canonical label string, or None if unrecognised.
    """
    raw = raw.strip()
    if raw in _LABEL_NORM:
        return _LABEL_NORM[raw]
    # try stripping trailing ' xN' / '_xN' quantity suffix
    stripped = re.sub(r"[\s_]x\d+$", "", raw).strip()
    if stripped in _LABEL_NORM:
        return _LABEL_NORM[stripped]
    # normalise underscores → spaces
    spaced = stripped.replace("_", " ")
    return _LABEL_NORM.get(spaced)


def collect_test_files(test_dir: Path) -> List[Tuple[Path, str]]:
    """Recursively collect (wav_path, canonical_label) pairs from *test_dir*.

    Skips files inside ``scr/`` and ``src/`` sub-folders (source originals).

    Args:
        test_dir: Root of the test split directory.

    Returns:
        List of (absolute Path, canonical label) tuples.
    """
    samples: List[Tuple[Path, str]] = []
    skipped_src: List[str] = []
    skipped_unknown: List[str] = []

    for wav in sorted(test_dir.rglob("*.wav")):
        parts = wav.parts

        # Skip source-original folders
        if "scr" in parts or "src" in parts:
            skipped_src.append(str(wav.relative_to(test_dir)))
            continue

        label: Optional[str] = None

        for i, part in enumerate(parts):
            if part in ("commands", "samples") and i + 1 < len(parts):
                label = _normalise_label(parts[i + 1])
                break
            elif part == "negatives":
                label = "другие слова"
                break

        if label is None:
            skipped_unknown.append(str(wav.relative_to(test_dir)))
        else:
            samples.append((wav, label))

    if skipped_src:
        logger.info("Skipped %d source-original files (scr/src).", len(skipped_src))
    if skipped_unknown:
        logger.warning(
            "Skipped %d files with unrecognised label path:\n  %s",
            len(skipped_unknown),
            "\n  ".join(skipped_unknown[:10]),
        )

    return samples


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _build_ort_session(
    onnx_path: Path,
) -> "onnxruntime.InferenceSession":  # type: ignore[name-defined]
    """Create an ORT session on CPU.

    Args:
        onnx_path: Path to the .onnx model file.

    Returns:
        Initialised ORT InferenceSession.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = os.cpu_count() or 1
    opts.intra_op_num_threads = os.cpu_count() or 1
    logger.info("Loading ONNX session: %s", onnx_path)
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    e = np.exp(x - x.max())
    return e / e.sum()


def _infer_file(
    wav_path: Path,
    session: "onnxruntime.InferenceSession",  # type: ignore[name-defined]
    input_name: str,
    id2label: Dict[int, str],
    max_samples: int,
) -> Tuple[str, float, float]:
    """Load one audio file and run a single ONNX forward pass.

    Args:
        wav_path:    Path to the .wav file.
        session:     Active ORT InferenceSession.
        input_name:  Name of the model's input tensor.
        id2label:    Integer index → class label mapping.
        max_samples: Canonical window length in samples.

    Returns:
        (predicted_label, confidence, latency_ms)
    """
    try:
        waveform, _ = load_wav(str(wav_path), target_sr=SR)
    except Exception as exc:
        logger.warning("Cannot load %s: %s — using silence.", wav_path.name, exc)
        waveform = np.zeros(max_samples, dtype=np.float32)

    window = prepare_window(waveform, target_samples=max_samples, do_normalize=True)
    batch = window[np.newaxis, :]  # shape: (1, max_samples)

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: batch})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs[0][0].astype(np.float32)
    probs = _softmax(logits)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    predicted_label = id2label[pred_idx]

    return predicted_label, confidence, latency_ms


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    normalise: bool = True,
) -> None:
    """Render and save a confusion matrix heat-map.

    Args:
        cm:        Square count matrix (rows = true, cols = predicted).
        labels:    Class label names (ordered to match matrix axes).
        title:     Figure title string.
        out_path:  Destination file path for the PNG.
        normalise: If True, normalise each row by its true-class count.
    """
    if normalise:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
        fmt_fn = lambda v: f"{v:.2f}"
        vmin, vmax = 0.0, 1.0
        cbar_label = "Recall (row-normalised)"
    else:
        cm_plot = cm.astype(float)
        fmt_fn = lambda v: str(int(v))
        vmin, vmax = 0.0, float(cm.max())
        cbar_label = "Count"

    n = len(labels)
    fig_size = max(6, n * 1.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [lbl if len(lbl) <= 20 else lbl[:18] + "…" for lbl in labels]
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    thresh = cm_plot.max() * 0.55
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            txt = fmt_fn(val)
            color = "white" if val > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    # Per-class support annotation on y-axis
    row_totals = cm.sum(axis=1)
    for i, total in enumerate(row_totals):
        ax.text(
            n + 0.05, i, f"n={total}",
            va="center", ha="left", fontsize=8, color="grey",
            transform=ax.get_yaxis_transform(),
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_eval(
    onnx_path: Path,
    onnx_cfg_path: Path,
    test_dir: Path,
    tag: str,
    output_dir: Path,
) -> None:
    """Run the full evaluation pipeline.

    Args:
        onnx_path:     Path to the ONNX model.
        onnx_cfg_path: Path to onnx_config.json (provides labels & window size).
        test_dir:      Root of clf_dset/test.
        tag:           Short identifier appended to output file names.
        output_dir:    Directory where PNG and JSON outputs are written.
    """
    # ── Load model config ──────────────────────────────────────────────────
    with open(onnx_cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    labels: List[str] = cfg["labels"]
    max_samples: int = int(cfg.get("win_samples", cfg.get("sr", SR) * cfg.get("window_s", 1.0)))
    id2label: Dict[int, str] = {i: lbl for i, lbl in enumerate(labels)}
    label2id: Dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}

    logger.info("Labels (%d): %s", len(labels), labels)
    logger.info("Window: %d samples (%.2f s)", max_samples, max_samples / SR)

    # ── Collect test files ────────────────────────────────────────────────
    samples = collect_test_files(test_dir)
    if not samples:
        logger.error("No .wav files found in %s", test_dir)
        sys.exit(1)

    logger.info("Test set: %d files", len(samples))

    # Per-class file count summary
    class_counts: Dict[str, int] = defaultdict(int)
    for _, lbl in samples:
        class_counts[lbl] += 1
    logger.info(
        "Class distribution:\n  %s",
        "\n  ".join(f"{k:<30}: {v:4d}" for k, v in sorted(class_counts.items())),
    )

    # Validate that all true labels exist in the model's label set
    unknown = {lbl for _, lbl in samples} - set(labels)
    if unknown:
        logger.error("True labels not in model vocabulary: %s", sorted(unknown))
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────
    session = _build_ort_session(onnx_path)
    input_name: str = session.get_inputs()[0].name

    y_true: List[int] = []
    y_pred: List[int] = []
    confs:  List[float] = []
    lats:   List[float] = []
    per_file_results = []

    for wav_path, true_label in tqdm(samples, desc="Inferring", unit="file"):
        pred_label, conf, lat_ms = _infer_file(
            wav_path, session, input_name, id2label, max_samples
        )
        y_true.append(label2id[true_label])
        y_pred.append(label2id[pred_label])
        confs.append(conf)
        lats.append(lat_ms)
        per_file_results.append({
            "file":       str(wav_path.relative_to(test_dir)),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": round(conf, 4),
            "latency_ms": round(lat_ms, 2),
            "correct":    pred_label == true_label,
        })

    # ── Build confusion matrix ────────────────────────────────────────────
    n_cls = len(labels)
    cm = np.zeros((n_cls, n_cls), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # ── Metrics ───────────────────────────────────────────────────────────
    accuracy = float(np.diag(cm).sum() / cm.sum())

    per_class_metrics: Dict[str, dict] = {}
    for i, lbl in enumerate(labels):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        fp = int(cm[:, i].sum()) - tp
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        per_class_metrics[lbl] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }

    macro_f1 = float(np.mean([v["f1"] for v in per_class_metrics.values()]))
    macro_precision = float(np.mean([v["precision"] for v in per_class_metrics.values()]))
    macro_recall    = float(np.mean([v["recall"]    for v in per_class_metrics.values()]))

    # Print readable report
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONFUSION MATRIX  [%s]", tag)
    logger.info("=" * 60)
    header = f"{'':30s}" + "".join(f"{lbl[:14]:>16s}" for lbl in labels)
    logger.info(header)
    for i, lbl in enumerate(labels):
        row = f"{lbl[:30]:<30s}" + "".join(f"{cm[i, j]:>16d}" for j in range(n_cls))
        logger.info(row)
    logger.info("-" * 60)
    logger.info("Accuracy:        %.4f  (%d / %d)", accuracy, int(np.diag(cm).sum()), int(cm.sum()))
    logger.info("Macro F1:        %.4f", macro_f1)
    logger.info("Macro Precision: %.4f", macro_precision)
    logger.info("Macro Recall:    %.4f", macro_recall)
    logger.info("Mean confidence: %.4f", float(np.mean(confs)))
    logger.info(
        "Latency (ms):    mean=%.1f  p50=%.1f  p95=%.1f",
        float(np.mean(lats)),
        float(np.percentile(lats, 50)),
        float(np.percentile(lats, 95)),
    )
    logger.info("")
    logger.info("Per-class metrics:")
    for lbl, m in per_class_metrics.items():
        logger.info(
            "  %-30s  P=%.3f  R=%.3f  F1=%.3f  support=%d",
            lbl, m["precision"], m["recall"], m["f1"], m["support"],
        )
    logger.info("=" * 60)

    # ── Save plots ────────────────────────────────────────────────────────
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_confusion_matrix(
        cm, labels,
        title=f"Confusion Matrix (normalised) — {tag}",
        out_path=plots_dir / f"confusion_matrix_{tag}.png",
        normalise=True,
    )
    _plot_confusion_matrix(
        cm, labels,
        title=f"Confusion Matrix (raw counts) — {tag}",
        out_path=plots_dir / f"confusion_matrix_{tag}_raw.png",
        normalise=False,
    )

    # ── Save JSON ─────────────────────────────────────────────────────────
    benchmarks_dir = output_dir / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    json_path = benchmarks_dir / f"confusion_matrix_{tag}.json"

    results = {
        "tag":            tag,
        "onnx_path":      str(onnx_path),
        "test_dir":       str(test_dir),
        "n_samples":      len(samples),
        "accuracy":       round(accuracy, 4),
        "macro_f1":       round(macro_f1, 4),
        "macro_precision":round(macro_precision, 4),
        "macro_recall":   round(macro_recall, 4),
        "mean_confidence":round(float(np.mean(confs)), 4),
        "mean_latency_ms":round(float(np.mean(lats)), 2),
        "p50_latency_ms": round(float(np.percentile(lats, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 2),
        "labels":         labels,
        "confusion_matrix": cm.tolist(),
        "per_class":      per_class_metrics,
        "per_file":       per_file_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("JSON results saved → %s", json_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a confusion matrix over clf_dset/test using an ONNX model."
    )
    p.add_argument(
        "--onnx_path",
        default="onnx_model/models/run_2026-05-22_09-50-17/model_int8.onnx",
        help="Path to the .onnx model file (relative to project root or absolute).",
    )
    p.add_argument(
        "--onnx_cfg",
        default="onnx_model/models/run_2026-05-22_09-50-17/onnx_config.json",
        help="Path to onnx_config.json accompanying the model.",
    )
    p.add_argument(
        "--test_dir",
        default="clf_dset/test",
        help="Root directory of the test split.",
    )
    p.add_argument(
        "--tag",
        default="int8_may22",
        help="Short identifier for output file names (e.g. int8_may22, fp32_may22).",
    )
    p.add_argument(
        "--output_dir",
        default="artifacts",
        help="Root directory where plots/ and benchmarks/ sub-folders are created.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    root = _PROJECT_ROOT

    onnx_path  = Path(args.onnx_path)
    onnx_cfg   = Path(args.onnx_cfg)
    test_dir   = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    # Resolve relative paths against project root
    if not onnx_path.is_absolute():
        onnx_path = root / onnx_path
    if not onnx_cfg.is_absolute():
        onnx_cfg = root / onnx_cfg
    if not test_dir.is_absolute():
        test_dir = root / test_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    for label, path in [("ONNX model", onnx_path), ("ONNX config", onnx_cfg), ("Test dir", test_dir)]:
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    run_eval(
        onnx_path=onnx_path,
        onnx_cfg_path=onnx_cfg,
        test_dir=test_dir,
        tag=args.tag,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
