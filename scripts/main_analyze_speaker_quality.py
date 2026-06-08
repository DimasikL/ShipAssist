"""
scripts/analyze_speaker_quality.py — Per-speaker quality analysis for the ONNX INT8 model.

Loads the test split from a metadata CSV (or scans ``clf_dset/test/`` directly),
runs inference with the quantised ONNX model, and produces:

  * Per-speaker macro F1 / weighted F1 / accuracy / confusion matrix (console + PDF)
  * Boxplot of per-class F1 distribution per speaker (horizontal, sorted by macro F1)
  * Heatmap of the mean normalised confusion matrix across all speakers
  * Scatter: N_examples vs macro F1 per speaker
  * Console summary: min/max/mean/std F1, worst-speaker drill-down, real vs synthetic split

Usage::

    python scripts/analyze_speaker_quality.py                    # auto-detect CSV & ONNX dir
    python scripts/analyze_speaker_quality.py --scan-dir         # scan clf_dset/test/ directly
    python scripts/analyze_speaker_quality.py --csv my.csv --onnx-dir onnx_model/models/run_X

CSV column defaults (pass --col-* to override)::

    filepath  → audio_path
    label     → class
    speaker   → audio_group
    split     → (derived from filepath if column absent)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

# ── Project root (scripts/ → parent) ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_utils import load_wav
from core.exceptions import ModelLoadError
from core.logger import get_logger
from core.onnx_engine import OnnxEngine

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Speaker types: prefix "silero" or "gtts" or "genwords" → synthetic.
_SYNTHETIC_PREFIXES: Tuple[str, ...] = ("silero", "gtts", "genwords")

# Canonical label set (must match onnx_config.json).  Filled after model load.
LABELS: List[str] = []

# Regex patterns → canonical label.  The folder names in clf_dset use Russian
# text with suffixes like "x9", "x18", underscores, and ё/е variants.
_LABEL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"другие\s*слова", re.IGNORECASE), "другие слова"),
    (re.compile(r"негатив|negative|negatives", re.IGNORECASE), "другие слова"),
    (re.compile(r"машина", re.IGNORECASE), "машина"),
    (re.compile(r"приготовить[_ ]машину|приготовить_машину", re.IGNORECASE), "приготовить машину"),
    (re.compile(r"самый[_ ]малый[_ ]вперед|самый[_ ]малый[_ ]вперёд", re.IGNORECASE), "самый малый вперед"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_label(raw: str) -> Optional[str]:
    """Map a raw folder / CSV class name to a canonical label string.

    Returns ``None`` when no canonical match is found (e.g. "scr", "src" folders).
    """
    for pattern, canonical in _LABEL_PATTERNS:
        if pattern.search(raw):
            return canonical
    return None


def _resolve_path(win_path: str) -> Path:
    """Translate a Windows absolute path from the CSV to the current filesystem.

    Replaces the Windows project root with PROJECT_ROOT so the same CSV works
    on Linux CI and the developer's Windows machine.
    """
    # Normalise separators first.
    normalised = win_path.replace("\\", "/")
    # Strip Windows drive prefix (C:/Users/.../ShipAssistant) and re-root.
    match = re.search(r"ShipAssistant/(.+)$", normalised)
    if match:
        return PROJECT_ROOT / match.group(1)
    # Fallback: treat as local path.
    return Path(win_path)


def _is_synthetic(speaker_id: str) -> bool:
    return any(speaker_id.lower().startswith(p) for p in _SYNTHETIC_PREFIXES)


def _split_from_path(p: str) -> str:
    """Extract 'test' / 'train' / 'val' from a file path string."""
    m = re.search(r"clf_dset[/\\](test|train_val|train|val)", p)
    if m:
        raw = m.group(1)
        return "train" if raw == "train_val" else raw
    return "unknown"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_from_csv(
    csv_path: Path,
    col_filepath: str,
    col_label: str,
    col_speaker: str,
    col_split: Optional[str],
    target_split: str,
) -> pd.DataFrame:
    """Load and filter the metadata CSV to the requested split.

    Returns a DataFrame with columns: filepath, label, speaker_id, split.
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded CSV: %s  (%d rows)", csv_path, len(df))

    # Validate required columns.
    for col in (col_filepath, col_label, col_speaker):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {csv_path}.  "
                f"Available: {df.columns.tolist()}"
            )

    # Derive split.
    if col_split and col_split in df.columns:
        df["_split"] = df[col_split].str.strip().str.lower()
    else:
        df["_split"] = df[col_filepath].apply(_split_from_path)

    out = df[df["_split"] == target_split].copy()
    logger.info("Rows in '%s' split: %d", target_split, len(out))

    out = out.rename(
        columns={
            col_filepath: "filepath",
            col_label: "label",
            col_speaker: "speaker_id",
        }
    )
    # Normalise labels to canonical form.
    out["label"] = out["label"].apply(lambda x: _normalize_label(str(x)) or str(x))
    # Resolve filepaths to current OS.
    out["filepath"] = out["filepath"].apply(lambda p: str(_resolve_path(str(p))))
    return out[["filepath", "label", "speaker_id"]].reset_index(drop=True)


def load_from_dir(test_dir: Path) -> pd.DataFrame:
    """Walk ``clf_dset/test/`` and build a DataFrame from the directory structure.

    Speaker ID is inferred from ``group=<speaker>`` folder names.
    Label is inferred from the immediate parent directory of each .wav file
    (normalised via ``_normalize_label``).  Files whose label cannot be
    resolved (e.g. "scr", "src" intermediate folders) are skipped with a
    warning.
    """
    rows: List[Dict] = []
    skipped = 0
    for wav_file in sorted(test_dir.rglob("*.wav")):
        parts = wav_file.parts
        # speaker_id: first path component that starts with "group="
        group_part = next((p for p in parts if p.startswith("group=")), None)
        speaker_id = group_part.replace("group=", "") if group_part else "unknown"
        # label: immediate parent directory name
        raw_label = wav_file.parent.name
        canonical = _normalize_label(raw_label)
        if canonical is None:
            logger.debug("Skipping unresolved label '%s': %s", raw_label, wav_file)
            skipped += 1
            continue
        rows.append(
            {"filepath": str(wav_file), "label": canonical, "speaker_id": speaker_id}
        )
    if skipped:
        logger.warning("Skipped %d files with unresolved labels.", skipped)
    df = pd.DataFrame(rows)
    logger.info("Scanned test dir: %d usable files.", len(df))
    return df


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    df: pd.DataFrame,
    engine: OnnxEngine,
) -> pd.DataFrame:
    """Run ONNX inference on every row and append 'pred' / 'confidence' columns.

    Files that fail to load are logged and skipped (dropped from the result).
    """
    preds: List[int] = []
    confs: List[float] = []
    keep: List[int] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference", unit="file"):
        try:
            audio, _ = load_wav(row["filepath"], target_sr=engine.sr)
            probs, _ = engine.predict(audio)
            pred_idx = int(np.argmax(probs))
            preds.append(pred_idx)
            confs.append(float(probs[pred_idx]))
            keep.append(idx)
        except FileNotFoundError:
            logger.warning("File not found, skipping: %s", row["filepath"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Inference error on %s: %s", row["filepath"], exc)

    result = df.loc[keep].copy()
    result["pred"] = preds
    result["confidence"] = confs
    # Map string labels → int indices using the engine's label list.
    label_to_idx = {lbl: i for i, lbl in enumerate(engine.labels)}
    result["true"] = result["label"].map(label_to_idx)
    # Log any unmapped labels (label in CSV not in model's label list).
    unmapped = result["true"].isna()
    if unmapped.any():
        bad = result.loc[unmapped, "label"].unique().tolist()
        logger.warning("Labels not in model vocabulary (will be dropped): %s", bad)
        result = result.dropna(subset=["true"])
    result["true"] = result["true"].astype(int)
    return result.reset_index(drop=True)


# ── Per-speaker metrics ───────────────────────────────────────────────────────

def compute_speaker_metrics(
    df: pd.DataFrame,
    labels: List[str],
) -> Dict[str, Dict]:
    """Compute per-speaker classification metrics.

    Returns a dict keyed by speaker_id with fields:
        n, macro_f1, weighted_f1, accuracy,
        per_class_f1 (np.ndarray, shape=[n_labels]),
        confusion_matrix (np.ndarray, shape=[n_labels, n_labels]),
        is_synthetic (bool)
    """
    n_labels = len(labels)
    results: Dict[str, Dict] = {}

    for speaker, grp in df.groupby("speaker_id"):
        y_true = grp["true"].values
        y_pred = grp["pred"].values
        n = len(grp)

        macro_f1 = f1_score(
            y_true, y_pred, labels=list(range(n_labels)),
            average="macro", zero_division=0,
        )
        weighted_f1 = f1_score(
            y_true, y_pred, labels=list(range(n_labels)),
            average="weighted", zero_division=0,
        )
        acc = accuracy_score(y_true, y_pred)
        per_class_f1 = f1_score(
            y_true, y_pred, labels=list(range(n_labels)),
            average=None, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_labels)))

        results[str(speaker)] = {
            "n": n,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": acc,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "is_synthetic": _is_synthetic(str(speaker)),
        }

    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def _fig_boxplot(metrics: Dict[str, Dict], labels: List[str]) -> plt.Figure:
    """Horizontal boxplot of per-class F1 distribution, sorted by macro F1."""
    speakers = sorted(metrics.keys(), key=lambda s: metrics[s]["macro_f1"])
    data = [metrics[s]["per_class_f1"] for s in speakers]
    macro_f1s = [metrics[s]["macro_f1"] for s in speakers]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(speakers) + 1)))

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        notch=False,
        widths=0.5,
    )
    # Colour each box by macro F1 (low → red, high → green).
    cmap = plt.cm.RdYlGn
    for patch, f1 in zip(bp["boxes"], macro_f1s):
        patch.set_facecolor(cmap(f1))
        patch.set_alpha(0.75)

    ax.set_yticks(range(1, len(speakers) + 1))
    ax.set_yticklabels(speakers, fontsize=9)
    ax.set_xlabel("Per-class F1 score")
    ax.set_title("Per-speaker F1 distribution\n(sorted by macro F1, coloured low→high)")
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(np.mean(macro_f1s), color="steelblue", linestyle="--",
               linewidth=1, label=f"Mean macro F1 = {np.mean(macro_f1s):.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _fig_heatmap(metrics: Dict[str, Dict], labels: List[str]) -> plt.Figure:
    """Heatmap of the mean row-normalised confusion matrix across all speakers."""
    n = len(labels)
    cumulative = np.zeros((n, n), dtype=float)
    for m in metrics.values():
        cm = m["confusion_matrix"].astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero for absent classes
        cumulative += cm / row_sums

    mean_cm = cumulative / len(metrics)

    short_labels = [lbl[:20] for lbl in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        mean_cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="gray",
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Mean normalised confusion matrix\n(averaged across all speakers)")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    return fig


def _fig_scatter(metrics: Dict[str, Dict]) -> plt.Figure:
    """Scatter: number of test examples vs macro F1 per speaker."""
    speakers = list(metrics.keys())
    ns = np.array([metrics[s]["n"] for s in speakers])
    f1s = np.array([metrics[s]["macro_f1"] for s in speakers])
    is_synth = np.array([metrics[s]["is_synthetic"] for s in speakers])

    fig, ax = plt.subplots(figsize=(8, 5))
    for synth, color, marker, label in [
        (True, "darkorange", "^", "Synthetic"),
        (False, "steelblue", "o", "Real"),
    ]:
        mask = is_synth == synth
        if mask.any():
            ax.scatter(ns[mask], f1s[mask], c=color, marker=marker,
                       s=80, label=label, zorder=3, edgecolors="white", linewidths=0.5)
            for i, spk in enumerate(speakers):
                if is_synth[i] == synth:
                    ax.annotate(
                        spk,
                        (ns[i], f1s[i]),
                        textcoords="offset points",
                        xytext=(6, 4),
                        fontsize=7,
                        color=color,
                    )

    # Correlation annotation.
    if len(ns) >= 3:
        corr = float(np.corrcoef(ns, f1s)[0, 1])
        ax.text(
            0.97, 0.05,
            f"Pearson r = {corr:.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9, color="gray",
        )

    ax.set_xlabel("Number of test examples (N)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Data volume vs. model performance per speaker")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ── Console output ────────────────────────────────────────────────────────────

def _top_errors(
    speaker: str,
    df: pd.DataFrame,
    labels: List[str],
    top_k: int = 3,
) -> List[str]:
    """Return the top-k most frequent (true_label → predicted_label) error pairs."""
    grp = df[df["speaker_id"] == speaker]
    errors = grp[grp["true"] != grp["pred"]]
    if errors.empty:
        return ["(no errors)"]
    pairs = (
        errors.apply(lambda r: f"{labels[r['true']]} → {labels[r['pred']]}", axis=1)
        .value_counts()
        .head(top_k)
    )
    return [f"{pair} ({cnt}x)" for pair, cnt in pairs.items()]


def print_summary(
    metrics: Dict[str, Dict],
    df: pd.DataFrame,
    labels: List[str],
) -> None:
    """Print a structured summary to stdout."""
    all_f1 = np.array([m["macro_f1"] for m in metrics.values()])
    speakers = list(metrics.keys())

    print("\n" + "=" * 65)
    print("  PER-SPEAKER MACRO F1 SUMMARY")
    print("=" * 65)
    print(f"  Speakers evaluated : {len(speakers)}")
    print(f"  Total test examples: {sum(m['n'] for m in metrics.values())}")
    print(f"  Min  F1 : {all_f1.min():.4f}  ({speakers[all_f1.argmin()]})")
    print(f"  Max  F1 : {all_f1.max():.4f}  ({speakers[all_f1.argmax()]})")
    print(f"  Mean F1 : {all_f1.mean():.4f}")
    print(f"  Std  F1 : {all_f1.std():.4f}")

    # Per-speaker table.
    print("\n  Speaker".ljust(28) + "  N   macro_F1  wt_F1  Acc   Type")
    print("  " + "-" * 60)
    for spk in sorted(speakers, key=lambda s: metrics[s]["macro_f1"], reverse=True):
        m = metrics[spk]
        kind = "synth" if m["is_synthetic"] else "real"
        print(
            f"  {spk[:24]:<24}  {m['n']:>3}  "
            f"{m['macro_f1']:.4f}    {m['weighted_f1']:.4f}  "
            f"{m['accuracy']:.4f}  {kind}"
        )

    # Worst speaker drill-down.
    worst = speakers[all_f1.argmin()]
    print(f"\n  ▶ Worst speaker: '{worst}'  (macro F1 = {all_f1.min():.4f})")
    top_err = _top_errors(worst, df, labels)
    for i, err in enumerate(top_err, 1):
        print(f"    {i}. {err}")

    # Real vs synthetic split.
    real_f1 = [m["macro_f1"] for m in metrics.values() if not m["is_synthetic"]]
    synth_f1 = [m["macro_f1"] for m in metrics.values() if m["is_synthetic"]]
    print("\n  ── Real vs Synthetic speakers ──")
    if real_f1:
        print(f"  Real    : n={len(real_f1)}  "
              f"mean={np.mean(real_f1):.4f}  std={np.std(real_f1):.4f}")
    else:
        print("  Real    : (none in this split)")
    if synth_f1:
        print(f"  Synth   : n={len(synth_f1)}  "
              f"mean={np.mean(synth_f1):.4f}  std={np.std(synth_f1):.4f}")
    else:
        print("  Synth   : (none in this split)")

    print("=" * 65 + "\n")


# ── PDF assembly ──────────────────────────────────────────────────────────────

def save_pdf(
    figures: List[plt.Figure],
    output_path: Path,
) -> None:
    """Save all figures into a single multi-page PDF."""
    with pdf_backend.PdfPages(str(output_path)) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    logger.info("Saved PDF: %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-speaker quality analysis for the ShipAssistant ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to metadata CSV.  Auto-detects the most recent dset_meta_only*.csv "
             "in PROJECT_ROOT when omitted.",
    )
    p.add_argument(
        "--scan-dir",
        action="store_true",
        help="Scan clf_dset/test/ directly instead of using the CSV.  "
             "Derives speaker_id and label from folder names.",
    )
    p.add_argument(
        "--onnx-dir",
        type=Path,
        default=PROJECT_ROOT / "onnx_model" / "models" / "run_2026-05-22_09-50-17",
        help="Directory containing onnx_config.json and model_int8.onnx.",
    )
    p.add_argument(
        "--precision",
        choices=["int8", "fp32", "fp16"],
        default="int8",
        help="ONNX model precision to load.",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Which data split to evaluate (only used in CSV mode).",
    )
    p.add_argument("--col-filepath", default="audio_path", metavar="COL")
    p.add_argument("--col-label", default="class", metavar="COL")
    p.add_argument("--col-speaker", default="audio_group", metavar="COL")
    p.add_argument(
        "--col-split",
        default=None,
        metavar="COL",
        help="CSV column holding split labels.  "
             "When absent, split is derived from the filepath.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "speaker_analysis.pdf",
        help="Output PDF path.",
    )
    p.add_argument(
        "--min-examples",
        type=int,
        default=5,
        help="Skip speakers with fewer examples than this threshold.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _auto_detect_csv() -> Path:
    candidates = sorted(PROJECT_ROOT.glob("dset_meta_only_*.csv"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No dset_meta_only_*.csv found in PROJECT_ROOT.  "
            "Pass --csv explicitly."
        )
    chosen = candidates[0]
    logger.info("Auto-detected CSV: %s", chosen)
    return chosen


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    # 1. Load data ─────────────────────────────────────────────────────────────
    if args.scan_dir:
        test_dir = PROJECT_ROOT / "clf_dset" / "test"
        if not test_dir.is_dir():
            logger.error("Test directory not found: %s", test_dir)
            sys.exit(1)
        df = load_from_dir(test_dir)
    else:
        csv_path = args.csv or _auto_detect_csv()
        df = load_from_csv(
            csv_path,
            col_filepath=args.col_filepath,
            col_label=args.col_label,
            col_speaker=args.col_speaker,
            col_split=args.col_split,
            target_split=args.split,
        )

    if df.empty:
        logger.error(
            "No data found for split='%s'.  "
            "Try --scan-dir or check --col-* arguments.",
            args.split,
        )
        sys.exit(1)

    # 2. Load ONNX model ───────────────────────────────────────────────────────
    try:
        engine = OnnxEngine(str(args.onnx_dir), precision=args.precision)
    except ModelLoadError as exc:
        logger.error("Failed to load ONNX model: %s", exc)
        sys.exit(1)

    global LABELS
    LABELS = engine.labels
    logger.info("Model labels: %s", LABELS)

    # 3. Run inference ─────────────────────────────────────────────────────────
    df_results = run_inference(df, engine)
    if df_results.empty:
        logger.error("All inference attempts failed.  Check file paths and model.")
        sys.exit(1)

    n_correct = (df_results["true"] == df_results["pred"]).sum()
    logger.info(
        "Overall accuracy: %.4f  (%d / %d)",
        n_correct / len(df_results), n_correct, len(df_results),
    )

    # 4. Per-speaker metrics ───────────────────────────────────────────────────
    metrics = compute_speaker_metrics(df_results, LABELS)

    # Drop speakers with too few examples.
    thin = [s for s, m in metrics.items() if m["n"] < args.min_examples]
    if thin:
        logger.warning(
            "Dropping %d speaker(s) with < %d examples: %s",
            len(thin), args.min_examples, thin,
        )
        for s in thin:
            del metrics[s]

    if not metrics:
        logger.error(
            "No speakers remain after filtering.  Lower --min-examples."
        )
        sys.exit(1)

    # 5. Console summary ───────────────────────────────────────────────────────
    print_summary(metrics, df_results, LABELS)

    # 6. Build figures ─────────────────────────────────────────────────────────
    fig_box = _fig_boxplot(metrics, LABELS)
    fig_heat = _fig_heatmap(metrics, LABELS)
    fig_scatter = _fig_scatter(metrics)

    # 7. Save PDF ──────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_pdf([fig_box, fig_heat, fig_scatter], args.output)
    print(f"  PDF saved → {args.output}")


if __name__ == "__main__":
    main()
