"""Evaluate ONNX INT8 Wav2Vec2+LoRA on the DEMAND OOD mixed dataset.

Loads the mixed CSV from prepare_demand_ood.py, runs the ONNX model,
computes macro-F1 grouped by SNR and noise environment, and saves a
two-panel PDF report.

Audio preprocessing uses core.audio_utils.prepare_window — the same path
used by the production ONNX engine — to guarantee inference parity.

Usage (from project root):
    python scripts/evaluate_demand_ood.py
    python scripts/evaluate_demand_ood.py \\
        --mixed-csv  artifacts/demand_ood_test.csv \\
        --onnx-model onnx_model/models/run_2026-04-30_23-34-27/model_int8.onnx \\
        --onnx-cfg   onnx_model/models/run_2026-04-30_23-34-27/onnx_config.json \\
        --output-pdf artifacts/plots/demand_snr_analysis.pdf \\
        --baseline   artifacts/benchmarks/f1_vs_snr_full2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import

import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Project root on sys.path → core.* imports work from any cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths (relative to project root) ─ override via CLI
_DEFAULT_ONNX_RUN = "onnx_model/models/run_2026-04-30_23-34-27"
_DEFAULT_ONNX_MODEL = f"{_DEFAULT_ONNX_RUN}/model_int8.onnx"
_DEFAULT_ONNX_CFG   = f"{_DEFAULT_ONNX_RUN}/onnx_config.json"
_DEFAULT_MIXED_CSV  = "artifacts/demand_ood_test.csv"
_DEFAULT_BASELINE   = "artifacts/benchmarks/f1_vs_snr_full2.csv"
_DEFAULT_OUTPUT_PDF = "artifacts/plots/demand_snr_analysis.pdf"

F1_TARGET = 0.95  # horizontal reference line

# Plot colours
_DEMAND_COLOR    = "#2563eb"   # blue  — DEMAND OOD curve
_BASELINE_COLOR  = "#dc2626"   # red   — in-distribution baseline
_TARGET_COLOR    = "#16a34a"   # green — F1 = 0.95 line
_BELOW_COLOR     = "#f97316"   # orange — bars below target


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------

def load_onnx_session(model_path: Path) -> ort.InferenceSession:
    """Load an ONNX model on CPU with all graph optimisations enabled.

    Mirrors _build_ort_session in eval_confusion_matrix.py.

    Args:
        model_path: Path to the .onnx weight file.

    Returns:
        Configured InferenceSession.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.inter_op_num_threads = os.cpu_count() or 1
    opts.intra_op_num_threads = os.cpu_count() or 1
    opts.log_severity_level = 3  # suppress verbose ORT output

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info(
        "ONNX session ready — input: %s, outputs: %s",
        session.get_inputs()[0].name,
        [o.name for o in session.get_outputs()],
    )
    return session


def load_model_config(cfg_path: Path) -> Tuple[List[str], int, int]:
    """Read labels, sample rate and window length from onnx_config.json.

    Args:
        cfg_path: Path to onnx_config.json inside the model run directory.

    Returns:
        Tuple of (labels, sr, win_samples).

    Raises:
        FileNotFoundError: If cfg_path does not exist.
    """
    if not cfg_path.exists():
        raise FileNotFoundError(f"ONNX config not found: {cfg_path}")

    with cfg_path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)

    labels: List[str] = cfg["labels"]
    sr: int = int(cfg["sr"])
    win_samples: int = int(cfg["win_samples"])
    logger.info(
        "Model config: %d labels, sr=%d, win_samples=%d", len(labels), sr, win_samples
    )
    return labels, sr, win_samples


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mixed_csv(csv_path: Path) -> List[Dict]:
    """Load rows from the mixed DEMAND CSV.

    Args:
        csv_path: Path to CSV produced by prepare_demand_ood.py.

    Returns:
        List of row dicts: filepath_mixed, label, snr_db, noise_env.

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Mixed CSV not found: {csv_path}")

    rows: List[Dict] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "filepath_mixed": row["filepath_mixed"],
                "label":          row["label"],
                "snr_db":         float(row["snr_db"]),
                "noise_env":      row["noise_env"],
            })
    logger.info("Loaded %d rows from %s", len(rows), csv_path)
    return rows


def load_baseline(baseline_path: Optional[Path]) -> Optional[Dict[float, float]]:
    """Load in-distribution SNR→F1 baseline from f1_vs_snr_full2.csv.

    The file contains per-noise-type rows.  We exclude the clean/inf row
    and average f1_mean across noise types per SNR level.

    Args:
        baseline_path: Path to the benchmark CSV (None → skip).

    Returns:
        Dict mapping snr_db → mean macro-F1, or None if unavailable.
    """
    if baseline_path is None or not baseline_path.exists():
        logger.info("No baseline CSV available, skipping.")
        return None

    buckets: Dict[float, List[float]] = defaultdict(list)
    with baseline_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                snr = float(row["snr_db"])
            except ValueError:
                continue  # skip "inf" / non-numeric
            noise_type = row.get("noise_type", "")
            if noise_type == "clean":
                continue
            buckets[snr].append(float(row["f1_mean"]))

    result = {snr: float(np.mean(vals)) for snr, vals in sorted(buckets.items())}
    logger.info("Baseline loaded: %d SNR points", len(result))
    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    e = np.exp(logits - logits.max())
    return e / e.sum()


def infer_file(
    wav_path: Path,
    session: ort.InferenceSession,
    input_name: str,
    labels: List[str],
    win_samples: int,
    sr: int,
) -> Tuple[str, float]:
    """Load one audio file and run a single ONNX forward pass.

    Preprocessing goes through core.audio_utils.prepare_window — the same
    normalisation used by core.onnx_engine.OnnxEngine — so the logit
    distribution is identical to the production pipeline.

    Args:
        wav_path:   Path to the WAV file.
        session:    Active ORT InferenceSession.
        input_name: Name of the model's input tensor (typically "input_values").
        labels:     List of label strings indexed by class position.
        win_samples: Canonical window length (samples).
        sr:         Target sample rate.

    Returns:
        Tuple of (predicted_label, confidence).

    Raises:
        FileNotFoundError: Propagated from load_wav if the file is missing.
        RuntimeError:      On ORT inference failure.
    """
    waveform, _ = load_wav(wav_path, target_sr=sr)
    prepared = prepare_window(waveform, target_samples=win_samples, do_normalize=True)
    inp = prepared.reshape(1, -1)  # (1, win_samples)

    logits = session.run(None, {input_name: inp})[0][0]  # (num_labels,)
    probs = _softmax(logits.astype(np.float32))
    pred_idx = int(np.argmax(probs))
    return labels[pred_idx], float(probs[pred_idx])


def evaluate_all(
    rows: List[Dict],
    session: ort.InferenceSession,
    labels: List[str],
    win_samples: int,
    sr: int,
) -> List[Dict]:
    """Run inference on every row and return enriched records.

    Args:
        rows:        Mixed dataset rows (from load_mixed_csv).
        session:     Active ORT InferenceSession.
        labels:      Class label list from onnx_config.json.
        win_samples: Canonical window length in samples.
        sr:          Model sample rate.

    Returns:
        Input rows augmented with 'pred_label', 'confidence', and 'correct'.
    """
    input_name = session.get_inputs()[0].name
    results: List[Dict] = []
    latencies: List[float] = []
    n = len(rows)

    logger.info("Running inference on %d examples …", n)
    for i, row in enumerate(rows):
        path = Path(row["filepath_mixed"])
        try:
            t0 = time.perf_counter()
            pred_label, confidence = infer_file(
                path, session, input_name, labels, win_samples, sr
            )
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference error for %s: %s", path.name, exc)
            continue

        r = dict(row)
        r["pred_label"] = pred_label
        r["confidence"] = confidence
        r["correct"] = int(pred_label == row["label"])
        results.append(r)

        if (i + 1) % 500 == 0:
            logger.info(
                "  %d / %d  |  avg latency %.1f ms  |  p95 %.1f ms",
                i + 1, n, np.mean(latencies), np.percentile(latencies, 95),
            )

    if latencies:
        logger.info(
            "Inference complete — avg %.2f ms, p95 %.2f ms, n=%d",
            np.mean(latencies), np.percentile(latencies, 95), len(latencies),
        )
    return results


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _macro_f1(results: List[Dict]) -> float:
    """Compute macro-F1 from a list of result rows."""
    return float(f1_score(
        [r["label"] for r in results],
        [r["pred_label"] for r in results],
        average="macro",
        zero_division=0,
    ))


def group_by_snr(results: List[Dict]) -> Dict[float, float]:
    """Compute macro-F1 per SNR level, sorted ascending.

    Args:
        results: Evaluated rows with snr_db, label, pred_label.

    Returns:
        Sorted dict mapping snr_db → macro_f1.
    """
    buckets: Dict[float, List[Dict]] = defaultdict(list)
    for r in results:
        buckets[r["snr_db"]].append(r)

    snr_f1: Dict[float, float] = {}
    for snr in sorted(buckets):
        f1 = _macro_f1(buckets[snr])
        logger.info("SNR %+5.1f dB → F1 = %.4f  (n=%d)", snr, f1, len(buckets[snr]))
        snr_f1[snr] = f1
    return snr_f1


def group_by_env(results: List[Dict], target_snr: float) -> Dict[str, float]:
    """Compute macro-F1 per noise environment at *target_snr*, sorted ascending.

    Args:
        results:    Evaluated rows.
        target_snr: SNR level to filter on.

    Returns:
        Dict mapping noise_env → macro_f1 sorted by F1 ascending (hardest first).
    """
    filtered = [r for r in results if r["snr_db"] == target_snr]
    if not filtered:
        logger.warning("No results at SNR=%.1f dB", target_snr)
        return {}

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in filtered:
        buckets[r["noise_env"]].append(r)

    env_f1 = {env: _macro_f1(rows) for env, rows in buckets.items()}
    env_f1 = dict(sorted(env_f1.items(), key=lambda kv: kv[1]))  # ascending

    hardest = next(iter(env_f1))
    logger.info(
        "Hardest env at SNR=%.1f dB: %s (F1=%.4f)",
        target_snr, hardest, env_f1[hardest],
    )
    return env_f1


def find_threshold_breach(
    snr_f1: Dict[float, float], threshold: float = F1_TARGET
) -> Optional[float]:
    """Return the first SNR (sorted ascending) where F1 drops below *threshold*.

    Args:
        snr_f1:    Dict mapping snr_db → macro_f1.
        threshold: Target F1 value.

    Returns:
        SNR value in dB, or None if F1 never drops below the threshold.
    """
    for snr in sorted(snr_f1):
        if snr_f1[snr] < threshold:
            return snr
    return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_f1_vs_snr(
    ax: plt.Axes,
    snr_f1: Dict[float, float],
    baseline_f1: Optional[Dict[float, float]],
) -> None:
    """Line chart: macro F1 vs SNR with optional baseline and target line.

    Args:
        ax:          Matplotlib Axes.
        snr_f1:      DEMAND OOD results.
        baseline_f1: Optional in-distribution baseline.
    """
    snrs = sorted(snr_f1)
    f1s  = [snr_f1[s] for s in snrs]

    ax.plot(snrs, f1s, marker="o", lw=2, ms=6, color=_DEMAND_COLOR,
            label="DEMAND (OOD)")

    if baseline_f1:
        bsnrs = sorted(baseline_f1)
        bf1s  = [baseline_f1[s] for s in bsnrs]
        ax.plot(bsnrs, bf1s, marker="s", lw=2, ms=6, color=_BASELINE_COLOR,
                linestyle="--", label="In-distribution baseline")

    ax.axhline(F1_TARGET, color=_TARGET_COLOR, lw=1.5, ls=":",
               label=f"Target F1 = {F1_TARGET}")

    for snr, f1 in zip(snrs, f1s):
        ax.annotate(f"{f1:.3f}", xy=(snr, f1), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=7, color=_DEMAND_COLOR)

    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Macro F1 vs. SNR — DEMAND OOD Evaluation",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(max(0.0, min(f1s) - 0.05), 1.03)
    ax.set_xticks(snrs)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_env_bar(
    ax: plt.Axes,
    env_f1: Dict[str, float],
    snr_db: float,
) -> None:
    """Horizontal bar chart: F1 per DEMAND environment at a fixed SNR.

    Args:
        ax:     Matplotlib Axes.
        env_f1: Mapping noise_env → macro_f1 sorted ascending.
        snr_db: SNR level displayed in the subtitle.
    """
    envs = list(env_f1.keys())
    f1s  = [env_f1[e] for e in envs]
    colors = [_DEMAND_COLOR if v >= F1_TARGET else _BELOW_COLOR for v in f1s]

    bars = ax.barh(envs, f1s, color=colors, edgecolor="white", height=0.65)
    for bar, val in zip(bars, f1s):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#1e293b")

    ax.axvline(F1_TARGET, color=_TARGET_COLOR, lw=1.5, ls=":",
               label=f"Target F1 = {F1_TARGET}")
    ax.set_xlabel("Macro F1", fontsize=11)
    ax.set_title(f"F1 per DEMAND Environment  (SNR = {snr_db:+.0f} dB)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.06)
    ax.legend(fontsize=9)
    ax.grid(axis="x", ls="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_pdf_report(
    snr_f1: Dict[float, float],
    env_f1_at_zero: Dict[str, float],
    baseline_f1: Optional[Dict[float, float]],
    output_pdf: Path,
    breach_snr: Optional[float],
) -> None:
    """Render two-panel figure and save to PDF.

    Args:
        snr_f1:         DEMAND macro-F1 per SNR.
        env_f1_at_zero: Macro-F1 per environment at SNR=0 dB.
        baseline_f1:    Optional baseline F1 per SNR.
        output_pdf:     Destination path.
        breach_snr:     SNR where F1 first drops below F1_TARGET.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if breach_snr is not None:
        subtitle = f"F1 drops below {F1_TARGET} at SNR = {breach_snr:+.1f} dB"
    else:
        subtitle = f"F1 stays ≥ {F1_TARGET} across all tested SNR levels"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.38)
    fig.suptitle(
        f"DEMAND OOD Robustness Report — ShipAssistant (ONNX INT8)\n{subtitle}",
        fontsize=11, fontstyle="italic", color="#475569", y=1.02,
    )

    _plot_f1_vs_snr(axes[0], snr_f1, baseline_f1)
    _plot_env_bar(axes[1], env_f1_at_zero, snr_db=0.0)

    fig.tight_layout()
    with pdf_backend.PdfPages(str(output_pdf)) as pp:
        pp.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    logger.info("PDF saved → %s", output_pdf)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    snr_f1: Dict[float, float],
    env_f1_at_zero: Dict[str, float],
    breach_snr: Optional[float],
) -> None:
    """Print results table to stdout.

    Args:
        snr_f1:         Macro-F1 per SNR.
        env_f1_at_zero: Macro-F1 per env at SNR=0 dB.
        breach_snr:     First SNR below F1_TARGET.
    """
    sep = "─" * 44
    print(f"\n{sep}")
    print("  DEMAND OOD Evaluation — ShipAssistant ONNX INT8")
    print(sep)
    print(f"  {'SNR (dB)':>10}  {'Macro F1':>10}")
    print(f"  {'─'*10}  {'─'*10}")
    for snr in sorted(snr_f1):
        flag = "  ← BELOW TARGET" if snr_f1[snr] < F1_TARGET else ""
        print(f"  {snr:>+10.1f}  {snr_f1[snr]:>10.4f}{flag}")
    print(sep)

    if breach_snr is not None:
        print(f"\n  ⚠  F1 < {F1_TARGET} first at SNR = {breach_snr:+.1f} dB")
    else:
        print(f"\n  ✓  F1 ≥ {F1_TARGET} at all tested SNR levels")

    print(f"\n  Hardest DEMAND environments at SNR=0 dB (top 5):")
    for env, f1 in list(env_f1_at_zero.items())[:5]:
        print(f"    {env:<14}  F1 = {f1:.4f}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate ONNX INT8 Wav2Vec2+LoRA on DEMAND OOD test set."
    )
    p.add_argument("--mixed-csv", type=Path,
                   default=_PROJECT_ROOT / _DEFAULT_MIXED_CSV,
                   help="Mixed dataset CSV from prepare_demand_ood.py.")
    p.add_argument("--onnx-model", type=Path,
                   default=_PROJECT_ROOT / _DEFAULT_ONNX_MODEL,
                   help="Path to ONNX weight file (.onnx).")
    p.add_argument("--onnx-cfg", type=Path,
                   default=_PROJECT_ROOT / _DEFAULT_ONNX_CFG,
                   help="Path to onnx_config.json (provides labels, sr, win_samples).")
    p.add_argument("--output-pdf", type=Path,
                   default=_PROJECT_ROOT / _DEFAULT_OUTPUT_PDF,
                   help="Output PDF path.")
    p.add_argument("--baseline", type=Path,
                   default=_PROJECT_ROOT / _DEFAULT_BASELINE,
                   help="In-distribution baseline CSV (snr_db, noise_type, f1_mean).")
    p.add_argument("--snr-at-env", type=float, default=0.0,
                   help="SNR level (dB) used for the per-environment bar chart.")
    return p.parse_args()


def main() -> None:
    """Full evaluation pipeline."""
    args = parse_args()

    # 1. Model
    logger.info("=== Loading model ===")
    labels, sr, win_samples = load_model_config(args.onnx_cfg)
    session = load_onnx_session(args.onnx_model)

    # 2. Dataset
    logger.info("=== Loading mixed dataset ===")
    rows = load_mixed_csv(args.mixed_csv)

    # 3. Inference
    logger.info("=== Inference ===")
    results = evaluate_all(rows, session, labels, win_samples, sr)
    if not results:
        logger.error("No results — check that mixed audio files exist.")
        sys.exit(1)

    overall_f1 = _macro_f1(results)
    logger.info("Overall macro F1 (all conditions): %.4f", overall_f1)

    # 4. Aggregate
    logger.info("=== F1 by SNR ===")
    snr_f1 = group_by_snr(results)

    logger.info("=== F1 by environment at SNR=%.1f dB ===", args.snr_at_env)
    env_f1 = group_by_env(results, target_snr=args.snr_at_env)

    baseline_f1 = load_baseline(args.baseline)
    breach_snr = find_threshold_breach(snr_f1)

    # 5. Output
    print_summary(snr_f1, env_f1, breach_snr)
    save_pdf_report(snr_f1, env_f1, baseline_f1, args.output_pdf, breach_snr)


if __name__ == "__main__":
    main()
