"""
eval_onnx_model.py — Evaluate a quantised ONNX checkpoint against the test split.

Produces the same classification report as eval_lora_model.py so you can
do a direct PyTorch ↔ ONNX accuracy comparison.

Usage
-----
    # Evaluate INT8 (default):
    python scripts/train/eval_onnx_model.py \\
        --onnx_path  onnx_model/quant_benchmark/model_int8.onnx \\
        --run_dir    lora_tune/models/run_2026-04-30_23-34-27 \\
        --data_csv   dset_meta_only_2026-04-30_15-46-30.csv

    # Compare against a previously saved PyTorch eval:
    python scripts/train/eval_onnx_model.py \\
        --onnx_path  onnx_model/quant_benchmark/model_int8.onnx \\
        --run_dir    lora_tune/models/run_2026-04-30_23-34-27 \\
        --data_csv   dset_meta_only_2026-04-30_15-46-30.csv \\
        --compare_pt lora_tune/models/run_2026-04-30_23-34-27/eval_results.json

Preprocessing note
------------------
Audio is loaded via ``core.audio_utils.load_wav`` and normalised via
``core.audio_utils.prepare_window`` — the same path used by ``OnnxEngine``
at runtime.  This guarantees the latency benchmark and the accuracy eval
see identical input tensors.  Do NOT use Wav2Vec2FeatureExtractor here;
that would introduce a numeric discrepancy between eval and production.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Ensure the project root (two levels up from scripts/train/) is on sys.path
# so that `core.*` absolute imports resolve correctly when running this script
# directly (e.g. `python scripts/train/eval_onnx_model.py`).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test-split definition  (must stay in sync with eval_lora_model.py)
# ---------------------------------------------------------------------------
TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]

SR: int = 16_000


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _load_and_prepare(
    audio_path: str,
    max_samples: int,
) -> np.ndarray:
    """Load one audio file and apply the canonical inference preprocessing.

    Uses ``core.audio_utils`` — the same path as ``OnnxEngine.predict_logits``
    — so input tensors are byte-identical between eval and production.

    Args:
        audio_path:  Path to the audio file.
        max_samples: Window length in samples (= max_seconds * SR).

    Returns:
        1-D float32 numpy array of length ``max_samples``.
    """
    from core.audio_utils import load_wav, prepare_window

    try:
        waveform, _ = load_wav(audio_path, target_sr=SR)
    except Exception as exc:
        logger.warning("Could not load %s: %s — using silence.", audio_path, exc)
        waveform = np.zeros(max_samples, dtype=np.float32)

    return prepare_window(waveform, target_samples=max_samples, do_normalize=True)


# ---------------------------------------------------------------------------
# ONNX session helpers
# ---------------------------------------------------------------------------

def _build_session(onnx_path: str) -> "onnxruntime.InferenceSession":  # type: ignore[name-defined]
    """Create an ORT InferenceSession on CPUExecutionProvider."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = os.cpu_count() or 1
    opts.intra_op_num_threads = os.cpu_count() or 1
    return ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_onnx_eval(
    session: "onnxruntime.InferenceSession",  # type: ignore[name-defined]
    df: pd.DataFrame,
    label2id: Dict[str, int],
    id2label: Dict[str, str],
    max_samples: int,
    batch_size: int = 8,
) -> Tuple[dict, float]:
    """Run full evaluation and return (metrics_dict, mean_latency_ms).

    Batching is supported — all samples are padded/truncated to ``max_samples``
    by ``prepare_window``, so the batch tensor is always rectangular.

    Args:
        session:     Active ORT InferenceSession.
        df:          DataFrame with ``audio_path`` and ``class`` columns.
        label2id:    Class-name → int index mapping.
        id2label:    Int index → class-name mapping.
        max_samples: Canonical window length (= max_seconds * SR).
        batch_size:  Number of samples per ORT call.

    Returns:
        Tuple of (metrics dict, mean per-sample latency in ms).
    """
    input_name = session.get_inputs()[0].name
    label_names = [id2label[str(i)] for i in range(len(id2label))]

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_confs: List[float] = []
    latencies_ms: List[float] = []

    rows = df.reset_index(drop=True)
    n = len(rows)
    indices = list(range(n))

    for start in tqdm(range(0, n, batch_size), desc="Evaluating ONNX"):
        batch_idx = indices[start : start + batch_size]
        batch_audio = np.stack(
            [
                _load_and_prepare(rows.iloc[i]["audio_path"], max_samples)
                for i in batch_idx
            ]
        )  # shape: (B, max_samples)
        batch_labels = [label2id[rows.iloc[i]["class"]] for i in batch_idx]

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: batch_audio})
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        per_sample_ms = elapsed_ms / len(batch_idx)
        latencies_ms.extend([per_sample_ms] * len(batch_idx))

        logits: np.ndarray = outputs[0]  # shape: (B, num_classes)
        for i, logit_row in enumerate(logits):
            probs = _softmax(logit_row.astype(np.float32))
            pred = int(np.argmax(probs))
            conf = float(probs.max())
            all_preds.append(pred)
            all_targets.append(batch_labels[i])
            all_confs.append(conf)

    report_str = classification_report(
        all_targets, all_preds, target_names=label_names, zero_division=0
    )
    logger.info("\n%s", report_str)

    metrics = {
        "accuracy":           accuracy_score(all_targets, all_preds),
        "macro_f1":           f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "weighted_f1":        f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "macro_precision":    precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "macro_recall":       recall_score(all_targets, all_preds, average="macro", zero_division=0),
        "mean_confidence":    float(np.mean(all_confs)),
        "n_samples":          len(all_targets),
        "n_classes":          len(set(all_targets)),
        "per_class": {
            label_names[i]: {
                "accuracy": float(
                    np.mean([p == t for p, t in zip(all_preds, all_targets) if t == i])
                ) if sum(1 for t in all_targets if t == i) > 0 else 0.0,
                "support": int(sum(1 for t in all_targets if t == i)),
            }
            for i in range(len(label_names))
        },
        "classification_report": report_str,
    }
    mean_latency_ms = float(np.mean(latencies_ms))
    p50_latency_ms  = float(np.percentile(latencies_ms, 50))
    p95_latency_ms  = float(np.percentile(latencies_ms, 95))
    p99_latency_ms  = float(np.percentile(latencies_ms, 99))
    metrics["p50_latency_ms"] = p50_latency_ms
    metrics["p95_latency_ms"] = p95_latency_ms
    metrics["p99_latency_ms"] = p99_latency_ms
    metrics["all_latencies_ms"] = latencies_ms

    return metrics, mean_latency_ms


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------

def _print_comparison(pt: dict, onnx: dict, onnx_label: str) -> None:
    """Print a side-by-side PT ↔ ONNX metric table."""
    metrics = [
        ("Accuracy",        "accuracy"),
        ("Macro F1",        "macro_f1"),
        ("Weighted F1",     "weighted_f1"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall",    "macro_recall"),
        ("Mean Confidence", "mean_confidence"),
    ]

    col_w = 18
    header_pt   = "PyTorch FP32".center(col_w)
    header_onnx = onnx_label.center(col_w)

    print()
    width = 2 + 22 + col_w * 2 + 10
    print("=" * width)
    print(f"  {'Metric':<22}{header_pt}{header_onnx}{'Delta':>10}")
    print("=" * width)

    for name, key in metrics:
        v_pt   = pt.get(key, float("nan"))
        v_onnx = onnx.get(key, float("nan"))
        delta  = v_onnx - v_pt
        print(
            f"  {name:<22}"
            f"{v_pt:>{col_w}.4f}"
            f"{v_onnx:>{col_w}.4f}"
            f"{delta:>+10.4f}"
        )

    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a quantised ONNX model against the test split."
    )
    parser.add_argument(
        "--onnx_path",
        required=True,
        help="Path to the ONNX model file (e.g. onnx_model/quant_benchmark/model_int8.onnx).",
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the LoRA run directory (contains best_model/config.json with id2label).",
    )
    parser.add_argument(
        "--data_csv",
        required=True,
        help="Path to the dataset metadata CSV (audio_path, audio_group, class columns required).",
    )
    parser.add_argument(
        "--compare_pt",
        default=None,
        help="(Optional) Path to eval_results.json from eval_lora_model.py for side-by-side comparison.",
    )
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--max_seconds",  type=float, default=3.0)
    args = parser.parse_args()

    onnx_path  = Path(args.onnx_path)
    run_dir    = Path(args.run_dir)
    best_model = run_dir / "best_model"

    # Derive a short label from the filename for reports (e.g. "ONNX INT8")
    stem = onnx_path.stem  # e.g. "model_int8"
    precision_tag = stem.replace("model_", "").upper()  # → "INT8"
    onnx_label = f"ONNX {precision_tag}"

    results_path = run_dir / f"eval_onnx_{precision_tag.lower()}_results.json"

    if not onnx_path.exists():
        logger.error("ONNX file not found: %s", onnx_path)
        sys.exit(1)

    if not best_model.exists():
        logger.error("best_model not found at %s", best_model)
        sys.exit(1)

    # ── Load label map from checkpoint config ──
    with open(best_model / "config.json", encoding="utf-8") as f:
        model_cfg = json.load(f)

    id2label: Dict[str, str] = model_cfg.get("id2label", {})
    if not id2label:
        logger.error("id2label not found in config.json.")
        sys.exit(1)

    label2id: Dict[str, int] = {v: int(k) for k, v in id2label.items()}
    max_samples = int(args.max_seconds * SR)

    # ── Build test split ──
    df = pd.read_csv(args.data_csv)
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)

    logger.info("Test split: %d samples", len(test_df))
    logger.info("Class distribution:\n%s", test_df["class"].value_counts().to_string())

    missing_classes = set(label2id.keys()) - set(test_df["class"].unique())
    if missing_classes:
        logger.warning(
            "Test split is missing classes: %s. Consider expanding TEST_GROUPS.",
            sorted(missing_classes),
        )

    # ── Load ONNX session ──
    logger.info("Loading ONNX session: %s", onnx_path)
    session = _build_session(str(onnx_path))

    # ── Run eval ──
    logger.info("Running evaluation (%s)...", onnx_label)
    metrics, mean_latency_ms = run_onnx_eval(
        session,
        test_df,
        label2id,
        id2label,
        max_samples=max_samples,
        batch_size=args.batch_size,
    )

    # ── Print summary ──
    logger.info("=" * 60)
    logger.info("Model:           %s", onnx_label)
    logger.info("Accuracy:        %.4f", metrics["accuracy"])
    logger.info("Macro F1:        %.4f", metrics["macro_f1"])
    logger.info("Weighted F1:     %.4f", metrics["weighted_f1"])
    logger.info("Mean confidence: %.4f", metrics["mean_confidence"])
    logger.info(
        "Latency (ms):    mean=%.1f  P50=%.1f  P95=%.1f  P99=%.1f",
        mean_latency_ms,
        metrics["p50_latency_ms"],
        metrics["p95_latency_ms"],
        metrics["p99_latency_ms"],
    )
    logger.info("Samples:         %d  |  Classes: %d", metrics["n_samples"], metrics["n_classes"])
    logger.info("=" * 60)

    # ── Optional side-by-side comparison ──
    if args.compare_pt:
        pt_path = Path(args.compare_pt)
        if not pt_path.exists():
            logger.warning("--compare_pt file not found: %s", pt_path)
        else:
            with open(pt_path, encoding="utf-8") as f:
                pt_metrics = json.load(f)
            _print_comparison(pt_metrics, metrics, onnx_label)

    # ── Save results ──
    # Exclude raw per-sample latency list from JSON (reported via benchmark_stats.py).
    metrics_for_json = {k: v for k, v in metrics.items() if k != "all_latencies_ms"}
    output = {
        "onnx_path":       str(onnx_path),
        "onnx_label":      onnx_label,
        "mean_latency_ms": mean_latency_ms,
        **metrics_for_json,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Results saved → %s", results_path)


if __name__ == "__main__":
    main()
