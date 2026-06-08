"""
scripts/benchmark_gigaam_v2_int8.py — Latency & Quality benchmark for GigaAM-v2.

Tests three backends in a single run:

  • PyTorch FP32  — stock gigaam.load_model("v2_ctc") + model.transcribe()
  • PyTorch INT8  — same model after torch.quantization.quantize_dynamic
  • ONNX INT8     — exported with export_gigaam_onnx_int8.py, run via
                    gigaam.onnx_utils.load_onnx / infer_onnx

Pipeline: audio → transcription → rapidfuzz match to command → label

Typical workflow
----------------
    # Step 1: export ONNX (only once)
    python scripts/export_gigaam_onnx_int8.py

    # Step 2: benchmark all three backends
    python scripts/benchmark_gigaam_v2_int8.py

    # Benchmark only ONNX INT8 (skip PyTorch runs):
    python scripts/benchmark_gigaam_v2_int8.py --backends onnx_int8

    # Only latency, no quality eval:
    python scripts/benchmark_gigaam_v2_int8.py --skip-quality

    # Custom fuzzy threshold:
    python scripts/benchmark_gigaam_v2_int8.py --fuzzy-th 65

    # Quick smoke-test:
    python scripts/benchmark_gigaam_v2_int8.py --n-warmup 2 --n-bench 10

Dependencies
------------
    gigaam, rapidfuzz, torch, torchaudio, onnxruntime

Results
-------
    artifacts/benchmarks/gigaam_v2_int8_benchmark.json
    artifacts/benchmarks/gigaam_v2_int8_benchmark.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gigaam_bench")

# ── Project paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Target commands ───────────────────────────────────────────────────────────

COMMANDS: List[str] = [
    "машина",
    "приготовить машину",
    "самый малый вперед",
]
NEGATIVE_LABEL = "другие слова"
ALL_LABELS     = [NEGATIVE_LABEL] + COMMANDS

SR = 16_000


# ── Dataset helpers ───────────────────────────────────────────────────────────

_LABEL_NORM: Dict[str, str] = {
    "другие слова":       NEGATIVE_LABEL,
    "negatives":          NEGATIVE_LABEL,
    "машина":             "машина",
    "приготовить машину": "приготовить машину",
    "приготовить_машину": "приготовить машину",
    "самый малый вперед": "самый малый вперед",
    "самый_малый_вперед": "самый малый вперед",
    "самый малый вперёд": "самый малый вперед",
    "самый_малый_вперёд": "самый малый вперед",
}


def _normalise_label(raw: str) -> Optional[str]:
    raw = raw.strip()
    if raw in _LABEL_NORM:
        return _LABEL_NORM[raw]
    stripped = re.sub(r"[\s_]x\d+$", "", raw).strip()
    if stripped in _LABEL_NORM:
        return _LABEL_NORM[stripped]
    return _LABEL_NORM.get(stripped.replace("_", " "))


def collect_test_files(test_dir: Path) -> List[Tuple[Path, str]]:
    """Walk clf_dset/test and return (wav_path, canonical_label) pairs.

    Args:
        test_dir: Root of the test split.

    Returns:
        List of (Path, label) tuples, skipping source originals.
    """
    samples: List[Tuple[Path, str]] = []
    skipped = 0
    for wav in sorted(test_dir.rglob("*.wav")):
        parts = wav.parts
        if "scr" in parts or "src" in parts:
            skipped += 1
            continue
        label: Optional[str] = None
        for i, part in enumerate(parts):
            if part in ("commands", "samples") and i + 1 < len(parts):
                label = _normalise_label(parts[i + 1])
                break
            elif part == "negatives":
                label = NEGATIVE_LABEL
                break
        if label is None:
            skipped += 1
        else:
            samples.append((wav, label))
    if skipped:
        log.info("Skipped %d files (unknown label or source dir).", skipped)
    return samples


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def _match(text: str, fuzzy_th: float) -> str:
    """Match transcription to a command via rapidfuzz.

    Args:
        text:     Lowercased ASR output.
        fuzzy_th: Minimum score threshold (0–100).

    Returns:
        Matched command label or NEGATIVE_LABEL.
    """
    if not text:
        return NEGATIVE_LABEL
    from rapidfuzz import fuzz, process

    best_match, score, _ = process.extractOne(
        query=text, choices=COMMANDS, scorer=fuzz.ratio
    )
    return best_match if score >= fuzzy_th else NEGATIVE_LABEL


# ── Backend: PyTorch (FP32 and INT8) ─────────────────────────────────────────

def _load_pytorch(model_mode: str, quantize: bool):
    """Load GigaAM PyTorch model, optionally applying dynamic INT8 quantisation.

    Args:
        model_mode: GigaAM model variant (e.g. ``"v2_ctc"``).
        quantize:   If True, apply ``quantize_dynamic`` to Linear layers.

    Returns:
        GigaAM model object.
    """
    import gigaam
    import torch

    label = "INT8" if quantize else "FP32"
    log.info("Loading GigaAM-v2 PyTorch %s [%s] …", label, model_mode)
    model = gigaam.load_model(model_mode)
    model.eval()

    if quantize:
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        log.info("  Dynamic INT8 quantisation applied (Linear layers).")

    return model


def _pytorch_transcribe_fn(model) -> Callable[[Path], str]:
    """Return a transcription callable for a PyTorch GigaAM model.

    Args:
        model: GigaAM model with ``.transcribe()`` method.

    Returns:
        Function that takes a wav path and returns a lowercased string.
    """
    def _fn(wav_path: Path) -> str:
        try:
            result = model.transcribe(str(wav_path))
            if isinstance(result, dict):
                return result.get("text", "").strip().lower()
            return str(result).strip().lower()
        except Exception as exc:
            log.warning("Transcription failed (%s): %s", wav_path.name, exc)
            return ""
    return _fn


# ── Backend: ONNX INT8 ────────────────────────────────────────────────────────

def _load_onnx(onnx_dir: Path, onnx_filename: str) -> list:
    """Load a GigaAM CTC ONNX session pinned to CPUExecutionProvider.

    Forces CPU-only execution by:
    1. Passing ``providers=["CPUExecutionProvider"]`` to ORT — CUDA is never tried.
    2. Setting ``intra_op_num_threads`` so ORT uses all available cores.

    Args:
        onnx_dir:      Directory containing the ONNX file.
        onnx_filename: Filename (e.g. ``"v2_ctc_int8.onnx"``).

    Returns:
        Single-element list with the ORT InferenceSession.
    """
    import os
    import onnxruntime as ort

    # Hide all GPUs from ORT — prevents "trying CUDA, falling back to CPU" warnings
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model_path = onnx_dir / onnx_filename
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3  # suppress INFO-level ORT logs

    log.info(
        "Loading GigaAM ONNX session [CPU, %d threads]: %s …",
        opts.intra_op_num_threads,
        model_path.name,
    )
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
        sess_options=opts,
    )
    actual = session.get_providers()
    log.info("  ORT providers active: %s", actual)
    return [session]


def _onnx_transcribe_fn(sessions) -> Callable[[Path], str]:
    """Return a CPU-only transcription callable via ``gigaam.onnx_utils.transcribe_sample``.

    Passes an explicitly-CPU preprocessor to ``transcribe_sample`` so no
    PyTorch tensor is ever moved to GPU.

    Args:
        sessions: List containing a single ORT InferenceSession (CTC model).

    Returns:
        Function that takes a Path and returns a lowercased transcription string.
    """
    import torch
    import gigaam.preprocess as _gp
    from gigaam.onnx_utils import transcribe_sample, SAMPLE_RATE, FEAT_IN

    # Build preprocessor once, keep on CPU
    preprocessor = _gp.FeatureExtractor(SAMPLE_RATE, FEAT_IN)

    def _fn(wav_path: Path) -> str:
        try:
            with torch.no_grad():
                text = transcribe_sample(
                    str(wav_path), "ctc", sessions, preprocessor=preprocessor
                )
            return text.strip().lower()
        except Exception as exc:
            log.warning("ONNX inference failed (%s): %s", wav_path.name, exc)
            return ""
    return _fn


# ── Latency benchmark ─────────────────────────────────────────────────────────

def _get_ref_audio(test_dir: Optional[Path] = None) -> Path:
    """Find a reference .wav file for latency benchmarking.

    Prefers a real file from clf_dset/test; falls back to writing white noise
    to a temp file so latency can still be measured without a dataset.

    Args:
        test_dir: Optional path to clf_dset/test.

    Returns:
        Path to a usable .wav file.
    """
    if test_dir and test_dir.exists():
        wavs = list(test_dir.rglob("*.wav"))
        if wavs:
            return wavs[0]

    import soundfile as sf
    import tempfile

    tmp = Path(tempfile.mktemp(suffix=".wav"))
    rng = np.random.default_rng(42)
    wav = rng.standard_normal(SR * 3).astype(np.float32)
    wav /= (np.sqrt(np.mean(wav ** 2)) + 1e-8)
    sf.write(str(tmp), wav, SR, subtype="PCM_16")
    log.info("  No ref audio found — using synthetic white noise: %s", tmp)
    return tmp


def benchmark_latency(
    transcribe_fn: Callable[[Path], str],
    ref_audio: Path,
    n_warmup: int,
    n_bench: int,
) -> Dict[str, float]:
    """Measure transcription latency over a fixed reference file.

    Args:
        transcribe_fn: Callable that takes a Path and returns a string.
        ref_audio:     Reference audio file (reused every run).
        n_warmup:      Warm-up runs (discarded).
        n_bench:       Timed runs.

    Returns:
        Dict with avg, std, min, max, P50, P95, P99 in milliseconds.
    """
    log.info("  Warm-up: %d runs …", n_warmup)
    for _ in range(n_warmup):
        transcribe_fn(ref_audio)

    log.info("  Benchmark: %d runs …", n_bench)
    lats: List[float] = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        transcribe_fn(ref_audio)
        lats.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(lats)
    return {
        "avg": round(float(arr.mean()), 1),
        "std": round(float(arr.std()),  1),
        "min": round(float(arr.min()),  1),
        "max": round(float(arr.max()),  1),
        "P50": round(float(np.percentile(arr, 50)), 1),
        "P95": round(float(np.percentile(arr, 95)), 1),
        "P99": round(float(np.percentile(arr, 99)), 1),
        "n_warmup": n_warmup,
        "n_bench":  n_bench,
        "ref_audio": str(ref_audio.name),
    }


# ── Quality evaluation ────────────────────────────────────────────────────────

def evaluate_quality(
    transcribe_fn: Callable[[Path], str],
    test_dir: Path,
    fuzzy_th: float,
) -> Dict:
    """Evaluate command recognition on clf_dset/test.

    Args:
        transcribe_fn: Transcription callable.
        test_dir:      Root of clf_dset/test.
        fuzzy_th:      Fuzzy-match threshold (0–100).

    Returns:
        Dict with accuracy, macro metrics, per-class breakdown, confusion matrix.
    """
    samples = collect_test_files(test_dir)
    if not samples:
        log.error("No test files found in %s", test_dir)
        return {}

    log.info("  Quality eval: %d files, fuzzy_th=%.1f …", len(samples), fuzzy_th)

    label2id = {lbl: i for i, lbl in enumerate(ALL_LABELS)}

    y_true: List[int] = []
    y_pred: List[int] = []
    lats:   List[float] = []
    per_file: List[dict] = []

    for wav_path, true_label in samples:
        t0 = time.perf_counter()
        text = transcribe_fn(wav_path)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        pred_label = _match(text, fuzzy_th)
        lats.append(lat_ms)
        y_true.append(label2id[true_label])
        y_pred.append(label2id.get(pred_label, 0))

        per_file.append({
            "file":          str(wav_path.name),
            "true_label":    true_label,
            "transcription": text,
            "pred_label":    pred_label,
            "latency_ms":    round(lat_ms, 1),
            "correct":       pred_label == true_label,
        })

    n_cls = len(ALL_LABELS)
    cm = np.zeros((n_cls, n_cls), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = float(np.diag(cm).sum() / cm.sum())

    per_class: Dict[str, dict] = {}
    for i, lbl in enumerate(ALL_LABELS):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i].sum()) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   int(cm[i].sum()),
        }

    macro_f1   = np.mean([v["f1"]        for v in per_class.values()])
    macro_prec = np.mean([v["precision"] for v in per_class.values()])
    macro_rec  = np.mean([v["recall"]    for v in per_class.values()])

    return {
        "n_samples":       len(samples),
        "fuzzy_th":        fuzzy_th,
        "accuracy":        round(accuracy,         4),
        "macro_f1":        round(float(macro_f1),  4),
        "macro_precision": round(float(macro_prec),4),
        "macro_recall":    round(float(macro_rec), 4),
        "mean_latency_ms": round(float(np.mean(lats)),           1),
        "p50_latency_ms":  round(float(np.percentile(lats, 50)), 1),
        "p95_latency_ms":  round(float(np.percentile(lats, 95)), 1),
        "labels":          ALL_LABELS,
        "confusion_matrix":cm.tolist(),
        "per_class":       per_class,
        "per_file":        per_file,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _format_report(results: dict) -> str:
    """Render a human-readable benchmark report."""
    sep   = "=" * 72
    lines = [
        sep,
        "GigaAM-v2 Benchmark  (PyTorch FP32 / PyTorch INT8 / ONNX INT8)",
        f"model_mode: {results['model_mode']}    fuzzy_th: {results['fuzzy_th']}",
        sep,
        "",
        "── LATENCY (ms) ──────────────────────────────────────────────────────",
        f"  {'Backend':<16} {'avg':>7} {'P50':>7} {'P95':>7} {'P99':>7} {'std':>6}",
        f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}",
    ]
    for key in ("pytorch_fp32", "pytorch_int8", "onnx_int8"):
        d = results.get(f"latency_{key}")
        if d:
            lines.append(
                f"  {key:<16} {d['avg']:>7.1f} {d['P50']:>7.1f} "
                f"{d['P95']:>7.1f} {d['P99']:>7.1f} {d['std']:>6.1f}"
            )

    # Speedups
    fp32_avg = results.get("latency_pytorch_fp32", {}).get("avg")
    for key in ("pytorch_int8", "onnx_int8"):
        d = results.get(f"latency_{key}")
        if d and fp32_avg:
            lines.append(f"  → {key} speedup vs FP32: {fp32_avg / d['avg']:.2f}×")

    lines.append("")

    # Quality
    for key, label in [
        ("pytorch_fp32", "PyTorch FP32"),
        ("pytorch_int8", "PyTorch INT8"),
        ("onnx_int8",    "ONNX INT8"),
    ]:
        q = results.get(f"quality_{key}")
        if not q:
            continue
        lines += [
            f"── QUALITY [{label}]  fuzzy_th={q['fuzzy_th']} ─────────────────────",
            f"  Accuracy:  {q['accuracy']:.4f}   "
            f"Macro F1: {q['macro_f1']:.4f}   "
            f"P: {q['macro_precision']:.4f}   R: {q['macro_recall']:.4f}",
            f"  n_samples: {q['n_samples']}",
            "",
            "  Per-class:",
        ]
        for lbl, m in q["per_class"].items():
            lines.append(
                f"    {lbl:<30s}  P={m['precision']:.3f}  R={m['recall']:.3f}"
                f"  F1={m['f1']:.3f}  n={m['support']}"
            )
        lines.append("")

        # Compact confusion matrix
        labels = q["labels"]
        cm = np.array(q["confusion_matrix"])
        lines.append("  Confusion matrix (rows=true, cols=pred):")
        header = "  " + " " * 24 + "".join(f"{lbl[:12]:>14s}" for lbl in labels)
        lines.append(header)
        for i, lbl in enumerate(labels):
            row = f"  {lbl[:24]:<24s}" + "".join(f"{cm[i,j]:>14d}" for j in range(len(labels)))
            lines.append(row)
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Benchmark GigaAM-v2: PyTorch FP32 / PyTorch INT8 / ONNX INT8."
    )
    p.add_argument(
        "--model-mode", default="v2_ctc",
        choices=["v2_ctc", "v2_rnnt", "ctc", "rnnt"],
        help="GigaAM model variant (default: v2_ctc).",
    )
    p.add_argument(
        "--fuzzy-th", type=float, default=78.0,
        help="Fuzzy-match threshold 0–100 (default: 78.0, tuned for FP/recall balance).",
    )
    p.add_argument(
        "--backends", nargs="+",
        default=["pytorch_fp32", "pytorch_int8", "onnx_int8"],
        choices=["pytorch_fp32", "pytorch_int8", "onnx_int8"],
        help="Which backends to test (default: all three).",
    )
    p.add_argument(
        "--onnx-dir", default="onnx_model/gigaam_v2",
        help="Directory with ONNX artefacts (from export_gigaam_onnx_int8.py).",
    )
    p.add_argument(
        "--onnx-version", default=None,
        help="ONNX model name stem (default: auto-detect v2_ctc_int8).",
    )
    p.add_argument(
        "--test-dir", default="clf_dset/test",
        help="Root of clf_dset/test (default: clf_dset/test).",
    )
    p.add_argument(
        "--n-warmup", type=int, default=3,
        help="Warm-up runs for latency (default: 3).",
    )
    p.add_argument(
        "--n-bench", type=int, default=30,
        help="Timed runs for latency (default: 30).",
    )
    p.add_argument(
        "--skip-quality", action="store_true",
        help="Only measure latency, skip quality eval.",
    )
    p.add_argument(
        "--output-dir", default="artifacts/benchmarks",
        help="Output dir for JSON/TXT results.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    test_dir   = PROJECT_ROOT / args.test_dir
    output_dir = PROJECT_ROOT / args.output_dir
    onnx_dir   = PROJECT_ROOT / args.onnx_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_quality and not test_dir.exists():
        log.error("Test dir not found: %s — use --skip-quality or --test-dir.", test_dir)
        sys.exit(1)

    results: dict = {
        "model_mode": args.model_mode,
        "fuzzy_th":   args.fuzzy_th,
        "backends":   args.backends,
    }

    ref_audio = _get_ref_audio(test_dir if not args.skip_quality else None)

    # ── PyTorch FP32 ──────────────────────────────────────────────────────────
    if "pytorch_fp32" in args.backends:
        log.info("")
        log.info("━━━  PyTorch FP32  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        model = _load_pytorch(args.model_mode, quantize=False)
        fn    = _pytorch_transcribe_fn(model)

        results["latency_pytorch_fp32"] = benchmark_latency(
            fn, ref_audio, args.n_warmup, args.n_bench
        )
        log.info(
            "[FP32] Latency  avg=%.1f ms  P50=%.1f  P95=%.1f  P99=%.1f",
            results["latency_pytorch_fp32"]["avg"],
            results["latency_pytorch_fp32"]["P50"],
            results["latency_pytorch_fp32"]["P95"],
            results["latency_pytorch_fp32"]["P99"],
        )

        if not args.skip_quality:
            results["quality_pytorch_fp32"] = evaluate_quality(
                fn, test_dir, args.fuzzy_th
            )
            q = results["quality_pytorch_fp32"]
            log.info("[FP32] Quality  acc=%.4f  macro_F1=%.4f", q["accuracy"], q["macro_f1"])

        del model

    # ── PyTorch INT8 ──────────────────────────────────────────────────────────
    if "pytorch_int8" in args.backends:
        log.info("")
        log.info("━━━  PyTorch INT8 (dynamic)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        model = _load_pytorch(args.model_mode, quantize=True)
        fn    = _pytorch_transcribe_fn(model)

        results["latency_pytorch_int8"] = benchmark_latency(
            fn, ref_audio, args.n_warmup, args.n_bench
        )
        log.info(
            "[INT8-PT] Latency  avg=%.1f ms  P50=%.1f  P95=%.1f  P99=%.1f",
            results["latency_pytorch_int8"]["avg"],
            results["latency_pytorch_int8"]["P50"],
            results["latency_pytorch_int8"]["P95"],
            results["latency_pytorch_int8"]["P99"],
        )

        if not args.skip_quality:
            results["quality_pytorch_int8"] = evaluate_quality(
                fn, test_dir, args.fuzzy_th
            )
            q = results["quality_pytorch_int8"]
            log.info("[INT8-PT] Quality  acc=%.4f  macro_F1=%.4f", q["accuracy"], q["macro_f1"])

        del model

    # ── ONNX INT8 ─────────────────────────────────────────────────────────────
    if "onnx_int8" in args.backends:
        log.info("")
        log.info("━━━  ONNX INT8 (ORT)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Auto-detect ONNX version name
        onnx_version = args.onnx_version
        if onnx_version is None:
            candidate = f"{args.model_mode}_int8"
            if (onnx_dir / f"{candidate}.onnx").exists():
                onnx_version = candidate
            else:
                log.error(
                    "ONNX INT8 model not found: %s/%s.onnx\n"
                    "Run: python scripts/export_gigaam_onnx_int8.py first.",
                    onnx_dir, candidate,
                )
                log.info("Skipping ONNX INT8 backend.")
                onnx_version = None

        if onnx_version:
            sessions = _load_onnx(onnx_dir, f"{onnx_version}.onnx")
            fn = _onnx_transcribe_fn(sessions)

            results["latency_onnx_int8"] = benchmark_latency(
                fn, ref_audio, args.n_warmup, args.n_bench
            )
            log.info(
                "[INT8-ORT] Latency  avg=%.1f ms  P50=%.1f  P95=%.1f  P99=%.1f",
                results["latency_onnx_int8"]["avg"],
                results["latency_onnx_int8"]["P50"],
                results["latency_onnx_int8"]["P95"],
                results["latency_onnx_int8"]["P99"],
            )

            if not args.skip_quality:
                results["quality_onnx_int8"] = evaluate_quality(
                    fn, test_dir, args.fuzzy_th
                )
                q = results["quality_onnx_int8"]
                log.info("[INT8-ORT] Quality  acc=%.4f  macro_F1=%.4f", q["accuracy"], q["macro_f1"])

    # ── Speedups ──────────────────────────────────────────────────────────────
    fp32_avg = results.get("latency_pytorch_fp32", {}).get("avg")
    if fp32_avg:
        for key in ("pytorch_int8", "onnx_int8"):
            d = results.get(f"latency_{key}")
            if d:
                results[f"speedup_{key}_vs_fp32"] = round(fp32_avg / d["avg"], 3)

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = output_dir / "gigaam_v2_int8_benchmark.json"
    txt_path  = output_dir / "gigaam_v2_int8_benchmark.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("JSON → %s", json_path)

    report = _format_report(results)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    print()
    print(report)
    log.info("TXT  → %s", txt_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
