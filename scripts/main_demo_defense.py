"""
scripts/demo_defense.py — Thesis defense demonstration CLI for ShipAssistant.

Usage
-----
  python scripts/demo_defense.py --mode bench      # synthetic-noise benchmark
  python scripts/demo_defense.py --mode metrics    # per-class P/R on 4 phrases
  python scripts/demo_defense.py --mode realtime   # live mic recognition loop
  python scripts/demo_defense.py --mode api        # start FastAPI server

Bench mode
----------
Runs N synthetic inference cycles (N = cfg.benchmark.samples) using a random
1-second waveform at the model's target sample rate.  Prints a formatted table
to stdout and saves the full result set to:
    artifacts/benchmarks/defense_metrics.json

Metrics mode
------------
Walks ``--data-dir`` (default: artifacts/eval) where each subdirectory name is
one of the 4 target phrases (== a class label) and contains evaluation .wav
clips. Computes per-class precision / recall / F1 / mean confidence and prints
a defense-ready table. Saves results to:
    artifacts/benchmarks/defense_pr.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Resolve project root so absolute imports work when the script is invoked
# directly (python scripts/demo_defense.py) without package installation.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import settings
from core.engine import AudioEngine, create_engine
from core.logger import get_logger

logger = get_logger("demo_defense")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _memory_mb() -> float:
    """Return current RSS memory usage in MB, or -1 if psutil is unavailable."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1_048_576, 2)
    except ImportError:
        return -1.0


def _separator(widths: List[int], char: str = "─") -> str:
    return "  " + "  ".join(char * w for w in widths)


def _row(values: List[str], widths: List[int]) -> str:
    cells = [str(v).ljust(w) for v, w in zip(values, widths)]
    return "  " + "  ".join(cells)


# ── Bench mode ────────────────────────────────────────────────────────────────

def run_bench(engine: AudioEngine, n_samples: int) -> None:
    """
    Run *n_samples* synthetic inference cycles and print a results table.

    A random waveform (Gaussian noise, shape=(sample_rate,)) is used as input
    so the benchmark can run without a microphone or real audio files.
    The randomness is seeded for reproducibility.
    """
    rng = np.random.default_rng(seed=42)
    sr: int = settings.audio.sample_rate
    engine_type: str = settings.model.type.upper()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         ShipAssistant — Thesis Defense Benchmark             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Engine      : {engine_type}  "
          f"(precision={settings.onnx.precision}, T={settings.onnx.temperature}, "
          f"providers={settings.onnx.providers})")
    print(f"  Model path  : {settings.paths.onnx_model}")
    print(f"  Sample rate : {sr} Hz  |  Window: {settings.audio.window_seconds} s  "
          f"({int(sr * settings.audio.window_seconds)} samples)")
    print(f"  Runs        : {n_samples}")
    print()

    col_headers = ["Run", "Label", "Confidence", "Latency (ms)", "Memory (MB)"]
    col_widths   = [4,    22,      10,           12,            11]

    print(_row(col_headers, col_widths))
    print(_separator(col_widths))

    rows: List[Dict[str, Any]] = []

    for i in range(1, n_samples + 1):
        waveform = rng.standard_normal(sr).astype(np.float32)
        mem_before = _memory_mb()

        result = engine.predict(waveform)

        mem_after = _memory_mb()
        mem_delta = mem_after if mem_before < 0 else mem_after

        row_data = {
            "run": i,
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "latency_ms": result["latency_ms"],
            "memory_mb": mem_delta,
        }
        rows.append(row_data)

        print(_row(
            [
                str(i),
                result["label"],
                f"{result['confidence']:.4f}",
                f"{result['latency_ms']:.2f}",
                f"{mem_delta:.1f}" if mem_delta >= 0 else "n/a",
            ],
            col_widths,
        ))

    print(_separator(col_widths))

    # Summary statistics
    latencies = [r["latency_ms"] for r in rows]
    confidences = [r["confidence"] for r in rows]
    memories = [r["memory_mb"] for r in rows if r["memory_mb"] >= 0]

    avg_lat = round(float(np.mean(latencies)), 3)
    p95_lat = round(float(np.percentile(latencies, 95)), 3)
    avg_conf = round(float(np.mean(confidences)), 4)
    avg_mem = round(float(np.mean(memories)), 2) if memories else -1.0

    print(_row(
        ["AVG", "—", f"{avg_conf:.4f}", f"{avg_lat:.2f}", f"{avg_mem:.1f}" if avg_mem >= 0 else "n/a"],
        col_widths,
    ))
    print(_row(
        ["P95", "—", "—", f"{p95_lat:.2f}", "—"],
        col_widths,
    ))
    print()

    # Save JSON
    summary = {
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "avg_confidence": avg_conf,
        "avg_memory_mb": avg_mem,
    }
    _save_json(engine_type, n_samples, rows, summary)


def _save_json(
    engine_type: str,
    n_samples: int,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    """Persist benchmark results to artifacts/benchmarks/defense_metrics.json."""
    out_dir = settings.paths.artifacts_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "defense_metrics.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "engine_type": engine_type,
        "model_path": str(settings.paths.onnx_model),
        "use_int8": settings.onnx.use_int8,
        "sample_rate": settings.audio.sample_rate,
        "samples": n_samples,
        "results": rows,
        "summary": summary,
    }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"  Results saved → {out_path}")
    print()


# ── Metrics mode (per-class P/R on the 4 target phrases) ─────────────────────

def _iter_eval_clips(data_dir: Path) -> List[Tuple[str, Path]]:
    """Walk *data_dir* and return ``[(true_label, wav_path), ...]``.

    Layout::

        data_dir/
          phrase_a/
            clip_001.wav
            clip_002.wav
          phrase_b/
            ...
    """
    pairs: List[Tuple[str, Path]] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for wav in sorted(class_dir.glob("*.wav")):
            pairs.append((class_dir.name, wav))
    return pairs


def run_metrics(engine: AudioEngine, data_dir: Path) -> None:
    """Compute and print per-class precision / recall on the 4 target phrases.

    Uses the unified preprocessing pipeline from ``core.audio_utils`` to
    guarantee parity with the PyTorch / ONNX backends.
    """
    from core.audio_utils import load_wav, prepare_window

    pairs = _iter_eval_clips(data_dir)
    if not pairs:
        print(f"\n  [ERROR] No .wav clips found under {data_dir}\n"
              f"  Expected layout: {data_dir}/<phrase>/clip_*.wav\n")
        sys.exit(1)

    sr = settings.audio.sample_rate
    target_samples = int(settings.audio.window_seconds * sr)
    target_labels = settings.recognition.per_label_thresholds.keys() or engine.labels
    target_labels = list(target_labels)

    # Per-class counters: TP / FP / FN
    tp: Dict[str, int] = {l: 0 for l in engine.labels}
    fp: Dict[str, int] = {l: 0 for l in engine.labels}
    fn: Dict[str, int] = {l: 0 for l in engine.labels}
    conf_by_class: Dict[str, List[float]] = {l: [] for l in engine.labels}

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   ShipAssistant — Per-class Precision / Recall (4 phrases)   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Engine    : {settings.model.type.upper()} "
          f"(precision={settings.onnx.precision}, T={settings.onnx.temperature})")
    print(f"  Data dir  : {data_dir}")
    print(f"  Clips     : {len(pairs)} across {len(set(t for t, _ in pairs))} class(es)")
    print()

    for true_label, wav_path in pairs:
        if true_label not in engine.labels:
            logger.warning(
                "Skipping clip with unknown label %r (path=%s)",
                true_label, wav_path,
            )
            continue
        waveform, _ = load_wav(wav_path, target_sr=sr)
        prepared = prepare_window(waveform, target_samples=target_samples)
        result = engine.predict(prepared)
        pred_label: str = result["label"]
        confidence: float = float(result["confidence"])

        if pred_label == true_label:
            tp[true_label] += 1
            conf_by_class[true_label].append(confidence)
        else:
            fn[true_label] += 1
            fp[pred_label] += 1

    # Render table
    headers = ["Label", "TP", "FP", "FN", "Precision", "Recall", "F1", "Mean conf"]
    widths = [22, 4, 4, 4, 9, 7, 6, 9]
    print(_row(headers, widths))
    print(_separator(widths))

    rows: List[Dict[str, Any]] = []
    macro_p = macro_r = macro_f = 0.0
    counted = 0
    for label in engine.labels:
        if (tp[label] + fp[label] + fn[label]) == 0:
            continue                                    # never seen this class
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        mean_conf = float(np.mean(conf_by_class[label])) if conf_by_class[label] else 0.0

        macro_p += precision
        macro_r += recall
        macro_f += f1
        counted += 1

        rows.append({
            "label": label, "tp": tp[label], "fp": fp[label], "fn": fn[label],
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "mean_confidence": round(mean_conf, 4),
        })
        print(_row(
            [label, str(tp[label]), str(fp[label]), str(fn[label]),
             f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", f"{mean_conf:.3f}"],
            widths,
        ))

    print(_separator(widths))
    if counted:
        macro_p /= counted
        macro_r /= counted
        macro_f /= counted
        print(_row(
            ["MACRO", "—", "—", "—",
             f"{macro_p:.3f}", f"{macro_r:.3f}", f"{macro_f:.3f}", "—"],
            widths,
        ))
    print()

    out_dir = settings.paths.artifacts_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "defense_pr.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "engine": settings.model.type,
                "precision": settings.onnx.precision,
                "temperature": settings.onnx.temperature,
                "data_dir": str(data_dir),
                "per_class": rows,
                "macro": {
                    "precision": round(macro_p, 4),
                    "recall": round(macro_r, 4),
                    "f1": round(macro_f, 4),
                } if counted else {},
            }, fh, indent=2, ensure_ascii=False,
        )
    print(f"  Results saved → {out_path}")
    print()


# ── Realtime mode ─────────────────────────────────────────────────────────────

def run_realtime(engine: AudioEngine) -> None:
    """Start the live microphone recognition loop (delegates to RealTimeRecognizer)."""
    from core.recognizer import RealTimeRecognizer

    recognizer = RealTimeRecognizer(
        sample_rate=settings.audio.sample_rate,
        window_s=settings.audio.window_seconds,
        stride_s=settings.audio.stride_seconds,
    )

    def _on_audio(audio_chunk: np.ndarray) -> None:
        audio_data = audio_chunk.flatten().astype(np.float32)
        result = engine.predict(audio_data)
        label: str = result["label"]
        conf: float = result["confidence"]
        threshold = settings.recognition.per_label_thresholds.get(
            label, settings.recognition.default_confidence
        )
        if conf >= threshold:
            print(f"  >>> ДЕТЕКЦИЯ: {label.upper()}  (conf={conf:.3f})")

    try:
        recognizer.start_stream(callback=_on_audio)
        print("  Система слушает...  (Ctrl+C для выхода)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Остановка по команде пользователя.")
    finally:
        recognizer.stop()


# ── API mode ──────────────────────────────────────────────────────────────────

def run_api() -> None:
    """Launch the FastAPI server via uvicorn."""
    import uvicorn

    print(f"  Запуск API на {settings.api.host}:{settings.api.port} ...")
    uvicorn.run(
        "src.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ShipAssistant — thesis defense demo script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/demo_defense.py --mode bench\n"
            "  python scripts/demo_defense.py --mode realtime\n"
            "  python scripts/demo_defense.py --mode api\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["bench", "metrics", "realtime", "api"],
        default="bench",
        help="Operation mode (default: bench)",
    )
    parser.add_argument(
        "--engine",
        choices=["onnx", "torch"],
        default=None,
        help="Override cfg.model.type for this run",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override cfg.benchmark.samples for bench mode",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Eval clips directory for --mode metrics. "
            "Default: cfg.paths.artifacts_dir / 'eval'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    # Engine factory — CLI flag overrides config
    logger.info(f"demo_defense starting — mode={args.mode!r} engine_override={args.engine!r}")

    if args.mode == "api":
        # API mode does not need a pre-loaded engine; api.py manages its own lifecycle
        run_api()
        return

    try:
        engine = create_engine(settings, mode=args.engine)
    except Exception as exc:
        logger.error(f"Engine init failed: {exc}")
        print(f"\n  [ERROR] Could not load model: {exc}\n")
        sys.exit(1)

    n_samples = args.samples if args.samples is not None else settings.benchmark.samples

    if args.mode == "bench":
        run_bench(engine, n_samples)
    elif args.mode == "metrics":
        data_dir = args.data_dir or (settings.paths.artifacts_dir / "eval")
        run_metrics(engine, Path(data_dir))
    elif args.mode == "realtime":
        run_realtime(engine)


if __name__ == "__main__":
    main()
