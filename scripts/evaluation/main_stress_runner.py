"""
scripts/vkr/stress_runner.py — Standalone 24-hour load / memory test.

Measures RSS memory consumption and per-inference latency over an
extended run (default: 86 400 s = 24 h) and writes the time-series to
a CSV file suitable for VKR Figure 4.4.

Why a separate script (not tests/stress_test.py)
-------------------------------------------------
* ``tests/stress_test.py`` is a *pytest unit test* — it can't accept
  ``--duration`` / ``--output`` CLI arguments and crashes outside pytest
  because ``sys.path`` does not include PROJECT_ROOT.
* This script fixes both: it adds PROJECT_ROOT to ``sys.path`` at
  startup and is designed to run from any working directory.

Usage
-----
    # From ANY directory — path resolution is automatic
    python scripts/vkr/stress_runner.py

    # With explicit args
    python scripts/vkr/stress_runner.py \\
        --model-dir onnx_model/models/run_2026-05-22_09-50-17 \\
        --precision int8 \\
        --duration 86400 \\
        --interval 60 \\
        --output logs/memory_24h.csv

    # Quick smoke test (5 minutes)
    python scripts/vkr/stress_runner.py --duration 300 --interval 10

Output CSV columns
------------------
    elapsed_s   — seconds since test start
    timestamp   — ISO-8601 wall-clock time
    rss_mb      — RSS memory of this process (MB)
    vms_mb      — VMS / virtual memory (MB)
    inferences  — total inferences completed so far
    lat_ms      — latency of the *last* inference call (ms)
    lat_mean_ms — rolling mean latency over this interval window
    errors      — total inference errors so far
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Fix sys.path so that ``import core`` works regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve()          # scripts/vkr/stress_runner.py
PROJECT_ROOT = _SCRIPT_DIR.parents[2]           # project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Now we can import project modules
# ---------------------------------------------------------------------------
import numpy as np
import psutil

from core.onnx_engine import OnnxEngine  # noqa: E402  (after sys.path fix)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_onnx_dir() -> Optional[Path]:
    """Return the most recent onnx_model/models/run_* directory."""
    base = PROJECT_ROOT / "onnx_model" / "models"
    if not base.exists():
        return None
    candidates = sorted(
        (d for d in base.iterdir() if d.is_dir() and (d / "onnx_config.json").exists()),
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def _fake_audio(sr: int = 16000) -> np.ndarray:
    """Generate 1 second of white noise at *sr* Hz."""
    rng = np.random.default_rng(seed=42)
    return rng.uniform(-1.0, 1.0, sr).astype(np.float32)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_stress_test(
    engine: OnnxEngine,
    duration_s: int,
    interval_s: int,
    output_path: Path,
    dry_run: bool = False,
) -> None:
    """Run the load test and write results to *output_path*.

    Args:
        engine:       Initialised OnnxEngine to benchmark.
        duration_s:   Total test duration in seconds.
        interval_s:   How often (seconds) to write a row to the CSV.
        output_path:  Destination CSV path.
        dry_run:      If True, run for max 60 s and print to stdout only.
    """
    if dry_run:
        duration_s = min(duration_s, 60)
        log.info("DRY-RUN mode: capping duration to %d s", duration_s)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "elapsed_s", "timestamp", "rss_mb", "vms_mb",
        "inferences", "lat_ms", "lat_mean_ms", "errors",
    ]

    process = psutil.Process(os.getpid())
    audio = _fake_audio(sr=engine.sr)

    # Warm-up (not counted in stats)
    log.info("Разогрев — 20 итераций...")
    for _ in range(20):
        engine.predict(audio)
    gc.collect()

    total_inferences = 0
    total_errors = 0
    last_lat_ms = 0.0
    interval_lats: list[float] = []

    start_wall = time.monotonic()
    next_log_at = start_wall + interval_s

    log.info(
        "Старт нагрузочного теста: duration=%d s, interval=%d s, output=%s",
        duration_s, interval_s, output_path,
    )

    with open(output_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        csv_f.flush()

        while True:
            now = time.monotonic()
            elapsed = now - start_wall

            if elapsed >= duration_s:
                break

            # --- single inference ---
            t0 = time.perf_counter()
            try:
                engine.predict(audio)
                last_lat_ms = (time.perf_counter() - t0) * 1000.0
                total_inferences += 1
                interval_lats.append(last_lat_ms)
            except Exception as exc:  # noqa: BLE001
                total_errors += 1
                log.warning("Inference error (#%d): %s", total_errors, exc)

            # --- periodic logging ---
            if time.monotonic() >= next_log_at:
                mem = process.memory_info()
                rss_mb = mem.rss / (1024 * 1024)
                vms_mb = mem.vms / (1024 * 1024)
                lat_mean = (
                    sum(interval_lats) / len(interval_lats)
                    if interval_lats else 0.0
                )

                row = {
                    "elapsed_s":   int(elapsed),
                    "timestamp":   datetime.now().isoformat(timespec="seconds"),
                    "rss_mb":      round(rss_mb, 2),
                    "vms_mb":      round(vms_mb, 2),
                    "inferences":  total_inferences,
                    "lat_ms":      round(last_lat_ms, 3),
                    "lat_mean_ms": round(lat_mean, 3),
                    "errors":      total_errors,
                }
                writer.writerow(row)
                csv_f.flush()

                log.info(
                    "t=%6d s | RSS=%.1f MB | inferences=%d | "
                    "lat_mean=%.1f ms | errors=%d",
                    int(elapsed), rss_mb, total_inferences, lat_mean, total_errors,
                )

                interval_lats = []
                next_log_at += interval_s

        # Final row at end of test
        mem = process.memory_info()
        writer.writerow({
            "elapsed_s":   int(time.monotonic() - start_wall),
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "rss_mb":      round(mem.rss / 1024 / 1024, 2),
            "vms_mb":      round(mem.vms / 1024 / 1024, 2),
            "inferences":  total_inferences,
            "lat_ms":      round(last_lat_ms, 3),
            "lat_mean_ms": 0.0,
            "errors":      total_errors,
        })

    log.info(
        "Тест завершён. Итог: %d инференсов, %d ошибок, лог: %s",
        total_inferences, total_errors, output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "24-hour load test for ShipAssistant ONNX engine. "
            "Writes RSS memory + latency time-series to a CSV file."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Path to ONNX bundle directory (contains onnx_config.json). "
            "Relative paths are resolved from PROJECT_ROOT. "
            "Default: latest run in onnx_model/models/."
        ),
    )
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["int8", "fp32"],
        help="ONNX precision to load (default: int8).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=86400,
        help="Test duration in seconds (default: 86400 = 24 h).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Logging interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--output",
        default="logs/memory_24h.csv",
        help="Output CSV path (relative to PROJECT_ROOT, default: logs/memory_24h.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Cap duration to 60 s for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    # Resolve model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
    else:
        model_dir = _find_latest_onnx_dir()
        if model_dir is None:
            log.error(
                "Не найдена ни одна модель в onnx_model/models/. "
                "Укажите --model-dir явно."
            )
            sys.exit(1)

    if not (model_dir / "onnx_config.json").exists():
        log.error("onnx_config.json не найден в: %s", model_dir)
        sys.exit(1)

    # Resolve output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    log.info("PROJECT_ROOT : %s", PROJECT_ROOT)
    log.info("Модель       : %s  (precision=%s)", model_dir, args.precision)
    log.info("Длительность : %d s  (%.1f ч)", args.duration, args.duration / 3600)
    log.info("Интервал лога: %d s", args.interval)
    log.info("Выходной файл: %s", output_path)

    engine = OnnxEngine(str(model_dir), precision=args.precision)

    run_stress_test(
        engine=engine,
        duration_s=args.duration,
        interval_s=args.interval,
        output_path=output_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
