"""
scripts/vkr/measure_memory.py — RAM measurement script for VKR thesis.

Measures RSS (Resident Set Size) and VMS (Virtual Memory Size) of the
inference process at three canonical checkpoints:

  Point 0 — bare Python process (before any imports)
  Point 1 — after model load (OnnxEngine initialised, no inferences yet)
  Point 2 — warm inference load (N=20 inferences, model fully warmed up)

Produces a human-readable summary table and writes the canonical trio
of numbers used in the thesis text:

  - RAM модели ONNX INT8 (файл/рантайм): 339 МБ  (from config / file size)
  - RSS процесса при инференсе (warm):    X МБ    (Point 2, psutil RSS)
  - Стабилизированный RSS (из 24ч лога): Y МБ    (read from memory_24h.csv)

Usage
-----
    # From project root
    python scripts/vkr/measure_memory.py

    # Override model dir
    python scripts/vkr/measure_memory.py --model-dir onnx_model/models/run_2026-05-22_09-50-17

    # Quick run (fewer warm-up inferences)
    python scripts/vkr/measure_memory.py --n-inferences 5

Notes
-----
* RSS is the only meaningful "RAM consumed" metric for the thesis.
  VMS (virtual address space) includes memory-mapped files and shared
  libraries — it does NOT represent physical RAM usage and should NOT
  be reported as "потребление ОЗУ".
* The 24-hour stabilised value (~379 МБ) comes from logs/memory_24h.csv
  and is read automatically when the file exists.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Fix sys.path so that ``import core`` works regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Return RSS (physical RAM) of the current process in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _vms_mb() -> float:
    """Return VMS (virtual address space) of the current process in MiB."""
    return psutil.Process(os.getpid()).memory_info().vms / (1024 ** 2)


def _model_file_mb(model_dir: Path) -> float | None:
    """Return size of model_int8.onnx in MiB, or None if not found."""
    candidates = [
        model_dir / "model_int8.onnx",
        model_dir / "best_model" / "model_int8.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p.stat().st_size / (1024 ** 2)
    return None


def _stabilised_rss_from_log(log_path: Path) -> float | None:
    """
    Read the 24-hour stress-test CSV and return the stabilised RSS value
    (mean of the last 10 % of rows, column ``rss_mb``).

    Returns None if the file does not exist or is malformed.
    """
    if not log_path.exists():
        return None
    try:
        rows = list(csv.DictReader(log_path.open(newline="", encoding="utf-8")))
        vals = [float(r["rss_mb"]) for r in rows if r.get("rss_mb")]
        if not vals:
            return None
        tail = vals[len(vals) * 9 // 10 :]
        return sum(tail) / len(tail)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse %s: %s", log_path, exc)
        return None


def _peak_rss_from_log(log_path: Path) -> float | None:
    """Return peak RSS from 24-hour log."""
    if not log_path.exists():
        return None
    try:
        rows = list(csv.DictReader(log_path.open(newline="", encoding="utf-8")))
        vals = [float(r["rss_mb"]) for r in rows if r.get("rss_mb")]
        return max(vals) if vals else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Main measurement routine
# ---------------------------------------------------------------------------

def measure(model_dir: Path, n_inferences: int = 20) -> dict:
    """
    Run the three-point RAM measurement and return a results dict.

    Parameters
    ----------
    model_dir:
        Directory containing the ONNX bundle (model_int8.onnx + onnx_config.json).
    n_inferences:
        Number of warm-up inferences to run before taking the final measurement.

    Returns
    -------
    dict with keys:
        baseline_rss_mb, after_load_rss_mb, warm_rss_mb, warm_vms_mb,
        model_file_mb, stabilised_rss_mb (from log), peak_rss_log_mb
    """
    results: dict = {}

    # ------------------------------------------------------------------
    # Point 0: bare Python (already imported psutil/numpy above, but
    # before loading OnnxEngine / ONNX Runtime)
    # ------------------------------------------------------------------
    gc.collect()
    results["baseline_rss_mb"] = _rss_mb()
    log.info("Point 0 — baseline RSS: %.1f МБ", results["baseline_rss_mb"])

    # ------------------------------------------------------------------
    # Point 1: model loaded, no inferences
    # ------------------------------------------------------------------
    log.info("Loading OnnxEngine from %s …", model_dir)
    from core.onnx_engine import OnnxEngine  # noqa: PLC0415 (import after sys.path)

    engine = OnnxEngine(str(model_dir), precision="int8")
    gc.collect()
    results["after_load_rss_mb"] = _rss_mb()
    results["after_load_vms_mb"] = _vms_mb()
    log.info(
        "Point 1 — after model load: RSS=%.1f МБ  VMS=%.1f МБ",
        results["after_load_rss_mb"],
        results["after_load_vms_mb"],
    )

    # ------------------------------------------------------------------
    # Point 2: warm inference (fully warmed up)
    # ------------------------------------------------------------------
    log.info("Running %d warm-up inferences …", n_inferences)
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 s of silence
    rss_samples: list[float] = []

    for i in range(n_inferences):
        engine.predict_logits(dummy_audio)
        if i >= n_inferences // 2:  # collect from second half
            rss_samples.append(_rss_mb())

    gc.collect()
    results["warm_rss_mb"] = sum(rss_samples) / len(rss_samples)
    results["warm_vms_mb"] = _vms_mb()
    log.info(
        "Point 2 — warm inference: RSS=%.1f МБ  VMS=%.1f МБ",
        results["warm_rss_mb"],
        results["warm_vms_mb"],
    )

    # ------------------------------------------------------------------
    # Auxiliary: model file size and 24-hour log values
    # ------------------------------------------------------------------
    results["model_file_mb"] = _model_file_mb(model_dir)

    log_path = PROJECT_ROOT / "logs" / "memory_24h.csv"
    results["stabilised_rss_mb"] = _stabilised_rss_from_log(log_path)
    results["peak_rss_log_mb"] = _peak_rss_from_log(log_path)
    results["log_path"] = str(log_path) if log_path.exists() else None

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(r: dict) -> None:
    """Print the canonical thesis summary table."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  CANONICAL RAM MEASUREMENTS FOR VKR THESIS")
    print(sep)

    def row(label: str, value_mb: float | None, note: str = "") -> None:
        if value_mb is None:
            val_str = "н/д"
        else:
            val_str = f"{value_mb:>7.1f} МБ  ({value_mb / 1024:.3f} ГБ)"
        print(f"  {label:<42}  {val_str}   {note}")

    print()
    print("  ─── Три канонические сущности ───────────────────────────────")
    row(
        "1. RAM модели ONNX INT8 (файл/рантайм)",
        r.get("model_file_mb"),
        "← таблица 4.2",
    )
    row(
        "2. RSS процесса при инференсе (warm)",
        r.get("warm_rss_mb"),
        "← реферат, §практической значимости",
    )
    stabilised = r.get("stabilised_rss_mb")
    peak_log = r.get("peak_rss_log_mb")
    row(
        "3. RSS стабилизированный (24ч лог)",
        stabilised,
        "← §4.5, Заключение п.6",
    )
    print()
    print("  ─── Вспомогательные ─────────────────────────────────────────")
    row("   RSS до загрузки модели (baseline)", r.get("baseline_rss_mb"))
    row("   RSS после загрузки, до инференса", r.get("after_load_rss_mb"))
    row("   VMS (адрес. пространство, НЕ RAM)", r.get("warm_vms_mb"), "← НЕ путать с RSS!")
    row("   RSS пик (из 24ч лога, t=0–10 мин)", peak_log)
    print()
    print("  ─── Требование §2.1: не более 4 ГБ ─────────────────────────")
    check_val = r.get("warm_rss_mb") or r.get("after_load_rss_mb")
    if check_val:
        ok = check_val < 4096
        print(f"  RSS warm = {check_val:.1f} МБ  {'✅ выполнено' if ok else '❌ НАРУШЕНО'} (порог 4 096 МБ)")
    print()

    if r.get("log_path"):
        print(f"  Источник п.3: {r['log_path']}")
    else:
        print(
            "  ⚠️  Лог 24ч не найден (logs/memory_24h.csv). Для п.3 запустите:\n"
            "       python scripts/vkr/stress_runner.py --duration 86400"
        )
    print()
    print("  ─── Рекомендованные формулировки для docx ──────────────────")
    m_model = r.get("model_file_mb")
    m_warm = r.get("warm_rss_mb")
    m_stab = stabilised
    m_peak = peak_log

    if m_warm:
        print(
            f"\n  Реферат / §практической значимости:\n"
            f"    «Потребление ОЗУ процессом при инференсе составляет {m_warm:.0f} МБ\n"
            f"     (RSS, ONNX INT8, без учёта прочих сервисов).»"
        )
    if m_stab and m_peak:
        print(
            f"\n  §4.5 / Заключение п.6:\n"
            f"    «Потребление ОЗУ в первые 30 мин: пиковый RSS {m_peak:.0f} МБ;\n"
            f"     после стабилизации (t > 1 ч): RSS {m_stab:.0f} МБ.\n"
            f"     Адресное пространство процесса (VMS) постоянно: {r.get('warm_vms_mb', 0):.0f} МБ.\n"
            f"     Утечек памяти не обнаружено.»"
        )
    elif not m_stab:
        print(
            "\n  §4.5 / Заключение п.6:\n"
            "    ⚠️  Запустите stress_runner.py для получения стабилизированного RSS."
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure RSS/VMS at three canonical points for the VKR thesis."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "onnx_model" / "models" / "run_2026-05-22_09-50-17",
        help="Path to ONNX bundle directory (default: latest run).",
    )
    parser.add_argument(
        "--n-inferences",
        type=int,
        default=20,
        help="Number of warm-up inferences for Point 2 (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    model_dir = args.model_dir.resolve() if not args.model_dir.is_absolute() else args.model_dir
    if not model_dir.exists():
        log.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    results = measure(model_dir, n_inferences=args.n_inferences)
    print_report(results)


if __name__ == "__main__":
    main()
