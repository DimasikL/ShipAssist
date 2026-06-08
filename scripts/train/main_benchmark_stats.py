"""
scripts/train/benchmark_stats.py — Thesis-level statistical supplement.

What this script adds on top of benchmark_compare_all.py
---------------------------------------------------------
1. **Clopper-Pearson 95 % confidence intervals** for accuracy of each
   method in the comparison table (Table 4.3 of the thesis). The
   Clopper-Pearson interval is the standard exact binomial CI; it is
   conservative but appropriate for small samples (N=123).

2. **P50 / P95 / P99 latency** for the main ONNX INT8 model, computed
   from a dedicated warmup-excluded benchmark run (N_BENCH inferences on
   the same audio window). The 10-run figure in defense_metrics.json is
   insufficient for percentile stability — P95 from 10 samples has
   roughly ±35 % relative error.

Usage
-----
    # Quick mode: use existing JSONs, just add CIs (no latency rerun):
    python scripts/train/benchmark_stats.py --no_latency_rerun

    # Full mode (default): also re-runs latency benchmark:
    python scripts/train/benchmark_stats.py \
        --onnx_dir onnx_model/models/run_2026-02-25_19-07-15/best_model \
        --n_bench 300 \
        --warmup 20

Output
------
    artifacts/benchmarks/thesis_stats.json   — machine-readable
    artifacts/benchmarks/thesis_stats.txt    — formatted table for copy-paste into thesis

Table format (thesis_stats.txt)
--------------------------------
The printed table matches the structure of thesis Table 4.3 and is ready
for direct inclusion. Column headers are in Russian to match the document.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCH_DIR   = _PROJECT_ROOT / "artifacts" / "benchmarks"
LORA_DIR    = _PROJECT_ROOT / "lora_tune" / "models" / "run_2026-04-30_23-34-27"

# ── Clopper-Pearson ──────────────────────────────────────────────────────────

def clopper_pearson_ci(
    k: int,
    n: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Exact binomial (Clopper-Pearson) confidence interval.

    The interval is guaranteed to contain the true proportion with
    probability ≥ 1 − α for all sample sizes. It is slightly conservative
    (wider than necessary) but has no approximation error — appropriate
    for the committee when N=123.

    Args:
        k:     Number of successes (correct predictions).
        n:     Number of trials (total predictions).
        alpha: Significance level (0.05 → 95 % CI).

    Returns:
        Tuple ``(lower, upper)``.  Returns ``(0, upper)`` when k=0 and
        ``(lower, 1)`` when k=n.

    References:
        Clopper & Pearson (1934). "The Use of Confidence or Fiducial Limits
        Illustrated in the Case of the Binomial."
        Biometrika, 26(4), 404–413.
    """
    from scipy.stats import beta as beta_dist

    if n <= 0:
        return (0.0, 1.0)
    lo = float(beta_dist.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return (lo, hi)


# ── Latency benchmark ────────────────────────────────────────────────────────

def run_latency_benchmark(
    onnx_dir: str,
    n_bench: int = 300,
    warmup: int = 20,
    sr: int = 16_000,
) -> Dict[str, float]:
    """Run dedicated latency benchmark and return percentile statistics.

    Uses a fixed synthetic audio window (ones, normalized) so that the
    measured time is purely model latency with no I/O noise.

    Args:
        onnx_dir: Path to the ONNX model bundle directory
                  (contains ``onnx_config.json``).
        n_bench:  Number of inference calls AFTER warmup (default 300).
        warmup:   Number of warmup calls before timing starts (default 20).
        sr:       Sample rate in Hz (default 16 000).

    Returns:
        Dict with keys: ``mean_ms``, ``std_ms``, ``p50_ms``, ``p95_ms``,
        ``p99_ms``, ``min_ms``, ``max_ms``, ``n_bench``, ``warmup``.

    Raises:
        RuntimeError: If the ONNX engine fails to load.
    """
    from core.onnx_engine import OnnxEngine

    logger.info("Loading ONNX engine from %s …", onnx_dir)
    engine = OnnxEngine(onnx_dir=onnx_dir, precision="int8")

    # Synthetic audio: 1-second window of uniform noise, normalized
    rng = np.random.default_rng(42)
    audio = rng.uniform(-0.5, 0.5, sr).astype(np.float32)
    audio /= max(np.abs(audio).max(), 1e-8)

    logger.info("Warming up (%d calls) …", warmup)
    for _ in range(warmup):
        engine.predict_logits(audio)

    logger.info("Benchmarking (%d calls) …", n_bench)
    latencies: List[float] = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        engine.predict_logits(audio)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d done.", i + 1, n_bench)

    lat = np.array(latencies)
    return {
        "mean_ms": float(lat.mean()),
        "std_ms":  float(lat.std()),
        "p50_ms":  float(np.percentile(lat, 50)),
        "p95_ms":  float(np.percentile(lat, 95)),
        "p99_ms":  float(np.percentile(lat, 99)),
        "min_ms":  float(lat.min()),
        "max_ms":  float(lat.max()),
        "n_bench": n_bench,
        "warmup":  warmup,
    }


# ── Table 4.3 — method comparison with CIs ───────────────────────────────────

def _load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_table_43(
    latency_stats: Optional[Dict[str, float]],
) -> Tuple[List[dict], str]:
    """Build Table 4.3 rows with Clopper-Pearson CIs.

    Loads from the benchmark JSONs already present in artifacts/benchmarks/.

    Args:
        latency_stats: Output of ``run_latency_benchmark()``, or ``None``
                       to use the latency already stored in the JSON files.

    Returns:
        Tuple of ``(rows_list, formatted_text_table)``.
    """
    mfcc_clean = _load_json(BENCH_DIR / "mfcc_svm_results.json")
    mfcc_noisy = _load_json(BENCH_DIR / "mfcc_svm_noisy_results.json")
    wh_clean   = _load_json(BENCH_DIR / "whisper_tiny_results.json")
    wh_noisy   = _load_json(BENCH_DIR / "whisper_tiny_noisy_results.json")
    onnx_clean = _load_json(LORA_DIR / "eval_onnx_int8_results.json")

    sources = [
        ("MFCC + SVM",               mfcc_clean, mfcc_noisy),
        ("Whisper-tiny (zero-shot)",  wh_clean,   wh_noisy),
        ("LoRA-Wav2Vec2 + ONNX INT8", onnx_clean, None),
    ]

    rows = []
    for method_name, clean, noisy in sources:
        if clean is None:
            rows.append({"method": method_name, "status": "NOT RUN"})
            continue

        n = int(clean.get("n_samples", 0))
        acc = float(clean.get("accuracy", 0.0))
        k   = round(acc * n)
        ci_lo, ci_hi = clopper_pearson_ci(k, n, alpha=0.05)

        f1_clean = float(clean.get("macro_f1", 0.0))
        f1_noisy = float(noisy.get("macro_f1", 0.0)) if noisy else None

        # Latency: prefer re-run stats for the main ONNX model
        if latency_stats and method_name.startswith("LoRA"):
            lat_mean = latency_stats["mean_ms"]
            lat_p95  = latency_stats["p95_ms"]
            lat_p99  = latency_stats["p99_ms"]
        else:
            lat_mean = float(clean.get("mean_latency_ms", clean.get("latency_ms", 0.0)))
            lat_p95  = float(clean.get("p95_latency_ms", float("nan")))
            lat_p99  = float(clean.get("p99_latency_ms", float("nan")))

        rows.append({
            "method":       method_name,
            "n_samples":    n,
            "accuracy":     acc,
            "acc_ci_lo":    ci_lo,
            "acc_ci_hi":    ci_hi,
            "macro_f1":     f1_clean,
            "f1_noisy":     f1_noisy,
            "lat_mean_ms":  lat_mean,
            "lat_p95_ms":   lat_p95,
            "lat_p99_ms":   lat_p99,
            "k_correct":    k,
        })

    text = _format_table(rows, latency_stats is not None)
    return rows, text


def _fmt(v: Optional[float], fmt_str: str = ".4f") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(v, fmt_str)


def _format_table(rows: List[dict], has_latency_rerun: bool) -> str:
    """Render thesis-ready plaintext table."""
    divider = "=" * 110
    latency_note = (
        f"  ‡  P95/P99 latency from dedicated {rows[2].get('n_bench', '?')}-run benchmark "
        "(после 20 прогревочных запусков)."
        if has_latency_rerun else
        "  ‡  P95/P99 latency not re-run (use --no_latency_rerun=False to enable)."
    )

    header = (
        f"  {'Метод':<35} {'N':>5}  {'Acc':>7}  {'95% ДИ':>17}  "
        f"{'F1 clean':>9}  {'F1 noisy':>9}  "
        f"{'Lat mean':>9}  {'P95':>7}  {'P99':>7}"
    )
    lines = [
        divider,
        "  Таблица 4.3 — Сравнение методов распознавания речевых команд",
        "  (N=тест, 95% ДИ Клоппера–Пирсона для accuracy, латентность — мс/сэмпл)",
        divider,
        header,
        "-" * 110,
    ]

    for r in rows:
        if r.get("status") == "NOT RUN":
            lines.append(f"  {r['method']:<35}  [NOT RUN]")
            continue

        ci = f"[{r['acc_ci_lo']:.4f}; {r['acc_ci_hi']:.4f}]"
        lines.append(
            f"  {r['method']:<35} {r['n_samples']:>5}  "
            f"{r['accuracy']:>7.4f}  {ci:>17}  "
            f"{_fmt(r['macro_f1']):>9}  {_fmt(r['f1_noisy']):>9}  "
            f"{_fmt(r['lat_mean_ms'], '.1f'):>9}  "
            f"{_fmt(r['lat_p95_ms'], '.1f'):>7}  "
            f"{_fmt(r['lat_p99_ms'], '.1f'):>7}"
        )

    lines += [
        divider,
        "  Примечания:",
        "  * F1 noisy: SNR 12 dБ (морской шум по DEMAND/ESC-50).",
        "  * 95% ДИ — точный интервал Клоппера–Пирсона для доли правильных ответов.",
        latency_note,
        divider,
    ]
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Clopper-Pearson CIs and P95/P99 latency for the thesis."
    )
    parser.add_argument(
        "--onnx_dir",
        default="onnx_model/models/run_2026-02-25_19-07-15/best_model",
        help="ONNX bundle directory for latency benchmark.",
    )
    parser.add_argument(
        "--n_bench",
        type=int,
        default=300,
        help="Number of inferences for latency benchmark (default 300).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup inferences excluded from timing (default 20).",
    )
    parser.add_argument(
        "--no_latency_rerun",
        action="store_true",
        default=False,
        help="Skip latency benchmark (use CIs only). Faster but no P95/P99.",
    )
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # ── Latency benchmark ──────────────────────────────────────────────
    latency_stats: Optional[Dict[str, float]] = None
    if not args.no_latency_rerun:
        onnx_dir = Path(args.onnx_dir)
        if not onnx_dir.exists():
            onnx_dir = _PROJECT_ROOT / args.onnx_dir
        if not onnx_dir.exists():
            logger.error(
                "ONNX directory not found: %s\n"
                "Pass --no_latency_rerun to skip the latency benchmark.",
                onnx_dir,
            )
            sys.exit(1)
        latency_stats = run_latency_benchmark(
            str(onnx_dir),
            n_bench=args.n_bench,
            warmup=args.warmup,
        )
        logger.info(
            "Latency: mean=%.1f ms  P50=%.1f ms  P95=%.1f ms  P99=%.1f ms",
            latency_stats["mean_ms"],
            latency_stats["p50_ms"],
            latency_stats["p95_ms"],
            latency_stats["p99_ms"],
        )
    else:
        logger.info("Skipping latency rerun (--no_latency_rerun).")

    # ── Table 4.3 with CIs ─────────────────────────────────────────────
    rows, table_text = build_table_43(latency_stats)
    print("\n" + table_text + "\n")

    # ── Save outputs ───────────────────────────────────────────────────
    out = {
        "latency_benchmark": latency_stats,
        "table_4_3":         rows,
    }
    json_path = BENCH_DIR / "thesis_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON  → %s", json_path)

    txt_path = BENCH_DIR / "thesis_stats.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_text + "\n")
    logger.info("Saved TXT   → %s", txt_path)


if __name__ == "__main__":
    main()
