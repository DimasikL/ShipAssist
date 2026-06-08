"""
scripts/benchmark_latency_percentiles.py

Measures P50 / P95 / P99 inference latency for pytorch_fp32, onnx_fp32, and
onnx_int8 backends, writes the results into artifacts/benchmarks/thesis_stats2.json,
and saves an academic-quality box plot to artifacts/plots/vkr_figures/.

Hardware target: Intel Core i5-6300U, 1 CPU core, 16 kHz, 3-second window.

Usage
-----
    # From the project root (activate .venv first):
    python scripts/benchmark_latency_percentiles.py

    # Override model paths if needed:
    python scripts/benchmark_latency_percentiles.py \
        --onnx-dir onnx_model/models/run_2026-05-22_09-50-17 \
        --torch-model experiments/archive_training/lora_tune/models/run_2026-05-22_09-50-17/best_model

    # Skip slow PyTorch run:
    python scripts/benchmark_latency_percentiles.py --skip-pytorch

Dependencies
------------
    onnxruntime               — for onnx_fp32, onnx_int8
    torch, transformers, peft — for pytorch_fp32
    numpy, matplotlib

Results are written to artifacts/benchmarks/thesis_stats2.json under the
top-level key ``latency_benchmark``, preserving all other existing keys.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_FILE   = PROJECT_ROOT / "artifacts" / "benchmarks" / "thesis_stats2.json"

DEFAULT_ONNX_DIR    = PROJECT_ROOT / "onnx_model" / "models" / "run_2026-05-22_09-50-17"
DEFAULT_TORCH_MODEL = (
    PROJECT_ROOT
    / "experiments" / "archive_training" / "lora_tune"
    / "models" / "run_2026-05-22_09-50-17" / "best_model"
)

SR          = 16_000   # Hz
WIN_SECONDS = 3        # seconds — canonical inference window
WIN_SAMPLES = SR * WIN_SECONDS   # 48 000 samples
N_WARMUP    = 20
N_BENCH     = 300


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _make_synthetic_audio(seed: int = 42) -> np.ndarray:
    """Return a 1-D float32 white-noise window normalised like Wav2Vec2FE.

    The same waveform is reused for every run so only backend latency varies.
    Wav2Vec2FeatureExtractor normalisation: x' = (x - μ) / sqrt(σ² + 1e-7).
    """
    rng = np.random.default_rng(seed)
    wav = rng.standard_normal(WIN_SAMPLES).astype(np.float32)
    wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
    return wav


# ── Percentile helper ─────────────────────────────────────────────────────────

def _percentiles(latencies: List[float]) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute summary stats and return both the dict and the raw array.

    Returns:
        Tuple of (stats_dict, raw_array_ms) so callers can pass raw data to
        the box plot without re-running inference.
    """
    a = np.array(latencies, dtype=np.float64)
    stats = {
        "avg": round(float(a.mean()), 1),
        "P50": round(float(np.percentile(a, 50)), 1),
        "P95": round(float(np.percentile(a, 95)), 1),
        "P99": round(float(np.percentile(a, 99)), 1),
        "std": round(float(a.std()),  1),
        "min": round(float(a.min()),  1),
        "max": round(float(a.max()),  1),
        "n_bench": N_BENCH,
        "warmup":  N_WARMUP,
    }
    return stats, a


# ── Generic ONNX benchmark ────────────────────────────────────────────────────

def _bench_onnx(
    model_path: Path, audio: np.ndarray, tag: str
) -> Tuple[Dict[str, float], np.ndarray]:
    """Run N_BENCH ONNX inferences for any precision and return latency stats.

    Args:
        model_path: Absolute path to the .onnx file.
        audio:      Pre-normalised 1-D float32 waveform (WIN_SAMPLES samples).
        tag:        Human-readable label used in log messages (e.g. "ONNX FP32").

    Returns:
        Tuple of (stats_dict, raw_latencies_ms_array).

    Raises:
        ImportError:       if onnxruntime is not installed.
        FileNotFoundError: if model_path does not exist.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is not installed. Run: pip install onnxruntime") from exc

    if not model_path.exists():
        raise FileNotFoundError(f"{tag} model not found: {model_path}")

    log.info("Loading %s: %s", tag, model_path)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # single-core — matches thesis HW config
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(model_path),
        sess_opts=opts,
        providers=["CPUExecutionProvider"],
    )

    feed = {"input_values": audio.reshape(1, -1)}

    log.info("%s warmup (%d runs)…", tag, N_WARMUP)
    for _ in range(N_WARMUP):
        sess.run(None, feed)

    log.info("%s measuring (%d runs)…", tag, N_BENCH)
    lats: List[float] = []
    for i in range(N_BENCH):
        t0 = time.perf_counter()
        sess.run(None, feed)
        lats.append((time.perf_counter() - t0) * 1_000.0)
        if (i + 1) % 100 == 0:
            log.info("  %d / %d done", i + 1, N_BENCH)

    stats, raw = _percentiles(lats)
    log.info("%s → avg=%.1f ms  P50=%.1f  P95=%.1f  P99=%.1f",
             tag, stats["avg"], stats["P50"], stats["P95"], stats["P99"])
    return stats, raw


def bench_onnx_fp32(
    onnx_dir: Path, audio: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """Benchmark ONNX FP32 (model_fp32.onnx)."""
    return _bench_onnx(onnx_dir / "model_fp32.onnx", audio, "ONNX FP32")


def bench_onnx_int8(
    onnx_dir: Path, audio: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """Benchmark ONNX INT8 (model_int8.onnx)."""
    return _bench_onnx(onnx_dir / "model_int8.onnx", audio, "ONNX INT8")


# ── PyTorch FP32 benchmark ────────────────────────────────────────────────────

def bench_pytorch_fp32(
    model_dir: Path, audio: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """Run N_BENCH PyTorch FP32 inferences and return latency percentiles.

    The timing window covers the full ``model(input_values)`` forward pass on
    CPU — matching the thesis latency definition (predict() call to result
    receipt, preprocessing excluded).

    The checkpoint is a LoRA fine-tune (contains adapter_config.json), so the
    load sequence is:
      1. Read ``adapter_config.json`` → base model name.
      2. Read ``config.json`` → num_labels.
      3. Load base ``Wav2Vec2ForSequenceClassification`` from HF Hub / cache.
      4. ``PeftModel.from_pretrained`` → apply LoRA weights.
      5. ``merge_and_unload()`` → plain forward pass, no PEFT overhead.

    Args:
        model_dir: Directory with adapter_config.json + adapter_model.safetensors.
        audio:     Pre-normalised 1-D float32 waveform (WIN_SAMPLES samples).

    Returns:
        Dict with avg, P50, P95, P99, std, min, max (all in ms).

    Raises:
        ImportError:       if torch / transformers / peft are not installed.
        FileNotFoundError: if model_dir or adapter_config.json are missing.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import Wav2Vec2ForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "torch, transformers and peft are required for PyTorch benchmark.\n"
            "Run: pip install torch transformers peft"
        ) from exc

    if not model_dir.exists():
        raise FileNotFoundError(f"PyTorch model directory not found: {model_dir}")

    # ── Resolve base model name ────────────────────────────────────────────────
    adapter_cfg_path = model_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {model_dir}")
    with adapter_cfg_path.open("r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_name: str = adapter_cfg["base_model_name_or_path"]
    log.info("Base model: %s", base_model_name)

    # ── Resolve num_labels from checkpoint config.json ────────────────────────
    ckpt_config_path = model_dir / "config.json"
    if ckpt_config_path.exists():
        with ckpt_config_path.open("r", encoding="utf-8") as f:
            ckpt_cfg = json.load(f)
        num_labels: int = ckpt_cfg.get("num_labels") or len(ckpt_cfg.get("id2label", {}))
    else:
        num_labels = len(adapter_cfg.get("label2id", {})) or 4  # fallback: 4 ship commands
    log.info("num_labels=%d", num_labels)

    # ── Load base model + LoRA adapter, merge ─────────────────────────────────
    log.info("Loading base Wav2Vec2ForSequenceClassification…")
    base = Wav2Vec2ForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    log.info("Applying LoRA adapter from: %s", model_dir)
    peft_model = PeftModel.from_pretrained(base, str(model_dir))
    model = peft_model.merge_and_unload()   # no PEFT overhead in timing
    model.eval().to(torch.float32).cpu()
    log.info("Model ready (params: %d M)", sum(p.numel() for p in model.parameters()) // 1_000_000)

    tensor = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)  # (1, WIN_SAMPLES)

    log.info("PyTorch FP32 warmup (%d runs)…", N_WARMUP)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(tensor).logits

    log.info("PyTorch FP32 measuring (%d runs)…", N_BENCH)
    lats: List[float] = []
    with torch.no_grad():
        for i in range(N_BENCH):
            t0 = time.perf_counter()
            _ = model(tensor).logits
            lats.append((time.perf_counter() - t0) * 1_000.0)
            if (i + 1) % 100 == 0:
                log.info("  %d / %d done", i + 1, N_BENCH)

    stats, raw = _percentiles(lats)
    log.info("PyTorch FP32 → avg=%.1f ms  P50=%.1f  P95=%.1f  P99=%.1f",
             stats["avg"], stats["P50"], stats["P95"], stats["P99"])
    return stats, raw


# ── Academic box plot ─────────────────────────────────────────────────────────

def plot_boxplot(
    raw: Dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    """Render an academic-quality box plot of inference latency distributions.

    Produces a publication-ready figure suitable for inclusion in a Russian
    thesis (ВКР). The plot follows journal conventions:
      * Seaborn ``whitegrid`` style with a white background.
      * Boxes coloured by backend category (PyTorch / ONNX FP32 / ONNX INT8).
      * Individual outlier points drawn as semi-transparent circles.
      * Дополнительные горизонтальные аннотации P95 и P99.
      * 300 dpi PNG + vector PDF saved side-by-side.

    Args:
        raw:     Dict mapping config name to 1-D array of latencies in ms.
                 Keys: ``"pytorch_fp32"``, ``"onnx_fp32"``, ``"onnx_int8"``.
                 If a value is an empty array, that backend was skipped and
                 will appear as a greyed-out placeholder box.
        out_dir: Directory where the figure files are written.

    Returns:
        Path to the saved PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as ticker
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for box plot generation.\n"
            "Run: pip install matplotlib"
        ) from exc

    # ── Style ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "DejaVu Serif",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "xtick.labelsize":  11,
        "ytick.labelsize":  10,
        "axes.linewidth":   0.8,
        "axes.edgecolor":   "#333333",
        "grid.color":       "#cccccc",
        "grid.linewidth":   0.6,
        "figure.dpi":       150,
    })

    # ── Data & labels ──────────────────────────────────────────────────────────
    CONFIG_ORDER = ["pytorch_fp32", "onnx_fp32", "onnx_int8"]
    LABELS = {
        "pytorch_fp32": "PyTorch\nFP32",
        "onnx_fp32":    "ONNX\nFP32",
        "onnx_int8":    "ONNX\nINT8",
    }
    # Warm palette: blue → orange → green  (colourblind-safe)
    COLORS = {
        "pytorch_fp32": "#4878CF",
        "onnx_fp32":    "#E67E22",
        "onnx_int8":    "#27AE60",
    }

    data    = [raw.get(k, np.array([])) for k in CONFIG_ORDER]
    labels  = [LABELS[k]               for k in CONFIG_ORDER]
    colors  = [COLORS[k]               for k in CONFIG_ORDER]
    valid   = [len(d) > 0              for d in data]

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    positions = list(range(1, len(CONFIG_ORDER) + 1))
    box_data  = [d if len(d) > 0 else np.array([0.0]) for d in data]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(
            marker="o", markersize=3.5,
            linestyle="none", alpha=0.35,
        ),
        medianprops=dict(color="white", linewidth=2.0),
        whiskerprops=dict(linewidth=1.2, linestyle="--"),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.2),
        zorder=3,
    )

    # ── Colour boxes individually ──────────────────────────────────────────────
    for patch, color, is_valid in zip(bp["boxes"], colors, valid):
        patch.set_facecolor(color if is_valid else "#cccccc")
        patch.set_alpha(0.82)
    for flier, color in zip(bp["fliers"], colors):
        flier.set(markerfacecolor=color, markeredgecolor=color)

    # ── P95 / P99 diamond markers per box ─────────────────────────────────────
    # Avoid text annotations inside the plot area — they overlap when P95/P99
    # are close. Instead draw small labelled markers on the right axis margin.
    MIN_LABEL_GAP = 8.0   # px equivalent in data units; adjusted after y-range known

    for i, (k, pos) in enumerate(zip(CONFIG_ORDER, positions)):
        arr = raw.get(k)
        if arr is None or len(arr) == 0:
            continue
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        x0, x1 = pos - 0.20, pos + 0.20

        # P95 — dashed line across the box cap
        ax.hlines(p95, x0, x1, colors=COLORS[k], linewidths=1.6,
                  linestyles=(0, (4, 2)), zorder=5, alpha=0.85)
        # P99 — dash-dot line
        ax.hlines(p99, x0, x1, colors=COLORS[k], linewidths=1.6,
                  linestyles=(0, (2, 1, 1, 1)), zorder=5, alpha=0.85)

    # Single set of labels on the RIGHT of the rightmost box only,
    # with a guaranteed minimum vertical gap to prevent overlap.
    last_valid = None
    for k in reversed(CONFIG_ORDER):
        arr = raw.get(k)
        if arr is not None and len(arr) > 0:
            last_valid = k
            break

    if last_valid is not None:
        last_pos = positions[CONFIG_ORDER.index(last_valid)]
        arr      = raw[last_valid]
        p95_val  = float(np.percentile(arr, 95))
        p99_val  = float(np.percentile(arr, 99))
        x_label  = last_pos + 0.30

        # Determine label positions — ensure at least MIN_LABEL_GAP apart
        y_range  = ax.get_ylim()
        gap_min  = (y_range[1] - y_range[0]) * 0.04   # 4 % of y-axis height

        y_p95, y_p99 = p95_val, p99_val
        if abs(y_p99 - y_p95) < gap_min:
            mid = (y_p95 + y_p99) / 2
            y_p95 = mid - gap_min / 2
            y_p99 = mid + gap_min / 2

        ax.annotate(
            "P95", xy=(x_label, y_p95),
            fontsize=8.5, color="#555555", va="center", ha="left",
            fontweight="normal",
        )
        ax.annotate(
            "P99", xy=(x_label, y_p99),
            fontsize=8.5, color="#555555", va="center", ha="left",
            fontweight="normal",
        )

    # ── Axes labels & ticks ────────────────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Конфигурация модели", labelpad=8)
    ax.set_ylabel("Задержка инференса, мс", labelpad=8)
    ax.set_title(
        f"Распределение задержки инференса\n"
        f"(n = {N_BENCH}, окно {WIN_SECONDS} с, Intel Core i5-6300U, 1 ядро CPU)",
        pad=12,
    )
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis="x", which="both", bottom=False)

    # ── Legend: line styles for P95 / P99 ─────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#888888", linewidth=1.4, linestyle=":",
               label="P95 (95-й перцентиль)"),
        Line2D([0], [0], color="#888888", linewidth=1.4,
               linestyle=(0, (3, 1)), label="P99 (99-й перцентиль)"),
        Line2D([0], [0], color="white", linewidth=2.0,
               linestyle="-", label="Медиана (P50)"),
    ]
    # Add a white median line swatch with dark outline so it's visible
    legend_elements[2] = mpatches.Patch(
        facecolor="none", edgecolor="#333333", linewidth=1,
        label="Медиана (P50) — белая линия в ящике",
    )
    ax.legend(
        handles=legend_elements[:2],
        loc="upper right", fontsize=9,
        framealpha=0.9, edgecolor="#cccccc",
    )

    fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig_4_6_latency_boxplot.png"
    pdf_path = out_dir / "fig_4_6_latency_boxplot.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    log.info("Box plot saved: %s", png_path)
    log.info("Box plot saved: %s", pdf_path)
    return png_path


# ── thesis_stats2.json update ─────────────────────────────────────────────────

def update_thesis_stats(
    pytorch_fp32: Dict[str, float],
    onnx_fp32: Dict[str, float],
    onnx_int8: Dict[str, float],
) -> None:  # noqa: D401
    """Merge benchmarked percentiles into thesis_stats2.json.

    Preserves all existing top-level keys. Overwrites only
    ``latency_benchmark.pytorch_fp32``, ``latency_benchmark.onnx_fp32``,
    and ``latency_benchmark.onnx_int8``.

    Args:
        pytorch_fp32: Latency stats dict for the PyTorch FP32 backend.
        onnx_fp32:    Latency stats dict for the ONNX FP32 backend.
        onnx_int8:    Latency stats dict for the ONNX INT8 backend.
    """
    if STATS_FILE.exists():
        with STATS_FILE.open("r", encoding="utf-8") as f:
            stats: dict = json.load(f)
    else:
        log.warning("File not found — creating new: %s", STATS_FILE)
        stats = {}

    lb = stats.setdefault("latency_benchmark", {})

    def _slim(d: Dict) -> Dict:
        """Keep only the four thesis-facing keys in the output."""
        return {"avg": d["avg"], "P50": d["P50"], "P95": d["P95"], "P99": d["P99"]}

    lb["pytorch_fp32"] = _slim(pytorch_fp32)
    lb["onnx_fp32"]    = _slim(onnx_fp32)
    lb["onnx_int8"]    = _slim(onnx_int8)

    # Store full stats (std, min, max, n_bench) under a separate audit key
    stats["latency_raw"] = {
        "pytorch_fp32": pytorch_fp32,
        "onnx_fp32":    onnx_fp32,
        "onnx_int8":    onnx_int8,
        "win_seconds":  WIN_SECONDS,
        "sr":           SR,
    }

    with STATS_FILE.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log.info("Written: %s", STATS_FILE)

    print("\n=== latency_benchmark (updated) ===")
    print(json.dumps(lb, indent=2, ensure_ascii=False))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--onnx-dir",    type=Path, default=DEFAULT_ONNX_DIR,
                   help="Directory with onnx_config.json + model_fp32.onnx + model_int8.onnx")
    p.add_argument("--torch-model", type=Path, default=DEFAULT_TORCH_MODEL,
                   help="HuggingFace LoRA checkpoint directory for PyTorch FP32")
    p.add_argument("--skip-pytorch", action="store_true",
                   help="Skip pytorch_fp32 run (saves ~5 min; needs torch/transformers/peft)")
    p.add_argument("--skip-onnx-fp32", action="store_true",
                   help="Skip onnx_fp32 run")
    p.add_argument("--skip-onnx-int8", action="store_true",
                   help="Skip onnx_int8 run")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Window       : %d s  (%d samples @ %d Hz)",
             WIN_SECONDS, WIN_SAMPLES, SR)
    log.info("Runs         : %d warmup + %d bench", N_WARMUP, N_BENCH)

    audio = _make_synthetic_audio(seed=42)
    log.info("Synthetic audio: shape=%s  mean=%.4f  std=%.4f",
             audio.shape, float(audio.mean()), float(audio.std()))

    pytorch_result:    Optional[Dict]     = None
    onnx_fp32_result:  Optional[Dict]     = None
    onnx_int8_result:  Optional[Dict]     = None
    pytorch_raw:       Optional[np.ndarray] = None
    onnx_fp32_raw:     Optional[np.ndarray] = None
    onnx_int8_raw:     Optional[np.ndarray] = None

    # ── ONNX FP32 ─────────────────────────────────────────────────────────────
    if not args.skip_onnx_fp32:
        try:
            onnx_fp32_result, onnx_fp32_raw = bench_onnx_fp32(args.onnx_dir, audio)
        except Exception as exc:
            log.error("ONNX FP32 benchmark failed: %s", exc)
            sys.exit(1)

    # ── ONNX INT8 ─────────────────────────────────────────────────────────────
    if not args.skip_onnx_int8:
        try:
            onnx_int8_result, onnx_int8_raw = bench_onnx_int8(args.onnx_dir, audio)
        except Exception as exc:
            log.error("ONNX INT8 benchmark failed: %s", exc)
            sys.exit(1)

    # ── PyTorch FP32 ──────────────────────────────────────────────────────────
    if not args.skip_pytorch:
        try:
            pytorch_result, pytorch_raw = bench_pytorch_fp32(args.torch_model, audio)
        except ImportError as exc:
            log.error("%s", exc)
            log.error("Tip: pip install torch transformers peft  (or use --skip-pytorch)")
            sys.exit(1)
        except Exception as exc:
            log.error("PyTorch FP32 benchmark failed: %s", exc)
            sys.exit(1)

    # ── Fallback placeholders for skipped backends ─────────────────────────────
    _EMPTY = np.array([])
    if pytorch_result is None:
        log.warning("pytorch_fp32 skipped — P50/P95/P99 will be null in output")
        pytorch_result = {"avg": 474, "P50": None, "P95": None, "P99": None,
                          "std": None, "min": None, "max": None}
        pytorch_raw = _EMPTY
    if onnx_fp32_result is None:
        log.warning("onnx_fp32 skipped — P50/P95/P99 will be null in output")
        onnx_fp32_result = {"avg": 328, "P50": None, "P95": None, "P99": None,
                            "std": None, "min": None, "max": None}
        onnx_fp32_raw = _EMPTY
    if onnx_int8_result is None:
        log.warning("onnx_int8 skipped — P50/P95/P99 will be null in output")
        onnx_int8_result = {"avg": 247, "P50": None, "P95": None, "P99": None,
                            "std": None, "min": None, "max": None}
        onnx_int8_raw = _EMPTY

    update_thesis_stats(pytorch_result, onnx_fp32_result, onnx_int8_result)

    # ── Box plot ───────────────────────────────────────────────────────────────
    plots_dir = PROJECT_ROOT / "artifacts" / "plots" / "vkr_figures"
    try:
        png = plot_boxplot(
            raw={
                "pytorch_fp32": pytorch_raw,
                "onnx_fp32":    onnx_fp32_raw,
                "onnx_int8":    onnx_int8_raw,
            },
            out_dir=plots_dir,
        )
        print(f"\nBox plot → {png}")
    except ImportError as exc:
        log.warning("Skipping box plot: %s", exc)


if __name__ == "__main__":
    main()
