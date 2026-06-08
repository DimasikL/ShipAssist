"""
scripts/benchmark_onnx_providers.py

Comparative inference benchmark: CPUExecutionProvider vs OpenVINOExecutionProvider
for an INT8-quantized Wav2Vec2 + classification head ONNX model.

Target hardware: Intel i5-6300U (Skylake-U, 2 cores, no AVX-512).

Usage:
    python scripts/benchmark_onnx_providers.py \
        --model artifacts/models/wav2vec2_int8.onnx \
        --wav   artifacts/data/test_sample.wav

Requirements:
    pip install onnxruntime scipy numpy librosa matplotlib psutil
    # For OpenVINO EP (optional):
    pip install onnxruntime-openvino
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import psutil
import scipy.stats
import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000        # Hz expected by Wav2Vec2
EXPECTED_LENGTH: int = 16_000    # samples (1 second)
WARMUP_RUNS: int = 20
MEASURE_RUNS: int = 300
PLOT_OUTPUT: str = "latency_comparison.pdf"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav(wav_path: Path) -> np.ndarray:
    """Load a WAV file and return a float32 array of shape [1, EXPECTED_LENGTH].

    Args:
        wav_path: Path to the .wav file.

    Returns:
        Input tensor ready for ONNX session, shape [1, 16000].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If audio is too short after loading.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

    if len(audio) < EXPECTED_LENGTH:
        raise ValueError(
            f"WAV too short: got {len(audio)} samples, need {EXPECTED_LENGTH}."
        )

    # Take the first second, normalise to [-1, 1]
    audio = audio[:EXPECTED_LENGTH]
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val

    return audio.astype(np.float32)[np.newaxis, :]   # [1, 16000]


# ---------------------------------------------------------------------------
# ONNX session factory
# ---------------------------------------------------------------------------

def build_session(
    model_path: Path,
    providers: list[str],
) -> "onnxruntime.InferenceSession":  # noqa: F821
    """Create an ONNX Runtime InferenceSession with the given provider list.

    Args:
        model_path: Path to the .onnx model file.
        providers: Ordered list of execution providers, e.g.
                   ["CPUExecutionProvider"] or
                   ["OpenVINOExecutionProvider", "CPUExecutionProvider"].

    Returns:
        Configured InferenceSession.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    import onnxruntime as ort  # imported late so ImportError surfaces clearly

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2   # match i5-6300U physical core count
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=providers,
    )
    actual = session.get_providers()
    log.info("Requested providers : %s", providers)
    log.info("Active providers    : %s", actual)
    return session


# ---------------------------------------------------------------------------
# Single-provider benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    session: "onnxruntime.InferenceSession",  # noqa: F821
    input_tensor: np.ndarray,
    provider_label: str,
) -> tuple[list[float], int]:
    """Run warmup + timed inference loop for one provider configuration.

    Args:
        session: Configured ONNX InferenceSession.
        input_tensor: Float32 numpy array of shape [1, 16000].
        provider_label: Human-readable label for logging.

    Returns:
        Tuple of (latency_ms_list, peak_rss_bytes).
        latency_ms_list has length MEASURE_RUNS.
        peak_rss_bytes is the maximum RSS observed during the measurement loop
        (includes C++ heap used by ONNX Runtime, unlike tracemalloc).
    """
    input_name: str = session.get_inputs()[0].name
    feed: dict[str, np.ndarray] = {input_name: input_tensor}
    proc = psutil.Process(os.getpid())

    # --- Warmup ---------------------------------------------------------
    log.info("[%s] Warming up (%d runs)…", provider_label, WARMUP_RUNS)
    for _ in range(WARMUP_RUNS):
        session.run(None, feed)

    # --- Timed measurement with RSS tracking ----------------------------
    log.info("[%s] Measuring (%d runs)…", provider_label, MEASURE_RUNS)
    latencies: list[float] = []
    peak_rss: int = 0

    for i in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000.0)  # convert to ms

        rss = proc.memory_info().rss
        if rss > peak_rss:
            peak_rss = rss

        if (i + 1) % 100 == 0:
            log.info("  … %d / %d done", i + 1, MEASURE_RUNS)

    return latencies, peak_rss


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def print_stats(latencies: list[float], label: str, peak_bytes: int) -> None:
    """Print summary statistics for a latency distribution.

    Args:
        latencies: List of per-run latencies in milliseconds.
        label: Provider label for display.
        peak_bytes: Peak process RSS in bytes (from psutil), covering
                    both Python and C++ (ONNX Runtime) allocations.
    """
    arr = np.array(latencies)
    log.info(
        "\n=== %s ===\n"
        "  mean   : %.2f ms\n"
        "  median : %.2f ms\n"
        "  p95    : %.2f ms\n"
        "  p99    : %.2f ms\n"
        "  std    : %.2f ms\n"
        "  peak RAM (process RSS): %.1f MB",
        label,
        arr.mean(),
        np.median(arr),
        np.percentile(arr, 95),
        np.percentile(arr, 99),
        arr.std(),
        peak_bytes / 1_048_576,
    )


# ---------------------------------------------------------------------------
# Wilcoxon test
# ---------------------------------------------------------------------------

def wilcoxon_test(lat_a: list[float], lat_b: list[float]) -> None:
    """Run Wilcoxon signed-rank test between two paired latency distributions.

    Both lists must have the same length (MEASURE_RUNS = 300).

    Args:
        lat_a: Latencies for provider A (ms).
        lat_b: Latencies for provider B (ms).
    """
    if len(lat_a) != len(lat_b):
        log.warning(
            "Wilcoxon test requires equal-length samples; skipping "
            "(%d vs %d).", len(lat_a), len(lat_b)
        )
        return

    stat, p_value = scipy.stats.wilcoxon(lat_a, lat_b, alternative="two-sided")
    alpha = 0.05
    conclusion = (
        "Difference IS statistically significant (reject H₀)."
        if p_value < alpha
        else "Difference is NOT statistically significant (fail to reject H₀)."
    )
    log.info(
        "\n=== Wilcoxon Signed-Rank Test ===\n"
        "  statistic : %.4f\n"
        "  p-value   : %.6f\n"
        "  α = %.2f  → %s",
        stat, p_value, alpha, conclusion,
    )


# ---------------------------------------------------------------------------
# Boxplot
# ---------------------------------------------------------------------------

def save_boxplot(
    lat_cpu: list[float],
    lat_openvino: Optional[list[float]],
    output_path: Path,
) -> None:
    """Save a boxplot comparing latency distributions to a PDF file.

    Args:
        lat_cpu: Latencies for CPUExecutionProvider (ms).
        lat_openvino: Latencies for OpenVINOExecutionProvider (ms), or None
                      if OpenVINO EP was unavailable.
        output_path: Destination PDF file path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    data: list[list[float]] = [lat_cpu]
    labels: list[str] = ["CPU EP"]

    if lat_openvino is not None:
        data.append(lat_openvino)
        labels.append("OpenVINO EP")

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )

    colors = ["#4C72B0", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        "ONNX Inference Latency — i5-6300U\n"
        f"INT8 Wav2Vec2 · {MEASURE_RUNS} runs · input [1, {EXPECTED_LENGTH}]"
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    log.info("Boxplot saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with .model (Path) and .wav (Path).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ONNX Runtime inference: "
            "CPUExecutionProvider vs OpenVINOExecutionProvider."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to INT8-quantized Wav2Vec2 .onnx model file.",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        required=True,
        help="Path to a real .wav file used as inference input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(PLOT_OUTPUT),
        help="Output path for the latency comparison PDF boxplot.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main benchmark routine."""
    args = parse_args()

    log.info("Model : %s", args.model)
    log.info("WAV   : %s", args.wav)

    # Load real audio input (no zero tensors)
    input_tensor = load_wav(args.wav)
    log.info("Input tensor shape : %s  dtype : %s", input_tensor.shape, input_tensor.dtype)

    # -----------------------------------------------------------------------
    # 1. CPUExecutionProvider
    # -----------------------------------------------------------------------
    log.info("\n--- Provider 1: CPUExecutionProvider ---")
    cpu_session = build_session(args.model, providers=["CPUExecutionProvider"])
    lat_cpu, ram_cpu = run_benchmark(cpu_session, input_tensor, "CPU EP")
    print_stats(lat_cpu, "CPUExecutionProvider", ram_cpu)

    # -----------------------------------------------------------------------
    # 2. OpenVINOExecutionProvider (with graceful fallback)
    # -----------------------------------------------------------------------
    lat_openvino: Optional[list[float]] = None
    ram_openvino: int = 0

    log.info("\n--- Provider 2: OpenVINOExecutionProvider ---")
    try:
        import onnxruntime as ort  # noqa: F401 — trigger import once more to validate

        available = ort.get_available_providers()
        if "OpenVINOExecutionProvider" not in available:
            raise ImportError("OpenVINOExecutionProvider not in available providers.")

        # On Windows, onnxruntime_providers_openvino.dll depends on openvino.dll,
        # which lives inside the pip-installed openvino package but is NOT on the
        # system PATH. Python 3.8+ provides os.add_dll_directory() to tell the
        # Windows DLL loader where to search — use it before creating the session.
        if sys.platform == "win32":
            try:
                import openvino  # noqa: F401
                openvino_pkg_dir = Path(openvino.__file__).parent
                # DLLs are in the package root and/or a 'libs' subdirectory.
                for dll_dir in [openvino_pkg_dir, openvino_pkg_dir / "libs"]:
                    if dll_dir.is_dir():
                        os.add_dll_directory(str(dll_dir))
                        log.info("Added DLL search path: %s", dll_dir)
            except ImportError:
                pass  # openvino Python package not installed; DLL must be on PATH

        openvino_session = build_session(
            args.model,
            # OpenVINO EP first; ORT falls back to CPU for unsupported ops
            providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        )

        # Guard: if ORT silently fell back to CPU-only (e.g. openvino.dll still
        # missing), the benchmark would compare CPU vs CPU — meaningless.
        active = openvino_session.get_providers()
        if active == ["CPUExecutionProvider"]:
            raise ImportError(
                "OpenVINO EP was requested but ORT fell back to CPUExecutionProvider only. "
                "os.add_dll_directory() was applied but openvino.dll still could not be "
                "found. Check that 'pip install openvino' completed without errors and "
                "that the openvino package directory contains the DLL."
            )

        lat_openvino, ram_openvino = run_benchmark(
            openvino_session, input_tensor, "OpenVINO EP"
        )
        print_stats(lat_openvino, "OpenVINOExecutionProvider", ram_openvino)

    except ImportError as exc:
        log.warning(
            "\n[!] OpenVINOExecutionProvider is NOT available: %s\n"
            "\n    To install it, run one of the following:\n"
            "      pip install onnxruntime-openvino\n"
            "    or, for a specific OpenVINO version:\n"
            "      pip install openvino onnxruntime-openvino\n"
            "\n    Then verify with:\n"
            "      python -c \"import onnxruntime; "
            "print(onnxruntime.get_available_providers())\"\n"
            "\n    NOTE: OpenVINO EP requires OpenVINO Runtime ≥ 2022.1 "
            "and is NOT available via onnxruntime (CPU-only) package.\n",
            exc,
        )
        log.info("Skipping OpenVINO EP benchmark. Only CPU EP results will be plotted.")

    # -----------------------------------------------------------------------
    # 3. Statistical comparison (Wilcoxon)
    # -----------------------------------------------------------------------
    if lat_openvino is not None:
        wilcoxon_test(lat_cpu, lat_openvino)
    else:
        log.info("Wilcoxon test skipped — OpenVINO EP data not collected.")

    # -----------------------------------------------------------------------
    # 4. Boxplot → PDF
    # -----------------------------------------------------------------------
    save_boxplot(lat_cpu, lat_openvino, args.output)

    log.info("Benchmark complete.")


if __name__ == "__main__":
    main()
