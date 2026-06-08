"""
scripts/export_gigaam_onnx_int8.py — Export GigaAM-v2 to ONNX and quantise to INT8.

Uses GigaAM's native ``model.to_onnx()`` API (added in v2, Dec 2024) to
export the full encoder+decoder graph as FP32 ONNX, then applies
``onnxruntime.quantization.quantize_dynamic`` (weight-only INT8) to produce
the INT8 variant.

For CTC models the exported ONNX is a **single file** (encoder + CTC head),
usable directly with ``gigaam.onnx_utils.load_onnx`` / ``infer_onnx``.

Output layout in ``onnx_model/gigaam_v2/``:
    v2_ctc.onnx         — FP32 ONNX (from model.to_onnx)
    v2_ctc.yaml         — model config (required by load_onnx)
    v2_ctc_int8.onnx    — dynamic INT8 quantised model
    v2_ctc_int8.yaml    — copy of v2_ctc.yaml (load_onnx looks for matching yaml)

Usage
-----
    # From project root (activate .venv first):
    python scripts/export_gigaam_onnx_int8.py

    # RNNT variant:
    python scripts/export_gigaam_onnx_int8.py --model-mode v2_rnnt

    # Export to custom directory:
    python scripts/export_gigaam_onnx_int8.py --output-dir onnx_model/gigaam_v2

    # Only export FP32, skip INT8 quantisation:
    python scripts/export_gigaam_onnx_int8.py --skip-int8

Dependencies
------------
    gigaam, torch, onnx, onnxruntime (>=1.16)

Architecture note
-----------------
``model.to_onnx()`` exports the encoder + CTC head together (for CTC models),
so the resulting ONNX takes audio features as input and returns log-probs + lengths.
This is the correct graph for ``infer_onnx`` from ``gigaam.onnx_utils``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gigaam_export")

# ── Project paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SR = 16_000


# ── Export ────────────────────────────────────────────────────────────────────

def export_fp32(model_mode: str, onnx_dir: Path) -> Path:
    """Export GigaAM to FP32 ONNX using the native ``model.to_onnx()`` API.

    Args:
        model_mode: GigaAM variant, e.g. ``"v2_ctc"`` or ``"v2_rnnt"``.
        onnx_dir:   Target directory for ONNX artefacts.

    Returns:
        Path to the exported FP32 ONNX file.

    Raises:
        RuntimeError: on export failure.
    """
    import gigaam

    onnx_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading GigaAM [%s] …", model_mode)
    model = gigaam.load_model(model_mode)
    model.eval()

    log.info("Exporting FP32 ONNX → %s …", onnx_dir)
    model.to_onnx(str(onnx_dir))  # creates {model_mode}.onnx + {model_mode}.yaml

    # For CTC: single file; for RNNT: encoder/decoder/joint
    if "ctc" in model_mode:
        fp32_path = onnx_dir / f"{model_mode}.onnx"
    else:
        # RNNT encoder is the primary file
        fp32_path = onnx_dir / f"{model_mode}_encoder.onnx"

    if not fp32_path.exists():
        raise RuntimeError(f"Expected ONNX file not found: {fp32_path}")

    size_mb = fp32_path.stat().st_size / 1024 / 1024
    log.info("  Exported: %s (%.1f MB)", fp32_path.name, size_mb)
    return fp32_path


def quantize_int8(
    model_mode: str,
    onnx_dir: Path,
    fp32_path: Path,
) -> Path:
    """Apply dynamic INT8 quantisation to the FP32 ONNX model.

    Quantises MatMul / Gemm weights to INT8 (weight-only; activations FP32).
    Also copies the YAML config so ``load_onnx`` can find ``{int8_name}.yaml``.

    Args:
        model_mode: GigaAM variant string.
        onnx_dir:   Directory containing the FP32 ONNX and YAML.
        fp32_path:  Path to the FP32 ONNX file.

    Returns:
        Path to the INT8 ONNX file.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Derive INT8 filename: v2_ctc → v2_ctc_int8, v2_rnnt_encoder → v2_rnnt_encoder_int8
    stem = fp32_path.stem          # e.g. "v2_ctc" or "v2_rnnt_encoder"
    int8_name = f"{stem}_int8"
    int8_path = fp32_path.parent / f"{int8_name}.onnx"

    log.info("Quantising → %s …", int8_path.name)
    # Restrict to MatMul/Gemm only — ORT CPU does NOT support ConvInteger,
    # so quantising Conv layers causes NotImplemented at runtime.
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    size_fp32 = fp32_path.stat().st_size / 1024 / 1024
    size_int8 = int8_path.stat().st_size / 1024 / 1024
    log.info(
        "  FP32 %.1f MB → INT8 %.1f MB  (%.2f× compression)",
        size_fp32, size_int8, size_fp32 / size_int8,
    )

    # Copy YAML config with INT8 name so gigaam.onnx_utils.load_onnx can find it
    # load_onnx looks for {model_version}.yaml where model_version is the name
    # passed to load_onnx; we pass int8_name, so we need int8_name.yaml
    yaml_src = onnx_dir / f"{model_mode}.yaml"
    yaml_dst = onnx_dir / f"{int8_name}.yaml"
    if yaml_src.exists() and not yaml_dst.exists():
        shutil.copy2(yaml_src, yaml_dst)
        log.info("  Copied config → %s", yaml_dst.name)

    return int8_path


# ── ORT latency benchmark ─────────────────────────────────────────────────────

def benchmark_ort(
    onnx_dir: Path,
    model_version: str,
    n_warmup: int,
    n_bench: int,
    audio_path: str,
) -> dict:
    """Measure ORT full-pipeline latency using ``gigaam.onnx_utils``.

    Args:
        onnx_dir:      Directory with ONNX + YAML files.
        model_version: Name passed to ``load_onnx`` (e.g. ``"v2_ctc"``).
        n_warmup:      Warm-up passes (discarded).
        n_bench:       Timed passes.
        audio_path:    Path to a reference audio file for realistic latency.

    Returns:
        Dict with avg, P50, P95, P99 latency in milliseconds.
    """
    from gigaam.onnx_utils import infer_onnx, load_onnx

    sessions, model_cfg = load_onnx(str(onnx_dir), model_version)

    for _ in range(n_warmup):
        infer_onnx(audio_path, model_cfg, sessions, progress=False)

    lats = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        infer_onnx(audio_path, model_cfg, sessions, progress=False)
        lats.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(lats)
    return {
        "avg": round(float(arr.mean()), 1),
        "std": round(float(arr.std()),  1),
        "P50": round(float(np.percentile(arr, 50)), 1),
        "P95": round(float(np.percentile(arr, 95)), 1),
        "P99": round(float(np.percentile(arr, 99)), 1),
        "n_warmup": n_warmup,
        "n_bench":  n_bench,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Export GigaAM-v2 to ONNX FP32 then quantise to INT8."
    )
    p.add_argument(
        "--model-mode", default="v2_ctc",
        choices=["v2_ctc", "v2_rnnt", "ctc", "rnnt"],
        help="GigaAM model variant (default: v2_ctc).",
    )
    p.add_argument(
        "--output-dir", default="onnx_model/gigaam_v2",
        help="Output directory for ONNX artefacts.",
    )
    p.add_argument(
        "--skip-int8", action="store_true",
        help="Skip INT8 quantisation (export FP32 only).",
    )
    p.add_argument(
        "--skip-latency", action="store_true",
        help="Skip ORT latency benchmark after export.",
    )
    p.add_argument(
        "--ref-audio", default=None,
        help="Path to a reference .wav file for latency benchmark. "
             "If omitted, benchmark is skipped.",
    )
    p.add_argument(
        "--n-warmup", type=int, default=5,
        help="ORT warm-up runs (default: 5).",
    )
    p.add_argument(
        "--n-bench", type=int, default=30,
        help="ORT timed runs (default: 30).",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    onnx_dir = PROJECT_ROOT / args.output_dir

    # 1. Export FP32
    fp32_path = export_fp32(args.model_mode, onnx_dir)

    # 2. Quantise to INT8
    int8_path: Path | None = None
    if not args.skip_int8:
        int8_path = quantize_int8(args.model_mode, onnx_dir, fp32_path)

    # 3. Optional latency benchmark
    latency: dict = {}
    if not args.skip_latency and args.ref_audio:
        ref = args.ref_audio
        log.info("ORT latency benchmark (ref audio: %s) …", ref)

        fp32_version = fp32_path.stem  # "v2_ctc" or "v2_ctc_encoder"
        latency["fp32"] = benchmark_ort(
            onnx_dir, fp32_version, args.n_warmup, args.n_bench, ref
        )
        log.info(
            "  FP32  avg=%.1f ms  P50=%.1f  P95=%.1f",
            latency["fp32"]["avg"],
            latency["fp32"]["P50"],
            latency["fp32"]["P95"],
        )

        if int8_path is not None:
            int8_version = int8_path.stem   # "v2_ctc_int8"
            latency["int8"] = benchmark_ort(
                onnx_dir, int8_version, args.n_warmup, args.n_bench, ref
            )
            log.info(
                "  INT8  avg=%.1f ms  P50=%.1f  P95=%.1f",
                latency["int8"]["avg"],
                latency["int8"]["P50"],
                latency["int8"]["P95"],
            )
            speedup = latency["fp32"]["avg"] / latency["int8"]["avg"]
            log.info("  Speedup: %.2f×", speedup)
    elif not args.skip_latency and not args.ref_audio:
        log.info(
            "Skipping ORT latency benchmark (no --ref-audio provided). "
            "Use benchmark_gigaam_v2_int8.py for full latency + quality eval."
        )

    # 4. Save summary
    cfg_path = onnx_dir / "export_summary.json"
    summary = {
        "model_mode":  args.model_mode,
        "fp32_onnx":   str(fp32_path.name),
        "fp32_size_mb": round(fp32_path.stat().st_size / 1024 / 1024, 1),
        "int8_onnx":   str(int8_path.name) if int8_path else None,
        "int8_size_mb": round(int8_path.stat().st_size / 1024 / 1024, 1)
                        if int8_path else None,
        "ort_latency": latency,
        "usage": {
            "load_fp32":  f"load_onnx('{onnx_dir}', '{fp32_path.stem}')",
            "load_int8":  f"load_onnx('{onnx_dir}', '{int8_path.stem}')"
                          if int8_path else None,
            "infer":      "infer_onnx(audio_path, model_cfg, sessions)",
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("Done. Artefacts in: %s", onnx_dir)
    log.info("  FP32 → %s", fp32_path.name)
    if int8_path:
        log.info("  INT8 → %s", int8_path.name)
    log.info("")
    log.info("Next step — full quality+latency benchmark:")
    log.info(
        "  python scripts/benchmark_gigaam_v2_int8.py --onnx-dir %s",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
