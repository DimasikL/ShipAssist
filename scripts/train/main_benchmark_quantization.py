"""
benchmark_quantization.py — Export and CPU-benchmark FP32 / FP16 / UINT8 / INT8.

Produces a side-by-side table:

  Format   Size(MB)   Latency(ms)   Speedup   Pred   Confidence
  FP32       1204        850 ms       1.0×      ✓        0.991
  FP16        602       GPU only      —         —        —
  UINT8       340        230 ms       3.7×      ✓        0.988
  INT8        301        210 ms       4.0×      ✓        0.985

NOTE: INT16 has been removed.  ONNX Runtime's MatMulInteger operator only
accepts QInt8 / QUInt8 weight types; passing QInt16 produces an
INVALID_GRAPH error at session load time.  UINT8 is the correct unsigned
counterpart to INT8 and is fully supported on CPU.

Usage
-----
    python scripts/train/benchmark_quantization.py \\
        --model_dir lora_tune/models/run_2026-04-30_23-34-27/best_model \\
        --output_dir onnx_model/quant_benchmark \\
        --audio_sec 3.0 \\
        --n_iters 30

Requirements
------------
    pip install onnx onnxruntime onnxconverter-common
    (onnxruntime >= 1.16 recommended for INT16 support)
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pt_model(model_dir: str):
    """Load PyTorch model from a LoRA best_model checkpoint.

    The saved best_model folder contains adapter fields in config.json that
    cause HuggingFace's from_pretrained to call load_adapter() internally,
    which fails because adapter_attn_dim is not defined.  We detect this case
    via lora_info.json and fall back to a manual state_dict overlay.
    """
    import json
    import torch
    import transformers
    transformers.logging.set_verbosity_error()

    from transformers import Wav2Vec2ForSequenceClassification

    BASE_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path         = os.path.join(model_dir, "pytorch_model.bin")
    config_path      = os.path.join(model_dir, "config.json")
    lora_info_path   = os.path.join(model_dir, "lora_info.json")

    # Read label map from config.json
    with open(config_path) as f:
        cfg = json.load(f)
    id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    # Determine whether this is a fully-merged checkpoint
    is_merged = True
    if os.path.exists(lora_info_path):
        with open(lora_info_path) as f:
            is_merged = bool(json.load(f).get("merged", False))

    # Route 1: merged checkpoint — try standard from_pretrained(local)
    if is_merged:
        try:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            )
            model.eval()
            print(f"         Loaded via from_pretrained (merged).")
            return model
        except Exception as e:
            print(f"         from_pretrained(local) failed: {e}")
            print(f"         → falling back to state_dict overlay.")

    # Route 2: unmerged adapter save — load base model + apply saved weights
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    else:
        raise FileNotFoundError(f"No weights found in {model_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"         Missing keys: {missing}")
    if unexpected:
        print(f"         Unexpected keys: {unexpected}")

    model.eval()
    print(f"         Loaded via state_dict overlay ({sum(p.numel() for p in model.parameters()):,} params).")
    return model


def _make_dummy_input(audio_sec: float = 3.0, sr: int = 16_000):
    """Create a realistic dummy input (3 s of low-level noise)."""
    from transformers import Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    )
    rng = np.random.default_rng(42)
    waveform = (rng.standard_normal(int(audio_sec * sr)) * 0.01).astype(np.float32)
    inputs = fe(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    return inputs["input_values"]


def _file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _ort_session(path: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def _benchmark_session(session, np_input: np.ndarray, n_iters: int) -> float:
    """Return mean latency in ms over n_iters (3 warmup runs excluded)."""
    input_name = session.get_inputs()[0].name
    for _ in range(3):
        session.run(None, {input_name: np_input})
    t0 = time.perf_counter()
    for _ in range(n_iters):
        session.run(None, {input_name: np_input})
    return (time.perf_counter() - t0) / n_iters * 1000


def _benchmark_pytorch(model, input_values, n_iters: int) -> float:
    import torch
    with torch.no_grad():
        for _ in range(3):
            model(input_values)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(input_values)
    return (time.perf_counter() - t0) / n_iters * 1000


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_fp32(model, input_values, output_dir: str) -> str:
    """Export base FP32 ONNX model.

    Uses opset 14 — the highest opset that onnxruntime.quantization supports
    reliably without a separate pre-processing pass.
    """
    import torch
    import onnx

    path = os.path.join(output_dir, "model_fp32.onnx")
    if os.path.exists(path):
        print(f"  [FP32]  Reusing existing {path}")
        return path

    print("  [FP32]  Exporting (opset 14)...")

    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x).logits

    wrapper = _Wrapper(model).eval()

    torch.onnx.export(
        wrapper,
        (input_values,),
        path,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=14,          # 14 = safe for quantization tools
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(path))
    print(f"         → {_file_mb(path):.1f} MB  ✓")
    return path


def _preprocess_for_quant(fp32_path: str, output_dir: str) -> str:
    """Run onnxruntime's quant_pre_process (shape inference + graph optimisation).

    This is required before quantize_dynamic when the model uses opset >= 13
    with certain operator patterns.  Returns the path of the preprocessed model.
    """
    pre_path = os.path.join(output_dir, "model_fp32_pre.onnx")
    if os.path.exists(pre_path):
        return pre_path
    try:
        from onnxruntime.quantization import quant_pre_process
        quant_pre_process(fp32_path, pre_path, skip_optimization=False)
    except Exception as e:
        # quant_pre_process is optional — fall back to the original model
        print(f"         quant_pre_process skipped ({e}), using raw FP32.")
        return fp32_path
    return pre_path


def export_fp16(fp32_path: str, output_dir: str) -> Optional[str]:
    """Convert FP32 ONNX → FP16 for size comparison (GPU inference only).

    NOTE: ORT CPUExecutionProvider does not support float16 computation.
    This format is exported so you can report the model size, but its
    latency will NOT be benchmarked on CPU.
    """
    path = os.path.join(output_dir, "model_fp16.onnx")
    if os.path.exists(path):
        print(f"  [FP16]  Reusing existing {path}  (size only — GPU format)")
        return path

    try:
        import onnx
        from onnxconverter_common import float16
        model_fp32 = onnx.load(fp32_path)
        # keep_io_types=True keeps model inputs/outputs as float32 so the
        # model stays callable with standard numpy float32 arrays.
        model_fp16 = float16.convert_float_to_float16(
            model_fp32, keep_io_types=True
        )
        onnx.save(model_fp16, path)
        print(f"  [FP16]  → {_file_mb(path):.1f} MB  ✓  (size only — GPU format)")
        return path
    except ImportError:
        print("  [FP16]  ⚠ Skipped — pip install onnxconverter-common")
        return None
    except Exception as e:
        print(f"  [FP16]  ⚠ Failed: {e}")
        return None


def export_int8(fp32_path: str, output_dir: str) -> Optional[str]:
    """Dynamic INT8 quantization via onnxruntime.quantization."""
    path = os.path.join(output_dir, "model_int8.onnx")
    if os.path.exists(path):
        print(f"  [INT8]  Reusing existing {path}")
        return path

    pre_path = _preprocess_for_quant(fp32_path, output_dir)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=pre_path,
            model_output=path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        print(f"  [INT8]  → {_file_mb(path):.1f} MB  ✓")
        return path
    except ImportError:
        print("  [INT8]  ⚠ Skipped — onnxruntime not installed")
        return None
    except Exception as e:
        print(f"  [INT8]  ⚠ Failed: {e}")
        return None


def export_uint8(fp32_path: str, output_dir: str) -> Optional[str]:
    """Dynamic UINT8 quantization via onnxruntime.quantization.

    ONNX Runtime's MatMulInteger operator requires unsigned 8-bit (QUInt8) or
    signed 8-bit (QInt8) weight types.  INT16 is *not* supported by the op and
    produces an INVALID_GRAPH error at load time, so we replace that format
    with UINT8 which is fully supported on CPU and gives a complementary
    data-point alongside INT8.
    """
    path = os.path.join(output_dir, "model_uint8.onnx")
    if os.path.exists(path):
        print(f"  [UINT8] Reusing existing {path}")
        return path

    pre_path = _preprocess_for_quant(fp32_path, output_dir)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=pre_path,
            model_output=path,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        print(f"  [UINT8] → {_file_mb(path):.1f} MB  ✓")
        return path
    except ImportError:
        print("  [UINT8] ⚠ Skipped — onnxruntime not installed")
        return None
    except Exception as e:
        print(f"  [UINT8] ⚠ Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FP32 / FP16 / INT16 / INT8 ONNX models on CPU."
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to the best_model folder (HF checkpoint).",
    )
    parser.add_argument(
        "--output_dir", default="onnx_model/quant_benchmark",
        help="Where to save ONNX files and results.",
    )
    parser.add_argument("--audio_sec", type=float, default=3.0,
                        help="Dummy input length in seconds (default 3.0).")
    parser.add_argument("--n_iters", type=int, default=30,
                        help="Number of benchmark iterations per format.")
    parser.add_argument("--skip_pytorch", action="store_true",
                        help="Skip the PyTorch CPU baseline (slow on large models).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  Quantization Benchmark")
    print(f"  Model : {args.model_dir}")
    print(f"  Input : {args.audio_sec}s  |  Iters: {args.n_iters}")
    print("=" * 65)

    # ── Load PyTorch model ──
    print("\n[1/3] Loading PyTorch model...")
    model = _load_pt_model(args.model_dir)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    # ── Dummy input ──
    print("[2/3] Building dummy input...")
    input_values = _make_dummy_input(args.audio_sec)
    np_input = input_values.numpy().astype(np.float32)

    # Ground-truth prediction from PyTorch (FP32)
    import torch
    with torch.no_grad():
        pt_logits = model(input_values).logits.numpy()
    ref_pred = int(np.argmax(pt_logits))
    ref_conf = float(_softmax(pt_logits.flatten()).max())

    # ── Export all formats ──
    print("\n[3/3] Exporting formats...")
    fp32_path  = export_fp32(model, input_values, args.output_dir)
    fp16_path  = export_fp16(fp32_path, args.output_dir)
    int8_path  = export_int8(fp32_path, args.output_dir)
    uint8_path = export_uint8(fp32_path, args.output_dir)

    # ── Benchmark ──
    print(f"\n{'─'*65}")
    print(f"  Benchmarking on CPU  ({os.cpu_count()} logical cores, {args.n_iters} iters)")
    print(f"{'─'*65}")

    rows = []

    # PyTorch FP32 baseline
    if not args.skip_pytorch:
        pt_ms = _benchmark_pytorch(model, input_values, args.n_iters)
        rows.append({
            "Format": "PyTorch FP32",
            "Size (MB)": f"{_file_mb(fp32_path):.0f}",
            "Latency (ms)": f"{pt_ms:.0f}",
            "Speedup": "1.0×",
            "Pred OK": "✓",
            "Confidence": f"{ref_conf:.4f}",
        })
        baseline_ms = pt_ms
    else:
        baseline_ms = None

    # FP16 is a GPU-only format on ORT — report size but skip latency benchmark
    if fp16_path and os.path.exists(fp16_path):
        rows.append({
            "Format":       "ONNX FP16",
            "Size (MB)":    f"{_file_mb(fp16_path):.0f}",
            "Latency (ms)": "GPU only",
            "Speedup":      "—",
            "Pred OK":      "—",
            "Confidence":   "—",
        })

    for label, path in [
        ("ONNX FP32",  fp32_path),
        ("ONNX UINT8", uint8_path),
        ("ONNX INT8",  int8_path),
    ]:
        if path is None or not os.path.exists(path):
            continue

        try:
            sess     = _ort_session(path)
            lat_ms   = _benchmark_session(sess, np_input, args.n_iters)
            logits   = sess.run(None, {sess.get_inputs()[0].name: np_input})[0]
            pred     = int(np.argmax(logits))
            conf     = float(_softmax(logits.flatten()).max())
            pred_ok  = "✓" if pred == ref_pred else f"✗ ({id2label.get(pred, pred)})"

            if baseline_ms is None and label == "ONNX FP32":
                baseline_ms = lat_ms

            speedup = f"{baseline_ms / lat_ms:.1f}×" if baseline_ms else "—"

            rows.append({
                "Format":       label,
                "Size (MB)":    f"{_file_mb(path):.0f}",
                "Latency (ms)": f"{lat_ms:.0f}",
                "Speedup":      speedup,
                "Pred OK":      pred_ok,
                "Confidence":   f"{conf:.4f}",
            })
        except Exception as e:
            print(f"  [{label}] ⚠ Benchmark failed: {e}")

    # ── Print table ──
    cols = ["Format", "Size (MB)", "Latency (ms)", "Speedup", "Pred OK", "Confidence"]
    col_w = {c: max(len(c), max(len(r[c]) for r in rows)) + 2 for c in cols}

    header = "  " + "".join(c.ljust(col_w[c]) for c in cols)
    sep    = "  " + "".join("─" * col_w[c] for c in cols)

    print()
    print(header)
    print(sep)
    for r in rows:
        print("  " + "".join(r[c].ljust(col_w[c]) for c in cols))
    print()

    # ── Save JSON results ──
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "ref_label": id2label.get(ref_pred, str(ref_pred))}, f, indent=2)
    print(f"  Results saved → {results_path}")
    print()

    # ── Recommendation ──
    # Only consider rows that have a real numeric latency (skip GPU-only / failed entries).
    def _numeric_latency(r: dict) -> Optional[float]:
        try:
            return float(r["Latency (ms)"])
        except (ValueError, TypeError):
            return None

    benchmarked = [r for r in rows if _numeric_latency(r) is not None]
    if benchmarked:
        accurate = [r for r in benchmarked if r["Pred OK"] == "✓"]
        candidates = accurate if accurate else benchmarked
        best = min(candidates, key=lambda r: _numeric_latency(r))
        print(f"  Recommendation: use  {best['Format']}  "
              f"({best['Latency (ms)']} ms, size {best['Size (MB)']} MB, "
              f"pred {best['Pred OK']})")
        print()


if __name__ == "__main__":
    main()
