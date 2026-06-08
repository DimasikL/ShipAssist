"""
prepare_lora_for_deployment.py

Prepares a LoRA (or already-merged) Wav2Vec2 model for deployment:
  1. Detects whether the model is a merged checkpoint or an unmerged
     PEFT adapter, and merges LoRA weights into the base model if needed.
  2. Exports the merged model to ONNX FP32.
  3. Optionally quantizes to INT8 and runs a latency benchmark.
  4. Writes onnx_config.json + preprocessor_config.json to the output dir.

Usage:
    # Minimal (auto-derives output dir):
    python -m scripts.train.prepare_lora_for_deployment \
        --model_dir lora_tune/models/run_2026-04-30_23-34-27/best_model

    # Full control:
    python -m scripts.train.prepare_lora_for_deployment \
        --model_dir lora_tune/models/run_2026-04-30_23-34-27/best_model \
        --output_dir onnx_model/models/run_2026-04-30_23-34-27 \
        --quantize \
        --benchmark

After this script finishes, run the live recognizer with:
    python -m scripts.utils.main_calib_live_outdet_run run \
        --model_dir <output_dir>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
DEFAULT_SR = 16_000
DEFAULT_WINDOW_S = 1.0
DEFAULT_STRIDE_MS = 500.0
DEFAULT_BENCHMARK_ITERS = 20


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _ExportWrapper(torch.nn.Module):
    """Strips the HuggingFace output dataclass for ONNX tracing.

    Returns (logits, mean-pooled embedding) as plain tensors.
    """

    def __init__(self, model: Wav2Vec2ForSequenceClassification) -> None:
        super().__init__()
        self.wav2vec2 = model.wav2vec2
        self.projector = model.projector
        self.classifier = model.classifier

    def forward(self, input_values: torch.Tensor):
        """Forward pass returning (logits, embedding).

        Args:
            input_values: Raw waveform tensor of shape (batch, time).

        Returns:
            Tuple of (logits, mean-pooled embedding).
        """
        hidden = self.wav2vec2(input_values)[0]
        projected = self.projector(hidden)
        embedding = projected.mean(dim=1)
        logits = self.classifier(embedding)
        return logits, embedding


# ─────────────────────────────────────────────────────────────────────────────
#  ORT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_ort():
    """Imports onnxruntime lazily to avoid hard dependency at module level.

    Returns:
        The onnxruntime module, or None if not installed.
    """
    try:
        import onnxruntime
        return onnxruntime
    except ImportError:
        print("[WARN] onnxruntime not installed — skipping ORT steps.")
        print("       pip install onnxruntime")
        return None


def _make_session(ort, model_path: str):
    """Creates an optimised ORT InferenceSession.

    Args:
        ort: The onnxruntime module.
        model_path: Path to the .onnx file.

    Returns:
        onnxruntime.InferenceSession
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        model_path, opts, providers=["CPUExecutionProvider"]
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Step helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_unmerged_lora(model_dir: Path) -> bool:
    """Returns True if the directory contains an unmerged PEFT adapter.

    A directory is considered unmerged when it has adapter_model.safetensors
    (or adapter_model.bin) but lora_info.json reports merged=False (or is
    absent).

    Args:
        model_dir: Path to the model directory.

    Returns:
        True if LoRA weights need to be merged before export.
    """
    has_adapter = (
        (model_dir / "adapter_model.safetensors").exists()
        or (model_dir / "adapter_model.bin").exists()
    )
    if not has_adapter:
        return False

    lora_info_path = model_dir / "lora_info.json"
    if lora_info_path.exists():
        info = json.loads(lora_info_path.read_text(encoding="utf-8"))
        return not info.get("merged", False)

    # adapter file present but no lora_info.json → assume unmerged
    return True


def _merge_lora(model_dir: Path, output_dir: Path) -> Path:
    """Merges PEFT LoRA adapters into the base model and saves the result.

    Args:
        model_dir: Directory containing the unmerged PEFT adapter.
        output_dir: Directory to write the merged model to.

    Returns:
        Path to the merged model directory (a sub-folder of output_dir).

    Raises:
        ImportError: If the peft package is not installed.
        RuntimeError: If the merge fails for any reason.
    """
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "peft is required to merge LoRA adapters.\n"
            "  pip install peft"
        ) from exc

    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MERGE] Loading base model from adapter config …")
    adapter_cfg_path = model_dir / "adapter_config.json"
    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    base_name = adapter_cfg.get("base_model_name_or_path", DEFAULT_BASE_MODEL)
    print(f"[MERGE] Base model: {base_name}")

    base = Wav2Vec2ForSequenceClassification.from_pretrained(base_name)
    peft_model = PeftModel.from_pretrained(base, str(model_dir))

    print("[MERGE] Merging LoRA weights into base model …")
    merged = peft_model.merge_and_unload()
    merged.eval()

    print(f"[MERGE] Saving merged model → {merged_dir}")
    merged.save_pretrained(str(merged_dir))
    return merged_dir


def _load_model(model_dir: Path) -> Wav2Vec2ForSequenceClassification:
    """Loads a merged Wav2Vec2ForSequenceClassification model.

    When the directory also contains PEFT adapter files (adapter_model.safetensors),
    newer Transformers versions try to call load_adapter() on the path and fail.
    We avoid that by building the model from its config and loading the merged
    state dict from model.safetensors directly.

    Args:
        model_dir: Path to the merged model directory.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If neither model.safetensors nor pytorch_model.bin
            is present in model_dir.
    """
    from transformers import Wav2Vec2Config

    config = Wav2Vec2Config.from_pretrained(str(model_dir))
    model = Wav2Vec2ForSequenceClassification(config)

    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
        print(f"  Loading weights from model.safetensors")
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
        print(f"  Loading weights from pytorch_model.bin")
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    model.eval()
    return model


def _export_onnx(
    model: Wav2Vec2ForSequenceClassification,
    feature_extractor: Wav2Vec2FeatureExtractor,
    output_dir: Path,
    sr: int,
    window_s: float,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Exports the model to ONNX FP32 and verifies it against PyTorch.

    Args:
        model: Merged PyTorch model in eval mode.
        feature_extractor: HuggingFace feature extractor.
        output_dir: Directory to write model_fp32.onnx into.
        sr: Sample rate in Hz.
        window_s: Audio window length in seconds (determines dummy input size).

    Returns:
        Tuple of (fp32_path, np_input, pt_logits_np) for downstream steps.

    Raises:
        AssertionError: If the ORT output deviates more than 1e-3 from PyTorch.
        SystemExit: If the ONNX export itself fails.
    """
    fp32_path = output_dir / "model_fp32.onnx"
    win_samples = int(window_s * sr)

    dummy = np.random.randn(win_samples).astype(np.float32) * 0.01
    inputs = feature_extractor(
        dummy, sampling_rate=sr, return_tensors="pt", padding=True
    )
    input_values: torch.Tensor = inputs["input_values"]

    wrapper = _ExportWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        logits, embedding = wrapper(input_values)

    # Sanity-check wrapper matches original
    with torch.no_grad():
        orig_logits = model(input_values).logits
    diff = float((logits - orig_logits).abs().max())
    assert diff < 1e-5, f"ExportWrapper mismatch: {diff:.2e}"
    print(f"  Wrapper vs original logits diff: {diff:.2e}  ✓")
    print(f"  Logits shape:    {list(logits.shape)}")
    print(f"  Embedding shape: {list(embedding.shape)}")

    try:
        torch.onnx.export(
            wrapper,
            (input_values,),
            str(fp32_path),
            input_names=["input_values"],
            output_names=["logits", "embedding"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    except Exception as exc:
        print(f"\n[ERROR] ONNX export failed: {exc}")
        sys.exit(1)

    fp32_mb = fp32_path.stat().st_size / (1024 ** 2)
    print(f"  Exported: {fp32_path.name}  ({fp32_mb:.1f} MB)")

    import onnx
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    print("  ONNX graph check: ✓")

    # ORT verification
    ort = _load_ort()
    if ort is not None:
        np_input = input_values.numpy()
        pt_logits_np = logits.detach().numpy()
        sess = _make_session(ort, str(fp32_path))
        ort_logits, _ = sess.run(None, {"input_values": np_input})
        ort_diff = float(np.abs(pt_logits_np - ort_logits).max())
        print(f"  ORT vs PyTorch logits diff: {ort_diff:.2e}", end="")
        assert ort_diff < 1e-3, f"ORT verification failed! diff={ort_diff}"
        print("  ✓")
        return fp32_path, np_input, pt_logits_np

    return fp32_path, input_values.numpy(), logits.detach().numpy()


def _quantize_int8(
    ort,
    output_dir: Path,
    fp32_path: Path,
    np_input: np.ndarray,
    pt_logits_np: np.ndarray,
) -> Optional[Path]:
    """Quantizes the FP32 ONNX model to INT8 dynamic quantization.

    Args:
        ort: The onnxruntime module.
        output_dir: Directory to write model_int8.onnx into.
        fp32_path: Path to the FP32 ONNX model.
        np_input: Dummy numpy input for verification.
        pt_logits_np: PyTorch logits for class-match verification.

    Returns:
        Path to the INT8 model, or None if quantization failed.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("  [WARN] onnxruntime.quantization not available — skipping.")
        return None

    int8_path = output_dir / "model_int8.onnx"
    try:
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
    except Exception as exc:
        print(f"  [WARN] Quantization failed: {exc}")
        return None

    fp32_mb = fp32_path.stat().st_size / (1024 ** 2)
    int8_mb = int8_path.stat().st_size / (1024 ** 2)
    print(f"  INT8: {int8_mb:.1f} MB  (compression {fp32_mb / int8_mb:.1f}×)")

    sess_q = _make_session(ort, str(int8_path))
    q_logits, _ = sess_q.run(None, {"input_values": np_input})

    fp32_pred = int(np.argmax(pt_logits_np))
    int8_pred = int(np.argmax(q_logits))
    match = fp32_pred == int8_pred

    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    prob_diff = float(
        np.abs(_softmax(pt_logits_np.flatten()) - _softmax(q_logits.flatten())).max()
    )
    print(
        f"  Predicted class: FP32={fp32_pred} INT8={int8_pred} "
        f"→ {'✓ match' if match else '⚠ mismatch'}  "
        f"softmax diff={prob_diff:.4f}"
    )
    return int8_path


def _benchmark(
    ort,
    model: Wav2Vec2ForSequenceClassification,
    input_values: torch.Tensor,
    np_input: np.ndarray,
    fp32_path: Path,
    int8_path: Optional[Path],
    n_iters: int,
    stride_ms: float,
) -> dict:
    """Benchmarks PyTorch, ONNX FP32, and ONNX INT8 inference latency.

    Args:
        ort: The onnxruntime module.
        model: PyTorch model in eval mode.
        input_values: PyTorch input tensor.
        np_input: Numpy input array (same data).
        fp32_path: Path to FP32 ONNX model.
        int8_path: Path to INT8 ONNX model, or None.
        n_iters: Number of iterations to average over.
        stride_ms: Audio stride in ms (used to assess real-time capability).

    Returns:
        Dict with latency values for each backend.
    """
    results: dict = {}

    # PyTorch
    with torch.no_grad():
        for _ in range(3):
            model(input_values)
        t0 = time.monotonic()
        for _ in range(n_iters):
            model(input_values)
        pt_ms = (time.monotonic() - t0) / n_iters * 1000
    print(f"  PyTorch FP32:  {pt_ms:.0f} ms")
    results["pytorch_fp32_ms"] = round(pt_ms, 1)

    # ONNX FP32
    sess_fp32 = _make_session(ort, str(fp32_path))
    for _ in range(3):
        sess_fp32.run(None, {"input_values": np_input})
    t0 = time.monotonic()
    for _ in range(n_iters):
        sess_fp32.run(None, {"input_values": np_input})
    ort_ms = (time.monotonic() - t0) / n_iters * 1000
    print(f"  ONNX FP32:     {ort_ms:.0f} ms  (×{pt_ms / ort_ms:.1f})")
    results["onnx_fp32_ms"] = round(ort_ms, 1)

    # ONNX INT8
    if int8_path and int8_path.exists():
        sess_int8 = _make_session(ort, str(int8_path))
        for _ in range(3):
            sess_int8.run(None, {"input_values": np_input})
        t0 = time.monotonic()
        for _ in range(n_iters):
            sess_int8.run(None, {"input_values": np_input})
        q_ms = (time.monotonic() - t0) / n_iters * 1000
        print(f"  ONNX INT8:     {q_ms:.0f} ms  (×{pt_ms / q_ms:.1f})")
        results["onnx_int8_ms"] = round(q_ms, 1)

    best_ms = min(results.values())
    rt_ok = best_ms < stride_ms
    print(
        f"  Stride: {stride_ms:.0f} ms  →  "
        f"{'✓ real-time OK' if rt_ok else '⚠ slower than stride'} "
        f"(best {best_ms:.0f} ms)"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare(args: argparse.Namespace) -> None:
    """End-to-end pipeline: merge → export → quantize → benchmark.

    Args:
        args: Parsed CLI arguments.
    """
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"[ERROR] model_dir not found: {model_dir}")
        sys.exit(1)

    # Auto-derive output dir from model_dir if not specified.
    # e.g. lora_tune/models/run_2026-04-30_23-34-27/best_model
    #   →  onnx_model/models/run_2026-04-30_23-34-27
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        run_name = model_dir.parent.name  # e.g. run_2026-04-30_23-34-27
        output_dir = Path("onnx_model") / "models" / run_name
        output_dir = output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  LoRA → ONNX Deployment Preparation")
    print("=" * 62)
    print(f"\n  Source : {model_dir}")
    print(f"  Output : {output_dir}\n")

    # ── Step 1: Resolve merged model directory ────────────────────────────
    print("── [1/5] Model resolution ───────────────────────────────────")
    if _is_unmerged_lora(model_dir):
        print("  Detected unmerged PEFT adapter. Merging LoRA weights …")
        merged_dir = _merge_lora(model_dir, output_dir)
        print(f"  Merged model saved → {merged_dir}")
    else:
        merged_dir = model_dir
        print("  Detected merged model. Using directory directly.")

    # ── Step 2: Load model & feature extractor ────────────────────────────
    print("\n── [2/5] Loading model ──────────────────────────────────────")
    model = _load_model(merged_dir)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.base_model_name
    )
    feature_extractor.save_pretrained(str(output_dir))

    id2label = model.config.id2label
    if id2label:
        keys = list(id2label.keys())
        labels = (
            [id2label[i] for i in range(len(id2label))]
            if isinstance(keys[0], int)
            else [id2label[str(i)] for i in range(len(id2label))]
        )
    else:
        labels = [f"class_{i}" for i in range(model.config.num_labels)]
    print(f"  Labels ({len(labels)}): {labels}")

    # ── Step 3: ONNX FP32 export ─────────────────────────────────────────
    print("\n── [3/5] ONNX FP32 export ───────────────────────────────────")
    fp32_path, np_input, pt_logits_np = _export_onnx(
        model, feature_extractor, output_dir,
        sr=args.sr, window_s=args.window_s,
    )
    input_values_tensor = torch.from_numpy(np_input)

    # ── Step 4: INT8 quantization ─────────────────────────────────────────
    int8_path: Optional[Path] = None
    ort = _load_ort()

    if args.quantize:
        print("\n── [4/5] INT8 quantization ──────────────────────────────────")
        if ort is None:
            print("  Skipped (onnxruntime not installed).")
        else:
            int8_path = _quantize_int8(
                ort, output_dir, fp32_path, np_input, pt_logits_np
            )
    else:
        print("\n── [4/5] INT8 quantization  ─────────────────────────── skipped")
        print("  Pass --quantize to enable.")

    # ── Step 5: Benchmark ─────────────────────────────────────────────────
    benchmark_results: dict = {}

    if args.benchmark:
        print("\n── [5/5] Latency benchmark ──────────────────────────────────")
        if ort is None:
            print("  Skipped (onnxruntime not installed).")
        else:
            benchmark_results = _benchmark(
                ort=ort,
                model=model,
                input_values=input_values_tensor,
                np_input=np_input,
                fp32_path=fp32_path,
                int8_path=int8_path,
                n_iters=args.benchmark_iters,
                stride_ms=args.stride_ms,
            )
    else:
        print("\n── [5/5] Benchmark  ─────────────────────────────────── skipped")
        print("  Pass --benchmark to enable.")

    # ── Write onnx_config.json ────────────────────────────────────────────
    win_samples = int(args.window_s * args.sr)
    fp32_mb = fp32_path.stat().st_size / (1024 ** 2)
    int8_mb = int8_path.stat().st_size / (1024 ** 2) if int8_path else None

    config = {
        "labels": labels,
        "sr": args.sr,
        "window_s": args.window_s,
        "win_samples": win_samples,
        "embedding_dim": int(model.config.hidden_size),
        "num_labels": len(labels),
        "base_model_name": args.base_model_name,
        "source_model_dir": str(model_dir),
        "model_fp32": "model_fp32.onnx",
        "model_int8": "model_int8.onnx" if int8_path else None,
        "fp32_size_mb": round(fp32_mb, 1),
        "int8_size_mb": round(int8_mb, 1) if int8_mb else None,
        "benchmark": benchmark_results,
    }
    config_path = output_dir / "onnx_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Done! Output files:")
    print("=" * 62)
    print(f"\n  {output_dir}/")
    print(f"    model_fp32.onnx            {fp32_mb:.0f} MB")
    if int8_path:
        print(f"    model_int8.onnx            {int8_mb:.0f} MB  ← recommended")
    print(f"    onnx_config.json")
    print(f"    preprocessor_config.json")
    print()
    print("  Run live recognizer (PyTorch):")
    print(
        f"    python -m scripts.utils.main_calib_live_outdet_run run \\\n"
        f"        --model_dir {merged_dir}"
    )
    if int8_path:
        print()
        print("  Run live recognizer (ONNX — not yet wired in the run script):")
        print(
            f"    RealTimeRecognizer(model_dir=..., onnx_dir='{output_dir}', "
            f"onnx_use_int8=True)"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Populated argparse.Namespace.
    """
    p = argparse.ArgumentParser(
        description="Prepare a LoRA Wav2Vec2 model for deployment (ONNX export).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_dir",
        required=True,
        help="Path to the LoRA best_model directory.",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Where to write ONNX files. "
            "Defaults to onnx_model/models/<run_name>/ derived from model_dir."
        ),
    )
    p.add_argument(
        "--base_model_name",
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace base model name (used for feature extractor).",
    )
    p.add_argument(
        "--window_s",
        type=float,
        default=DEFAULT_WINDOW_S,
        help="Audio window length in seconds (must match training config).",
    )
    p.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SR,
        help="Sample rate in Hz.",
    )
    p.add_argument(
        "--stride_ms",
        type=float,
        default=DEFAULT_STRIDE_MS,
        help="Audio stride in ms (used only for real-time benchmark display).",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Also produce a dynamic INT8-quantized ONNX model.",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark (PyTorch vs ONNX FP32 vs INT8).",
    )
    p.add_argument(
        "--benchmark_iters",
        type=int,
        default=DEFAULT_BENCHMARK_ITERS,
        help="Number of iterations for the latency benchmark.",
    )
    return p.parse_args()


if __name__ == "__main__":
    prepare(_parse_args())
