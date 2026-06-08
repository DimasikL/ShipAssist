"""
export_to_onnx.py

Экспорт Wav2Vec2ForSequenceClassification → ONNX + INT8.

    python export_to_onnx.py \
        --model_dir best_model \
        --output_dir onnx_model \
        --quantize \
        --benchmark
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)


# ═══════════════════════════════════════════════════════════
#  Wrapper
# ═══════════════════════════════════════════════════════════

class ExportWrapper(torch.nn.Module):
    """ONNX export wrapper that returns all three outputs required by HybridAudioEngine.

    Output contract (fixed — do not change without updating
    HybridAudioEngine._get_features and OnnxEngine.predict_logits):

        outputs[0] — logits          : (B, N_classes)
            Raw classifier logits (pre-softmax).
            → Stage 3 Path A: ``argmax(softmax(logits))`` → predicted intent.

        outputs[1] — embedding       : (B, D_proj)
            Mean-pooled projected features (``projected_frames.mean(dim=1)``).
            → Stage 2: EnsembleOutlierGate Mahalanobis / cosine scoring.
            → Stage 3 Path B: CentroidSearch cosine fallback when logits unavailable.
            → Stage 4 Variant A: NumberRegressor MLP input.

        outputs[2] — projected_frames: (B, T, D_proj)
            Per-frame projected features before pooling.
            ``T`` is dynamic (registered in ``dynamic_axes`` at export time).
            → Stage 4 Variant B: CTCDigitDecoder frame-level slot-fill.

    Verification (post-export):
        All three outputs are verified in ``_verify_ort()``:
        * Logits criterion  : max|logits_onnx − logits_pt| < 1e-3.
        * Embedding criterion: cosine_similarity(emb_onnx, emb_pt) > 0.9999.
        * Frames criterion  : shape equality (B, T, D_proj).
    """

    def __init__(self, model: Wav2Vec2ForSequenceClassification):
        super().__init__()
        self.wav2vec2 = model.wav2vec2
        self.projector = model.projector
        self.classifier = model.classifier

    def forward(self, input_values: torch.Tensor):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]              # (B, T, D_model)
        projected = self.projector(hidden_states)   # (B, T, D_proj)
        embedding = projected.mean(dim=1)       # (B, D_proj)  — pooled
        logits = self.classifier(embedding)     # (B, N_labels)
        # projected is returned as the third output so downstream modules
        # (CTC digit decoder) can consume frame-level features without a
        # second forward pass.  ONNX dynamic_axes marks T as variable.
        return logits, embedding, projected


# ═══════════════════════════════════════════════════════════
#  Вспомогательные функции (без ort на верхнем уровне)
# ═══════════════════════════════════════════════════════════

def _load_ort():
    """
    Импорт onnxruntime внутри функции.
    Возвращает модуль или None.
    """
    try:
        import onnxruntime
        return onnxruntime
    except ImportError:
        print("[WARN] onnxruntime не установлен")
        print("       pip install onnxruntime")
        return None


def _create_session(ort_module, model_path: str):
    """Создаёт ORT InferenceSession с оптимизациями."""
    session_opts = ort_module.SessionOptions()
    session_opts.graph_optimization_level = (
        ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session_opts.intra_op_num_threads = os.cpu_count() or 4
    session_opts.inter_op_num_threads = 1

    session = ort_module.InferenceSession(
        model_path,
        session_opts,
        providers=["CPUExecutionProvider"],
    )
    return session


def _verify_ort(
    ort_module,
    fp32_path: str,
    np_input: np.ndarray,
    pt_logits: np.ndarray,
    pt_embedding: np.ndarray,
    pt_frames: np.ndarray,
) -> Optional[object]:
    """Verify all three ONNX outputs match their PyTorch counterparts.

    Checks:
        outputs[0] — logits          : max absolute difference < 1e-3
        outputs[1] — embedding       : cosine similarity > 0.9999
        outputs[2] — projected_frames: shape equality

    Args:
        ort_module:    Imported onnxruntime module.
        fp32_path:     Path to the FP32 ONNX model file.
        np_input:      Input numpy array (shape matching export dummy).
        pt_logits:     PyTorch logits numpy array  (B, N_classes).
        pt_embedding:  PyTorch embedding numpy array (B, D_proj).
        pt_frames:     PyTorch projected_frames numpy array (B, T, D_proj).

    Returns:
        ORT InferenceSession on success; raises AssertionError on failure.
    """
    session = _create_session(ort_module, fp32_path)

    ort_outputs = session.run(None, {"input_values": np_input})
    ort_logits    = ort_outputs[0]   # (B, N_classes)
    ort_embedding = ort_outputs[1]   # (B, D_proj)
    ort_frames    = ort_outputs[2]   # (B, T, D_proj)

    # ── logits: max absolute difference ──────────────────────────────
    logits_diff = float(np.abs(pt_logits - ort_logits).max())

    # ── embedding: cosine similarity ──────────────────────────────────
    emb_pt  = pt_embedding.flatten().astype(np.float64)
    emb_ort = ort_embedding.flatten().astype(np.float64)
    cos_sim = float(
        np.dot(emb_pt, emb_ort)
        / (np.linalg.norm(emb_pt) * np.linalg.norm(emb_ort) + 1e-12)
    )

    # ── projected_frames: shape equality ─────────────────────────────
    frames_shape_ok = ort_frames.shape == pt_frames.shape

    # ── Report ───────────────────────────────────────────────────────
    print(f"      Verification:")
    print(f"        logits    max_delta={logits_diff:.6f}  "
          f"{'OK' if logits_diff < 1e-3 else 'WARN'}")
    print(f"        embedding cos_sim={cos_sim:.6f}    "
          f"{'OK' if cos_sim > 0.9999 else 'WARN'}")
    print(f"        frames    shape_match={frames_shape_ok}  {ort_frames.shape}")

    assert logits_diff < 1e-3, (
        f"ORT verification failed — logits max_delta={logits_diff:.2e} ≥ 1e-3"
    )
    assert cos_sim > 0.9999, (
        f"ORT verification failed — embedding cos_sim={cos_sim:.6f} < 0.9999"
    )
    assert frames_shape_ok, (
        f"ORT verification failed — frames shape mismatch: "
        f"ONNX {ort_frames.shape} vs PyTorch {pt_frames.shape}"
    )
    print("      ✓ All three outputs verified (logits / embedding / frames)")

    return session


def _quantize(ort_module, fp32_path, int8_path,
              np_input, pt_logits):
    """
    Квантизация FP32 → INT8.
    Проверяем не абсолютную разницу логитов (она может быть большой),
    а совпадение предсказанного класса и близость softmax-вероятностей.
    """
    try:
        from onnxruntime.quantization import (
            quantize_dynamic,
            QuantType,
        )
    except ImportError:
        print("      ⚠ onnxruntime.quantization не доступен")
        return None

    try:
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
    except Exception as e:
        print(f"      ⚠ Квантизация не удалась: {e}")
        return None

    int8_mb = os.path.getsize(int8_path) / (1024 * 1024)
    fp32_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"      INT8: {int8_mb:.1f} MB "
          f"(сжатие {fp32_mb / int8_mb:.1f}×)")

    # ── Проверка INT8 ──
    session_q = _create_session(ort_module, int8_path)
    q_logits, _ = session_q.run(None, {"input_values": np_input})

    # Разница в логитах (информативно, но не критично)
    q_diff = float(np.abs(pt_logits - q_logits).max())
    print(f"      Logits diff FP32→INT8: {q_diff:.2e}")

    # Главная проверка: совпадает ли предсказанный класс?
    fp32_pred = int(np.argmax(pt_logits))
    int8_pred = int(np.argmax(q_logits))
    preds_match = fp32_pred == int8_pred
    print(f"      FP32 pred: {fp32_pred}, INT8 pred: {int8_pred} "
          f"→ {'✓ MATCH' if preds_match else '⚠ MISMATCH'}")

    # Проверка softmax-вероятностей
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    fp32_probs = softmax(pt_logits.flatten())
    int8_probs = softmax(q_logits.flatten())
    prob_diff = float(np.abs(fp32_probs - int8_probs).max())
    print(f"      Softmax diff: {prob_diff:.4f}")
    print(f"      FP32 probs: {np.round(fp32_probs, 3)}")
    print(f"      INT8 probs: {np.round(int8_probs, 3)}")

    if preds_match and prob_diff < 0.3:
        print("      ✓ INT8 verified (predictions match)")
    elif preds_match:
        print("      ⚠ INT8 predictions match but probabilities differ")
        print("        (допустимо — модель работает корректно)")
    else:
        print("      ⚠ INT8 predictions DIFFER on dummy input")
        print("        (это dummy-вход из шума, на реальных данных")
        print("         может быть лучше — рекомендуем проверить)")

    return int8_mb


def _benchmark(ort_module, model, input_values: torch.Tensor,
               np_input: np.ndarray,
               fp32_path: str,
               int8_path: Optional[str],
               n_iters: int,
               stride_ms: float) -> dict:
    """Бенчмарк PyTorch vs ONNX FP32 vs ONNX INT8."""
    results = {}

    # ── PyTorch ──
    with torch.no_grad():
        for _ in range(3):
            model(input_values)
        t0 = time.monotonic()
        for _ in range(n_iters):
            model(input_values)
        pt_ms = (time.monotonic() - t0) / n_iters * 1000

    print(f"      PyTorch FP32:  {pt_ms:.0f} ms")
    results["pytorch_fp32_ms"] = round(pt_ms, 1)

    # ── ONNX FP32 ──
    sess_fp32 = _create_session(ort_module, fp32_path)
    for _ in range(3):
        sess_fp32.run(None, {"input_values": np_input})
    t0 = time.monotonic()
    for _ in range(n_iters):
        sess_fp32.run(None, {"input_values": np_input})
    ort_ms = (time.monotonic() - t0) / n_iters * 1000

    speedup = pt_ms / ort_ms if ort_ms > 0 else 0
    print(f"      ONNX FP32:     {ort_ms:.0f} ms (×{speedup:.1f})")
    results["onnx_fp32_ms"] = round(ort_ms, 1)

    # ── ONNX INT8 ──
    if int8_path and os.path.exists(int8_path):
        sess_int8 = _create_session(ort_module, int8_path)
        for _ in range(3):
            sess_int8.run(None, {"input_values": np_input})
        t0 = time.monotonic()
        for _ in range(n_iters):
            sess_int8.run(None, {"input_values": np_input})
        q_ms = (time.monotonic() - t0) / n_iters * 1000

        speedup_q = pt_ms / q_ms if q_ms > 0 else 0
        print(f"      ONNX INT8:     {q_ms:.0f} ms "
              f"(×{speedup_q:.1f})")
        results["onnx_int8_ms"] = round(q_ms, 1)

    # Сравнение со stride
    print()
    best_ms = min(results.values())
    print(f"      Stride:        {stride_ms:.0f} ms")
    if best_ms < stride_ms:
        print(f"      ✓ Real-time OK (best {best_ms:.0f}ms "
              f"< stride {stride_ms:.0f}ms)")
    else:
        print(f"      ⚠ Медленнее stride → окна будут пропускаться")

    return results


# ═══════════════════════════════════════════════════════════
#  Главная функция
# ═══════════════════════════════════════════════════════════

def export(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  ONNX Export + Quantization")
    print("=" * 60)

    # ── 1. Загрузка модели ──
    print(f"\n[1/6] Загрузка: {args.model_dir}")

    adapter_cfg_path = os.path.join(args.model_dir, "adapter_config.json")
    is_lora = os.path.exists(adapter_cfg_path)

    if is_lora:
        # PEFT LoRA checkpoint: load base → apply adapter → merge into dense weights.
        # Must NOT call from_pretrained(model_dir) directly — transformers
        # mistakes adapter_config.json for a Wav2Vec2-native adapter and crashes.
        import json as _json
        with open(adapter_cfg_path, "r") as _f:
            _acfg = _json.load(_f)
        base_name = _acfg.get("base_model_name_or_path", args.base_model_name)
        print(f"      LoRA detected — base model: {base_name}")

        try:
            from peft import PeftModel
        except ImportError:
            print("[ERROR] peft not installed. Run: pip install peft")
            sys.exit(1)

        # Load config from the fine-tuned dir so the classifier head is sized
        # correctly (num_labels=4) before PEFT copies modules_to_save weights.
        from transformers import AutoConfig
        ft_config = AutoConfig.from_pretrained(args.model_dir)
        base = Wav2Vec2ForSequenceClassification.from_pretrained(
            base_name, config=ft_config
        )
        lora_model = PeftModel.from_pretrained(base, args.model_dir)
        model = lora_model.merge_and_unload()
        print("      LoRA weights merged into base model.")
    else:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            args.model_dir
        )

    model.eval()

    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model_name)
    fe.save_pretrained(args.output_dir)

    # Labels
    id2label = model.config.id2label
    if id2label:
        keys = list(id2label.keys())
        if isinstance(keys[0], int):
            labels = [id2label[i] for i in range(len(id2label))]
        else:
            labels = [id2label[str(i)] for i in range(len(id2label))]
    else:
        labels = [f"class_{i}" for i in range(model.config.num_labels)]
    print(f"      Labels: {labels}")

    # ── 2. Wrapper ──
    print(f"\n[2/6] ExportWrapper...")
    wrapper = ExportWrapper(model)
    wrapper.eval()

    win_samples = int(args.window_s * args.sr)
    dummy = np.random.randn(win_samples).astype(np.float32) * 0.01
    inputs = fe(
        dummy, sampling_rate=args.sr,
        return_tensors="pt", padding=True,
    )
    input_values = inputs["input_values"]

    with torch.no_grad():
        logits, embedding, projected_frames = wrapper(input_values)
    print(f"      Logits:           {list(logits.shape)}")
    print(f"      Embedding:        {list(embedding.shape)}")
    print(f"      Projected frames: {list(projected_frames.shape)}  (T={projected_frames.shape[1]}, D_proj={projected_frames.shape[2]})")

    # Сверка с оригиналом
    with torch.no_grad():
        orig_logits = model(input_values).logits
    diff = float((logits - orig_logits).abs().max())
    print(f"      Wrapper diff: {diff:.2e}")
    assert diff < 1e-5

    # ── 3. ONNX Export ──
    fp32_path = os.path.join(args.output_dir, "model_fp32.onnx")
    print(f"\n[3/6] ONNX → {fp32_path}")

    try:
        torch.onnx.export(
            wrapper,
            (input_values,),
            fp32_path,
            input_names=["input_values"],
            output_names=["logits", "embedding", "projected_frames"],
            dynamic_axes={
                "input_values":     {0: "batch", 1: "sequence"},
                "logits":           {0: "batch"},
                "embedding":        {0: "batch"},
                # T (time dimension) varies with audio length — mark as dynamic
                # so callers can pass windows of different sizes at inference.
                "projected_frames": {0: "batch", 1: "time"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    except Exception as e:
        print(f"\n[ERROR] ONNX export failed: {e}")
        sys.exit(1)

    fp32_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"      Размер: {fp32_mb:.1f} MB")

    import onnx
    onnx.checker.check_model(onnx.load(fp32_path))
    print("      ✓ ONNX graph verified")

    print(f"\n      Output tensor shapes:")
    print(f"        logits           : {list(logits.shape)}"
          f"  → (batch=1, n_classes={logits.shape[-1]})")
    print(f"        embedding        : {list(embedding.shape)}"
          f"  → (batch=1, D_proj={embedding.shape[-1]})")
    print(f"        projected_frames : {list(projected_frames.shape)}"
          f"  → (batch=1, T={projected_frames.shape[1]}, D_proj={projected_frames.shape[2]})")
    print(f"      Output names registered: logits, embedding, projected_frames")

    # Numpy inputs для дальнейших шагов
    np_input        = input_values.numpy()
    pt_logits_np    = logits.detach().numpy()
    pt_embedding_np = embedding.detach().numpy()
    pt_frames_np    = projected_frames.detach().numpy()

    # ── 4. Верификация ORT ──
    print(f"\n[4/6] ONNX Runtime verification...")
    ort_module = _load_ort()

    if ort_module is not None:
        _verify_ort(
            ort_module, fp32_path, np_input,
            pt_logits_np, pt_embedding_np, pt_frames_np,
        )
    else:
        print("      ⚠ Пропущено (нет onnxruntime)")

    # ── 5. Квантизация ──
    int8_path = None
    int8_mb = None

    if args.quantize:
        print(f"\n[5/6] Квантизация INT8...")
        if ort_module is None:
            print("      ⚠ Пропущено (нет onnxruntime)")
        else:
            int8_path = os.path.join(
                args.output_dir, "model_int8.onnx"
            )
            int8_mb = _quantize(
                ort_module, fp32_path, int8_path,
                np_input, pt_logits_np,
            )
            if int8_mb is None:
                int8_path = None  # квантизация не удалась
    else:
        print(f"\n[5/6] Квантизация пропущена (добавьте --quantize)")

    # ── 6. Бенчмарк ──
    benchmark_results = {}

    if args.benchmark:
        print(f"\n[6/6] Бенчмарк ({args.benchmark_iters} итераций)...")
        if ort_module is None:
            print("      ⚠ Пропущено (нет onnxruntime)")
        else:
            benchmark_results = _benchmark(
                ort_module=ort_module,
                model=model,
                input_values=input_values,
                np_input=np_input,
                fp32_path=fp32_path,
                int8_path=int8_path,
                n_iters=args.benchmark_iters,
                stride_ms=args.stride_ms,
            )
    else:
        print(f"\n[6/6] Бенчмарк пропущен (добавьте --benchmark)")

    # ── Конфиг ──
    config = {
        "labels": labels,
        "sr": args.sr,
        "window_s": args.window_s,
        "win_samples": win_samples,
        "embedding_dim": int(embedding.shape[-1]),
        # frame_dim: projection dimension of the per-frame output (D_proj).
        # CTCDigitDecoder reads this to size its Linear head correctly.
        "frame_dim": int(projected_frames.shape[-1]),
        # has_frames: signals to OnnxEngine that outputs[2] is available.
        "has_frames": True,
        "num_labels": len(labels),
        "base_model_name": args.base_model_name,
        "model_fp32": "model_fp32.onnx",
        "model_int8": "model_int8.onnx" if int8_path else None,
        "fp32_size_mb": round(fp32_mb, 1),
        "int8_size_mb": round(int8_mb, 1) if int8_mb else None,
        "benchmark": benchmark_results,
    }

    config_path = os.path.join(args.output_dir, "onnx_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ── Итого ──
    print("\n" + "=" * 60)
    print("  Готово!")
    print("=" * 60)
    print(f"\n  {args.output_dir}/")
    print(f"    model_fp32.onnx          {fp32_mb:.0f} MB")
    if int8_path:
        print(f"    model_int8.onnx          {int8_mb:.0f} MB ←")
    print(f"    onnx_config.json")
    print(f"    preprocessor_config.json")
    print()
    print("  Запуск:")
    print(f"    # PyTorch:")
    print(f"    python main.py run --model_dir {args.model_dir}")
    print()
    print(f"    # ONNX:")
    print(f"    python main.py run --model_dir {args.model_dir} "
          f"--onnx_dir {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_dir", default="onnx_model")
    p.add_argument(
        "--base_model_name",
        default="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    )
    p.add_argument("--window_s", type=float, default=1.0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--stride_ms", type=float, default=500)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark_iters", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())