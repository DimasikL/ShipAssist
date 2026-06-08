"""
core/realtime_recognizer_2.py

Два бэкенда через один класс:
  - PyTorch: RealTimeRecognizer(model_dir="best_model/")
  - ONNX:    RealTimeRecognizer(model_dir="best_model/", onnx_dir="onnx_model/")

Архитектура (для обоих):
  - audio callback → ring buffer (мгновенно)
  - inference thread → берёт последнее окно (отдельный поток)
"""

import os
import sys
import time
import threading
from typing import List, Dict, Optional, Callable

import numpy as np
import sounddevice as sd

# PyTorch — всегда доступен (нужен для PyTorch-бэкенда)
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

# ONNX — опционально
try:
    from core.onnx_engine import OnnxEngine, HAS_ORT
except ImportError:
    HAS_ORT = False
    OnnxEngine = None

# Outlier detection — опционально
try:
    from scripts.utils.outlier_detection import OutlierDetector, EmbeddingExtractor
    HAS_OUTLIER = True
except ImportError:
    HAS_OUTLIER = False


# ═══════════════════════════════════════════════════════════
#  Model loading helper
# ═══════════════════════════════════════════════════════════

def _load_wav2vec2_model(
    model_dir: str,
) -> Wav2Vec2ForSequenceClassification:
    """Loads a Wav2Vec2ForSequenceClassification from a local directory.

    Newer Transformers versions call load_adapter() when they find
    adapter_model.safetensors in the directory, which fails for PEFT LoRA
    checkpoints that lack ``adapter_attn_dim`` in their config.  This helper
    bypasses that path by building the model from its config and loading the
    merged state dict (model.safetensors / pytorch_model.bin) directly.

    Args:
        model_dir: Path to the model directory (merged LoRA or plain checkpoint).

    Returns:
        Wav2Vec2ForSequenceClassification in eval mode with weights loaded.

    Raises:
        FileNotFoundError: If no weight file is found in model_dir.
    """
    import pathlib
    from transformers import Wav2Vec2Config

    mdir = pathlib.Path(model_dir)
    config = Wav2Vec2Config.from_pretrained(str(mdir))
    model = Wav2Vec2ForSequenceClassification(config)

    safetensors_path = mdir / "model.safetensors"
    bin_path = mdir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No weight file found in {mdir}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[REC] [WARN] Missing keys ({len(missing)}): {missing[:3]} …")
    if unexpected:
        print(f"[REC] [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")

    return model


# ═══════════════════════════════════════════════════════════
#  Ring Buffer
# ═══════════════════════════════════════════════════════════

class RingBuffer:
    """Потокобезопасный кольцевой буфер float32."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._pos = 0
        self._total = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        with self._lock:
            if n >= self._cap:
                self._buf[:] = data[-self._cap:]
                self._pos = 0
            else:
                end = self._pos + n
                if end <= self._cap:
                    self._buf[self._pos:end] = data
                    self._pos = end % self._cap
                else:
                    first = self._cap - self._pos
                    self._buf[self._pos:] = data[:first]
                    self._buf[:n - first] = data[first:]
                    self._pos = n - first
            self._total += n

    def read_last(self, n: int) -> Optional[np.ndarray]:
        with self._lock:
            if self._total < n:
                return None
            end = self._pos
            if end >= n:
                return self._buf[end - n:end].copy()
            else:
                first = n - end
                return np.concatenate([
                    self._buf[self._cap - first:],
                    self._buf[:end]
                ]).copy()

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total


# ═══════════════════════════════════════════════════════════
#  RealTimeRecognizer
# ═══════════════════════════════════════════════════════════

class RealTimeRecognizer:
    """
    Примеры создания:

        # PyTorch (как раньше)
        rec = RealTimeRecognizer(model_dir="best_model/", labels=labels, ...)

        # ONNX (быстрее на CPU в 2-4×)
        rec = RealTimeRecognizer(model_dir="best_model/", labels=labels,
                                  onnx_dir="onnx_model/", ...)

    Всё остальное (start_stream, callback, ...) — одинаково.
    """

    def __init__(
            self,
            model_dir: str,
            labels: Optional[List[str]] = None,
            sr: int = 16000,
            window_s: float = 1.0,
            stride_s: float = 0.5,
            energy_th: float = 1e-4,
            conf_th: float = 0.6,
            conf_th_per_label: Optional[Dict[str, float]] = None,
            debounce_s: float = 1.0,
            device: Optional[str] = None,
            outlier_detector: Optional[str] = None,
            buffer_duration_s: float = 10.0,
            sd_blocksize: int = 8000,
            warmup_s: float = 1.5,
            report_other: bool = False,
            base_model_name: str = (
                    "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
            ),
            # ─── ONNX параметры (None = PyTorch) ───
            onnx_dir: Optional[str] = None,
            onnx_use_int8: bool = True,
            onnx_num_threads: Optional[int] = None,
    ):
        # Общие параметры
        self.sr = sr
        self.window_s = window_s
        self.stride_s = stride_s
        self.win_samples = int(window_s * sr)
        self.stride_samples = int(stride_s * sr)
        self.energy_th = energy_th
        self.base_conf_th = conf_th
        self.conf_th_per_label = conf_th_per_label or {}
        self.debounce_s = debounce_s
        self.report_other = report_other
        self.warmup_samples = int(warmup_s * sr)
        self.buf_capacity = int(buffer_duration_s * sr)
        self.sd_blocksize = sd_blocksize

        # ════════════════════════════════════════════
        #  Инициализация бэкенда
        # ════════════════════════════════════════════

        self._use_onnx = False
        self._onnx_engine: Optional[OnnxEngine] = None
        self._outlier_detector: Optional[OutlierDetector] = None
        self._emb_extractor = None  # только для PyTorch

        # Пытаемся ONNX
        if onnx_dir is not None:
            if not os.path.isdir(onnx_dir):
                print(f"[REC] ⚠ onnx_dir не найден: {onnx_dir}")
                print(f"[REC]   Сначала: python export_to_onnx.py "
                      f"--model_dir {model_dir} --output_dir {onnx_dir} "
                      f"--quantize --benchmark")
                print(f"[REC]   Используем PyTorch бэкенд.")
            elif not HAS_ORT:
                print("[REC] ⚠ onnxruntime не установлен!")
                print("[REC]   pip install onnxruntime")
                print("[REC]   Используем PyTorch бэкенд.")
            else:
                self._init_onnx(
                    onnx_dir, onnx_use_int8, onnx_num_threads,
                    outlier_detector
                )

        # Fallback: PyTorch
        if not self._use_onnx:
            self._init_pytorch(
                model_dir, base_model_name, labels,
                device, outlier_detector
            )

        # Лог
        backend = "ONNX Runtime" if self._use_onnx else "PyTorch"
        print(f"\n[REC] ══════════════════════════════════════")
        print(f"[REC] Backend:   {backend}")
        print(f"[REC] Labels:    {self.labels}")
        print(f"[REC] Window:    {window_s}s ({self.win_samples} samples)")
        print(f"[REC] Stride:    {stride_s}s")
        print(f"[REC] Debounce:  {debounce_s}s")
        print(f"[REC] Energy th: {energy_th:.2e}")
        print(f"[REC] Conf th:   {conf_th:.2f}")
        if self.conf_th_per_label:
            for lbl, th in self.conf_th_per_label.items():
                print(f"[REC]   {lbl}: {th:.3f}")
        print(f"[REC] Outlier:   "
              f"{'ON' if self._outlier_detector else 'OFF'}")
        print(f"[REC] ══════════════════════════════════════\n")

        # Runtime state
        self._ring: Optional[RingBuffer] = None
        self._stop_event = threading.Event()
        self._callback: Optional[Callable] = None
        self._overflow_count = 0

    # ─────────── ONNX init ───────────

    def _init_onnx(self, onnx_dir, use_int8, num_threads,
                   outlier_detector):
        print(f"[REC] Инициализация ONNX бэкенда: {onnx_dir}")
        try:
            # OnnxEngine uses `precision` ("int8"/"fp32"/"fp16") and
            # `providers` instead of the old use_int8/num_threads/verbose API.
            precision = "int8" if use_int8 else "fp32"
            providers = ["CPUExecutionProvider"]
            if num_threads is not None:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = num_threads
                sess_opts.inter_op_num_threads = num_threads
            else:
                sess_opts = None

            self._onnx_engine = OnnxEngine(
                onnx_dir=onnx_dir,
                precision=precision,
                providers=providers,
            )
            self._use_onnx = True
            self.labels = self._onnx_engine.labels
            # Sync window size from ONNX config (overrides constructor arg).
            # The exported model was fixed at export time — mismatching here
            # silently produces wrong-length inputs.
            if self._onnx_engine.win_samples != self.win_samples:
                import warnings
                warnings.warn(
                    f"[REC] window_s mismatch: constructor gave "
                    f"{self.win_samples} samples but ONNX model expects "
                    f"{self._onnx_engine.win_samples}. "
                    f"Overriding to match the model.",
                    stacklevel=3,
                )
                self.win_samples = self._onnx_engine.win_samples
                self.window_s = self.win_samples / self._onnx_engine.sr

            # Outlier detector (embedding из ONNX)
            if (outlier_detector and HAS_OUTLIER
                    and os.path.exists(outlier_detector)):
                self._outlier_detector = OutlierDetector.load(
                    outlier_detector
                )
                print(f"[REC] Outlier detector: {outlier_detector}")

        except Exception as e:
            print(f"[REC] ⚠ ONNX init failed: {e}")
            print(f"[REC]   Используем PyTorch бэкенд.")
            self._use_onnx = False

    # ─────────── PyTorch init ───────────

    def _init_pytorch(self, model_dir, base_model_name, labels,
                      device, outlier_detector):
        self.device_str = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(self.device_str)
        print(f"[REC] Инициализация PyTorch бэкенда: {self.device_str}")

        self.feature_extractor = (
            Wav2Vec2FeatureExtractor.from_pretrained(base_model_name)
        )
        self.model = _load_wav2vec2_model(model_dir)
        self.model.to(self.device).eval()

        # Labels из модели
        id2label = self.model.config.id2label
        if id2label:
            keys = list(id2label.keys())
            if isinstance(keys[0], int):
                self.labels = [id2label[i]
                               for i in range(len(id2label))]
            else:
                self.labels = [id2label[str(i)]
                               for i in range(len(id2label))]
        else:
            self.labels = labels or [
                f"class_{i}"
                for i in range(self.model.config.num_labels)
            ]

        # Outlier detector (PyTorch embedding)
        if (outlier_detector and HAS_OUTLIER
                and os.path.exists(outlier_detector)):
            self._outlier_detector = OutlierDetector.load(outlier_detector)
            self._emb_extractor = EmbeddingExtractor(
                model=self.model,
                feature_extractor=self.feature_extractor,
                device=self.device,
                config=self._outlier_detector.config,
            )
            print(f"[REC] Outlier detector: {outlier_detector}")

        # Warmup
        print("[REC] PyTorch warmup...", end=" ", flush=True)
        t0 = time.monotonic()
        self._run_inference(
            np.zeros(self.win_samples, dtype=np.float32)
        )
        warmup_ms = (time.monotonic() - t0) * 1000
        print(f"{warmup_ms:.0f}ms")

        if warmup_ms > self.stride_s * 1000:
            print(f"[REC] ⚠ Inference ({warmup_ms:.0f}ms) > "
                  f"stride ({self.stride_s * 1000:.0f}ms)")
            print(f"[REC]   Рекомендация: используйте --onnx_dir "
                  f"для ускорения в 2-4×")

    # ═══════════════════════════════════════════════════════
    #  Unified inference (автовыбор бэкенда)
    # ═══════════════════════════════════════════════════════

    def _run_inference(self, audio: np.ndarray):
        """
        Returns: (probs, is_outlier, outlier_info)
        Автоматически ONNX или PyTorch.
        """
        if self._use_onnx:
            return self._infer_onnx(audio)
        else:
            return self._infer_pytorch(audio)

    def _infer_onnx(self, audio):
        probs, embedding = self._onnx_engine.predict(audio)

        is_outlier = False
        outlier_info = None
        if self._outlier_detector and embedding is not None:
            try:
                outlier_info = (
                    self._outlier_detector.score_with_details(embedding)
                )
                is_outlier = outlier_info.get("is_outlier", False)
            except Exception as e:
                print(f"[REC] OOD error: {e}", file=sys.stderr)

        return probs, is_outlier, outlier_info

    @torch.inference_mode()
    def _infer_pytorch(self, audio):
        inputs = self.feature_extractor(
            audio, sampling_rate=self.sr,
            return_tensors="pt", padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        outputs = self.model(input_values=input_values)
        probs = torch.softmax(
            outputs.logits[0], dim=-1
        ).cpu().numpy()

        is_outlier = False
        outlier_info = None
        if self._outlier_detector and self._emb_extractor:
            try:
                emb = self._emb_extractor._captured_embedding
                emb = self._emb_extractor._pool_embedding(
                    emb, attention_mask=None
                )
                emb_np = emb.squeeze(0).cpu().numpy()
                outlier_info = (
                    self._outlier_detector.score_with_details(emb_np)
                )
                is_outlier = outlier_info.get("is_outlier", False)
            except Exception as e:
                print(f"[REC] OOD error: {e}", file=sys.stderr)

        return probs, is_outlier, outlier_info

    # ═══════════════════════════════════════════════════════
    #  Audio callback + Inference loop (без изменений)
    # ═══════════════════════════════════════════════════════

    def _audio_cb(self, indata, frames, time_info, status):
        if status and status.input_overflow:
            self._overflow_count += 1
        self._ring.write(indata[:, 0])

    def _inference_loop(self):
        last_processed_pos = 0
        last_det_label = None
        last_det_time = 0.0
        start_time = time.time()

        n_processed = 0
        n_detected = 0
        inf_times = []

        # Ждём заполнения
        while not self._stop_event.is_set():
            if self._ring.total_written >= max(
                    self.win_samples, self.warmup_samples
            ):
                break
            time.sleep(0.05)

        if self._stop_event.is_set():
            return

        backend = "ONNX" if self._use_onnx else "PyTorch"
        print(f"[REC] Inference loop started ({backend})")

        while not self._stop_event.is_set():
            current_pos = self._ring.total_written
            new_samples = current_pos - last_processed_pos

            if new_samples < self.stride_samples:
                remaining = self.stride_samples - new_samples
                time.sleep(max(0.01, remaining / self.sr * 0.8))
                continue

            # Берём ПОСЛЕДНЕЕ окно
            window = self._ring.read_last(self.win_samples)
            last_processed_pos = self._ring.total_written

            if window is None:
                time.sleep(0.05)
                continue

            # Energy gate
            energy = float(np.mean(window ** 2))
            if energy < self.energy_th:
                print(f"[DBG] energy={energy:.2e} < th={self.energy_th:.2e} SKIP")
                continue

            # Inference
            t0 = time.monotonic()
            probs, is_outlier, _ = self._run_inference(window)
            inf_ms = (time.monotonic() - t0) * 1000
            inf_times.append(inf_ms)
            n_processed += 1

            best_idx = int(np.argmax(probs))
            best_label = self.labels[best_idx]
            best_prob = float(probs[best_idx])
            print(f"[DBG] energy={energy:.2e} | {best_label}={best_prob:.3f} "
                  f"| outlier={is_outlier} | {inf_ms:.0f}ms")

            if is_outlier or probs is None:
                continue

            # Фильтр "другие слова"
            if best_label == "другие слова" and not self.report_other:
                continue

            # Порог
            th = self.conf_th_per_label.get(
                best_label, self.base_conf_th
            )
            if best_prob < th:
                continue

            # Debounce
            now = time.time()
            if (best_label == last_det_label
                    and (now - last_det_time) < self.debounce_s):
                continue

            # ── Детекция ──
            last_det_label = best_label
            last_det_time = now
            n_detected += 1
            t_rel = now - start_time

            print(
                f"[DETECTED] {best_label} (prob={best_prob:.3f}) "
                f"t={t_rel:.1f}s [{inf_ms:.0f}ms {backend}]"
            )

            if self._callback:
                try:
                    self._callback({
                        "label": best_label,
                        "prob": best_prob,
                        "time": now,
                        "time_relative": t_rel,
                        "inference_ms": inf_ms,
                        "backend": backend,
                    })
                except Exception as e:
                    print(f"[REC] Callback error: {e}",
                          file=sys.stderr)

        # Статистика
        if inf_times:
            avg_ms = np.mean(inf_times)
            min_ms = np.min(inf_times)
            max_ms = np.max(inf_times)
            print(f"\n[REC] ═══ Session Stats ═══")
            print(f"  Backend:    {backend}")
            print(f"  Processed:  {n_processed}")
            print(f"  Detected:   {n_detected}")
            print(f"  Inference:  avg={avg_ms:.0f}ms  "
                  f"min={min_ms:.0f}ms  max={max_ms:.0f}ms")
            print(f"  Stride:     {self.stride_s * 1000:.0f}ms")
            print(f"  Overflows:  {self._overflow_count}")
            if avg_ms > self.stride_s * 1000:
                ratio = avg_ms / (self.stride_s * 1000)
                print(f"  ⚠ Inference {ratio:.1f}× slower than stride")
                if not self._use_onnx:
                    print(f"  💡 Попробуйте --onnx_dir для "
                          f"ускорения в 2-4×")

    # ═══════════════════════════════════════════════════════
    #  Публичный API
    # ═══════════════════════════════════════════════════════

    def start_stream(
            self,
            callback_on_detection: Optional[Callable[[dict], None]] = None,
            device: Optional[int] = None,
    ):
        self._callback = callback_on_detection
        self._stop_event.clear()
        self._overflow_count = 0
        self._ring = RingBuffer(self.buf_capacity)

        inf_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="inference"
        )
        inf_thread.start()

        try:
            with sd.InputStream(
                    device=device,
                    samplerate=self.sr,
                    channels=1,
                    dtype="float32",
                    blocksize=self.sd_blocksize,
                    callback=self._audio_cb,
            ):
                print("[REC] ▶ Listening... Ctrl+C to stop\n")
                while not self._stop_event.is_set():
                    time.sleep(1.0)
                    if self._overflow_count > 0:
                        print(f"[REC] overflow ×{self._overflow_count}")
                        self._overflow_count = 0
        except KeyboardInterrupt:
            print("\n[REC] Stopped by user")
        finally:
            self._stop_event.set()
            inf_thread.join(timeout=10)

    def stop(self):
        self._stop_event.set()