"""
core/engine.py — Unified inference engine abstraction for ShipAssistant.

Provides:
  - AudioEngine: Abstract base class defining the inference contract.
  - OnnxAudioEngine: Production ONNX backend; wraps core.onnx_engine.OnnxEngine
    and adds temperature scaling, adaptive thresholding, and drift detection
    so INT8 quantisation never silently degrades the 4 target phrases.
  - TorchAudioEngine: Development/fallback backend for Wav2Vec2 checkpoints.
    Use for debugging or post-fine-tune validation only.
  - create_engine(): Factory keyed on cfg.model.type (override via ``mode``).

All callers (api.py, inference.py, demo_defense.py) depend ONLY on AudioEngine,
never on the concrete backend — making runtime swaps fully transparent.
"""

from __future__ import annotations

import collections
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional

import numpy as np

from core.exceptions import ModelLoadError
from core.logger import get_logger

if TYPE_CHECKING:
    from core.config import Settings

logger = get_logger(__name__)


# ── Math helpers ──────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logit vector."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32, copy=False)


def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by 1/T before softmax. T > 1 softens, T < 1 sharpens."""
    if temperature <= 0.0 or temperature == 1.0:
        return logits
    return (logits / float(temperature)).astype(np.float32, copy=False)


# ── Abstract contract ─────────────────────────────────────────────────────────

class AudioEngine(ABC):
    """Abstract base class for all inference backends.

    Every concrete engine must implement ``load()``, ``predict()``, and the
    ``labels`` property so callers are fully decoupled from the ML runtime.
    """

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load or re-initialise model weights from *model_path*."""

    @abstractmethod
    def predict(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Run inference on a single audio window.

        Returns a dict containing at minimum:
            label       (str)          — top-1 predicted command label
            confidence  (float)        — softmax probability of top-1 class
            probs       (np.ndarray)   — full softmax probability vector
            logits      (np.ndarray)   — raw logits (pre-softmax)
            latency_ms  (float)        — wall-clock inference time (ms)
        """

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """Ordered list of class labels as declared in the model config."""


# ── Adaptive threshold mixin ──────────────────────────────────────────────────

class _AdaptiveThresholdMixin:
    """Rolling-median confidence tracker.

    When the median confidence over the last ``window`` predictions drops
    under ``default_threshold * 0.8``, ``effective_threshold()`` returns
    a 15% relaxed threshold and a warning is logged once per regime
    transition. This protects the 4 target phrases from disappearing
    behind a too-strict cut-off after INT8 calibration drift.
    """

    def __init__(self, default_threshold: float, window: int, enabled: bool) -> None:
        self._default_threshold: float = float(default_threshold)
        self._window: int = max(1, int(window))
        self._enabled: bool = bool(enabled)
        self._history: Deque[float] = collections.deque(maxlen=self._window)
        self._relaxed: bool = False                                 # current regime

    def _record_confidence(self, confidence: float) -> None:
        if not self._enabled:
            return
        self._history.append(float(confidence))

    def effective_threshold(self, label_threshold: float) -> float:
        """Return the threshold to actually compare against for *label_threshold*.

        ``label_threshold`` is the per-label override (or default) coming
        from ``cfg.recognition.per_label_thresholds``.
        """
        if not self._enabled or len(self._history) < self._window:
            return label_threshold

        median = float(np.median(self._history))
        trigger = self._default_threshold * 0.8

        if median < trigger:
            if not self._relaxed:
                logger.warning(
                    "Адаптивный порог: median_confidence=%.3f < %.3f "
                    "(80%% от default_confidence=%.3f). "
                    "Временно снижаю порог на 15%%.",
                    median, trigger, self._default_threshold,
                )
                self._relaxed = True
            return label_threshold * 0.85

        if self._relaxed:
            logger.info(
                "Адаптивный порог: median_confidence=%.3f восстановилась, "
                "возвращаю штатный порог.", median,
            )
            self._relaxed = False
        return label_threshold


# ── ONNX backend ──────────────────────────────────────────────────────────────

class OnnxAudioEngine(AudioEngine, _AdaptiveThresholdMixin):
    """Production backend: delegates to core.onnx_engine.OnnxEngine.

    Adds three INT8-recovery layers on top of the raw ONNX session:

    1. **Temperature scaling** — divides logits by ``cfg.onnx.temperature``
       before softmax. Compensates for the logit "flattening" produced by
       dynamic INT8 quantisation of MatMul/Gemm layers.
    2. **Adaptive thresholding** — see ``_AdaptiveThresholdMixin``.
    3. **Drift detection** — at startup, logs a single
       ``"PT→ONNX drift detected"`` warning when configuration suggests
       calibration is needed (T != 1.0 or recalibrate flag set).

    Trade-off note: ONNX Runtime is chosen over native PyTorch for production
    because it exposes INT8 quantisation and graph-level optimisations that
    reduce CPU latency by ~2-3× with negligible accuracy delta when the
    above safeguards are configured.
    """

    def __init__(
        self,
        onnx_dir: str,
        precision: str = "int8",
        providers: Optional[List[str]] = None,
        temperature: float = 1.0,
        adaptive_threshold: bool = True,
        adaptive_window: int = 20,
        default_confidence: float = 0.8,
        recalibrate: bool = False,
    ) -> None:
        _AdaptiveThresholdMixin.__init__(
            self,
            default_threshold=default_confidence,
            window=adaptive_window,
            enabled=adaptive_threshold,
        )
        self._onnx_dir: str = onnx_dir
        self._precision: str = precision
        self._providers: Optional[List[str]] = providers
        self._temperature: float = float(temperature)
        self._recalibrate: bool = bool(recalibrate)
        self._inner: Optional[Any] = None                           # OnnxEngine
        self.load(onnx_dir)
        self._announce_drift()

    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        """Instantiate the underlying OnnxEngine session."""
        from core.onnx_engine import OnnxEngine as _OnnxEngine

        try:
            self._inner = _OnnxEngine(
                onnx_dir=model_path,
                precision=self._precision,
                providers=self._providers,
            )
            logger.info(
                "OnnxAudioEngine loaded — path=%r precision=%s temperature=%.3f "
                "adaptive_threshold=%s",
                model_path, self._precision, self._temperature,
                self._enabled,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"OnnxAudioEngine failed to load from {model_path!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def _announce_drift(self) -> None:
        """Emit a single 'PT→ONNX drift detected' warning when configured."""
        suspicious = (
            self._precision == "int8"
            and (self._temperature != 1.0 or self._recalibrate)
        )
        if suspicious:
            logger.warning(
                "PT→ONNX drift detected — precision=%s temperature=%.3f "
                "recalibrate=%s. Это сигнал, что INT8-калибровка может быть "
                "нерепрезентативна. Запустите scripts/debug_pt_vs_onnx.py "
                "для подтверждения.",
                self._precision, self._temperature, self._recalibrate,
            )
        if self._recalibrate:
            logger.warning(
                "onnx.recalibrate=true → сгенерируйте calib_data из 4 целевых фраз:\n"
                "  1. Соберите 30–50 .wav-файлов на каждую целевую фразу.\n"
                "  2. Положите их в artifacts/calib_data/<phrase>/*.wav.\n"
                "  3. Перезапустите scripts/train/main_export_to_onnx.py с\n"
                "     --quantize и custom CalibrationDataReader (см. RUNBOOK.md)."
            )

    # ------------------------------------------------------------------
    def predict(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Run inference, apply temperature scaling, return unified result dict.

        Latency is measured around the ONNX session.run() call only; audio
        I/O and pre-processing overhead measured by the caller are NOT
        included here.
        """
        t0 = time.perf_counter()
        logits, embedding, _frames = self._inner.predict_logits(waveform)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        scaled = _apply_temperature(logits, self._temperature)
        probs = _softmax(scaled)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        self._record_confidence(confidence)

        result: Dict[str, Any] = {
            "label": self._inner.labels[idx],
            "confidence": confidence,
            "probs": probs,
            "logits": logits,
            "latency_ms": latency_ms,
        }
        if embedding is not None:
            result["embedding"] = embedding

        logger.debug(
            "[OnnxAudioEngine] label=%r conf=%.3f latency=%.1fms T=%.2f",
            result["label"], confidence, latency_ms, self._temperature,
        )
        return result

    # ------------------------------------------------------------------
    @property
    def labels(self) -> List[str]:
        return self._inner.labels


# ── Torch backend ─────────────────────────────────────────────────────────────

class TorchAudioEngine(AudioEngine):
    """Development / fallback backend for Wav2Vec2ForSequenceClassification.

    Use when ``cfg.model.type == "torch"`` or ``--mode torch`` is passed.
    ~2-3× slower than OnnxAudioEngine on CPU; no INT8 support. Useful for
    post-fine-tune sanity-checks before ONNX export and for the diagnostic
    script ``scripts/debug_pt_vs_onnx.py``.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path: str = model_path
        self._model: Optional[Any] = None
        self._device: str = "cpu"
        self._id2label: Dict[int, str] = {}
        self.load(model_path)

    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        try:
            import torch
            from transformers import Wav2Vec2ForSequenceClassification

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
            model.eval().to(device)
            self._model = model
            self._id2label = {int(k): v for k, v in model.config.id2label.items()}
            logger.info(
                "TorchAudioEngine loaded — path=%r device=%s",
                model_path, device,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"TorchAudioEngine failed to load from {model_path!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def predict(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Forward pass through the Wav2Vec2 model with softmax output."""
        import torch

        from core.audio_utils import prepare_window
        from core.config import settings

        target_samples = int(
            settings.audio.window_seconds * settings.audio.sample_rate
        )
        prepared = prepare_window(
            waveform.astype(np.float32, copy=False),
            target_samples=target_samples,
            do_normalize=True,
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            tensor = torch.from_numpy(prepared).unsqueeze(0).to(self._device)
            logits = self._model(tensor).logits
            logits_np: np.ndarray = logits.cpu().numpy()[0]
            probs: np.ndarray = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        idx = int(np.argmax(probs))
        result: Dict[str, Any] = {
            "label": self._id2label[idx],
            "confidence": float(probs[idx]),
            "probs": probs,
            "logits": logits_np.astype(np.float32, copy=False),
            "latency_ms": latency_ms,
        }
        logger.debug(
            "[TorchAudioEngine] label=%r conf=%.3f latency=%.1fms",
            result["label"], result["confidence"], latency_ms,
        )
        return result

    # ------------------------------------------------------------------
    @property
    def labels(self) -> List[str]:
        return [self._id2label[i] for i in sorted(self._id2label)]


# ── Factory ───────────────────────────────────────────────────────────────────

def create_engine(cfg: "Settings", mode: Optional[str] = None) -> AudioEngine:
    """Instantiate and return the correct engine backend.

    Resolution order for engine type:
      1. ``mode`` argument (explicit CLI override, e.g. ``--mode torch``)
      2. ``cfg.model.type`` (from configs/model.yaml, default ``"onnx"``)

    All ONNX-related knobs (precision, temperature, adaptive threshold)
    are pulled from ``cfg.onnx`` — no values are hard-coded here.

    Raises:
        ModelLoadError: If the engine type string is unknown or model
                        files are missing / unreadable.
    """
    engine_type = (mode or cfg.model.type).lower()
    logger.info(f"create_engine — type={engine_type!r}")

    if engine_type == "onnx":
        # Backwards compatibility: respect cfg.onnx.use_int8 if precision is
        # left at default and use_int8 is False.
        precision = cfg.onnx.precision
        if precision == "int8" and not cfg.onnx.use_int8:
            precision = "fp32"

        return OnnxAudioEngine(
            onnx_dir=str(cfg.paths.onnx_model),
            precision=precision,
            providers=list(cfg.onnx.providers),
            temperature=cfg.onnx.temperature,
            adaptive_threshold=cfg.onnx.adaptive_threshold,
            adaptive_window=cfg.onnx.adaptive_window,
            default_confidence=cfg.recognition.default_confidence,
            recalibrate=cfg.onnx.recalibrate,
        )

    if engine_type == "torch":
        return TorchAudioEngine(model_path=str(cfg.paths.best_model))

    raise ModelLoadError(
        f"Unknown engine type: {engine_type!r}. "
        f"Supported values: 'onnx', 'torch'."
    )
