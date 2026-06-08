"""
core/onnx_engine.py — Low-level ONNX Runtime wrapper for ShipAssistant.

This module owns the actual ``InferenceSession``. Higher-level concerns
(temperature scaling, adaptive thresholds, drift detection) live in
``core/engine.py`` and operate on the *raw logits* returned here.

Preprocessing parity
--------------------
All audio normalisation/padding goes through ``core.audio_utils.prepare_window``
to guarantee the input tensor is byte-identical to the one used during
PyTorch training (modulo float rounding). That is the most common cause of
"PT works, ONNX doesn't" regressions, so the contract is enforced here.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.audio_utils import prepare_window
from core.exceptions import ModelLoadError
from core.logger import get_logger

logger = get_logger(__name__)

# Exported so callers can do: from core.onnx_engine import OnnxEngine, HAS_ORT
try:
    import onnxruntime as _ort_probe  # noqa: F401
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ── Precision resolution ──────────────────────────────────────────────────────

_PRECISION_TO_KEY: Dict[str, str] = {
    "int8": "model_int8",
    "fp32": "model_fp32",
    "fp16": "model_fp16",
}


def _resolve_model_file(config: Dict[str, Any], precision: str) -> str:
    """Pick the ONNX weight file matching *precision* with safe fallback.

    Falls back to FP32 if the requested precision is missing in the
    bundle's ``onnx_config.json`` — and logs a warning so the operator
    knows quantisation was silently skipped.
    """
    key = _PRECISION_TO_KEY.get(precision.lower(), "model_int8")
    requested = config.get(key)
    if requested:
        return requested

    logger.warning(
        "ONNX-бандл не содержит precision=%s, переключаюсь на FP32. "
        "Перезапустите экспорт с --quantize или обновите onnx_config.json.",
        precision,
    )
    fallback = config.get("model_fp32")
    if not fallback:
        raise ModelLoadError(
            f"ONNX bundle is missing both '{key}' and 'model_fp32' entries"
        )
    return fallback


# ── Engine ────────────────────────────────────────────────────────────────────

class OnnxEngine:
    """Direct wrapper around ``onnxruntime.InferenceSession``.

    Returns *raw logits* (NOT softmax probabilities) so that callers can
    apply temperature scaling before the final softmax. This is critical
    for INT8-quantised models, where logit magnitudes shrink and
    confidences flatten without temperature correction.
    """

    def __init__(
        self,
        onnx_dir: str,
        precision: str = "int8",
        providers: Optional[List[str]] = None,
    ) -> None:
        """Initialise the ONNX session.

        Args:
            onnx_dir:  Directory containing ``onnx_config.json`` and the
                       weight file (model_int8.onnx / model_fp32.onnx).
            precision: One of ``"int8"``, ``"fp32"``, ``"fp16"``.
            providers: Execution providers in priority order. Defaults to
                       ``["CPUExecutionProvider"]`` when ``None``.
        """
        import onnxruntime as ort

        config_path = os.path.join(onnx_dir, "onnx_config.json")
        if not os.path.exists(config_path):
            raise ModelLoadError(f"Файл конфигурации ONNX не найден: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = json.load(f)

        self.labels: List[str] = self.config["labels"]
        self.sr: int = int(self.config["sr"])
        self.win_samples: int = int(self.config["win_samples"])
        self.precision: str = precision.lower()
        # frame_dim is present in bundles exported after the Variant-B upgrade.
        # Older bundles lack the key; in that case frames output is unavailable.
        self.frame_dim: Optional[int] = self.config.get("frame_dim")
        self.has_frames: bool = bool(self.config.get("has_frames", False))

        model_file = _resolve_model_file(self.config, self.precision)
        model_path = os.path.join(onnx_dir, model_file)
        if not os.path.exists(model_path):
            raise ModelLoadError(f"Файл модели ONNX не найден: {model_path}")

        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=providers or ["CPUExecutionProvider"],
            )
            logger.info(
                "ONNX сессия создана: %s (precision=%s, win_samples=%d, sr=%d)",
                model_path, self.precision, self.win_samples, self.sr,
            )
        except Exception as exc:                                 # pragma: no cover
            raise ModelLoadError(f"Ошибка инициализации ONNX: {exc}") from exc

    # ------------------------------------------------------------------
    def predict_logits(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run inference and return **raw logits** (not softmax).

        Input audio is canonicalised through ``core.audio_utils`` so the
        normalisation matches the training-time ``Wav2Vec2FeatureExtractor``
        bit-for-bit (within float32 rounding).

        Args:
            audio_data: 1-D float32 array of samples at ``self.sr``.

        Returns:
            ``(logits, embedding, frames)`` where:

            * ``logits``    — 1-D float32 array of shape ``(N_labels,)``.
            * ``embedding`` — 1-D float32 pooled embedding ``(D_proj,)``, or
                              ``None`` when the exported graph has only one output.
            * ``frames``    — 2-D float32 per-frame tensor ``(T, D_proj)``, or
                              ``None`` for older bundles without the third output.
                              Used by ``CTCDigitDecoder`` for slot-fill (Variant B).

        Note:
            Callers that currently unpack two values still work because Python
            allows ``logits, embedding = engine.predict_logits(audio)`` only if
            the caller ignores the third element via ``logits, embedding, *_ =``
            or by using ``predict_logits`` with explicit indexing.  The
            ``predict()`` wrapper below is unchanged.
        """
        prepared = prepare_window(
            audio_data.astype(np.float32, copy=False),
            target_samples=self.win_samples,
            do_normalize=True,
        ).reshape(1, -1)

        outputs = self.session.run(None, {"input_values": prepared})
        logits: np.ndarray = outputs[0][0]
        embedding: Optional[np.ndarray] = outputs[1][0] if len(outputs) > 1 else None
        # outputs[2] = projected_frames (B, T, D_proj); present only in bundles
        # exported with the Variant-B ExportWrapper (has_frames=True in config).
        frames: Optional[np.ndarray] = (
            outputs[2][0].astype(np.float32) if len(outputs) > 2 else None
        )
        return logits.astype(np.float32, copy=False), embedding, frames

    # ------------------------------------------------------------------
    def predict(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Backwards-compatible API: return softmax probabilities + embedding.

        Use ``predict_logits()`` from new code so temperature scaling can
        be applied before the softmax.
        """
        logits, embedding, _frames = self.predict_logits(audio_data)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        return probs.astype(np.float32, copy=False), embedding
