"""
core/hybrid/number_regressor.py — Numeric slot-filling regressor wrapper.

Architecture role
-----------------
Stage 3 (optional) in the hybrid pipeline. When the centroid search identifies
an intent that contains an open numeric slot (e.g. ``"курс УГОЛ градусов"``
or ``"скорость УГОЛ узлов"``), this module predicts the numeric value from the
same Wav2Vec2 embedding vector.

How it works
------------
Wav2Vec2 encodes phonetic and prosodic variation correlated with numeric
magnitude in Russian speech (duration, vowel quality, intonation contour of
number words). A lightweight MLP regressor (``TorchClfBase`` in regression
mode) trained on ~60–100 samples covering the target range can resolve
spoken numbers with sufficient accuracy for maritime commands.

This class is a thin wrapper around ``core.model_triplet.TorchClfBase``
(``problem_mode='reg'``) that adds:
  - joblib-based save/load (consistent with the rest of the project).
  - Optional value clipping to ``[min_val, max_val]``.
  - Graceful no-op fallback when no model file is found.

Training data guidance
----------------------
  - Aim for ~60–100 samples spanning the target range *uniformly*.
  - Avoid clustering: if the range is [1, 360], record values at 1, 30, 45, 60,
    90, 120, 135, 180, 270, 315, 360 rather than only common headings.
  - Use TTS augmentation (``scripts/generation/main_audio_generate_tts.py``)
    to fill sparse regions cheaply.
  - Split 80/20 train/val; validate with MAE (mean absolute error in degrees).

Persistence
-----------
    # Training (handled by scripts/hybrid/train_regressor.py):
    reg = NumberRegressor(min_val=1.0, max_val=360.0)
    reg.fit(embeddings_train, values_train, embeddings_val, values_val)
    reg.save("artifacts/hybrid/regressors/курс_УГОЛ_градусов.pkl")

    # Inference:
    reg = NumberRegressor.load("artifacts/hybrid/regressors/курс_УГОЛ_градусов.pkl")
    predicted_value = reg.predict(embedding)   # e.g. 245.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class NumberRegressor:
    """Thin wrapper for ``TorchClfBase`` (regression mode) with clip + persist.

    Args:
        min_val:     Lower bound for predicted value clipping.
        max_val:     Upper bound for predicted value clipping.
        hidden_neurons: Hidden layer size for the MLP (forwarded to TorchClfBase).
        hidden_layers:  Number of hidden layers (0 = linear regressor).
        epochs:      Training epochs.
        lr:          Learning rate.
        device:      ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 360.0,
        hidden_neurons: int = 128,
        hidden_layers: int = 2,
        epochs: int = 200,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.min_val: float = float(min_val)
        self.max_val: float = float(max_val)
        self.hidden_neurons: int = hidden_neurons
        self.hidden_layers: int = hidden_layers
        self.epochs: int = epochs
        self.lr: float = lr
        self.device: str = device

        self._model = None          # TorchClfBase, set after fit() or load()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings_train: np.ndarray,
        values_train: np.ndarray,
        embeddings_val: np.ndarray,
        values_val: np.ndarray,
    ) -> "NumberRegressor":
        """Train the regressor on the given data split.

        Args:
            embeddings_train: Float32 array of shape ``(N_train, D)``.
            values_train:     Float32 array of shape ``(N_train,)`` — numeric targets.
            embeddings_val:   Float32 array of shape ``(N_val, D)``.
            values_val:       Float32 array of shape ``(N_val,)`` — numeric targets.

        Returns:
            Self (enables method chaining).

        Raises:
            ImportError: If PyTorch or the core.model_triplet module is unavailable.
        """
        from core.model_triplet import LinearOrMlpModel  # local import to keep ONNX path light

        X_train = np.asarray(embeddings_train, dtype=np.float32)
        y_train = np.asarray(values_train, dtype=np.float32).flatten()
        X_val = np.asarray(embeddings_val, dtype=np.float32)
        y_val = np.asarray(values_val, dtype=np.float32).flatten()

        embedding_dim = X_train.shape[1]

        self._model = LinearOrMlpModel(
            x_val=X_val,
            y_val=y_val,
            embedding_dim=embedding_dim,
            n_out=1,
            device=self.device,
            epochs=self.epochs,
            batch_size=min(32, max(4, len(X_train) // 8)),
            lr=self.lr,
            norm_x=True,
            norm_rows=False,
            hidden_neurons=self.hidden_neurons,
            hidden_layers=self.hidden_layers,
            weight_decay=1e-4,
            save_best_val=True,
            problem_mode="reg",
            dropout_rate=0.1,
            verbose=True,
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True

        # Quick validation MAE for logging
        preds = self._raw_predict(X_val)
        mae = float(np.abs(preds - y_val).mean())
        logger.info(
            "NumberRegressor fitted: D=%d, n_train=%d, val_MAE=%.2f "
            "(range=[%.1f, %.1f])",
            embedding_dim, len(X_train), mae, self.min_val, self.max_val,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, embedding: np.ndarray) -> Optional[float]:
        """Predict the numeric slot value from *embedding*.

        Args:
            embedding: 1-D float32 embedding of shape ``(D,)``.

        Returns:
            Predicted numeric value clipped to ``[min_val, max_val]``,
            or ``None`` if the regressor is not fitted (graceful fallback).
        """
        if not self._is_fitted or self._model is None:
            logger.warning(
                "NumberRegressor.predict() called on unfitted model — returning None."
            )
            return None

        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        raw = float(self._raw_predict(emb)[0])
        clipped = float(np.clip(raw, self.min_val, self.max_val))
        logger.debug(
            "NumberRegressor: raw=%.2f → clipped=%.2f (range=[%.1f, %.1f])",
            raw, clipped, self.min_val, self.max_val,
        )
        return clipped

    def predict_with_confidence(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """Predict value and return a normalised confidence score.

        Confidence is computed as ``1 - |raw - clipped| / (max_val - min_val)``,
        which captures how far the raw prediction was from the clipping bounds.
        A value of 1.0 means the prediction was well within bounds.

        Args:
            embedding: 1-D float32 embedding of shape ``(D,)``.

        Returns:
            Tuple of ``(value, confidence)``.
        """
        if not self._is_fitted or self._model is None:
            return None, 0.0
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        raw = float(self._raw_predict(emb)[0])
        clipped = float(np.clip(raw, self.min_val, self.max_val))
        span = max(self.max_val - self.min_val, 1e-6)
        confidence = max(0.0, 1.0 - abs(raw - clipped) / span)
        return clipped, float(confidence)

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the model without clipping.

        Args:
            X: Float32 array of shape ``(N, D)``.

        Returns:
            Float32 array of shape ``(N,)`` with raw regression outputs.
        """
        assert self._model is not None
        result = self._model.predict(X)
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        return np.asarray(result, dtype=np.float32).flatten()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted regressor to *path* using joblib.

        Args:
            path: Destination ``.pkl`` file path.

        Raises:
            RuntimeError: If the regressor has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted NumberRegressor.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info("NumberRegressor saved to %s", p)

    @staticmethod
    def load(path: str | Path) -> "NumberRegressor":
        """Load a fitted regressor from *path*.

        Args:
            path: Path to a ``.pkl`` file produced by ``NumberRegressor.save()``.

        Returns:
            Fitted ``NumberRegressor`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"NumberRegressor file not found: {p}")
        reg: NumberRegressor = joblib.load(p)
        logger.info(
            "NumberRegressor loaded from %s (range=[%.1f, %.1f])",
            p, reg.min_val, reg.max_val,
        )
        return reg
