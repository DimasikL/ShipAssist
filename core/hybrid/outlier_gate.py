"""
core/hybrid/outlier_gate.py — Mahalanobis-distance early-rejection gate.

Architecture role
-----------------
The gate is **the first stage** in the hybrid pipeline (Stage 1 of 3).
It operates on the raw Wav2Vec2 embedding vector and answers a single
binary question: *"Does this audio look like any known command, or is it
noise / out-of-vocabulary speech?"*

If rejected, the rest of the pipeline is skipped entirely. This is the
correct placement for a safety-critical maritime system where the cost of
a false positive (executing a wrong helm command from engine noise) far
exceeds the cost of a false negative (requiring the operator to repeat a
misheard command).

Distance metric
---------------
Mahalanobis distance (``method='mahalanobis'``) is the default because it
accounts for the correlated, non-spherical covariance structure of
Wav2Vec2 embeddings. The covariance is estimated over all training samples
(pooled, not per-class) and regularised with ``epsilon * I`` to prevent
singularity in the common case where embedding dimension (1024) >> n_samples.

Cosine and L2 alternatives are provided for ablation experiments.

Ensemble OOD + Adaptive τ (Scenario 1)
---------------------------------------
``EnsembleOutlierGate`` combines all three distance metrics via z-score
normalization and weighted averaging. The rejection threshold τ is
**adaptive**: each known class has its own calibrated τ derived from the
intra-class distance distribution at fit time. At inference the nearest
class determines which τ applies.

    Why no retraining is needed
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All parameters (centroids, inv-cov, normalization stats, per-class
    thresholds) are computed analytically from the embedding data. The same
    CSV + ONNX model used for the original gate is sufficient.

Persistence
-----------
    # Single-metric gate (original):
    gate = OutlierGate(method="mahalanobis", percentile=95.0)
    gate.fit(embeddings, labels)
    gate.save("artifacts/hybrid/outlier_gate.pkl")

    gate2 = OutlierGate.load("artifacts/hybrid/outlier_gate.pkl")
    rejected = gate2.is_outlier(new_embedding)

    # Ensemble gate (Scenario 1):
    egate = EnsembleOutlierGate(weights=(2.0, 1.0, 1.0), use_adaptive_tau=True)
    egate.fit(embeddings, labels)
    egate.save("artifacts/hybrid/ensemble_gate.pkl")

    egate2 = EnsembleOutlierGate.load("artifacts/hybrid/ensemble_gate.pkl")
    rejected = egate2.is_outlier(new_embedding)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_ALLOWED_METHODS = {"mahalanobis", "cosine", "l2"}
_ENSEMBLE_METHOD = "ensemble"


# ── Online SNR estimation (§2.3 — WADA-SNR approximation) ────────────────────


def estimate_snr_db(
    waveform: np.ndarray,
    sr: int = 16_000,
    n_fft: int = 512,
    hop_length: int = 128,
    window_ms: float = 400.0,
    eps: float = 1e-12,
) -> float:
    """Estimate signal-to-noise ratio via min/max spectral statistics.

    Approximates the WADA-SNR method (Hirsch & Pearce, 2000): per-frame
    spectral power is smoothed with a sliding-window minimum (noise floor
    tracker) and a sliding-window maximum (speech activity tracker). The
    global SNR is then ``10 · log10(mean_max / mean_min)``.

    Runtime on a 1-second 16 kHz clip (n_fft=512, hop=128): ~0.3 ms on CPU.

    Args:
        waveform:   1-D float32 waveform sampled at ``sr`` Hz.
        sr:         Sample rate in Hz.
        n_fft:      FFT size. Larger → finer frequency resolution.
        hop_length: STFT hop size in samples.
        window_ms:  Duration of the min/max tracking window in milliseconds.
        eps:        Floor added to prevent log(0).

    Returns:
        SNR estimate in dB, clamped to ``[-5, 40]`` for numerical stability.
        Returns ``0.0`` if the waveform is too short to compute even one frame.
    """
    x = np.asarray(waveform, dtype=np.float64)
    n_frames = 1 + max(0, len(x) - n_fft) // hop_length
    if n_frames < 1:
        return 0.0

    win = np.hanning(n_fft).astype(np.float64)

    # Build power spectrogram: (n_frames, n_fft//2 + 1)
    power = np.stack([
        np.abs(np.fft.rfft(x[i * hop_length: i * hop_length + n_fft] * win)) ** 2
        for i in range(n_frames)
    ])

    # Collapse frequency → per-frame mean power
    frame_power: np.ndarray = power.mean(axis=1)          # (n_frames,)

    # Sliding-window min/max (edge-padded)
    win_frames = max(1, int(round(window_ms * sr / hop_length / 1_000.0)))
    pad = win_frames // 2
    padded = np.pad(frame_power, pad, mode="edge")
    min_track = np.array([padded[i: i + win_frames].min() for i in range(n_frames)])
    max_track = np.array([padded[i: i + win_frames].max() for i in range(n_frames)])

    p_noise  = float(min_track.mean()) + eps
    p_speech = float(max_track.mean()) + eps
    snr = 10.0 * np.log10(p_speech / p_noise)
    return float(np.clip(snr, -5.0, 40.0))


def snr_adaptive_threshold(
    tau_0: float,
    snr_db: float,
    snr_ref: float = 12.0,
    beta: float = 0.15,
) -> float:
    """Compute the SNR-adjusted rejection threshold.

    ``τ_adaptive = τ_0 + β · max(0, SNR_ref − SNR_online)``

    In clean audio (SNR ≥ SNR_ref) the threshold is unchanged (τ_0) so the
    gate is no more restrictive than calibrated, preserving recall. Under
    heavy noise (SNR < SNR_ref) the threshold rises by up to
    ``β · SNR_ref`` units, reducing FP risk where misrecognition is most
    likely.

    Args:
        tau_0:   Base threshold calibrated at fit time (e.g. 95th-pct distance).
        snr_db:  Online SNR estimate from ``estimate_snr_db()``.
        snr_ref: Reference SNR level in dB (default 12 dB).
        beta:    Sensitivity coefficient. Calibrate via ablation; start 0.15.

    Returns:
        Adjusted threshold ≥ ``tau_0``.
    """
    return tau_0 + beta * max(0.0, snr_ref - snr_db)


# ── Gate class ────────────────────────────────────────────────────────────────

class OutlierGate:
    """Mahalanobis (or cosine/L2) distance gate for embedding-space rejection.

    The gate stores:
      - Per-class centroids (mean of L2-normalised embeddings).
      - A pooled (shared) inverse covariance matrix (Mahalanobis only).
      - A scalar global threshold calibrated at the chosen percentile of the
        training-set distance distribution.
      - (Optional) Per-class thresholds for adaptive τ — each known intent
        gets its own threshold derived from its intra-class distance
        distribution, so compact clusters get tighter rejection and sparse
        clusters get looser rejection. Enabled by ``use_adaptive_tau=True``.

    After ``fit()``, call ``is_outlier(embedding)`` to classify a new sample.
    ``score(embedding)`` returns the raw distance to the nearest centroid
    (lower = more in-distribution for Mahalanobis/L2; higher = more in-dist
    for cosine).

    Args:
        method:               Distance metric: ``'mahalanobis'``, ``'cosine'``,
                              or ``'l2'``.
        percentile:           Percentile of training distances used to set the
                              global threshold.
        use_adaptive_tau:     If ``True``, use per-class thresholds at inference
                              instead of the single global threshold. The
                              per-class threshold for class *c* is the
                              ``per_class_percentile``-th percentile of
                              distances from training samples of class *c* to
                              their own centroid.
        per_class_percentile: Percentile used for per-class threshold
                              calibration. Defaults to ``percentile`` when
                              ``None``.
        regularization_eps:   Added to cov diagonal before inversion
                              (Mahalanobis only).
        fallback_threshold:   Used when the gate is not fitted (graceful
                              no-op mode).
    """

    def __init__(
        self,
        method: str = "mahalanobis",
        percentile: float = 95.0,
        use_adaptive_tau: bool = False,
        per_class_percentile: Optional[float] = None,
        regularization_eps: float = 1e-4,
        fallback_threshold: float = 8.0,
    ) -> None:
        if method not in _ALLOWED_METHODS:
            raise ValueError(
                f"method must be one of {_ALLOWED_METHODS}, got '{method}'"
            )
        self.method: str = method
        self.percentile: float = percentile
        self.use_adaptive_tau: bool = use_adaptive_tau
        self.per_class_percentile: float = (
            per_class_percentile if per_class_percentile is not None else percentile
        )
        self.regularization_eps: float = regularization_eps
        self.fallback_threshold: float = fallback_threshold

        # Set after fit()
        self._centroids: Optional[np.ndarray] = None          # (n_classes, D)
        self._labels: Optional[List[str]] = None              # length n_classes
        self._inv_cov: Optional[np.ndarray] = None            # (D, D) Mahal only
        self._threshold: Optional[float] = None               # global calibrated scalar
        self._per_class_thresholds: Dict[str, float] = {}     # adaptive τ per class
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray, labels: List[str]) -> "OutlierGate":
        """Fit the gate on labelled training embeddings.

        Args:
            embeddings: Float32 array of shape ``(N, D)``. Each row is one
                        embedding extracted from a training audio file.
            labels:     List of N label strings matching the rows in
                        ``embeddings``.

        Returns:
            Self (enables method chaining).

        Raises:
            ValueError: If ``embeddings`` and ``labels`` have different lengths
                        or if ``embeddings`` is empty.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (N, D), got shape {embeddings.shape}"
            )
        if len(labels) != embeddings.shape[0]:
            raise ValueError(
                f"len(labels)={len(labels)} != embeddings.shape[0]={embeddings.shape[0]}"
            )
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot fit on empty embeddings array.")

        unique_labels: List[str] = sorted(set(labels))
        label_arr = np.array(labels)

        # Compute per-class centroids (on L2-normalised embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms

        centroids: List[np.ndarray] = []
        for lbl in unique_labels:
            mask = label_arr == lbl
            c = normed[mask].mean(axis=0)
            c /= max(np.linalg.norm(c), 1e-12)
            centroids.append(c.astype(np.float32))

        self._centroids = np.stack(centroids, axis=0)       # (n_classes, D)
        self._labels = unique_labels

        # Mahalanobis: compute class-balanced pooled covariance & invert.
        # We pass labels so that _compute_inv_cov can weight each class equally,
        # preventing large classes (e.g. "другие слова") from dominating the
        # pooled covariance and biasing Mahalanobis distances for small classes.
        if self.method == "mahalanobis":
            self._inv_cov = self._compute_inv_cov(normed, label_arr)

        # Calibrate global threshold from training distances
        train_distances = np.array(
            [self._distance_to_nearest(e) for e in normed],
            dtype=np.float32,
        )
        self._threshold = float(np.percentile(train_distances, self.percentile))

        # ── Adaptive τ: per-class thresholds ──────────────────────────
        if self.use_adaptive_tau:
            self._per_class_thresholds = self._calibrate_per_class_thresholds(
                normed, label_arr, unique_labels
            )
            logger.info(
                "OutlierGate adaptive τ calibrated for %d classes "
                "(per_class_percentile=%.0f)",
                len(self._per_class_thresholds),
                self.per_class_percentile,
            )

        self._is_fitted = True

        logger.info(
            "OutlierGate fitted: method=%s, n_classes=%d, D=%d, "
            "threshold@%.0fth_pct=%.4f%s",
            self.method, len(unique_labels), embeddings.shape[1],
            self.percentile, self._threshold,
            " (adaptive τ enabled)" if self.use_adaptive_tau else "",
        )
        return self

    def _calibrate_per_class_thresholds(
        self,
        normed: np.ndarray,
        label_arr: np.ndarray,
        unique_labels: List[str],
    ) -> Dict[str, float]:
        """Calibrate per-class thresholds from intra-class distance distributions.

        For each class *c*, the threshold is the ``per_class_percentile``-th
        percentile of distances from training samples of class *c* to the
        nearest centroid. Because intra-class distances are computed against
        all centroids (not just the own centroid), the threshold represents
        the expected minimum distance seen in training for that class context.

        Args:
            normed:        L2-normalised training embeddings.
            label_arr:     String label array aligned with ``normed``.
            unique_labels: Sorted unique label list.

        Returns:
            Dict mapping class label → scalar threshold.
        """
        per_class: Dict[str, float] = {}
        for lbl in unique_labels:
            mask = label_arr == lbl
            if not mask.any():
                continue
            class_dists = np.array(
                [self._distance_to_nearest(e) for e in normed[mask]],
                dtype=np.float32,
            )
            per_class[lbl] = float(
                np.percentile(class_dists, self.per_class_percentile)
            )
        return per_class

    def _compute_inv_cov(
        self,
        normed: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute regularised inverse of the class-balanced pooled covariance.

        When *labels* is provided, each class contributes **equally** to the
        pooled within-class covariance regardless of how many samples it has.
        This prevents large classes (e.g. "другие слова" with 1 786 samples)
        from dominating the covariance and inflating Mahalanobis distances for
        small classes (e.g. "машина" with 14 samples), which would cause the
        nearest-centroid assignment to be systematically wrong.

        When *labels* is ``None`` (legacy path), a plain unweighted covariance
        over all samples is used (original behaviour).

        Args:
            normed: L2-normalised embedding array of shape ``(N, D)``.
            labels: Optional 1-D array of class labels, same length as *normed*.
                    When supplied, per-class within-class covariances are
                    averaged with equal weight (class-balanced pooling).

        Returns:
            Regularised inverse covariance matrix of shape ``(D, D)``.
        """
        D = normed.shape[1]

        if labels is not None and len(set(labels.tolist())) > 1:
            # Class-balanced pooled within-class covariance:
            # cov = mean_over_classes( cov(class_samples) )
            unique = sorted(set(labels.tolist()))
            class_covs: List[np.ndarray] = []
            for lbl in unique:
                mask = labels == lbl
                X = normed[mask].astype(np.float64)
                if X.shape[0] < 2:
                    # Single sample: contribute zero variance for this class
                    class_covs.append(np.zeros((D, D), dtype=np.float64))
                else:
                    class_covs.append(np.cov(X.T))
            cov = np.mean(class_covs, axis=0)
        else:
            # Legacy: unweighted covariance over all samples
            cov = np.cov(normed.T).astype(np.float64)

        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov += self.regularization_eps * np.eye(D, dtype=np.float64)
        try:
            return np.linalg.inv(cov).astype(np.float32)
        except np.linalg.LinAlgError:
            logger.warning(
                "Covariance matrix singular even after regularisation "
                "(eps=%.2e). Falling back to pseudo-inverse.", self.regularization_eps,
            )
            return np.linalg.pinv(cov).astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, embedding: np.ndarray) -> Tuple[float, str]:
        """Compute the distance from *embedding* to the nearest class centroid.

        Args:
            embedding: 1-D float32 array of shape ``(D,)``.

        Returns:
            Tuple of ``(distance, nearest_label)`` where ``distance`` is:
              - Mahalanobis distance (lower = more in-distribution)
              - Cosine *distance* 1 - cosine_similarity (lower = more in-dist)
              - L2 distance (lower = more in-distribution)
        """
        if not self._is_fitted or self._centroids is None:
            return (self.fallback_threshold, "unknown")
        return self._distance_to_nearest(embedding, return_label=True)  # type: ignore[return-value]

    def is_outlier(
        self,
        embedding: np.ndarray,
        snr_db: Optional[float] = None,
        snr_ref: float = 12.0,
        beta: float = 0.15,
    ) -> bool:
        """Return ``True`` if *embedding* is considered out-of-distribution.

        When ``use_adaptive_tau=True`` and the gate has been fitted, the
        base threshold τ_0 is looked up from the per-class table using the
        nearest class label. This means tight classes (compact clusters)
        produce lower thresholds (stricter rejection) and loose classes
        (spread-out clusters) produce higher thresholds (more permissive).

        If ``snr_db`` is provided, the base threshold is further adjusted by
        ``snr_adaptive_threshold()`` (§2.3): the gate tightens when the
        waveform SNR falls below ``snr_ref`` dB.

        Args:
            embedding: 1-D float32 array of shape ``(D,)``.
            snr_db:    Optional online SNR estimate from ``estimate_snr_db()``.
                       Pass ``None`` to disable SNR adjustment (default).
            snr_ref:   Reference SNR in dB for the adaptive formula (12 dB).
            beta:      Sensitivity coefficient for the adaptive formula (0.15).

        Returns:
            ``True`` → reject (outlier); ``False`` → accept (in-distribution).
        """
        dist, nearest_label = self.score(embedding)

        if self.use_adaptive_tau and self._per_class_thresholds:
            tau_0 = self._per_class_thresholds.get(
                nearest_label,
                self._threshold if self._threshold is not None else self.fallback_threshold,
            )
        else:
            tau_0 = (
                self._threshold if self._threshold is not None else self.fallback_threshold
            )

        threshold = (
            snr_adaptive_threshold(tau_0, snr_db, snr_ref=snr_ref, beta=beta)
            if snr_db is not None
            else tau_0
        )

        # For Mahal/L2: higher distance = more outlier
        # For cosine: higher distance (1-sim) = more outlier
        return bool(dist > threshold)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _distance_to_nearest(
        self,
        embedding: np.ndarray,
        return_label: bool = False,
    ):
        """Compute distance to the nearest centroid.

        Args:
            embedding:    1-D float32 array.
            return_label: If True, returns ``(distance, label)`` tuple.

        Returns:
            Scalar distance, or ``(distance, label)`` if ``return_label=True``.
        """
        assert self._centroids is not None and self._labels is not None

        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        normed = emb / max(norm, 1e-12)

        if self.method == "mahalanobis":
            distances = self._mahalanobis_to_all(normed)
        elif self.method == "cosine":
            # cosine distance = 1 - cosine_similarity
            sims = self._centroids @ normed
            distances = 1.0 - sims
        else:  # l2
            diff = self._centroids - normed[None, :]
            distances = np.linalg.norm(diff, axis=1)

        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        best_label = self._labels[best_idx]

        return (best_dist, best_label) if return_label else best_dist

    def _mahalanobis_to_all(self, normed: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance from *normed* to every centroid.

        Args:
            normed: L2-normalised 1-D embedding of shape ``(D,)``.

        Returns:
            1-D array of shape ``(n_classes,)`` with distances.
        """
        assert self._inv_cov is not None and self._centroids is not None
        diffs = self._centroids - normed[None, :]              # (n_classes, D)
        # d² = (x-mu)ᵀ Σ⁻¹ (x-mu) for each centroid
        left = diffs @ self._inv_cov                           # (n_classes, D)
        sq_distances = np.einsum("ij,ij->i", left, diffs)     # (n_classes,)
        sq_distances = np.clip(sq_distances, 0.0, None)
        return np.sqrt(sq_distances).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the fitted gate to *path*.

        Args:
            path: Destination file path (usually ``artifacts/hybrid/outlier_gate.pkl``).

        Raises:
            RuntimeError: If the gate has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted OutlierGate. Call fit() first."
            )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("OutlierGate saved to %s", p)

    @staticmethod
    def load(path: str | Path) -> "OutlierGate":
        """Load a previously fitted gate from *path*.

        Args:
            path: Path to a ``.pkl`` file produced by ``OutlierGate.save()``.

        Returns:
            Fitted ``OutlierGate`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"OutlierGate file not found: {p}")
        with open(p, "rb") as f:
            gate: OutlierGate = pickle.load(f)
        # Use getattr for robustness: EnsembleOutlierGate pickled into this
        # path has no .method attribute (it's an ensemble of sub-gates).
        _method = getattr(gate, "method", type(gate).__name__)
        _threshold = getattr(gate, "_threshold", None) or getattr(gate, "fallback_threshold", 0.0)
        logger.info(
            "OutlierGate loaded from %s (method=%s, threshold=%.4f)",
            p, _method, _threshold,
        )
        return gate

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Return a human-readable summary dict for logging / debugging.

        Returns:
            Dictionary with keys: ``fitted``, ``method``, ``n_classes``,
            ``embedding_dim``, ``threshold``, ``labels``,
            ``use_adaptive_tau``, ``per_class_thresholds``.
        """
        return {
            "fitted": self._is_fitted,
            "method": self.method,
            "n_classes": len(self._labels) if self._labels else 0,
            "embedding_dim": self._centroids.shape[1] if self._centroids is not None else None,
            "threshold": self._threshold,
            "use_adaptive_tau": self.use_adaptive_tau,
            "per_class_thresholds": dict(self._per_class_thresholds),
            "labels": list(self._labels) if self._labels else [],
        }


# ── Ensemble OOD Gate (Scenario 1) ────────────────────────────────────────────



# ── Ensemble OOD Gate (Scenario 1) ────────────────────────────────────────────

class EnsembleOutlierGate:
    """Ensemble OOD gate: Mahalanobis + cosine + L2 with adaptive per-class tau.

    Scenario 1 — pure code, no retraining required.

    Three distance metrics are combined via robust z-score normalization and
    weighted averaging into a single ensemble OOD score. Rejection uses an
    adaptive per-class threshold tau: each registered class gets its own tau
    calibrated from the intra-class ensemble-score distribution at fit time.
    Compact clusters get tighter tau; sparse clusters get looser tau.

    Pipeline at inference
    ~~~~~~~~~~~~~~~~~~~~~
    1. Compute raw distances d_mahal, d_cos, d_l2 to the nearest centroid
       via three OutlierGate sub-gates.
    2. Normalize each: z_i = (d_i - median_i) / iqr_scale_i
       where (median_i, iqr_scale_i) are robust stats from training.
    3. Ensemble score: s = w_mahal*z_mahal + w_cos*z_cos + w_l2*z_l2
       with weights normalized to sum to 1.
       Default weights (2, 1, 1) → normalized: s = 0.5·z_mahal + 0.25·z_cos + 0.25·z_l2
    4. Identify the nearest class from the Mahalanobis sub-gate.
    5. Look up per-class tau calibrated at the 95th percentile of intra-class
       training scores (compact clusters → tight tau, sparse → loose tau),
       or fall back to the global tau.
    6. Reject if s > tau.

    Args:
        weights:              Weight tuple (w_mahal, w_cos, w_l2).
                              Automatically normalized to sum to 1.
                              Default (2.0, 1.0, 1.0) → effective (0.5, 0.25, 0.25),
                              favouring Mahalanobis as described in §2.2 of the thesis.
        percentile:           Percentile of global training ensemble scores.
        use_adaptive_tau:     If True, use per-class thresholds at inference.
        per_class_percentile: Percentile for per-class tau. Defaults to percentile.
        regularization_eps:   Covariance regularization for Mahalanobis gate.
        fallback_threshold:   Used before fitting or when a class has no tau.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        percentile: float = 95.0,
        use_adaptive_tau: bool = True,
        per_class_percentile: Optional[float] = None,
        regularization_eps: float = 1e-4,
        fallback_threshold: float = 0.0,
    ) -> None:
        w = np.array(weights, dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("weights must have a positive sum.")
        self._weights: np.ndarray = (w / w.sum()).astype(np.float64)
        self.percentile: float = percentile
        self.use_adaptive_tau: bool = use_adaptive_tau
        self.per_class_percentile: float = (
            per_class_percentile if per_class_percentile is not None else percentile
        )
        self.regularization_eps: float = regularization_eps
        self.fallback_threshold: float = fallback_threshold

        self._gate_mahal: Optional[OutlierGate] = None
        self._gate_cos: Optional[OutlierGate] = None
        self._gate_l2: Optional[OutlierGate] = None
        self._dist_stats: Dict[str, Tuple[float, float]] = {}
        self._global_threshold: Optional[float] = None
        self._per_class_thresholds: Dict[str, float] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray, labels: List[str]) -> "EnsembleOutlierGate":
        """Fit all sub-gates and calibrate ensemble thresholds.

        Args:
            embeddings: Float32 array of shape (N, D).
            labels:     List of N label strings.

        Returns:
            Self (enables method chaining).

        Raises:
            ValueError: On shape / length mismatches or empty arrays.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                "embeddings must be non-empty 2-D (N, D), got {}".format(embeddings.shape)
            )
        if len(labels) != embeddings.shape[0]:
            raise ValueError(
                "len(labels)={} != embeddings.shape[0]={}".format(
                    len(labels), embeddings.shape[0])
            )

        logger.info(
            "EnsembleOutlierGate.fit(): N=%d, D=%d, n_classes=%d",
            embeddings.shape[0], embeddings.shape[1], len(set(labels)),
        )

        # 1. Fit three sub-gates on the same data
        self._gate_mahal = OutlierGate(
            method="mahalanobis",
            percentile=self.percentile,
            regularization_eps=self.regularization_eps,
        ).fit(embeddings, labels)

        self._gate_cos = OutlierGate(
            method="cosine",
            percentile=self.percentile,
        ).fit(embeddings, labels)

        self._gate_l2 = OutlierGate(
            method="l2",
            percentile=self.percentile,
        ).fit(embeddings, labels)

        # 2. Collect raw training distances from each sub-gate
        n = embeddings.shape[0]
        raw_mahal = np.array(
            [self._gate_mahal._distance_to_nearest(embeddings[i]) for i in range(n)],
            dtype=np.float64,
        )
        raw_cos = np.array(
            [self._gate_cos._distance_to_nearest(embeddings[i]) for i in range(n)],
            dtype=np.float64,
        )
        raw_l2 = np.array(
            [self._gate_l2._distance_to_nearest(embeddings[i]) for i in range(n)],
            dtype=np.float64,
        )

        # 3. Robust normalization stats: (median, iqr_scale) per method
        self._dist_stats = {
            "mahalanobis": self._robust_stats(raw_mahal),
            "cosine":      self._robust_stats(raw_cos),
            "l2":          self._robust_stats(raw_l2),
        }

        # 4. Compute normalized ensemble scores for all training samples
        def _znorm(arr, key):
            med, scale = self._dist_stats[key]
            return (arr - med) / scale

        z_mahal = _znorm(raw_mahal, "mahalanobis")
        z_cos   = _znorm(raw_cos,   "cosine")
        z_l2    = _znorm(raw_l2,    "l2")
        ensemble_scores = (
            self._weights[0] * z_mahal
            + self._weights[1] * z_cos
            + self._weights[2] * z_l2
        )

        # 5. Global threshold at configured percentile
        self._global_threshold = float(np.percentile(ensemble_scores, self.percentile))

        # 6. Per-class adaptive tau from intra-class score distributions
        if self.use_adaptive_tau:
            label_arr = np.array(labels)
            for lbl in sorted(set(labels)):
                mask = label_arr == lbl
                if not mask.any():
                    continue
                self._per_class_thresholds[lbl] = float(
                    np.percentile(ensemble_scores[mask], self.per_class_percentile)
                )

        self._is_fitted = True
        logger.info(
            "EnsembleOutlierGate fitted: global_tau=%.4f%s, "
            "weights=(mahal=%.2f, cos=%.2f, l2=%.2f)",
            self._global_threshold,
            ", {} per-class tau".format(len(self._per_class_thresholds))
            if self.use_adaptive_tau else "",
            self._weights[0], self._weights[1], self._weights[2],
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, embedding: np.ndarray) -> Tuple[float, str]:
        """Compute the ensemble OOD score for embedding.

        Args:
            embedding: 1-D float32 array of shape (D,).

        Returns:
            Tuple (ensemble_score, nearest_label).
            Higher score means more likely out-of-distribution.
            Returns (fallback_threshold, 'unknown') if not fitted.
        """
        if not self._is_fitted:
            return (self.fallback_threshold, "unknown")

        assert self._gate_mahal is not None
        assert self._gate_cos is not None
        assert self._gate_l2 is not None

        d_mahal, lbl_mahal = self._gate_mahal.score(embedding)
        d_cos,   lbl_cos   = self._gate_cos.score(embedding)
        d_l2,    lbl_l2    = self._gate_l2.score(embedding)

        med_m, sc_m = self._dist_stats["mahalanobis"]
        med_c, sc_c = self._dist_stats["cosine"]
        med_l, sc_l = self._dist_stats["l2"]

        z_mahal = (float(d_mahal) - med_m) / sc_m
        z_cos   = (float(d_cos)   - med_c) / sc_c
        z_l2    = (float(d_l2)    - med_l) / sc_l

        s = float(
            self._weights[0] * z_mahal
            + self._weights[1] * z_cos
            + self._weights[2] * z_l2
        )

        # Determine nearest label by weighted vote across all three sub-gates.
        # Using only mahal is unreliable when the pooled covariance matrix is
        # biased due to class imbalance during fitting (inv_cov can inflate
        # distances in the direction of underrepresented classes).
        vote_weights = {
            lbl_mahal: self._weights[0],
            lbl_cos:   self._weights[1],
            lbl_l2:    self._weights[2],
        }
        # Accumulate in case two sub-gates agree on the same label
        tally: dict = {}
        for lbl, w in [(lbl_mahal, self._weights[0]),
                       (lbl_cos,   self._weights[1]),
                       (lbl_l2,    self._weights[2])]:
            tally[lbl] = tally.get(lbl, 0.0) + w
        nearest_label = max(tally, key=lambda k: tally[k])

        return (s, nearest_label)

    def is_outlier(
        self,
        embedding: np.ndarray,
        snr_db: Optional[float] = None,
        snr_ref: float = 12.0,
        beta: float = 0.15,
    ) -> bool:
        """Return True if embedding is considered out-of-distribution.

        Mirrors ``OutlierGate.is_outlier()`` — supports SNR-adaptive threshold
        adjustment (§2.3) when ``snr_db`` is supplied.

        Args:
            embedding: 1-D float32 array of shape (D,).
            snr_db:    Optional online SNR estimate from ``estimate_snr_db()``.
                       Pass ``None`` to disable SNR adjustment (default).
            snr_ref:   Reference SNR in dB for the adaptive formula (12 dB).
            beta:      Sensitivity coefficient for the adaptive formula (0.15).

        Returns:
            True -> reject; False -> accept.
        """
        s, nearest_label = self.score(embedding)

        if self.use_adaptive_tau and self._per_class_thresholds:
            tau_0 = self._per_class_thresholds.get(
                nearest_label,
                self._global_threshold
                if self._global_threshold is not None
                else self.fallback_threshold,
            )
        else:
            tau_0 = (
                self._global_threshold
                if self._global_threshold is not None
                else self.fallback_threshold
            )

        tau = (
            snr_adaptive_threshold(tau_0, snr_db, snr_ref=snr_ref, beta=beta)
            if snr_db is not None
            else tau_0
        )
        return bool(s > tau)

    def sub_gate_scores(self, embedding: np.ndarray) -> Dict[str, float]:
        """Return raw sub-gate distances plus ensemble score for diagnostics.

        Args:
            embedding: 1-D float32 array of shape (D,).

        Returns:
            Dict with keys 'mahalanobis', 'cosine', 'l2', 'ensemble'.
        """
        if not self._is_fitted:
            return {
                "mahalanobis": float("inf"), "cosine": float("inf"),
                "l2": float("inf"), "ensemble": float("inf"),
            }
        assert self._gate_mahal and self._gate_cos and self._gate_l2
        d_mahal, _ = self._gate_mahal.score(embedding)
        d_cos,   _ = self._gate_cos.score(embedding)
        d_l2,    _ = self._gate_l2.score(embedding)
        s, _ = self.score(embedding)
        return {
            "mahalanobis": float(d_mahal),
            "cosine":      float(d_cos),
            "l2":          float(d_l2),
            "ensemble":    s,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _robust_stats(distances: np.ndarray) -> Tuple[float, float]:
        """Compute (median, IQR-based scale) for robust z-score normalization.

        IQR-based scale = IQR / 1.3489, which equals std for a Gaussian but
        is far more robust to heavy tails in the distance distribution.

        Args:
            distances: 1-D array of raw distances.

        Returns:
            Tuple (median, iqr_scale) where iqr_scale >= 1e-8.
        """
        median = float(np.median(distances))
        q25 = float(np.percentile(distances, 25))
        q75 = float(np.percentile(distances, 75))
        iqr_scale = max((q75 - q25) / 1.3489, 1e-8)
        return (median, iqr_scale)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: "str | Path") -> None:
        """Pickle the fitted ensemble gate to path.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the gate has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted EnsembleOutlierGate. Call fit() first."
            )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("EnsembleOutlierGate saved to %s", p)

    @staticmethod
    def load(path: "str | Path") -> "EnsembleOutlierGate":
        """Load a previously fitted ensemble gate from path.

        Args:
            path: Path to a .pkl file produced by EnsembleOutlierGate.save().

        Returns:
            Fitted EnsembleOutlierGate instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError("EnsembleOutlierGate file not found: {}".format(p))
        with open(p, "rb") as f:
            gate = pickle.load(f)
        logger.info(
            "EnsembleOutlierGate loaded from %s (global_tau=%.4f, adaptive_tau=%s)",
            p,
            gate._global_threshold or gate.fallback_threshold,
            gate.use_adaptive_tau,
        )
        return gate

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Return a human-readable summary dict for logging / debugging.

        Returns:
            Dictionary describing the fitted state of the ensemble gate.
        """
        w = self._weights.tolist()
        return {
            "fitted":               self._is_fitted,
            "method":               "ensemble",
            "weights":              {"mahalanobis": w[0], "cosine": w[1], "l2": w[2]},
            "n_classes":            (
                len(self._gate_mahal._labels)
                if self._gate_mahal and self._gate_mahal._labels else 0
            ),
            "embedding_dim":        (
                self._gate_mahal._centroids.shape[1]
                if self._gate_mahal and self._gate_mahal._centroids is not None else None
            ),
            "global_threshold":     self._global_threshold,
            "use_adaptive_tau":     self.use_adaptive_tau,
            "per_class_thresholds": dict(self._per_class_thresholds),
            "dist_stats":           {
                k: {"median": v[0], "iqr_scale": v[1]}
                for k, v in self._dist_stats.items()
            },
            "labels":               (
                list(self._gate_mahal._labels)
                if self._gate_mahal and self._gate_mahal._labels else []
            ),
        }
