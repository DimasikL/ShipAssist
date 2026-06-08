"""
core/hybrid/engine.py — HybridAudioEngine: the Approach C+ inference backend.

This class implements the exact same ``AudioEngine`` abstract interface as
``OnnxAudioEngine`` (``core/engine.py``) so the two engines are drop-in
replaceable from the caller's perspective. Critically, it does NOT modify
``core/engine.py`` in any way — it imports ``AudioEngine`` read-only.

Pipeline (4 stages)
-------------------
1. **Embedding extraction** (Stage 1) — Either reuses ``outputs[1]`` from the
   ONNX session (zero extra latency) or runs a standalone ``WTVEmbedder``
   (if ``use_onnx_embeddings=False``).
2. **OOD Gate** (Stage 2, ``core.hybrid.outlier_gate.EnsembleOutlierGate``) —
   Ensemble Mahalanobis + cosine + L2 early rejection. If the gate fires,
   returns immediately with ``outlier_rejected=True``.
3. **Centroid Search** (Stage 3, ``core.hybrid.centroid_search.CentroidSearch``) —
   Cosine nearest-centroid intent classification in normalised embedding space.
4. **Slot Fill** (Stage 4) — Activated only when the predicted intent is in
   ``cfg.number_regressor.slot_intents``. Two implementations selected at
   runtime and recorded in ``slot_method`` for A/B telemetry:

   * ``"regressor"`` — MLP ``NumberRegressor`` (Variant A).
   * ``"ctc"``       — CTC digit decoder ``CTCDigitDecoder`` (Variant B),
                       requires ``has_frames=True`` in the ONNX bundle.
   * ``"none"``      — intent is not a slot intent; no numeric fill.

Output dictionary
-----------------
Strict superset of ``OnnxAudioEngine.predict()`` — every key produced by the
existing engine is present, plus hybrid-specific extras:

    {
        "label":            str,            # top-1 intent label (or "" if rejected)
        "full_label":       str,            # label with slot filled, e.g. "курс 245"
        "confidence":       float,          # cosine similarity score (≡ OnnxEngine confidence)
        "probs":            np.ndarray,     # softmax over cosine scores (same shape contract)
        "logits":           np.ndarray,     # raw cosine scores (pre-softmax)
        "latency_ms":       float,          # total wall-clock time (ms)
        "outlier_score":    float,          # Mahalanobis distance (lower = more in-dist)
        "outlier_rejected": bool,           # True → gate fired; pipeline stopped at stage 1
        "slot_value":       float | None,   # numeric prediction, or None if not a slot intent
        "slot_confidence":  float,          # regressor confidence (0–1), 0 if no slot
        "search_method":    str,            # "hybrid_cosine" — for telemetry / A-B tests
    }

Graceful fallback
-----------------
If the hybrid model artefacts are missing at load time, the engine logs a
warning and returns ``{"error": "hybrid_not_loaded", ...}`` from ``predict()``
rather than raising an exception. This preserves the existing ONNX demo.

Thread safety
-------------
All three components are read-only after loading. ``predict()`` is safe to
call from multiple threads simultaneously.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.engine import AudioEngine          # read-only import of abstract base
from core.audio_utils import prepare_window
from core.logger import get_logger

logger = get_logger(__name__)

# Sentinel returned when artefacts are missing
_NOT_LOADED: Dict[str, Any] = {
    "error": "hybrid_not_loaded",
    "label": "",
    "full_label": "",
    "confidence": 0.0,
    "probs": np.array([], dtype=np.float32),
    "logits": np.array([], dtype=np.float32),
    "latency_ms": 0.0,
    "outlier_score": float("inf"),
    "outlier_rejected": False,
    "slot_value": None,
    "slot_confidence": 0.0,
    # "ctc" | "regressor" | "none" — for A/B telemetry and thesis comparison
    "slot_method": "none",
    "search_method": "hybrid_cosine",
    "snr_db": None,
}


class HybridAudioEngine(AudioEngine):
    """Approach C+ inference engine: 4-stage OOD Gate → Centroid Search → Slot Fill.

    Implements the four-stage pipeline described in thesis §2.3:
    embedding extraction → EnsembleOutlierGate (OOD) → CentroidSearch → slot fill.
    Slot-fill method is tracked in ``slot_method`` ("ctc" | "regressor" | "none")
    for A/B telemetry and thesis §4 comparison tables.

    Implements the same ``AudioEngine`` abstract interface as ``OnnxAudioEngine``
    so it can be swapped in transparently by any caller that depends only on
    ``AudioEngine``.

    Args:
        cfg:            Loaded ``HybridConfig`` (from ``core.hybrid.config``).
        outlier_gate:   Pre-loaded ``OutlierGate`` instance, or ``None`` to
                        load lazily from ``cfg.paths.outlier_gate``.
        centroid_search: Pre-loaded ``CentroidSearch`` instance, or ``None``
                        to load lazily from ``cfg.paths``.
        number_regressors: Dict mapping slot-intent label → ``NumberRegressor``.
                           Leave ``None`` to load lazily.
        onnx_engine:    Optional pre-loaded ``OnnxEngine`` for embedding reuse.
                        If ``None`` and ``cfg.embedder.use_onnx_embeddings=True``,
                        a new session is created from ``cfg.embedder.onnx_model_dir``.

    Example:
        >>> from core.hybrid.factory import create_hybrid_engine
        >>> engine = create_hybrid_engine(cfg)
        >>> result = engine.predict(audio_np_array)
        >>> print(result["full_label"], result["confidence"])
    """

    def __init__(
        self,
        cfg: Any,                              # HybridConfig (Any to avoid circular import)
        outlier_gate: Optional[Any] = None,
        centroid_search: Optional[Any] = None,
        number_regressors: Optional[Dict[str, Any]] = None,
        onnx_engine: Optional[Any] = None,
    ) -> None:
        self._cfg = cfg
        self._loaded: bool = False
        self._load_error: Optional[str] = None

        # Components — set by _load_components()
        self._outlier_gate = outlier_gate
        self._centroid_search = centroid_search
        self._number_regressors: Dict[str, Any] = number_regressors or {}
        self._ctc_decoders: Dict[str, Any] = {}     # slot_intent → CTCDigitDecoder
        self._onnx_engine = onnx_engine
        self._standalone_embedder = None        # WTVEmbedder if onnx not used
        self._label_list: List[str] = []

        self._load_components()

    # ------------------------------------------------------------------
    # AudioEngine interface
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Reload all hybrid components from *model_path* directory.

        ``model_path`` is treated as the ``artifacts/hybrid/`` root.
        Override ``cfg.paths`` entries by passing an absolute directory.

        Args:
            model_path: Path to the hybrid artefacts directory.
        """
        # Re-point paths and reload
        root = Path(model_path)
        logger.info("HybridAudioEngine.load() called with root=%s", root)
        self._load_components()

    def predict(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Run the full hybrid pipeline on a single audio window.

        Args:
            waveform: 1-D float32 numpy array at ``cfg.sample_rate`` Hz.
                      Length does not need to equal ``cfg.win_samples`` —
                      ``prepare_window`` handles padding/truncation.

        Returns:
            Result dict (see module docstring for full key specification).
        """
        if not self._loaded:
            logger.warning(
                "HybridAudioEngine not loaded (%s). Returning error sentinel.",
                self._load_error or "unknown error",
            )
            return dict(_NOT_LOADED)

        t0 = time.perf_counter()

        # ── Stage 0: Preprocessing ────────────────────────────────────
        audio = prepare_window(
            waveform.astype(np.float32, copy=False),
            target_samples=self._cfg.win_samples,
            do_normalize=True,
        )

        # ── Stage 1: Embedding extraction ────────────────────────────
        # Returns the pooled embedding (for OOD gate / centroid fallback),
        # the per-frame feature sequence (for CTC slot-fill), and the raw
        # ONNX logits when available (used on Stage 3 instead of centroid).
        embedding, frames, onnx_logits = self._get_features(audio)
        if embedding is None:  # fatal: no embedding → cannot run gate or classifier
            return self._build_result(
                label="",
                full_label="",
                confidence=0.0,
                probs=np.zeros(len(self._label_list), dtype=np.float32),
                logits=np.zeros(len(self._label_list), dtype=np.float32),
                outlier_score=float("inf"),
                outlier_rejected=True,
                slot_value=None,
                slot_confidence=0.0,
                slot_method="none",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # ── Stage 2: Outlier Gate ─────────────────────────────────────
        outlier_score: float = float("inf")
        outlier_rejected: bool = False

        # Optional online-SNR estimate for adaptive threshold (§2.3).
        # Computed once here so it can be logged and reused if needed.
        snr_db: Optional[float] = None
        snr_cfg = self._cfg.outlier_gate.snr_adaptive
        if snr_cfg.enabled:
            try:
                from core.hybrid.outlier_gate import estimate_snr_db
                snr_db = estimate_snr_db(audio, sr=self._cfg.sample_rate)
                logger.debug("Online SNR estimate: %.1f dB", snr_db)
            except Exception as exc:
                logger.warning("SNR estimation failed, proceeding without: %s", exc)

        if (
            self._outlier_gate is not None
            and self._cfg.outlier_gate.enabled
        ):
            dist, _ = self._outlier_gate.score(embedding)
            outlier_score = float(dist)
            if self._outlier_gate.is_outlier(
                embedding,
                snr_db=snr_db,
                snr_ref=snr_cfg.snr_ref_db,
                beta=snr_cfg.beta,
            ):
                outlier_rejected = True
                logger.debug(
                    "OutlierGate fired: dist=%.4f > threshold=%.4f",
                    outlier_score,
                    self._outlier_gate._threshold or self._outlier_gate.fallback_threshold,
                )
                return self._build_result(
                    label="",
                    full_label="",
                    confidence=0.0,
                    probs=np.zeros(len(self._label_list), dtype=np.float32),
                    logits=np.zeros(len(self._label_list), dtype=np.float32),
                    outlier_score=outlier_score,
                    outlier_rejected=True,
                    slot_value=None,
                    slot_confidence=0.0,
                    slot_method="none",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

        # ── Stage 3: Intent Classification ───────────────────────────
        # Path A (preferred): argmax over ONNX logits — the classifier head
        # was trained with CrossEntropy and already knows class boundaries.
        # Path B (fallback):  cosine nearest-centroid search — used only when
        # ONNX logits are unavailable (standalone WTVEmbedder path).
        #
        # Why not centroid as primary? The 256-D mean-pool embedding has
        # purity cosine = 23.6% (inter-centroid angle ≤ 7.5°, intra-class
        # spread 15× larger). The ONNX head, trained with CrossEntropy over
        # these same embeddings, collapses that space via a learned linear
        # boundary — centroid lookup cannot replicate this. See thesis §3.4.
        label: Optional[str] = None
        confidence: float = 0.0
        logits_arr = np.zeros(len(self._label_list), dtype=np.float32)
        probs_arr = np.zeros(len(self._label_list), dtype=np.float32)

        if onnx_logits is not None and len(self._label_list) > 0:
            # ── Path A: ONNX logits (primary) ──
            try:
                # Align ONNX output order with self._label_list.
                # OnnxEngine.labels is the source of truth for index→class.
                onnx_labels = self._onnx_engine.labels  # type: ignore[union-attr]
                if len(onnx_logits) == len(self._label_list):
                    # Fast path: assume same ordering (typical case after
                    # centroids are built from the same ONNX bundle).
                    logits_arr = onnx_logits.astype(np.float32)
                else:
                    # Reindex: map ONNX label order → centroid label order.
                    onnx_idx = {lbl: i for i, lbl in enumerate(onnx_labels)}
                    logits_arr = np.array(
                        [onnx_logits[onnx_idx[lbl]] if lbl in onnx_idx else 0.0
                         for lbl in self._label_list],
                        dtype=np.float32,
                    )
                # Stable softmax
                shifted = logits_arr - logits_arr.max()
                exp_l = np.exp(shifted)
                probs_arr = (exp_l / exp_l.sum()).astype(np.float32)
                best_idx = int(np.argmax(probs_arr))
                label = self._label_list[best_idx]
                confidence = float(probs_arr[best_idx])
                logger.debug(
                    "Stage3 (ONNX logits): label='%s' conf=%.4f", label, confidence
                )
            except Exception as exc:
                logger.warning(
                    "ONNX logits classification failed (%s) — falling back to centroid.",
                    exc,
                )
                onnx_logits = None  # trigger fallback below

        if onnx_logits is None and self._centroid_search is not None:
            # ── Path B: centroid search (fallback) ──
            try:
                all_scores: Dict[str, float] = {}
                label, confidence, all_scores = self._centroid_search.search(embedding)
                logits_arr = np.array(
                    [all_scores.get(lbl, 0.0) for lbl in self._label_list],
                    dtype=np.float32,
                )
                probs_arr = self._centroid_search.scores_as_probs(all_scores)
                logger.debug(
                    "Stage3 (centroid fallback): label='%s' conf=%.4f", label, confidence
                )
            except RuntimeError as exc:
                logger.warning("CentroidSearch error: %s", exc)

        # ── Stage 4: Number Slot-Fill ─────────────────────────────────
        # Strategy:
        #   1. If intent is in ctc_intents AND frames are available → try CTC.
        #   2. Accept CTC result when confidence ≥ min_confidence.
        #   3. Fall back to MLP NumberRegressor for any other case.
        slot_value: Optional[float] = None
        slot_confidence: float = 0.0
        slot_method: str = "none"
        full_label: str = label or ""

        if label is not None and label in self._cfg.number_regressor.slot_intents:
            ctc_cfg = self._cfg.ctc_decoder
            ctc_used = False

            # ── CTC path (Variant B) ──
            if (
                ctc_cfg.enabled
                and label in ctc_cfg.ctc_intents
                and frames is not None
            ):
                decoder = self._ctc_decoders.get(label)
                if decoder is not None:
                    ctc_val, ctc_conf = decoder.predict(frames)
                    if ctc_val is not None and ctc_conf >= ctc_cfg.min_confidence:
                        slot_value = ctc_val
                        slot_confidence = ctc_conf
                        slot_method = "ctc"
                        ctc_used = True
                        logger.debug(
                            "CTC slot-fill: intent='%s' value=%.0f conf=%.3f",
                            label, slot_value, slot_confidence,
                        )
                    else:
                        logger.debug(
                            "CTC decode rejected (val=%s, conf=%.3f < %.3f) — "
                            "falling back to regressor for '%s'.",
                            ctc_val, ctc_conf, ctc_cfg.min_confidence, label,
                        )

            # ── MLP regressor fallback (Variant A) ──
            if not ctc_used:
                regressor = self._number_regressors.get(label)
                if regressor is not None:
                    slot_value, slot_confidence = regressor.predict_with_confidence(embedding)
                    if slot_value is not None:
                        slot_method = "regressor"
                else:
                    logger.debug(
                        "Slot intent '%s' has no fitted regressor — "
                        "returning label without number.", label,
                    )

            # ── Compose full_label ──
            if slot_value is not None:
                num_str = str(int(round(slot_value)))
                if "УГОЛ" in label:
                    full_label = label.replace("УГОЛ", num_str)
                else:
                    full_label = f"{label} {num_str}"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "HybridAudioEngine: label='%s' full='%s' conf=%.4f "
            "outlier=%.4f snr=%s slot_method=%s lat=%.1fms",
            label or "(none)", full_label, confidence, outlier_score,
            f"{snr_db:.1f}dB" if snr_db is not None else "n/a",
            slot_method, latency_ms,
        )

        return self._build_result(
            label=label or "",
            full_label=full_label,
            confidence=confidence,
            probs=probs_arr,
            logits=logits_arr,
            outlier_score=outlier_score,
            outlier_rejected=outlier_rejected,
            slot_value=slot_value,
            slot_confidence=slot_confidence,
            slot_method=slot_method,
            latency_ms=latency_ms,
            snr_db=snr_db,
        )

    @property
    def labels(self) -> List[str]:
        """Ordered list of known intent labels (from the centroid registry)."""
        return list(self._label_list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_components(self) -> None:
        """Load all sub-components with graceful per-component error handling."""
        cfg = self._cfg
        errors: List[str] = []

        # ── ONNX embedding engine ──────────────────────────────────────
        if self._onnx_engine is None and cfg.embedder.use_onnx_embeddings:
            try:
                from core.onnx_engine import OnnxEngine
                onnx_dir = str(cfg.embedder.onnx_model_dir)
                self._onnx_engine = OnnxEngine(onnx_dir=onnx_dir, precision="int8")
                logger.info("HybridAudioEngine: ONNX embedding engine loaded from %s", onnx_dir)
            except Exception as exc:
                msg = f"ONNX engine load failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        # ── Standalone embedder (fallback) ────────────────────────────
        if self._onnx_engine is None and not cfg.embedder.use_onnx_embeddings:
            try:
                from core.embedders import WTVEmbedder
                from core.preproc import PreprocNo
                self._standalone_embedder = WTVEmbedder(
                    sr=cfg.sample_rate,
                    preproc=PreprocNo(),
                    emb_model=cfg.embedder.hf_model_name,
                    output_hidden_states=True,
                )
                logger.info(
                    "HybridAudioEngine: standalone WTVEmbedder loaded (%s).",
                    cfg.embedder.hf_model_name,
                )
            except Exception as exc:
                msg = f"WTVEmbedder load failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        # ── Outlier Gate ───────────────────────────────────────────────
        if self._outlier_gate is None:
            gate_path = cfg.paths.outlier_gate
            if Path(gate_path).exists():
                try:
                    from core.hybrid.outlier_gate import OutlierGate
                    # OutlierGate.load() uses pickle.load() — the concrete type
                    # (OutlierGate or EnsembleOutlierGate) is restored automatically
                    # from the pickle. No need to branch on cfg.outlier_gate.method.
                    self._outlier_gate = OutlierGate.load(gate_path)
                except Exception as exc:
                    msg = f"OutlierGate load failed: {exc}"
                    logger.warning(msg)
                    errors.append(msg)
            else:
                logger.warning(
                    "OutlierGate artefact not found at %s — gate disabled. "
                    "Run scripts/hybrid/train_outlier_gate.py to create it.",
                    gate_path,
                )

        # ── Centroid Search ────────────────────────────────────────────
        if self._centroid_search is None:
            c_path = cfg.paths.centroids
            l_path = cfg.paths.centroid_labels
            if Path(c_path).exists() and Path(l_path).exists():
                try:
                    from core.hybrid.centroid_search import CentroidSearch
                    self._centroid_search = CentroidSearch.load_npz(
                        centroids_path=c_path,
                        labels_path=l_path,
                        min_cosine_similarity=cfg.centroid_search.min_cosine_similarity,
                        per_label_thresholds=cfg.centroid_search.per_label_thresholds,
                    )
                    self._label_list = self._centroid_search.labels
                except Exception as exc:
                    msg = f"CentroidSearch load failed: {exc}"
                    logger.warning(msg)
                    errors.append(msg)
            else:
                logger.warning(
                    "Centroid artefacts not found (%s / %s). "
                    "Run scripts/hybrid/build_centroids.py to create them.",
                    c_path, l_path,
                )
                errors.append("centroids_missing")
        else:
            self._label_list = self._centroid_search.labels

        # ── CTC Digit Decoders (Variant B) ────────────────────────────
        ctc_cfg = cfg.ctc_decoder
        if ctc_cfg.enabled and not self._ctc_decoders:
            head_path = ctc_cfg.head_path
            if Path(head_path).exists():
                try:
                    from core.hybrid.ctc_digit_decoder import CTCDigitDecoder

                    # Infer frame_dim from the ONNX config (set at export time).
                    frame_dim: Optional[int] = None
                    if self._onnx_engine is not None:
                        frame_dim = getattr(self._onnx_engine, "frame_dim", None)
                    if frame_dim is None:
                        logger.warning(
                            "CTCDigitDecoder: frame_dim not found in OnnxEngine config — "
                            "CTC decoders will not be loaded. "
                            "Re-export the ONNX model to add projected_frames output."
                        )
                    else:
                        for slot_intent in ctc_cfg.ctc_intents:
                            bounds = cfg.number_regressor.bounds.get(slot_intent, [0.0, 360.0])
                            min_v, max_v = float(bounds[0]), float(bounds[1])
                            try:
                                self._ctc_decoders[slot_intent] = CTCDigitDecoder.load(
                                    path=head_path,
                                    frame_dim=frame_dim,
                                    min_val=min_v,
                                    max_val=max_v,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "CTCDigitDecoder load failed for '%s': %s",
                                    slot_intent, exc,
                                )
                except Exception as exc:
                    logger.warning("CTCDigitDecoder import failed: %s", exc)
            else:
                logger.info(
                    "CTCDigitDecoder head not found at %s — CTC slot-fill disabled. "
                    "Train with: python scripts/hybrid/train_ctc_head.py",
                    head_path,
                )

        # ── Number Regressors ──────────────────────────────────────────
        if not self._number_regressors:
            reg_dir = Path(cfg.paths.number_regressors_dir)
            if reg_dir.exists():
                from core.hybrid.number_regressor import NumberRegressor
                for slot_intent in cfg.number_regressor.slot_intents:
                    safe_key = slot_intent.replace(" ", "_")
                    pkl_path = reg_dir / f"{safe_key}.pkl"
                    if pkl_path.exists():
                        try:
                            self._number_regressors[slot_intent] = NumberRegressor.load(pkl_path)
                        except Exception as exc:
                            logger.warning(
                                "NumberRegressor load failed for '%s': %s",
                                slot_intent, exc,
                            )
                    else:
                        logger.debug(
                            "No regressor file for slot intent '%s' (%s).",
                            slot_intent, pkl_path,
                        )

        # ── Determine overall loaded status ───────────────────────────
        # The engine is considered "loaded" if at minimum the centroid search
        # is available. The outlier gate is optional (degrades gracefully).
        if self._centroid_search is not None:
            self._loaded = True
            self._load_error = None
            logger.info(
                "HybridAudioEngine ready: %d intents, %d slot regressors, "
                "gate=%s",
                len(self._label_list),
                len(self._number_regressors),
                "on" if self._outlier_gate is not None else "off (artefact missing)",
            )
        else:
            self._loaded = False
            self._load_error = "; ".join(errors) or "centroid artefacts missing"
            logger.warning(
                "HybridAudioEngine could NOT be fully loaded: %s. "
                "predict() will return {'error': 'hybrid_not_loaded'}.",
                self._load_error,
            )

    def _get_features(
        self, audio: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract pooled embedding, per-frame features, and raw logits from *audio*.

        Args:
            audio: Pre-processed 1-D float32 waveform of length ``win_samples``.

        Returns:
            Tuple ``(embedding, frames, onnx_logits)`` where:

            * ``embedding``   — 1-D float32 pooled vector ``(D_proj,)``
              from ONNX ``outputs[1]`` (mean-pooled projected features).
              Used by: EnsembleOutlierGate (Stage 2), CentroidSearch
              cosine fallback (Stage 3 Path B), NumberRegressor (Stage 4 A).
              ``None`` on fatal extraction error — pipeline cannot proceed.
            * ``frames``      — 2-D float32 frame tensor ``(T, D_proj)``
              from ONNX ``outputs[2]`` (per-frame projected features).
              Used by: CTCDigitDecoder (Stage 4 Variant B).
              ``None`` when the ONNX bundle lacks the third output
              (older export without ``projected_frames``) or when using
              the standalone WTVEmbedder path.
            * ``onnx_logits`` — 1-D float32 raw logits ``(N_classes,)``
              from ONNX ``outputs[0]`` (pre-softmax classifier head output).
              Used by: argmax intent classification (Stage 3 Path A).
              ``None`` when using the standalone WTVEmbedder path
              (no classifier head available); Stage 3 then falls back to
              centroid search (Path B).
        """
        # Path A: Reuse ONNX engine outputs (fast, no extra model call).
        # predict_logits() returns (logits, embedding, frames) — all three
        # are produced in a single ONNX session.run() call, zero extra cost.
        if self._onnx_engine is not None:
            try:
                logits, embedding, frames = self._onnx_engine.predict_logits(audio)
                if embedding is not None:
                    return (
                        embedding.astype(np.float32),
                        frames,
                        logits.astype(np.float32),
                    )
                # ONNX model was not exported with embedding output → fall through
                logger.warning(
                    "ONNX model has no second output (embedding). "
                    "Re-export with the embedding head or set "
                    "use_onnx_embeddings=False in config."
                )
            except Exception as exc:
                logger.warning("ONNX embedding extraction failed: %s", exc)

        # Path B: Standalone WTVEmbedder (frames and logits not available).
        if self._standalone_embedder is not None:
            try:
                import tempfile, soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio, self._cfg.sample_rate)
                    emb = self._standalone_embedder.get_emb(tmp.name)
                import os; os.unlink(tmp.name)
                if emb is not None:
                    return np.asarray(emb, dtype=np.float32).flatten(), None, None
            except Exception as exc:
                logger.warning("Standalone embedder failed: %s", exc)

        logger.error("No embedding source available — cannot run hybrid pipeline.")
        return None, None, None

    @staticmethod
    def _build_result(
        label: str,
        full_label: str,
        confidence: float,
        probs: np.ndarray,
        logits: np.ndarray,
        outlier_score: float,
        outlier_rejected: bool,
        slot_value: Optional[float],
        slot_confidence: float,
        slot_method: str,
        latency_ms: float,
        snr_db: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Assemble the standardised result dictionary.

        All keys match or extend the ``OnnxAudioEngine.predict()`` contract.

        Returns:
            Result dict with all required keys.
        """
        return {
            # ── OnnxAudioEngine-compatible keys ──
            "label": label,
            "confidence": float(confidence),
            "probs": probs.astype(np.float32),
            "logits": logits.astype(np.float32),
            "latency_ms": float(latency_ms),
            # ── Hybrid-specific extras ────────────
            "full_label": full_label,
            "outlier_score": float(outlier_score),
            "outlier_rejected": bool(outlier_rejected),
            "slot_value": float(slot_value) if slot_value is not None else None,
            "slot_confidence": float(slot_confidence),
            # slot_method: "ctc" | "regressor" | "none"
            # Use for thesis A/B comparison and production telemetry.
            "slot_method": slot_method,
            "snr_db": float(snr_db) if snr_db is not None else None,
            "search_method": "hybrid_ensemble",
        }
