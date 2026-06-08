"""
core/router.py — SmartRouter: confidence-aware delegation facade.

What this module does
---------------------
``SmartRouter`` sits in front of two independently loaded engines —
``OnnxAudioEngine`` (LoRA fine-tuned, high accuracy for fixed phrases) and
``HybridAudioEngine`` (cosine centroid search + slot-fill for dynamic phrases)
— and decides, per call, which engine's result to trust and return.

It implements the same ``AudioEngine`` abstract interface as both sub-engines
so any existing caller can swap to the router transparently.

Routing logic (decision tree)
------------------------------
Every ``predict()`` call walks this tree in order:

  1. Run ONNX engine first (always; it's fast ~106 ms).
     ├─ ONNX fires on a **known phrase** AND conf ≥ onnx_threshold
     │  └─ ✅ Return ONNX result immediately. Hybrid is NOT called.
     │     (Fast path — saves ~30–80 ms on the common case.)
     └─ Otherwise: continue to step 2.

  2. Run Hybrid engine.
     ├─ Hybrid **outlier-rejected** the audio
     │  ├─ ONNX had a (lower-confidence) known-phrase result
     │  │  └─ ✅ Return ONNX result  (engine_used="onnx_lora")
     │  └─ Neither engine is confident
     │     └─ ✅ Return outlier-rejection sentinel (engine_used="outlier_rejected")
     │
     ├─ Hybrid label is a **slot intent** (contains a number)
     │  └─ ✅ Return hybrid result  (engine_used="hybrid")
     │     (Hybrid is the only engine that can resolve numeric slots.)
     │
     ├─ ONNX label ∈ known_phrases AND onnx_conf ≥ onnx_threshold
     │  └─ ✅ Return ONNX result  (engine_used="onnx_lora")
     │
     ├─ hybrid_conf_mapped ≥ hybrid_threshold
     │  └─ ✅ Return hybrid result  (engine_used="hybrid")
     │
     └─ Tie-break: highest mapped confidence wins.

Confidence scale alignment
---------------------------
ONNX produces softmax probabilities ∈ [0, 1].
Hybrid produces cosine similarities ∈ [–1, 1], typically [0.5, 1.0] for speech.
A simple linear mapping makes them comparable for the tie-break:

    mapped_cosine = clip(0, 1,  (cosine – shift) × factor)

Defaults: shift=0.5, factor=2.0
  → cosine 0.50 → 0.00   (random similarity → zero confidence)
  → cosine 0.75 → 0.50   (moderate match → medium confidence)
  → cosine 1.00 → 1.00   (identical embedding → full confidence)

This is NOT a calibrated probability — it is a scale alignment sufficient
for tie-breaking. For a properly calibrated comparison, run
``scripts/router_demo.py --mode bench`` on labelled data and adjust
``cosine_shift`` and ``cosine_factor`` in ``configs/routing.yaml``.

Output dict contract
--------------------
Every key from both sub-engines is preserved and passed through unchanged.
The router adds three keys on top:

    engine_used     str    "onnx_lora" | "hybrid" | "outlier_rejected"
    router_latency_ms float  total wall-clock time including both engine calls
    confidence_mapped float  cosine → [0,1] mapped value (for ONNX: identity)

Graceful fallback
-----------------
If one engine is unavailable (None or raised on load), the router logs a
warning and routes 100% of traffic to the other. If both fail, ``predict()``
returns a null-result dict with ``engine_used="both_unavailable"``.

Zero changes to existing files
-------------------------------
This file imports ``AudioEngine`` from ``core.engine`` read-only.
It does NOT modify ``core/engine.py``, ``core/config.py``,
``core/hybrid/engine.py``, ``scripts/demo_defense.py``, or any config file.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from core.engine import AudioEngine      # read-only: abstract base class only
from core.logger import get_logger

logger = get_logger(__name__)

# ── Engine-used tags ──────────────────────────────────────────────────────────
TAG_ONNX = "onnx_lora"
TAG_HYBRID = "hybrid"
TAG_OUTLIER = "outlier_rejected"
TAG_BOTH_UNAVAIL = "both_unavailable"


# ── Configuration ─────────────────────────────────────────────────────────────

class RoutingConfig(BaseModel):
    """All routing parameters in one validated object.

    Load from ``configs/routing.yaml`` via ``RoutingConfig.from_yaml()``,
    or construct directly for tests:

        cfg = RoutingConfig(known_phrases=["машина", "самый малый вперед"])

    Attributes:
        known_phrases:              Labels owned by the ONNX engine. When ONNX
                                    is confident on one of these, the router
                                    returns immediately without calling hybrid.
        number_slot_intents:        Labels that contain an open numeric slot
                                    (e.g. "курс УГОЛ градусов"). Always routed
                                    to hybrid — ONNX cannot fill number slots.
        onnx_confidence_threshold:  Minimum ONNX softmax confidence to trust a
                                    known-phrase prediction. Below this the
                                    router continues to hybrid.
        hybrid_confidence_threshold: Minimum mapped cosine confidence to accept
                                    a hybrid prediction in the tie-break stage.
        cosine_shift:               Subtracted from raw cosine before scaling.
                                    Default 0.5 (random baseline for Wav2Vec2).
        cosine_factor:              Multiplied after shift. Default 2.0.
                                    mapped = clip(0,1, (cosine-shift)*factor).
        run_both_always:            If True, always call both engines regardless
                                    of ONNX result. Useful for A/B logging.
        prefer_onnx_for_known:      If False, always call hybrid even for known
                                    phrases. Use during shadow-mode evaluation.
    """

    known_phrases: List[str] = Field(
        default_factory=list,
        description="Labels owned by ONNX. High-confidence hits short-circuit hybrid.",
    )
    number_slot_intents: List[str] = Field(
        default_factory=list,
        description="Labels with open numeric slots — always routed to hybrid.",
    )
    onnx_confidence_threshold: float = Field(
        0.85, ge=0.0, le=1.0,
        description="Minimum ONNX softmax confidence to accept a known-phrase prediction.",
    )
    hybrid_confidence_threshold: float = Field(
        0.75, ge=0.0, le=1.0,
        description="Minimum mapped cosine confidence to accept a hybrid prediction.",
    )
    cosine_shift: float = Field(
        0.50,
        description="Baseline subtracted from cosine before linear scaling.",
    )
    cosine_factor: float = Field(
        2.0, gt=0.0,
        description="Multiplier applied after subtracting cosine_shift.",
    )
    run_both_always: bool = Field(
        False,
        description="Always run both engines. Disables the ONNX fast-path.",
    )
    prefer_onnx_for_known: bool = Field(
        True,
        description="Route confident known-phrase hits to ONNX without calling hybrid.",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RoutingConfig":
        """Load routing config from a YAML file.

        The file may contain either a top-level ``routing:`` key (for embedding
        in ``base.yaml``) or be a flat routing-only file.

        Args:
            path: Path to the YAML file.

        Returns:
            Validated ``RoutingConfig``.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        import yaml

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Routing config not found: {p.resolve()}")

        with open(p, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}

        # Support both flat and nested (routing: {...}) formats
        if "routing" in data:
            data = data["routing"]

        return cls.model_validate(data)

    @classmethod
    def from_yaml_or_default(cls, path: str | Path) -> "RoutingConfig":
        """Like ``from_yaml`` but returns defaults if the file is absent.

        Useful during development before configs are committed.

        Args:
            path: Path to attempt to load.

        Returns:
            ``RoutingConfig`` from file, or default instance with a warning.
        """
        try:
            return cls.from_yaml(path)
        except FileNotFoundError:
            logger.warning(
                "Routing config not found at %s — using defaults. "
                "Create configs/routing.yaml to customise routing behaviour.",
                path,
            )
            return cls()


# ── Null / sentinel results ───────────────────────────────────────────────────

def _null_result(router_latency_ms: float) -> Dict[str, Any]:
    """Return a fully-keyed result dict when both engines are unavailable.

    Args:
        router_latency_ms: Elapsed time to include in the dict.

    Returns:
        Result dict with all required keys set to neutral / zero values.
    """
    return {
        "label": "",
        "full_label": "",
        "confidence": 0.0,
        "confidence_mapped": 0.0,
        "probs": np.array([], dtype=np.float32),
        "logits": np.array([], dtype=np.float32),
        "latency_ms": 0.0,
        "outlier_score": float("inf"),
        "outlier_rejected": False,
        "slot_value": None,
        "slot_confidence": 0.0,
        "search_method": "none",
        "engine_used": TAG_BOTH_UNAVAIL,
        "router_latency_ms": float(router_latency_ms),
    }


def _normalise_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Add any keys that hybrid-only results may be missing (ONNX compat).

    Ensures the dict has all hybrid keys so callers can always do
    ``result["outlier_rejected"]`` regardless of which engine ran.

    Args:
        result: Raw result dict from either engine.

    Returns:
        Copy of *result* with all expected keys guaranteed present.
    """
    defaults: Dict[str, Any] = {
        "full_label": result.get("label", ""),
        "outlier_score": float("inf"),
        "outlier_rejected": False,
        "slot_value": None,
        "slot_confidence": 0.0,
        "search_method": "onnx_softmax",
    }
    out = dict(defaults)
    out.update(result)   # real values win over defaults
    return out


# ── SmartRouter ───────────────────────────────────────────────────────────────

class SmartRouter(AudioEngine):
    """Confidence-aware routing facade over OnnxAudioEngine + HybridAudioEngine.

    Implements the full ``AudioEngine`` interface so it can replace either
    sub-engine transparently in ``src/api.py``, ``src/inference.py``, or any
    other caller that depends only on ``AudioEngine``.

    Args:
        cfg:           Routing rules and thresholds. Use
                       ``RoutingConfig.from_yaml("configs/routing.yaml")``.
        onnx_engine:   Loaded ``OnnxAudioEngine`` (or any ``AudioEngine``).
                       Pass ``None`` to route 100% to hybrid.
        hybrid_engine: Loaded ``HybridAudioEngine`` (or any ``AudioEngine``).
                       Pass ``None`` to route 100% to ONNX.

    Example::

        from core.router import SmartRouter, RoutingConfig
        from core.engine import OnnxAudioEngine
        from core.hybrid.factory import create_hybrid_engine
        from core.hybrid.config import HybridConfig

        routing_cfg  = RoutingConfig.from_yaml("configs/routing.yaml")
        onnx_engine  = OnnxAudioEngine(onnx_dir="artifacts/models/onnx_model")
        hybrid_cfg   = HybridConfig.from_yaml("configs/hybrid/model.yaml",
                                              "configs/hybrid/thresholds.yaml")
        hybrid_engine = create_hybrid_engine(hybrid_cfg)

        router = SmartRouter(routing_cfg, onnx_engine, hybrid_engine)
        result = router.predict(audio_np)
        # result["engine_used"] → "onnx_lora" | "hybrid" | "outlier_rejected"
    """

    def __init__(
        self,
        cfg: RoutingConfig,
        onnx_engine: Optional[AudioEngine] = None,
        hybrid_engine: Optional[AudioEngine] = None,
    ) -> None:
        self._cfg: RoutingConfig = cfg
        self._onnx: Optional[AudioEngine] = onnx_engine
        self._hybrid: Optional[AudioEngine] = hybrid_engine

        # Derive availability
        self._onnx_available: bool = onnx_engine is not None
        self._hybrid_available: bool = hybrid_engine is not None

        # Build unified label list (ONNX labels first, then hybrid-only ones)
        self._label_list: List[str] = self._build_label_list()

        self._log_startup()

    # ------------------------------------------------------------------
    # AudioEngine interface
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """No-op on the router itself; sub-engines load independently.

        Call ``onnx_engine.load()`` or ``hybrid_engine.load()`` directly
        if you need to hot-reload a specific backend.

        Args:
            model_path: Unused by the router (kept for interface parity).
        """
        logger.debug(
            "SmartRouter.load() called with path=%r — "
            "router does not own model weights; ignoring.",
            model_path,
        )

    def predict(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Route a single audio window to the best available engine.

        Follows the decision tree described in the module docstring.
        Always adds ``engine_used`` and ``router_latency_ms`` to the
        returned dict on top of whatever the sub-engine returned.

        Args:
            waveform: 1-D float32 numpy array at 16 000 Hz, any length
                      (sub-engines handle padding/truncation internally).

        Returns:
            Result dict from the winning engine, augmented with:
              - ``engine_used`` (str): which engine produced this result.
              - ``router_latency_ms`` (float): total wall-clock time (ms).
              - ``confidence_mapped`` (float): cosine→[0,1] mapped value
                (identical to ``confidence`` for ONNX softmax outputs).
        """
        t0 = time.perf_counter()
        waveform = waveform.astype(np.float32, copy=False)

        # ── Degenerate cases: one or both engines unavailable ─────────
        if not self._onnx_available and not self._hybrid_available:
            logger.error("SmartRouter: both engines unavailable.")
            return _null_result((time.perf_counter() - t0) * 1_000)

        if not self._onnx_available:
            hybrid_result = self._call_hybrid(waveform)
            return self._tag(
                hybrid_result,
                TAG_HYBRID if not hybrid_result.get("outlier_rejected") else TAG_OUTLIER,
                t0,
            )

        if not self._hybrid_available:
            onnx_result = self._call_onnx(waveform)
            return self._tag(onnx_result, TAG_ONNX, t0)

        # ── Both engines available ────────────────────────────────────

        # Stage 1 — ONNX (always run first; it's the fast path)
        onnx_result = self._call_onnx(waveform)
        onnx_label = onnx_result.get("label", "")
        onnx_conf = float(onnx_result.get("confidence", 0.0))

        # Fast-path: ONNX owns this label and is confident
        if (
            not self._cfg.run_both_always
            and self._cfg.prefer_onnx_for_known
            and onnx_label in self._cfg.known_phrases
            and onnx_conf >= self._cfg.onnx_confidence_threshold
        ):
            logger.debug(
                "SmartRouter fast-path: ONNX owns '%s' (conf=%.4f ≥ %.4f).",
                onnx_label, onnx_conf, self._cfg.onnx_confidence_threshold,
            )
            return self._tag(onnx_result, TAG_ONNX, t0)

        # Stage 2 — Hybrid engine
        hybrid_result = self._call_hybrid(waveform)

        return self._route(onnx_result, onnx_label, onnx_conf, hybrid_result, t0)

    @property
    def labels(self) -> List[str]:
        """Union of all labels from both sub-engines, ONNX labels first."""
        return list(self._label_list)

    # ------------------------------------------------------------------
    # Routing decision logic
    # ------------------------------------------------------------------

    def _route(
        self,
        onnx_result: Dict[str, Any],
        onnx_label: str,
        onnx_conf: float,
        hybrid_result: Dict[str, Any],
        t0: float,
    ) -> Dict[str, Any]:
        """Apply the full routing decision tree to two candidate results.

        Args:
            onnx_result:   Raw result from the ONNX engine.
            onnx_label:    Pre-extracted ``label`` from onnx_result.
            onnx_conf:     Pre-extracted ``confidence`` from onnx_result.
            hybrid_result: Raw result from the hybrid engine.
            t0:            ``perf_counter()`` value at start of ``predict()``.

        Returns:
            Tagged result dict from the winning engine.
        """
        hybrid_rejected = hybrid_result.get("outlier_rejected", False)
        hybrid_label = hybrid_result.get("label", "")
        hybrid_conf_raw = float(hybrid_result.get("confidence", 0.0))
        hybrid_conf_mapped = self._map_cosine(hybrid_conf_raw)

        # ── Hybrid explicitly rejected → fall back to ONNX ───────────
        if hybrid_rejected:
            if onnx_label:
                logger.debug(
                    "SmartRouter: hybrid outlier-rejected. "
                    "Falling back to ONNX '%s' (conf=%.4f).",
                    onnx_label, onnx_conf,
                )
                return self._tag(onnx_result, TAG_ONNX, t0)
            logger.debug("SmartRouter: outlier-rejected, ONNX also blank.")
            return self._tag(hybrid_result, TAG_OUTLIER, t0)

        # ── Slot intent → hybrid always wins ─────────────────────────
        if hybrid_label in self._cfg.number_slot_intents:
            logger.debug(
                "SmartRouter: slot intent '%s' → hybrid (conf_mapped=%.4f).",
                hybrid_label, hybrid_conf_mapped,
            )
            return self._tag(hybrid_result, TAG_HYBRID, t0)

        # ── ONNX owns known phrase with sufficient confidence ─────────
        if (
            onnx_label in self._cfg.known_phrases
            and onnx_conf >= self._cfg.onnx_confidence_threshold
        ):
            logger.debug(
                "SmartRouter: ONNX owns '%s' (conf=%.4f ≥ %.4f).",
                onnx_label, onnx_conf, self._cfg.onnx_confidence_threshold,
            )
            return self._tag(onnx_result, TAG_ONNX, t0)

        # ── Hybrid meets its threshold ────────────────────────────────
        if hybrid_conf_mapped >= self._cfg.hybrid_confidence_threshold:
            logger.debug(
                "SmartRouter: hybrid '%s' meets threshold "
                "(mapped=%.4f ≥ %.4f).",
                hybrid_label, hybrid_conf_mapped, self._cfg.hybrid_confidence_threshold,
            )
            return self._tag(hybrid_result, TAG_HYBRID, t0)

        # ── Tie-break: highest mapped confidence ─────────────────────
        if onnx_conf >= hybrid_conf_mapped:
            tag = TAG_ONNX if onnx_label else TAG_OUTLIER
            logger.debug(
                "SmartRouter tie-break → ONNX '%s' "
                "(onnx=%.4f ≥ hybrid_mapped=%.4f).",
                onnx_label, onnx_conf, hybrid_conf_mapped,
            )
            return self._tag(onnx_result, tag, t0)
        else:
            tag = TAG_HYBRID if hybrid_label else TAG_OUTLIER
            logger.debug(
                "SmartRouter tie-break → hybrid '%s' "
                "(hybrid_mapped=%.4f > onnx=%.4f).",
                hybrid_label, hybrid_conf_mapped, onnx_conf,
            )
            return self._tag(hybrid_result, tag, t0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_onnx(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Call the ONNX engine, catching and logging any exception.

        Args:
            waveform: Audio input array.

        Returns:
            Engine result dict, or a blank null-result on failure.
        """
        assert self._onnx is not None
        try:
            result = self._onnx.predict(waveform)
            return _normalise_result(result)
        except Exception as exc:
            logger.error("ONNX engine raised: %s — returning blank result.", exc)
            self._onnx_available = False
            return _normalise_result({"label": "", "confidence": 0.0,
                                      "probs": np.array([], dtype=np.float32),
                                      "logits": np.array([], dtype=np.float32),
                                      "latency_ms": 0.0})

    def _call_hybrid(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Call the hybrid engine, catching and logging any exception.

        Args:
            waveform: Audio input array.

        Returns:
            Engine result dict, or a blank null-result on failure.
        """
        assert self._hybrid is not None
        try:
            result = self._hybrid.predict(waveform)
            return _normalise_result(result)
        except Exception as exc:
            logger.error("Hybrid engine raised: %s — returning blank result.", exc)
            self._hybrid_available = False
            return _normalise_result({"label": "", "confidence": 0.0,
                                      "probs": np.array([], dtype=np.float32),
                                      "logits": np.array([], dtype=np.float32),
                                      "latency_ms": 0.0,
                                      "outlier_rejected": True})

    def _map_cosine(self, cosine: float) -> float:
        """Map a raw cosine similarity to the [0, 1] confidence scale.

        Uses the linear formula from the module docstring:
            mapped = clip(0, 1, (cosine − shift) × factor)

        Args:
            cosine: Raw cosine similarity from the hybrid engine.

        Returns:
            Mapped confidence in [0, 1].
        """
        mapped = (cosine - self._cfg.cosine_shift) * self._cfg.cosine_factor
        return float(max(0.0, min(1.0, mapped)))

    def _tag(
        self,
        result: Dict[str, Any],
        engine_used: str,
        t0: float,
    ) -> Dict[str, Any]:
        """Attach router metadata to a sub-engine result dict.

        Args:
            result:      Result dict from a sub-engine.
            engine_used: One of the TAG_* constants.
            t0:          ``perf_counter()`` at start of ``predict()``.

        Returns:
            Shallow copy of *result* with three extra keys.
        """
        conf = float(result.get("confidence", 0.0))
        # For ONNX results, mapped confidence = raw softmax (already [0,1])
        if engine_used == TAG_ONNX:
            conf_mapped = conf
        else:
            conf_mapped = self._map_cosine(conf)

        router_lat = (time.perf_counter() - t0) * 1_000.0

        out = dict(result)
        out["engine_used"] = engine_used
        out["router_latency_ms"] = float(router_lat)
        out["confidence_mapped"] = float(conf_mapped)
        return out

    def _build_label_list(self) -> List[str]:
        """Build an ordered union of both engines' label lists.

        ONNX labels come first (they are the authoritative known phrases),
        followed by any hybrid-only labels.

        Returns:
            Deduplicated ordered list of label strings.
        """
        seen: set = set()
        out: List[str] = []

        for engine in (self._onnx, self._hybrid):
            if engine is None:
                continue
            try:
                for lbl in engine.labels:
                    if lbl not in seen:
                        seen.add(lbl)
                        out.append(lbl)
            except Exception:
                pass   # engine may not be loaded yet

        # Supplement with config-declared labels that might not be in either engine
        for lbl in self._cfg.known_phrases + self._cfg.number_slot_intents:
            if lbl not in seen:
                seen.add(lbl)
                out.append(lbl)

        return out

    def _log_startup(self) -> None:
        """Log a human-readable startup summary."""
        onnx_status = "ready" if self._onnx_available else "UNAVAILABLE"
        hybrid_status = "ready" if self._hybrid_available else "UNAVAILABLE"

        logger.info(
            "SmartRouter initialised: "
            "onnx=%s  hybrid=%s  "
            "known_phrases=%d  slot_intents=%d  "
            "thresholds(onnx=%.2f, hybrid=%.2f)  "
            "cosine_map=(shift=%.2f, factor=%.2f)  "
            "fast_path=%s",
            onnx_status, hybrid_status,
            len(self._cfg.known_phrases),
            len(self._cfg.number_slot_intents),
            self._cfg.onnx_confidence_threshold,
            self._cfg.hybrid_confidence_threshold,
            self._cfg.cosine_shift,
            self._cfg.cosine_factor,
            "on" if (self._cfg.prefer_onnx_for_known and not self._cfg.run_both_always) else "off",
        )
        if not self._onnx_available:
            logger.warning(
                "SmartRouter: ONNX engine unavailable — "
                "routing 100%% to hybrid. "
                "Known phrases %s will be handled by cosine search.",
                self._cfg.known_phrases,
            )
        if not self._hybrid_available:
            logger.warning(
                "SmartRouter: Hybrid engine unavailable — "
                "routing 100%% to ONNX. "
                "Slot intents %s will NOT be resolved.",
                self._cfg.number_slot_intents,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def routing_summary(self) -> Dict[str, Any]:
        """Return a snapshot of the router's current configuration.

        Returns:
            Dict with keys: ``onnx_available``, ``hybrid_available``,
            ``known_phrases``, ``number_slot_intents``,
            ``onnx_threshold``, ``hybrid_threshold``,
            ``cosine_map``, ``fast_path_enabled``, ``total_labels``.
        """
        return {
            "onnx_available": self._onnx_available,
            "hybrid_available": self._hybrid_available,
            "known_phrases": list(self._cfg.known_phrases),
            "number_slot_intents": list(self._cfg.number_slot_intents),
            "onnx_threshold": self._cfg.onnx_confidence_threshold,
            "hybrid_threshold": self._cfg.hybrid_confidence_threshold,
            "cosine_map": {
                "shift": self._cfg.cosine_shift,
                "factor": self._cfg.cosine_factor,
                "formula": "clip(0,1,(cosine-shift)*factor)",
            },
            "fast_path_enabled": (
                self._cfg.prefer_onnx_for_known
                and not self._cfg.run_both_always
            ),
            "total_labels": len(self._label_list),
        }


# ── Convenience factory ───────────────────────────────────────────────────────

def create_router(
    routing_yaml: str | Path = "configs/routing.yaml",
    onnx_engine: Optional[AudioEngine] = None,
    hybrid_engine: Optional[AudioEngine] = None,
) -> SmartRouter:
    """Convenience factory: load config from YAML and return a SmartRouter.

    Both engines are optional — pass ``None`` to disable one side.
    When both are ``None`` the router operates in null mode (every call
    returns ``engine_used="both_unavailable"``).

    Args:
        routing_yaml:  Path to ``configs/routing.yaml``.
        onnx_engine:   Pre-loaded ONNX backend, or ``None``.
        hybrid_engine: Pre-loaded hybrid backend, or ``None``.

    Returns:
        Configured ``SmartRouter`` instance.

    Example::

        from core.engine import OnnxAudioEngine
        from core.hybrid.factory import create_hybrid_engine
        from core.hybrid.config import HybridConfig
        from core.router import create_router

        onnx  = OnnxAudioEngine("artifacts/models/onnx_model")
        hybrid = create_hybrid_engine(
                     HybridConfig.from_yaml("configs/hybrid/model.yaml",
                                            "configs/hybrid/thresholds.yaml"))
        router = create_router("configs/routing.yaml", onnx, hybrid)

        import numpy as np
        result = router.predict(np.zeros(16_000, dtype=np.float32))
        print(result["engine_used"], result["full_label"], result["confidence"])
    """
    cfg = RoutingConfig.from_yaml_or_default(routing_yaml)
    return SmartRouter(cfg, onnx_engine=onnx_engine, hybrid_engine=hybrid_engine)
