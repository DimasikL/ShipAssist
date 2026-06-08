"""
core/hybrid/factory.py — Factory for the Hybrid C+ engine.

This is the recommended entry-point for any code that wants to instantiate
the hybrid engine. It keeps the caller decoupled from the internal loading
logic of each component and provides a clean, testable boundary.

Usage
-----
Minimal (loads everything from YAML + artefacts automatically):

    from core.hybrid.config import HybridConfig
    from core.hybrid.factory import create_hybrid_engine

    cfg = HybridConfig.from_yaml(
        "configs/hybrid/model.yaml",
        "configs/hybrid/thresholds.yaml",
    )
    engine = create_hybrid_engine(cfg)
    result = engine.predict(audio_array)

Advanced (inject pre-loaded components for testing or hot-reload):

    from core.hybrid.centroid_search import CentroidSearch
    from core.hybrid.outlier_gate import OutlierGate

    gate = OutlierGate.load("artifacts/hybrid/outlier_gate.pkl")
    search = CentroidSearch.load_npz(...)

    engine = create_hybrid_engine(
        cfg,
        outlier_gate=gate,
        centroid_search=search,
    )

This file MUST NOT modify ``core/engine.py``. It only imports the abstract
``AudioEngine`` class (read-only) to satisfy the type system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.hybrid.config import HybridConfig
    from core.hybrid.engine import HybridAudioEngine
    from core.hybrid.outlier_gate import OutlierGate
    from core.hybrid.centroid_search import CentroidSearch

logger = logging.getLogger(__name__)

# Default YAML config paths, relative to the project root
_DEFAULT_MODEL_YAML = Path("configs/hybrid/model.yaml")
_DEFAULT_THRESH_YAML = Path("configs/hybrid/thresholds.yaml")


def create_hybrid_engine(
    cfg: Optional["HybridConfig"] = None,
    *,
    model_yaml: Optional[str | Path] = None,
    thresholds_yaml: Optional[str | Path] = None,
    outlier_gate: Optional["OutlierGate"] = None,
    centroid_search: Optional["CentroidSearch"] = None,
    onnx_engine: Optional[object] = None,
) -> "HybridAudioEngine":
    """Instantiate and return a ready-to-use ``HybridAudioEngine``.

    The factory resolves the configuration in priority order:
      1. ``cfg`` (passed directly).
      2. ``model_yaml`` + ``thresholds_yaml`` (loaded from YAML).
      3. Default YAML paths (``configs/hybrid/model.yaml`` and
         ``configs/hybrid/thresholds.yaml``).

    Any component passed explicitly (``outlier_gate``, ``centroid_search``,
    ``onnx_engine``) bypasses the lazy-loading logic in ``HybridAudioEngine``
    and is used as-is. This is intended for unit tests and hot-reload scenarios.

    Args:
        cfg:             Pre-built ``HybridConfig`` instance. If ``None``,
                         config is loaded from YAML.
        model_yaml:      Path to ``configs/hybrid/model.yaml``. Only used when
                         ``cfg`` is ``None``.
        thresholds_yaml: Path to ``configs/hybrid/thresholds.yaml``. Only used
                         when ``cfg`` is ``None``.
        outlier_gate:    Pre-loaded ``OutlierGate`` — skips loading from disk.
        centroid_search: Pre-loaded ``CentroidSearch`` — skips loading from disk.
        onnx_engine:     Pre-loaded ``OnnxEngine`` instance — skips re-creating
                         the ONNX session (saves ~100ms startup time).

    Returns:
        ``HybridAudioEngine`` instance. If artefact files are missing, the
        engine loads in degraded mode and returns ``{"error": "hybrid_not_loaded"}``
        from ``predict()`` rather than raising.

    Raises:
        FileNotFoundError: If the YAML config files cannot be found and no
                           ``cfg`` was provided.

    Example:
        >>> cfg = HybridConfig.from_yaml("configs/hybrid/model.yaml",
        ...                              "configs/hybrid/thresholds.yaml")
        >>> engine = create_hybrid_engine(cfg)
        >>> import numpy as np
        >>> result = engine.predict(np.zeros(16000, dtype=np.float32))
        >>> print(result["full_label"], result["confidence"])
    """
    from core.hybrid.config import HybridConfig
    from core.hybrid.engine import HybridAudioEngine

    # ── Resolve configuration ──────────────────────────────────────────
    if cfg is None:
        m_yaml = Path(model_yaml) if model_yaml else _DEFAULT_MODEL_YAML
        t_yaml = Path(thresholds_yaml) if thresholds_yaml else _DEFAULT_THRESH_YAML

        if not m_yaml.exists():
            logger.warning(
                "model.yaml not found at %s. Using default HybridConfig values.",
                m_yaml.resolve(),
            )
            cfg = HybridConfig()
        elif not t_yaml.exists():
            logger.warning(
                "thresholds.yaml not found at %s. Using default threshold values.",
                t_yaml.resolve(),
            )
            cfg = HybridConfig()
        else:
            cfg = HybridConfig.from_yaml(m_yaml, t_yaml)

    logger.info(
        "create_hybrid_engine: sr=%d, win_samples=%d, gate=%s",
        cfg.sample_rate,
        cfg.win_samples,
        cfg.outlier_gate.method,
    )

    return HybridAudioEngine(
        cfg=cfg,
        outlier_gate=outlier_gate,
        centroid_search=centroid_search,
        onnx_engine=onnx_engine,
    )


def create_hybrid_engine_from_yaml(
    model_yaml: str | Path = _DEFAULT_MODEL_YAML,
    thresholds_yaml: str | Path = _DEFAULT_THRESH_YAML,
) -> "HybridAudioEngine":
    """Convenience wrapper: load config from YAML files and return engine.

    Args:
        model_yaml:      Path to the model config YAML.
        thresholds_yaml: Path to the thresholds config YAML.

    Returns:
        ``HybridAudioEngine`` instance.
    """
    return create_hybrid_engine(
        model_yaml=model_yaml,
        thresholds_yaml=thresholds_yaml,
    )
