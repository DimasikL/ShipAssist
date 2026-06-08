"""
core/hybrid/config.py — Pydantic configuration for the Hybrid C+ engine.

This module is intentionally self-contained. It does NOT extend or import
``core.config.Settings`` so it can be loaded in isolation for unit tests
and for the training scripts in ``scripts/hybrid/`` without bootstrapping
the full application config stack.

Usage
-----
    from core.hybrid.config import HybridConfig

    # Load from both YAML files:
    cfg = HybridConfig.from_yaml(
        model_yaml="configs/hybrid/model.yaml",
        thresholds_yaml="configs/hybrid/thresholds.yaml",
    )

    # Or use defaults for a dry-run / test:
    cfg = HybridConfig()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


# ── Sub-configs ───────────────────────────────────────────────────────────────

class HybridPathConfig(BaseModel):
    """Filesystem paths for hybrid model artefacts.

    All paths are relative to the project root until ``absolutize()`` is called.
    """

    centroids: Path = Field(
        Path("artifacts/hybrid/centroids.npy"),
        description=(
            "Numpy .npy file of shape (N_labels, D) — one L2-normalised "
            "centroid vector per registered phrase."
        ),
    )
    centroid_labels: Path = Field(
        Path("artifacts/hybrid/centroid_labels.json"),
        description=(
            "JSON list of N_labels strings mapping row index → phrase label. "
            "Must be in the same order as rows in centroids.npy."
        ),
    )
    outlier_gate: Path = Field(
        Path("artifacts/hybrid/outlier_gate.pkl"),
        description="Pickled ``OutlierGate`` instance (fitted).",
    )
    number_regressors_dir: Path = Field(
        Path("artifacts/hybrid/regressors"),
        description=(
            "Directory containing one ``<intent_key>.pkl`` file per slot intent. "
            "The intent_key is the URL-safe label with spaces replaced by underscores."
        ),
    )

    def absolutize(self, root: Path) -> "HybridPathConfig":
        """Return a copy with all relative paths resolved against *root*.

        Args:
            root: Project root directory (typically ``Path(__file__).parent.parent.parent``).

        Returns:
            New ``HybridPathConfig`` with absolute ``Path`` values.
        """
        data: Dict[str, Path] = {}
        for k, v in self.model_dump().items():
            p = Path(v)
            data[k] = (root / p).resolve() if not p.is_absolute() else p
        return HybridPathConfig(**data)


class SnrAdaptiveConfig(BaseModel):
    """Online-SNR-based dynamic gate threshold adjustment (§2.3).

    At inference time an SNR estimate is computed from the raw waveform using
    min/max spectral statistics (WADA-SNR approximation). The gate threshold
    is then raised by ``beta * max(0, snr_ref_db - snr)`` dB, making the gate
    *stricter* under heavy noise (where FP risk is highest) and leaving it
    unchanged in clean conditions (preserving recall).

    Formula:  τ_adaptive = τ_0 + β · max(0, SNR_ref − SNR_online)

    Calibration: sweep ``beta`` on the validation set while holding
    ``snr_ref_db`` fixed at 12 dB.  A good starting search grid is
    β ∈ {0.05, 0.10, 0.15, 0.20}.
    """

    enabled: bool = Field(
        False,
        description=(
            "Enable online-SNR adaptive threshold adjustment. Set True after "
            "calibrating beta on the validation set."
        ),
    )
    snr_ref_db: float = Field(
        12.0,
        description=(
            "Reference SNR level in dB. Gate tightens for inputs with SNR "
            "below this value. 12 dB corresponds to mild background noise."
        ),
    )
    beta: float = Field(
        0.15,
        ge=0.0,
        description=(
            "Sensitivity coefficient controlling how aggressively the threshold "
            "rises with falling SNR. Calibrate via ablation on validation set."
        ),
    )


class OutlierGateConfig(BaseModel):
    """Parameters for the OOD early-rejection gate.

    Supports three modes controlled by ``method``:
      - ``'mahalanobis'`` / ``'cosine'`` / ``'l2'``: single-metric
        ``OutlierGate``.
      - ``'ensemble'``: ``EnsembleOutlierGate`` — combines all three metrics
        with z-score normalization + optional adaptive per-class τ. This is
        **Scenario 1** from the research plan and is the recommended setting
        for production.
    """

    method: str = Field(
        "mahalanobis",
        description=(
            "Distance metric used to measure how far an embedding is from "
            "the known class clusters. One of: 'mahalanobis', 'cosine', 'l2', "
            "or 'ensemble' (Scenario 1: combines all three with adaptive τ)."
        ),
    )
    percentile: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description=(
            "During gate fitting, the threshold is set to this percentile of "
            "the training-set distance distribution. Raise to accept more "
            "inputs; lower to reject more aggressively."
        ),
    )
    fallback_threshold: float = Field(
        8.0,
        ge=0.0,
        description=(
            "Hard-coded threshold used when no calibrated value is stored in "
            "the fitted gate (e.g., gate not yet trained). Higher = more permissive."
        ),
    )
    regularization_eps: float = Field(
        1e-4,
        ge=0.0,
        description=(
            "Epsilon added to the diagonal of the covariance matrix before "
            "inversion. Prevents singularity with high-dimensional (1024-D) "
            "Wav2Vec2 embeddings when training-set size is small."
        ),
    )
    enabled: bool = Field(
        True,
        description=(
            "Set to False to bypass the gate entirely. Useful for ablation "
            "experiments comparing accuracy with and without the gate."
        ),
    )

    # ── Adaptive τ (applies to both OutlierGate and EnsembleOutlierGate) ──────

    use_adaptive_tau: bool = Field(
        False,
        description=(
            "If True, use per-class calibrated thresholds at inference instead "
            "of a single global threshold. Compact class clusters get tighter "
            "rejection; sparse clusters get looser rejection. Recommended when "
            "method='ensemble'. No retraining required — thresholds are derived "
            "analytically from the embedding data during fit()."
        ),
    )
    per_class_percentile: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description=(
            "Percentile used for per-class threshold calibration when "
            "use_adaptive_tau=True. Defaults to ``percentile`` when None."
        ),
    )

    # ── Ensemble-specific settings (method='ensemble') ────────────────────────

    ensemble_weights: List[float] = Field(
        default_factory=lambda: [2.0, 1.0, 1.0],
        description=(
            "Weights for [mahalanobis, cosine, l2] sub-gates in the ensemble "
            "OOD score. Automatically normalized to sum to 1. Default (2,1,1) "
            "gives Mahalanobis double weight — justified because it accounts "
            "for the correlated structure of Wav2Vec2 embeddings."
        ),
    )

    # ── Online-SNR adaptive threshold (§2.3) ─────────────────────────────────

    snr_adaptive: SnrAdaptiveConfig = Field(
        default_factory=SnrAdaptiveConfig,
        description=(
            "Online-SNR-based dynamic threshold adjustment. "
            "See SnrAdaptiveConfig for full documentation."
        ),
    )


class CentroidSearchConfig(BaseModel):
    """Parameters for cosine centroid nearest-neighbour search."""

    min_cosine_similarity: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Global minimum cosine similarity required to accept a prediction. "
            "Inputs where all centroids score below this value are rejected."
        ),
    )
    per_label_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-phrase cosine overrides. Keys must be exact label strings. "
            "Takes precedence over min_cosine_similarity for the matching label."
        ),
    )


class NumberRegressorConfig(BaseModel):
    """Configuration for the numeric slot-filling regressor."""

    slot_intents: List[str] = Field(
        default_factory=list,
        description=(
            "Exact label strings that trigger number slot-filling. When the "
            "centroid search returns one of these labels, the number regressor "
            "is invoked to predict the numeric value embedded in the utterance."
        ),
    )
    bounds: Dict[str, List[float]] = Field(
        default_factory=dict,
        description=(
            "Maps slot intent label → [min_value, max_value]. The regressor "
            "output is clipped to this range before returning. Example: "
            "{'курс УГОЛ градусов': [1.0, 360.0]}."
        ),
    )


class CTCDecoderConfig(BaseModel):
    """Configuration for the CTC digit head (Variant B slot-fill).

    When ``enabled=True`` and the ONNX bundle exposes ``projected_frames``
    (``has_frames=True`` in ``onnx_config.json``), the pipeline tries CTC
    decoding before falling back to the MLP ``NumberRegressor``.
    """

    enabled: bool = Field(
        False,
        description=(
            "Enable CTC digit decoding (Variant B). Set False (default) to keep only "
            "the MLP NumberRegressor (Variant A) for all slot intents. "
            "Set True only after training the CTC head via "
            "scripts/hybrid/train_ctc_head.py — the head checkpoint must exist "
            "at head_path before enabling."
        ),
    )
    head_path: Path = Field(
        Path("artifacts/hybrid/ctc_digit_head.pt"),
        description=(
            "Path to the trained CTC head checkpoint (.pt). "
            "Created by: python scripts/hybrid/train_ctc_head.py"
        ),
    )
    ctc_intents: List[str] = Field(
        default_factory=list,
        description=(
            "Slot intents that use CTC decoding. Intents not in this list "
            "always use the MLP regressor. Recommended: wide-range intents "
            "(курс 0–359°, скорость 0–30 kts). Narrow bounded ones (поворот) "
            "can stay on the regressor where training data is sparse."
        ),
    )
    min_confidence: float = Field(
        0.60,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum CTC confidence score to accept a decode. "
            "Below this threshold the pipeline falls back to NumberRegressor."
        ),
    )


class EmbedderConfig(BaseModel):
    """Controls how the embedding vector is obtained during inference."""

    use_onnx_embeddings: bool = Field(
        True,
        description=(
            "If True (recommended), the embedding is taken from ``outputs[1]`` "
            "of the existing OnnxEngine session — no additional model call needed. "
            "If False, a standalone WTVEmbedder (full PyTorch forward pass) is used."
        ),
    )
    onnx_model_dir: Path = Field(
        Path("onnx_model/models/run_2026-02-25_19-07-15/best_model"),
        description=(
            "Directory containing ``onnx_config.json`` and the ONNX weight file. "
            "Used when ``use_onnx_embeddings=True``."
        ),
    )
    hf_model_name: str = Field(
        "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        description=(
            "HuggingFace model identifier for the standalone WTVEmbedder. "
            "Only used when ``use_onnx_embeddings=False``."
        ),
    )


# ── Root config ───────────────────────────────────────────────────────────────

class HybridConfig(BaseModel):
    """Root configuration object for the Hybrid C+ engine.

    Intentionally does NOT extend ``core.config.Settings`` — the isolation
    means this config can be loaded without triggering the full app config
    validation (useful in training scripts and unit tests).

    Example:
        >>> cfg = HybridConfig.from_yaml(
        ...     "configs/hybrid/model.yaml",
        ...     "configs/hybrid/thresholds.yaml",
        ... )
        >>> cfg.outlier_gate.enabled
        True
    """

    paths: HybridPathConfig = Field(default_factory=HybridPathConfig)
    outlier_gate: OutlierGateConfig = Field(default_factory=OutlierGateConfig)
    centroid_search: CentroidSearchConfig = Field(default_factory=CentroidSearchConfig)
    number_regressor: NumberRegressorConfig = Field(default_factory=NumberRegressorConfig)
    ctc_decoder: CTCDecoderConfig = Field(default_factory=CTCDecoderConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    sample_rate: int = Field(
        16_000,
        ge=8_000,
        le=48_000,
        description="Waveform sample rate in Hz. Must match the ONNX model.",
    )
    win_samples: int = Field(
        16_000,
        ge=1_000,
        description=(
            "Canonical inference window length in samples "
            "(e.g. 16_000 = 1 second at 16 kHz)."
        ),
    )

    @classmethod
    def from_yaml(
        cls,
        model_yaml: str | Path,
        thresholds_yaml: str | Path,
        project_root: Optional[Path] = None,
    ) -> "HybridConfig":
        """Merge ``model.yaml`` + ``thresholds.yaml`` into a validated config.

        Args:
            model_yaml:      Path to ``configs/hybrid/model.yaml``.
            thresholds_yaml: Path to ``configs/hybrid/thresholds.yaml``.
            project_root:    Resolve relative artefact paths against this directory.
                             Defaults to the project root (3 levels up from this file).

        Returns:
            Validated and path-resolved ``HybridConfig`` instance.

        Raises:
            FileNotFoundError: If either YAML file does not exist.
            pydantic.ValidationError: If field values fail validation.
        """
        model_path = Path(model_yaml)
        thresh_path = Path(thresholds_yaml)

        if not model_path.exists():
            raise FileNotFoundError(f"model.yaml not found: {model_path.resolve()}")
        if not thresh_path.exists():
            raise FileNotFoundError(f"thresholds.yaml not found: {thresh_path.resolve()}")

        with open(model_path, "r", encoding="utf-8") as f:
            model_data: dict = yaml.safe_load(f) or {}
        with open(thresh_path, "r", encoding="utf-8") as f:
            thresh_data: dict = yaml.safe_load(f) or {}

        # Threshold sections override / merge on top of model data.
        # ctc_decoder lives in model.yaml; number_regressor thresholds are in
        # thresholds.yaml — each can override the other's defaults independently.
        merged: dict = dict(model_data)
        for section in ("outlier_gate", "centroid_search", "number_regressor"):
            if section in thresh_data:
                merged[section] = thresh_data[section]

        cfg = cls.model_validate(merged)

        if project_root is None:
            # core/hybrid/config.py → 3 levels up → project root
            project_root = Path(__file__).resolve().parent.parent.parent

        cfg.paths = cfg.paths.absolutize(project_root)

        # Resolve ctc_decoder.head_path against project root
        h = cfg.ctc_decoder.head_path
        if not Path(h).is_absolute():
            cfg.ctc_decoder.head_path = (project_root / h).resolve()

        return cfg
