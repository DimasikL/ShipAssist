import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.exceptions import ConfigError

logger = logging.getLogger(__name__)


# -- Audio --------------------------------------------------------------------

class AudioConfig(BaseModel):
    """Audio capture and preprocessing parameters."""
    sample_rate: int = Field(
        16_000, ge=8_000, le=48_000,
        description="Waveform sample rate in Hz. Model was trained at 16 000 Hz.",
    )
    channels: int = Field(
        1, ge=1, le=2,
        description="Number of input channels. 1 = mono (required by Wav2Vec2).",
    )
    window_seconds: float = Field(
        1.0, gt=0.0,
        description="Sliding inference window length in seconds.",
    )
    stride_seconds: float = Field(
        0.5, gt=0.0,
        description="Step between consecutive inference windows in seconds.",
    )
    threshold_db: float = Field(
        -50.0,
        description="Energy gate in dBFS. Frames below this level are skipped.",
    )
    max_duration: float = Field(
        3.0, gt=0.0,
        description="Maximum accepted audio clip length in seconds (API uploads).",
    )
    target_lufs: float = Field(
        -20.0,
        description="Target loudness for normalisation (EBU R128, LUFS).",
    )


# -- Paths --------------------------------------------------------------------

class PathConfig(BaseModel):
    """Filesystem paths. All values are relative to PROJECT_ROOT until absolutize() is called."""
    base_dir: Path = Field(
        Path("."),
        description="Project root anchor. Resolved to an absolute path at load time.",
    )
    artifacts_dir: Path = Field(
        Path("artifacts"),
        description="Top-level directory for all generated artefacts.",
    )
    models_dir: Path = Field(
        Path("artifacts/models"),
        description="Parent directory for model checkpoints and ONNX exports.",
    )
    onnx_model: Path = Field(
        Path("artifacts/models/onnx_model"),
        description="Directory containing onnx_config.json and the .onnx weight file.",
    )
    checkpoints: Path = Field(
        Path("artifacts/models"),
        description="Directory scanned for fine-tuning checkpoints.",
    )
    best_model: Path = Field(
        Path("artifacts/models/best_model"),
        description="Path to the best HuggingFace checkpoint (used by TorchAudioEngine).",
    )
    logs_dir: Path = Field(
        Path("logs"),
        description="Directory for rotating JSON application logs.",
    )
    dataset_csv: Path = Field(
        Path("artifacts/data/dataset.csv"),
        description="Metadata CSV produced by the data-preparation scripts.",
    )

    def absolutize(self, project_root: Path) -> "PathConfig":
        """Return a copy of this config with all relative Path fields resolved."""
        data = self.model_dump()
        for k, v in data.items():
            if isinstance(v, Path) and not v.is_absolute():
                data[k] = (project_root / v).resolve()
        return PathConfig(**data)


# -- Recognition --------------------------------------------------------------

class RecognitionConfig(BaseModel):
    """Confidence thresholds and post-processing parameters."""
    default_confidence: float = Field(
        0.8, ge=0.0, le=1.0,
        description="Fallback minimum softmax confidence to accept a prediction.",
    )
    fuzzy_threshold: float = Field(
        68.0, ge=0.0, le=100.0,
        description="Fuzzy-string similarity score (0-100) for transcript matching.",
    )
    debounce_seconds: float = Field(
        1.5, ge=0.0,
        description="Minimum gap between two accepted detections of the same label.",
    )
    outlier_percentile: float = Field(
        95.0, ge=0.0, le=100.0,
        description="Percentile used to flag outlier confidence values during calibration.",
    )
    noise_log_threshold: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Predictions above this value but below threshold are logged as noise.",
    )
    per_label_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-command confidence overrides. Keys are label strings.",
    )


# -- Model identity -----------------------------------------------------------

class ModelConfig(BaseModel):
    """Identifies the pre-trained model family and the default inference backend."""
    name: str = Field(
        "",
        description=(
            "HuggingFace model identifier (e.g. 'jonatasgrosman/wav2vec2-large-xlsr-53-russian'). "
            "Empty string means the value is sourced from training.model_name instead."
        ),
    )
    type: str = Field(
        "onnx",
        description="Default inference backend: 'onnx' (production) or 'torch' (development).",
    )

    @field_validator("type")
    @classmethod
    def type_must_be_known(cls, v: str) -> str:
        allowed = {"onnx", "torch"}
        if v.lower() not in allowed:
            raise ValueError(f"model.type must be one of {allowed}, got '{v}'")
        return v.lower()


# -- Benchmark ----------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """Parameters for the defense demo benchmark runner (scripts/demo_defense.py)."""
    samples: int = Field(
        10, ge=1,
        description="Number of synthetic inference cycles per benchmark run.",
    )


# -- Base runtime -------------------------------------------------------------

class BaseRuntimeConfig(BaseModel):
    device: str = Field(
        "auto",
        description="Compute device: 'auto' (prefer CUDA), 'cpu', or 'cuda'.",
    )


# -- ONNX ---------------------------------------------------------------------

class OnnxConfig(BaseModel):
    """ONNX Runtime backend settings."""
    use_int8: bool = Field(
        True,
        description=(
            "DEPRECATED -- kept for backwards compatibility. Prefer 'precision'. "
            "When 'precision' is unset, True maps to int8, False maps to fp32."
        ),
    )
    precision: str = Field(
        "int8",
        description="ONNX weight precision: 'int8' | 'fp32' | 'fp16'.",
    )
    providers: List[str] = Field(
        default_factory=lambda: ["CPUExecutionProvider"],
        description="ONNX Runtime execution providers in priority order.",
    )
    temperature: float = Field(
        1.0, gt=0.0,
        description=(
            "Temperature scaling factor applied as logits/=T before softmax. "
            "Use values > 1.0 to soften INT8-flattened distributions. "
            "1.0 disables scaling."
        ),
    )
    auto_temperature: bool = Field(
        False,
        description=(
            "If True, the engine estimates a temperature from the first N "
            "windows of the session and overrides 'temperature'."
        ),
    )
    adaptive_threshold: bool = Field(
        True,
        description=(
            "If True, when median(confidence) over a rolling window is "
            "below default_confidence * 0.8, log a warning and lower the "
            "effective threshold by 15% temporarily."
        ),
    )
    adaptive_window: int = Field(
        20, ge=1,
        description="Number of recent predictions used to compute median confidence.",
    )
    recalibrate: bool = Field(
        False,
        description=(
            "If True, log instructions for regenerating the INT8 calibration set. "
            "Does NOT re-quantise automatically."
        ),
    )

    @field_validator("precision")
    @classmethod
    def precision_must_be_known(cls, v: str) -> str:
        allowed = {"int8", "fp32", "fp16"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"onnx.precision must be one of {allowed}, got '{v}'")
        return lower


# -- Data splits --------------------------------------------------------------

class SplitsConfig(BaseModel):
    """Speaker-disjoint split configuration consumed by scripts/verify_splits.py.

    Group names must match the ``audio_group`` column values in the dataset CSV.
    verify_splits.py asserts that the three sets are disjoint at both the group
    and speaker-ID levels.
    """

    train_groups: List[str] = Field(
        default_factory=lambda: [
            "test user 1", "test user 2",
            "drug slova 2", "drug slova",
            "train user 5", "train user 6",
            "train user 7", "drug slova-hardneg1",
            "train user 1", "drug slova1",
            "test user 3", "drug slova 3",
        ],
        description="Audio groups assigned to the training split.",
    )
    val_groups: List[str] = Field(
        default_factory=lambda: ["train user 3", "drug slova3"],
        description="Audio groups assigned to the validation split.",
    )
    test_groups: List[str] = Field(
        default_factory=lambda: [
            "train user 2", "drug slova 2",
            "train user 2 new", "drug slova2-new",
            "train user 4", "drug slova4",
        ],
        description="Audio groups assigned to the test split.",
    )


# -- Augmentation -------------------------------------------------------------

class MaritimeNoiseAugConfig(BaseModel):
    """Controls the maritime / pink-noise surrogate augmentation."""

    enabled: bool = Field(
        True,
        description="Enable maritime noise augmentation during training.",
    )
    probability: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Per-sample probability of applying maritime noise (training only).",
    )
    target_snr_db: float = Field(
        15.0,
        description="Target signal-to-noise ratio in dB when mixing noise.",
    )


class AugmentationConfig(BaseModel):
    """Top-level augmentation settings. New aug types nest here."""

    maritime_noise: MaritimeNoiseAugConfig = Field(
        default_factory=MaritimeNoiseAugConfig,
        description="Maritime / pink-noise surrogate augmentation parameters.",
    )


# -- Monitor ------------------------------------------------------------------

class MonitorConfig(BaseModel):
    """Resource monitor settings consumed by core/monitor.py."""
    interval_seconds: int = Field(
        60, ge=1,
        description="Seconds between resource usage log entries.",
    )
    ram_alert_mb: float = Field(
        2000.0, gt=0.0,
        description="RSS threshold in MB above which a RAM alert is logged.",
    )


# -- LoRA ---------------------------------------------------------------------

class LoRAConfig(BaseModel):
    enabled: bool = Field(True, description="Enable LoRA adapters during fine-tuning.")
    r: int = Field(32, ge=1, description="LoRA rank.")
    alpha: int = Field(64, ge=1, description="LoRA scaling factor (alpha). Scale = alpha / r.")


# -- Inference (realtime) -----------------------------------------------------

class InferenceConfig(BaseModel):
    buffer_seconds: float = Field(6.0, gt=0.0, description="Ring-buffer size in seconds.")
    overlap: float = Field(0.5, ge=0.0, le=1.0, description="Window overlap fraction.")
    realtime_threshold: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Minimum confidence for a real-time detection event.",
    )


# -- Recognizer (live capture loop) ------------------------------------------

class RecognizerConfig(BaseModel):
    """Parameters for the RealTimeRecognizer audio capture and inference loop.

    These values control the ring buffer size, the sounddevice block size used
    for low-latency capture, and the retry / poll timings that govern
    microphone reconnect behaviour. All values are consumed exclusively by
    ``core/recognizer.py`` so they can be tuned without touching source code.
    """

    ring_buffer_seconds: float = Field(
        10.0, gt=0.0,
        description=(
            "Duration of the ring buffer in seconds. "
            "Must be larger than audio.window_seconds; 10 s is a safe default "
            "that absorbs mic latency spikes without excessive memory use."
        ),
    )
    blocksize: int = Field(
        2048, ge=64,
        description=(
            "sounddevice InputStream block size in samples. "
            "Smaller values reduce latency; larger values reduce CPU overhead. "
            "2048 samples ≈ 128 ms at 16 kHz — a good default for recognition."
        ),
    )
    mic_reconnect_delay: float = Field(
        2.0, ge=0.0,
        description=(
            "Seconds to wait before attempting to reconnect the microphone "
            "after an InputStream error. Set to 0 for immediate retry."
        ),
    )
    inference_poll_interval: float = Field(
        0.1, ge=0.001,
        description=(
            "Seconds between inference loop iteration polls. "
            "Lower values improve detection latency at the cost of CPU usage."
        ),
    )


# -- Training -----------------------------------------------------------------

class TrainingConfig(BaseModel):
    """Fine-tuning hyperparameters."""
    model_name: str = Field(
        "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        description="HuggingFace model identifier for the base model.",
    )
    batch_size: int = Field(16, ge=1, description="Per-device training batch size.")
    epochs: int = Field(50, ge=1, description="Maximum training epochs.")
    learning_rate: float = Field(2e-5, gt=0.0, description="Peak AdamW learning rate.")
    weight_decay: float = Field(1e-4, ge=0.0, description="L2 regularisation coefficient.")
    warmup_ratio: float = Field(
        0.05, ge=0.0, le=1.0,
        description="Fraction of total steps used for LR linear warm-up.",
    )
    early_stopping_patience: int = Field(
        10, ge=1,
        description="Eval steps without improvement before stopping.",
    )
    use_lora: bool = Field(True, description="Enable LoRA adapters instead of full fine-tuning.")
    lora_r: int = Field(32, ge=1, description="LoRA rank.")
    lora_alpha: int = Field(64, ge=1, description="LoRA scaling factor.")
    label_smoothing: float = Field(
        0.1, ge=0.0, le=1.0,
        description="Label-smoothing epsilon for cross-entropy loss.",
    )
    use_ema: bool = Field(True, description="Maintain EMA weights during training.")
    ema_decay: float = Field(
        0.999, gt=0.0, le=1.0,
        description="EMA decay coefficient. Higher = slower adaptation.",
    )


# -- API ----------------------------------------------------------------------

class ApiConfig(BaseModel):
    """FastAPI server settings."""
    model_config = ConfigDict(extra="ignore")

    host: str = Field(
        "127.0.0.1",
        description="Bind address. Use '0.0.0.0' to accept external connections.",
    )
    port: int = Field(
        8000, ge=1, le=65535,
        description="TCP port the FastAPI server listens on.",
    )
    max_file_size_mb: int = Field(
        5, ge=1,
        description="Maximum accepted audio upload size in megabytes.",
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost", "http://localhost:8000"],
        description=(
            "List of allowed CORS origins for CORSMiddleware. "
            "Use ['*'] for development; restrict to specific domains in production."
        ),
    )


# -- Logging ------------------------------------------------------------------

class LogConfig(BaseModel):
    level: str = Field(
        "INFO",
        description="Root log level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    max_bytes: int = Field(
        10_485_760, ge=1,
        description="Maximum size of a single log file before rotation (bytes).",
    )
    backup_count: int = Field(
        5, ge=0,
        description="Number of rotated log files to retain.",
    )

    @field_validator("level")
    @classmethod
    def level_must_be_valid(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"logging.level must be one of {allowed}, got '{v}'")
        return upper


# -- Settings (root) ----------------------------------------------------------

class Settings(BaseSettings):
    """
    Root configuration object. Populated by load_config() via a 3-YAML deep-merge,
    with individual field defaults as the final fallback tier.

    Override any value at runtime via environment variables:
        SHIP_<SECTION>__<KEY>=value
    Example: SHIP_MODEL__TYPE=torch  SHIP_API__PORT=9000
    """
    model_config = SettingsConfigDict(
        env_prefix="SHIP_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        extra="ignore",
    )

    audio: AudioConfig = Field(default_factory=AudioConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LogConfig = Field(default_factory=LogConfig)
    base: BaseRuntimeConfig = Field(default_factory=BaseRuntimeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    onnx: OnnxConfig = Field(default_factory=OnnxConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    recognizer: RecognizerConfig = Field(default_factory=RecognizerConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)

    @classmethod
    def load(cls, path: str = "configs/default.yaml") -> "Settings":
        """Legacy single-file loader for backwards compatibility."""
        if not os.path.exists(path):
            raise ConfigError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)


# -- Helpers ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict *b* into dict *a*. Values in *b* win."""
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def validate_runtime_paths(s: "Settings") -> None:
    """Assert that the two paths required for any inference run exist on disk.

    Raises:
        ConfigError: if onnx_model directory or logs_dir is missing.
    """
    if not s.paths.onnx_model.exists():
        raise ConfigError(
            f"ONNX model directory not found: {s.paths.onnx_model}\n"
            "Run scripts/train/main_export_to_onnx.py or copy a pre-built bundle."
        )
    if not s.paths.logs_dir.exists():
        raise ConfigError(
            f"Logs directory not found: {s.paths.logs_dir}\n"
            "Create it with: mkdir -p logs"
        )


def load_config(
    base_path: str | Path = "configs/base.yaml",
    model_path: str | Path = "configs/model.yaml",
    inference_path: str | Path = "configs/inference.yaml",
) -> Settings:
    """Build a Settings object from three YAML files plus environment variables.

    Merge order (later wins):
        1. Field defaults defined in each config class
        2. configs/base.yaml      -- paths, logging, splits, augmentation
        3. configs/model.yaml     -- model identity, recognition thresholds, ONNX flags
        4. configs/inference.yaml -- audio parameters
        5. SHIP_* environment variables (handled by pydantic-settings)

    All path values in YAML must be relative to PROJECT_ROOT; absolutize() is
    called automatically. Path existence is NOT validated here -- call
    validate_runtime_paths(settings) at application startup instead.
    """
    def _read_yaml(p: str | Path) -> Dict[str, Any]:
        p = Path(p)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Invalid YAML (expected a mapping) in {p}")
        return data

    merged: Dict[str, Any] = {}
    merged = _deep_merge(merged, _read_yaml(base_path))
    merged = _deep_merge(merged, _read_yaml(model_path))
    merged = _deep_merge(merged, _read_yaml(inference_path))

    # Compatibility shim: model.name -> training.model_name
    if "model" in merged and isinstance(merged["model"], dict) and merged["model"].get("name"):
        merged.setdefault("training", {})
        merged["training"].setdefault("model_name", merged["model"]["name"])

    settings_obj = Settings(**merged)

    # Resolve all relative Path fields to absolute paths anchored at PROJECT_ROOT
    settings_obj.paths = settings_obj.paths.absolutize(PROJECT_ROOT)

    return settings_obj


def get_settings() -> Settings:
    """Load settings using the following priority order:
      1. Canonical 3-YAML set  (configs/base.yaml + model.yaml + inference.yaml)
      2. Legacy single file     (configs/default.yaml) -- backwards compatibility
      3. Pure defaults          -- last resort, logs a warning
    """
    # --- attempt 1: canonical 3-YAML ---
    try:
        return load_config(
            os.getenv("SHIP_BASE_CONFIG", "configs/base.yaml"),
            os.getenv("SHIP_MODEL_CONFIG", "configs/model.yaml"),
            os.getenv("SHIP_INFERENCE_CONFIG", "configs/inference.yaml"),
        )
    except ConfigError as exc:
        logger.debug("Canonical config unavailable (%s), trying legacy path.", exc)

    # --- attempt 2: legacy single-file ---
    for p in [
        os.getenv("SHIP_CONFIG_PATH", "configs/default.yaml"),
        "configs/default.yaml",
        "../configs/default.yaml",
    ]:
        if os.path.exists(p):
            try:
                return Settings.load(p)
            except Exception as exc:
                logger.debug("Legacy config %s failed: %s", p, exc)

    # --- attempt 3: pure defaults (no YAML at all) ---
    logger.warning(
        "No config file found. Running with built-in defaults. "
        "Paths will be relative; call validate_runtime_paths() before inference."
    )
    return Settings()


settings = get_settings()
