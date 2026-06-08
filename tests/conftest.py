import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def test_config_path(tmp_path):
    """Creates a minimal valid config file in a temp directory."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "default.yaml"

    data = {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "window_seconds": 1.0,
            "stride_seconds": 0.5,
            "threshold_db": -50.0,
            "max_duration": 3.0,
            "target_lufs": -20.0,
        },
        "paths": {
            "base_dir": str(tmp_path),
            "models_dir": "models",
            "onnx_model": "models/onnx",
            "checkpoints": "models/checkpoints",
            "logs_dir": "logs",
            "dataset_csv": "data/meta.csv",
        },
        "recognition": {
            "default_confidence": 0.8,
            "fuzzy_threshold": 68.0,
            "debounce_seconds": 1.5,
            "outlier_percentile": 95.0,
            "per_label_thresholds": {"test": 0.9},
        },
        "training": {
            "model_name": "mock-model",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "early_stopping_patience": 5,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "label_smoothing": 0.1,
            "use_ema": True,
            "ema_decay": 0.99,
        },
        "api": {"host": "0.0.0.0", "port": 8000, "max_file_size_mb": 1},
        "logging": {"level": "DEBUG", "max_bytes": 1000, "backup_count": 1},
    }

    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return str(config_file)


@pytest.fixture
def raw_audio() -> np.ndarray:
    """440 Hz sine wave, 1 second at 16 000 Hz — clean test signal."""
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def mock_feature_extractor() -> MagicMock:
    """Mock for Wav2Vec2FeatureExtractor."""
    extractor = MagicMock()
    extractor.return_value = {
        "input_values": torch.randn(1, 16000),
        "attention_mask": torch.ones(1, 16000, dtype=torch.long),
    }
    return extractor
