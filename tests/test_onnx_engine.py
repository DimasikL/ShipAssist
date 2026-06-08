import pytest
import numpy as np
import json
import os
from unittest.mock import MagicMock, patch
from core.onnx_engine import OnnxEngine
from core.exceptions import ModelLoadError

@pytest.fixture
def mock_onnx_dir(tmp_path):
    """Создает временную структуру папок для ONNX."""
    onnx_path = tmp_path / "onnx_model"
    onnx_path.mkdir()

    config = {
        "labels": ["cmd1", "cmd2"],
        "sr": 16000,
        "win_samples": 16000,
        "model_int8": "model.onnx"
    }

    with open(onnx_path / "onnx_config.json", "w") as f:
        json.dump(config, f)

    (onnx_path / "model.onnx").write_text("fake binary")
    return str(onnx_path)

@patch('onnxruntime.InferenceSession')
@patch('transformers.Wav2Vec2FeatureExtractor.from_pretrained')
def test_onnx_predict_shape(mock_fe, mock_session, mock_onnx_dir):
    """Happy path: Проверка инференса и формы выходных данных."""
    # Настройка мока сессии
    session_inst = mock_session.return_value
    # Эмулируем logits [1, num_labels]
    session_inst.run.return_value = [np.array([[2.0, 1.0]]), np.array([[0.1]*256])]

    engine = OnnxEngine(onnx_dir=mock_onnx_dir, use_int8=True)

    fake_audio = np.zeros(16000, dtype=np.float32)
    probs, emb = engine.predict(fake_audio)

    assert len(probs) == 2
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0] > probs[1] # Т.к. логгет 2.0 > 1.0

def test_onnx_load_error(tmp_path):
    """Ошибка: Загрузка при отсутствии конфига."""
    with pytest.raises(ModelLoadError):
        OnnxEngine(onnx_dir=str(tmp_path))