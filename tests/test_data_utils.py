import pytest
import torch
import pandas as pd
from src.data_utils import CommandDataset
from unittest.mock import MagicMock

def test_dataset_item_shape(mocker):
    """Исправленный тест датасета с корректным возвратом мока."""
    df = pd.DataFrame({
        "audio_path": ["fake.wav"],
        "class": ["cmd"]
    })

    # 1. Мокаем загрузку аудио (Tensor + SR)
    mocker.patch('torchaudio.load', return_value=(torch.randn(1, 16000), 16000))

    # 2. Настраиваем мок для feature_extractor
    # Он должен возвращать объект, который можно вызвать как результат['input_values']
    mock_output = MagicMock()
    mock_output.input_values = torch.randn(1, 16000)
    mock_output.attention_mask = torch.ones(1, 16000)

    mock_fe = MagicMock()
    mock_fe.return_value = mock_output

    # 3. Инициализируем датасет
    ds = CommandDataset(df, mock_fe, {"cmd": 0})

    item = ds[0]
    assert "input_values" in item
    assert isinstance(item["input_values"], torch.Tensor)