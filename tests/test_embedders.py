import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from core.embedders import WTVEmbedder

@patch('transformers.Wav2Vec2Model.from_pretrained')
@patch('transformers.Wav2Vec2Processor.from_pretrained')
@patch('core.embedders.torchaudio.load')
def test_wtv_embedder_output(mock_load, mock_proc, mock_model, tmp_path, raw_audio):
    """Happy path: Проверка размерности эмбеддинга."""
    # Настройка моков
    mock_inst = mock_model.return_value
    mock_inst.to.return_value = mock_inst

    # Эмулируем выход модели (last_hidden_state)
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(1, 50, 768)
    mock_inst.return_value = mock_output

    # Эмулируем torchaudio.load
    mock_load.return_value = (torch.randn(1, 16000), 16000)

    embedder = WTVEmbedder(sr=16000, preproc=None, emb_model="fake", output_hidden_states=False)
    emb = embedder.get_emb(wav_path)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)