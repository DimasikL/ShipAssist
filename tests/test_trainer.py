import pytest
import torch
from unittest.mock import MagicMock
from src.train import Trainer

def test_trainer_step(mocker):
    """Один шаг обучения на моках."""
    # Мокаем модель и экстрактор
    mocker.patch('transformers.Wav2Vec2ForSequenceClassification.from_pretrained')
    mocker.patch('transformers.Wav2Vec2FeatureExtractor.from_pretrained')

    trainer = Trainer(label2id={"cmd": 0}, id2label={0: "cmd"})

    # Мокаем выход модели
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(0.5, requires_grad=True)
    trainer.model.return_value = mock_output

    batch = {
        "input_values": torch.randn(2, 16000),
        "attention_mask": torch.ones(2, 16000),
        "labels": torch.tensor([0, 0])
    }

    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
    scheduler = MagicMock()

    loss = trainer.train_epoch([batch], optimizer, scheduler)
    assert loss == 0.5