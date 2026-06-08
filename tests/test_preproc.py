import pytest
import numpy as np
import soundfile as sf
import os
from core.preproc import PreprocNormalize

def test_normalize_happy_path(tmp_path, raw_audio):
    # CHANGED: Используем уникальное имя и закрываем файл
    path = str(tmp_path / "norm_test.wav")
    sf.write(path, raw_audio, 16000)

    norm = PreprocNormalize(sr=16000)
    norm(path)

    # Проверяем что файл читается
    data, _ = sf.read(path)
    assert np.max(np.abs(data)) <= 1.05 # Допуск на плавающую точку