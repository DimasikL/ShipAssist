import os

import pytest
import numpy as np
import psutil
import gc
from core.onnx_engine import OnnxEngine
from unittest.mock import MagicMock, patch

@pytest.mark.skipif(not os.path.exists("models/onnx_v1"), reason="Нужна реальная модель для стресс-теста")
def test_memory_leak_inference():
    """
    Стресс-тест: 1000 итераций инференса.
    Проверка, что потребление памяти не растет линейно.
    """
    engine = OnnxEngine(onnx_dir="models/onnx_v1")
    fake_audio = np.random.uniform(-1, 1, 16000).astype(np.float32)

    process = psutil.Process()

    # Разогрев
    for _ in range(10): engine.predict(fake_audio)
    gc.collect()
    mem_start = process.memory_info().rss / (1024 * 1024)

    # Цикл
    for _ in range(1000):
        engine.predict(fake_audio)

    gc.collect()
    mem_end = process.memory_info().rss / (1024 * 1024)

    diff = mem_end - mem_start
    # Допускаем погрешность в 20МБ на фрагментацию, но не сотни МБ
    assert diff < 20, f"Обнаружена утечка памяти: рост на {diff:.2f} MB"