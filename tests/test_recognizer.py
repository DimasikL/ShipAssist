import pytest
import numpy as np
import time
from core.recognizer import RingBuffer, RealTimeRecognizer

def test_ring_buffer_overflow():
    """Ошибка: Проверка логики при переполнении буфера."""
    rb = RingBuffer(capacity=10)
    # Пишем 15 элементов
    rb.write(np.arange(15).astype(np.float32))

    # Должны получить последние 10 [5, 6...14]
    data = rb.read_last(10)
    assert data[0] == 5
    assert data[-1] == 14

def test_ring_buffer_partial_read():
    """Граничный случай: Чтение больше, чем есть в буфере."""
    rb = RingBuffer(capacity=100)
    rb.write(np.ones(10))
    assert rb.read_last(20) is None

# CHANGED: Вместо pytest.mock.patch используем фикстуру mocker (из pytest-mock)
@pytest.mark.timeout(5)
def test_recognizer_start_stop(mocker):
    """Happy path: Запуск и остановка потоков."""
    rec = RealTimeRecognizer(sample_rate=16000, window_s=1.0, stride_s=0.5)

    # CHANGED: Правильный синтаксис для pytest-mock
    mocker.patch('sounddevice.InputStream')

    rec.start_stream(callback=lambda x: print(x))
    assert rec.is_running is True
    time.sleep(0.2)
    rec.stop()
    assert rec.is_running is False