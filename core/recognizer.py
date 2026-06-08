import threading
import time
import logging
import numpy as np
import sounddevice as sd
from typing import List, Callable, Optional
from core.logger import get_logger
from core.config import settings

# CHANGED: Убрали лишние зависимости, оставили только ядро
logger = get_logger(__name__)

class RingBuffer:
    """
    Потокобезопасный кольцевой буфер для хранения аудио-потока.
    Используется для передачи данных между микрофоном и инференсом.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """Записывает новые сэмплы в буфер."""
        n = len(data)
        with self.lock:
            if n > self.capacity:
                data = data[-self.capacity:]
                n = self.capacity

            end = self.pos + n
            if end <= self.capacity:
                self.buffer[self.pos:end] = data
            else:
                first_part = self.capacity - self.pos
                self.buffer[self.pos:] = data[:first_part]
                self.buffer[:end % self.capacity] = data[first_part:]

            self.pos = (self.pos + n) % self.capacity
            self.total_written += n

    def read_last(self, n: int) -> Optional[np.ndarray]:
        """
        Возвращает последние n сэмплов.
        Возвращает None, если данных в буфере меньше, чем n.
        """
        with self.lock:
            if self.total_written < n:
                return None

            # Ограничиваем чтение размером буфера
            read_n = min(n, self.capacity)

            if self.pos >= read_n:
                return self.buffer[self.pos - read_n:self.pos].copy()
            else:
                return np.concatenate([
                    self.buffer[-(read_n - self.pos):],
                    self.buffer[:self.pos]
                ]).copy()


class RealTimeRecognizer:
    """
    Потоковый распознаватель с логикой автоматического перезапуска микрофона.

    Параметры захвата (размер буфера, blocksize, задержки) читаются из
    ``settings.recognizer`` (configs/base.yaml → секция recognizer).
    Это гарантирует, что изменения в конфиге не требуют правки кода.
    """
    def __init__(self, sample_rate: int, window_s: float, stride_s: float):
        self.sr = sample_rate
        self.win_len = int(window_s * sample_rate)
        self.stride_len = int(stride_s * sample_rate)

        # Ring buffer size sourced from settings.recognizer.ring_buffer_seconds
        # (configs/base.yaml → recognizer.ring_buffer_seconds, default 10.0 s).
        # Must be larger than window_s to avoid stale-data reads.
        _buf_seconds: float = settings.recognizer.ring_buffer_seconds
        self.ring_buffer = RingBuffer(int(sample_rate * _buf_seconds))

        self.stop_event = threading.Event()
        self.is_running = False

    def _audio_cb(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Внутренний колбэк sounddevice для записи звука."""
        if status:
            logger.warning(f"Статус аудио-потока: {status}")
        self.ring_buffer.write(indata[:, 0])

    def start_stream(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Запуск потока захвата звука и цикла инференса.

        Args:
            callback: Функция, которая будет вызываться с аудио-чанком для распознавания.
        """
        self.is_running = True
        self.stop_event.clear()

        # Поток инференса: вырезает окна из буфера и отдает в callback
        # Poll interval sourced from settings.recognizer.inference_poll_interval.
        _poll_interval: float = settings.recognizer.inference_poll_interval

        def inference_loop():
            last_processed_total = 0
            while self.is_running:
                current_total = self.ring_buffer.total_written
                # Проверяем, накопилось ли достаточно данных для нового шага (stride)
                if current_total - last_processed_total >= self.stride_len:
                    audio_window = self.ring_buffer.read_last(self.win_len)
                    if audio_window is not None:
                        try:
                            callback(audio_window)
                        except Exception as e:
                            logger.error(f"Ошибка в функции распознавания: {e}")
                    last_processed_total = current_total
                time.sleep(_poll_interval)

        threading.Thread(target=inference_loop, daemon=True, name="InferenceThread").start()

        # blocksize and reconnect delay sourced from settings.recognizer.
        # (configs/base.yaml → recognizer.blocksize / mic_reconnect_delay).
        _blocksize: int = settings.recognizer.blocksize
        _reconnect_delay: float = settings.recognizer.mic_reconnect_delay

        # Поток захвата микрофона с авто-реконнектом
        def capture_thread():
            while self.is_running:
                try:
                    with sd.InputStream(samplerate=self.sr, channels=1,
                                        callback=self._audio_cb,
                                        blocksize=_blocksize):
                        logger.info("Микрофон успешно подключен.")
                        while self.is_running:
                            time.sleep(0.5)
                except Exception as e:
                    logger.error(
                        f"Ошибка микрофона: {e}. "
                        f"Повторная попытка через {_reconnect_delay} сек..."
                    )
                    time.sleep(_reconnect_delay)

        threading.Thread(target=capture_thread, daemon=True, name="CaptureThread").start()

    def stop(self) -> None:
        """Полная остановка распознавания и очистка ресурсов."""
        self.is_running = False
        self.stop_event.set()
        logger.info("Система распознавания остановлена.")