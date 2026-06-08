import os
import sys
import time
import json
import collections
import threading
import queue

import torch
import torchaudio
import numpy as np
import pyaudio

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# =====================================================================
#  RealTimeRecognizer с интеграцией Silero VAD
# =====================================================================
class RealTimeRecognizerVAD:
    def __init__(
            self,
            model_dir,
            labels,
            window_s=3.0,       # Длина окна для классификатора (3 сек)
            stride_s=0.5,       # Сдвиг окна (чем меньше, тем чаще проверки, но выше нагрузка)
            sample_rate=16000,
            device=None,
            conf_th=0.9,        # Общий порог уверенности
            conf_th_per_label=None,
            debounce_s=1.0,     # Задержка между повторными срабатываниями
            vad_threshold=0.5   # Порог VAD (вероятность речи > 0.5)
    ):
        self.labels = labels
        self.window_len = int(window_s * sample_rate)
        self.stride_len = int(stride_s * sample_rate)
        self.sr = sample_rate
        self.debounce_samples = int(debounce_s * sample_rate)
        self.vad_threshold = vad_threshold

        self.conf_th_per_label = conf_th_per_label or {}
        self.default_conf_th = conf_th

        # --- Device ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Device: {self.device}")

        # --- Загрузка Wav2Vec2 (Классификатор) ---
        print(f"Loading Wav2Vec2 from {model_dir}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # --- Загрузка Silero VAD ---
        print("Loading Silero VAD...")
        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False  # Используем PyTorch версию (можно True для ускорения)
            )
            self.vad_model.to(self.device)
            self.use_vad = True
        except Exception as e:
            print(f"⚠️ Failed to load Silero VAD: {e}")
            print("Running WITHOUT VAD (prone to false positives).")
            self.use_vad = False

        # --- Buffer ---
        # Буфер хранит аудио. Мы пишем в конец, читаем окно с конца.
        self.buffer = collections.deque(maxlen=self.window_len * 2)
        # Заполняем нулями на старте
        self.buffer.extend([0.0] * self.window_len)

        self.running = False
        self.audio_thread = None
        self.proc_thread = None

        # Очередь для передачи данных из аудио-потока в поток обработки
        self.chunk_queue = queue.Queue()
        self.samples_since_last_detect = self.debounce_samples

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback для PyAudio"""
        if not self.running:
            return None, pyaudio.paComplete

        # Конвертация байт -> float32
        data_np = np.frombuffer(in_data, dtype=np.float32)
        self.chunk_queue.put(data_np)
        return in_data, pyaudio.paContinue

    def _processing_loop(self, callback):
        """Поток обработки: VAD -> Wav2Vec2"""
        # Накопитель для VAD (Silero требует чанки по 512, 1024 и т.д.)
        vad_buffer = []

        while self.running:
            try:
                chunk = self.chunk_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Добавляем в основной буфер
            self.buffer.extend(chunk)
            self.samples_since_last_detect += len(chunk)

            # Проверка VAD (только если набралось достаточно данных для проверки)
            # Берем последнее окно для классификации
            if len(self.buffer) >= self.window_len:
                # Берем последние window_s секунд
                window_data = list(self.buffer)[-self.window_len:]
                tensor_wav = torch.tensor(window_data, dtype=torch.float32)

                # --- 1. ПРОВЕРКА VAD ---
                is_speech = True
                # --- 1. ПРОВЕРКА VAD ---
            is_speech = True

            if self.use_vad:
                with torch.no_grad():

                    frame_size = 512  # для 16kHz
                    check_len = int(1.0 * self.sr)  # проверяем последнюю 1 сек
                    segment = tensor_wav[-check_len:]

                    # Разбиваем на фреймы по 512
                    speech_probs = []

                    for i in range(0, len(segment) - frame_size, frame_size):
                        frame = segment[i:i+frame_size]
                        frame = frame.unsqueeze(0).to(self.device)

                        prob = self.vad_model(frame, self.sr).item()
                        speech_probs.append(prob)

                    if len(speech_probs) == 0:
                        is_speech = False
                    else:
                        mean_prob = np.mean(speech_probs)
                        is_speech = mean_prob > self.vad_threshold

                # --- 2. КЛАССИФИКАЦИЯ ---
                # Запускаем Wav2Vec только если VAD сказал "Речь"
                # ИЛИ если прошло достаточно времени с прошлого запуска (stride)
                # Тут упростим: запускаем каждые stride_samples, но если VAD=False, то игнорим результат

                # Реализуем "прореживание" (stride) через ожидание накопления чанков
                # (В текущей реализации мы проверяем каждый чанк, это дорого.
                #  Лучше накапливать stride в chunk_queue. Но для простоты оставим так)

                if is_speech and (self.samples_since_last_detect > self.debounce_samples):

                    with torch.no_grad():
                        inputs = self.feature_extractor(
                            tensor_wav.numpy(),
                            sampling_rate=self.sr,
                            return_tensors="pt",
                            padding=True
                        )
                        input_values = inputs.input_values.to(self.device)

                        logits = self.model(input_values).logits
                        probs = torch.softmax(logits, dim=-1)[0]

                        score, pred_idx = torch.max(probs, dim=-1)
                        label = self.labels[pred_idx.item()]
                        prob = score.item()

                        # --- Пороги ---
                        threshold = self.conf_th_per_label.get(label, self.default_conf_th)

                        # Фильтр "другие слова" — мы не хотим их детектить как событие
                        if label == "другие слова":
                            # sys.stdout.write(f"\rOther words ({prob:.2f})   ")
                            pass
                        elif prob >= threshold:
                            # УСПЕХ!
                            event = {
                                "label": label,
                                "prob": prob,
                                "time": time.time()
                            }
                            callback(event)
                            self.samples_since_last_detect = 0 # Сброс дебаунса

                            # Очистка буфера (чтобы не сработало дважды на одной фразе)
                            # self.buffer.extend([0.0] * int(self.window_len / 2))

    def start_stream(self, callback_on_detection):
        self.running = True

        # PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=2048, # Размер чанка
            stream_callback=self._audio_callback
        )

        print("🎤 Listening... (Press Ctrl+C to stop)")

        self.proc_thread = threading.Thread(
            target=self._processing_loop,
            args=(callback_on_detection,)
        )
        self.proc_thread.start()

        try:
            while stream.is_active():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            p.terminate()
            if self.proc_thread:
                self.proc_thread.join()

# =====================================================================
#  MAIN RUN
# =====================================================================
if __name__ == "__main__":
    # --- НАСТРОЙКИ ---
    # Укажи путь к своей модели (слэши для Windows экранируй или r"")
    MODEL_DIR = r"lora_tune/models/run_2026-02-25_19-07-15/best_model"

    # Твои классы (порядок важен, должен совпадать с обучением!
    # Если в config.json есть id2label, лучше брать оттуда, но ты просил жестко)
    LABELS = ['другие слова', 'машина', 'приготовить машину', 'самый малый вперед']

    # Пороги уверенности
    CONF_THRESHOLDS = {
        "машина": 0.95,             # Короткое слово, ложных много -> высокий порог
        "приготовить машину": 0.96, # Часто путается с шумом -> очень высокий порог
        "самый малый вперед": 0.92, # Длинная уникальная фраза -> можно пониже
        "другие слова": 0.99        # Это мусор, порог не важен, мы его игнорим в коде
    }

    # Создаем распознавалку
    rec = RealTimeRecognizerVAD(
        model_dir=MODEL_DIR,
        labels=LABELS,
        window_s=2.5,        # 3.0 многовато для реалтайма, 2.0-2.5 отзывчивее
        stride_s=0.5,        # Проверять каждые 0.5 сек
        conf_th=0.9,         # Дефолтный порог
        conf_th_per_label=CONF_THRESHOLDS,
        debounce_s=2.0,      # Не повторять команду чаще чем раз в 2 сек
        vad_threshold=0.4    # Чувствительность к речи (0.3 - ловит шепот, 0.7 - только крик)
    )

    def on_detect(d):
        print(f"\n🚀 [COMMAND] {d['label'].upper()} (prob={d['prob']:.3f})")

    # Поехали
    rec.start_stream(callback_on_detection=on_detect)
