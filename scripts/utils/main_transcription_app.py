import sounddevice as sd
import threading
import time

from core.audio_detecting import AudioDetector
from experiments.best_params import get_model_best_params


def main():
    print("▶️ Старт. Говорите что-нибудь (Ctrl+C для выхода)...")
    # model_name = 'gigaam'
    # model_name = 'whisper'
    # model_name = 'wtv_clf'
    # model_name = 'wtv_asr'
    # model_name = 'w2v2'
    model_name = 'ft w2v2'
    detector = AudioDetector(
        **get_model_best_params(model_name),
        search_commands=[
            'машина', 'приготовить машину', 'самый малый вперед',
            #'поворот влево', 'поворот вправо'
            # *[f'поворот вправо на {angle}' for angle in range(5, 31)],
            # *[f'поворот влево на {angle}' for angle in range(5, 31)]
        ],
        model_name=model_name,
        verbose=True,
        reset_if_found=True
    )

    threading.Thread(target=detector.recognize_stream, daemon=True).start()

    with sd.InputStream(
            samplerate=detector.sr,
            channels=1,
            callback=detector.audio_callback,
            blocksize=detector.block_size
    ):
        while True:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
