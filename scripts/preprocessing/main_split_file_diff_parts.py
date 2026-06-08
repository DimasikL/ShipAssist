import os
import shutil
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm


def split_file(in_file: Path, min_silence_len=500, silence_thresh_offset=-16, keep_silence=200):
    """
    in_file: путь к файлу
    min_silence_len: минимальная длина тишины в мс (если тишина короче, она считается частью фразы)
    silence_thresh_offset: порог тишины относительно средней громкости файла (dB).
                           Например, если файл -20dB, а оффсет -16, то тишиной считается всё, что тише -36dB.
    keep_silence: сколько тишины оставлять в начале и конце вырезанного куска (мс).
    """

    # Подготовка папки для результатов
    res_dir = in_file.parent.parent.joinpath('samples').joinpath(in_file.stem)
    if res_dir.exists():
        shutil.rmtree(res_dir)
    res_dir.mkdir(exist_ok=True, parents=True)

    # Загрузка аудио
    try:
        audio = AudioSegment.from_wav(in_file)
    except Exception as e:
        print(f"Error loading {in_file}: {e}")
        return

    # Рассчитываем порог тишины относительно громкости конкретного файла
    # Это надежнее, чем фиксированное значение типа -40
    silence_thresh = audio.dBFS + silence_thresh_offset

    # Разбиваем по тишине
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )

    # Фильтрация совсем коротких "мусорных" кусков (например, щелчков короче 0.1с)
    # Если нужны абсолютно все звуки, можно убрать этот if
    min_meaningful_len = 100
    valid_chunks = [c for c in chunks if len(c) > min_meaningful_len]

    if not valid_chunks:
        print(f"Warning: No phrases found in {in_file.name} with current settings.")
        return

    # Экспорт
    for i, chunk in enumerate(valid_chunks):
        out_path = res_dir / f"{in_file.stem}_{i}.wav"
        chunk.export(out_path, format="wav")


def main():
    # Путь к датасету
    root_dir = 'clf_dset/train_val/group=drug slova-hardneg1'

    files_to_split = [
        Path(root) / file
        for root, _, files in os.walk(root_dir)
        for file in files
        if file.lower().endswith('.wav') # Добавлена проверка расширения
    ]

    # Настройки разбиения (можно покрутить эти цифры)
    # min_silence_len=400: пауза между словами меньше 0.4с не считается разрывом
    # silence_thresh_offset=-16: тишина это то, что на 16дБ тише среднего уровня файла
    for file in tqdm(files_to_split):
        split_file(file, min_silence_len=1000, silence_thresh_offset=-16, keep_silence=300)


if __name__ == '__main__':
    main()