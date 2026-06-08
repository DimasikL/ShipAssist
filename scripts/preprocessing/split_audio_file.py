from pathlib import Path
from pydub import AudioSegment

FOLDER_PATH = Path(r'C:\Users\Dmitriy\PycharmProjects\ShipAssistant\clf_dset\calibration\home\другие слова')
CHUNK_MS = 5_000
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}

def split_audio_file(file_path: Path, chunk_ms: int = CHUNK_MS, delete_original: bool = False):
    print(f'Обработка: {file_path.name}')
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)

    if duration_ms <= chunk_ms:
        print(f'  Файл короче или равен {chunk_ms/1000:.0f} сек, не режу.')
        return

    part = 1
    saved_any = False

    for start_ms in range(0, duration_ms, chunk_ms):
        end_ms = start_ms + chunk_ms
        chunk = audio[start_ms:end_ms]

        out_name = f"{file_path.stem}_part{part}{file_path.suffix}"
        out_path = file_path.with_name(out_name)

        chunk.export(out_path, format=file_path.suffix[1:])
        print(f'  Сохранён фрагмент: {out_path.name}')
        part += 1
        saved_any = True

    if delete_original and saved_any:
        print(f'  Удаляю оригинал: {file_path.name}')
        file_path.unlink()

def main():
    for file in FOLDER_PATH.iterdir():
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            split_audio_file(file, delete_original=True)  # тут включаем удаление

if __name__ == '__main__':
    main()