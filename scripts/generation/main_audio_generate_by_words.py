import itertools
from pathlib import Path
from pydub import AudioSegment
import sys

sys.path.append(r"C:\Users\Dmitriy\PycharmProjects\ShipAssistant")
from core.utils import convert_m4a_to_wav
BASE_DIR = Path(r"C:\Users\Dmitriy\PycharmProjects\ShipAssistant\clf_dset\train_val\group=genwords")
SCR_DIR = BASE_DIR / "scr"
OUT_DIR = BASE_DIR / "samples" / "genwords"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_word_files(word: str):
    word_dir = SCR_DIR / word
    files = list(word_dir.glob("*.mp3"))
    if not files:
        raise FileNotFoundError(f"Нет файлов для слова: {word}")
    return files

def make_phrase(file_paths: list[Path], pause_ms: int = 200) -> AudioSegment:
    phrase = AudioSegment.silent(duration=0)
    for f in file_paths:
        audio = AudioSegment.from_file(f)
        phrase += audio + AudioSegment.silent(duration=pause_ms)
    return phrase

if __name__ == "__main__":
    phrases = {
        "машина": ["машина"],
        "самый малый вперед": ["самый", "малый", "вперед"],
        "приготовить машину": ["приготовить", "машина"],
    }

    for name, words in phrases.items():
        word_files = [get_word_files(w) for w in words]
        for combo_idx, combo in enumerate(itertools.product(*word_files), start=1):
            phrase_audio = make_phrase(combo)
            wav_path = OUT_DIR / f"{name}_{combo_idx}.wav"
            convert_m4a_to_wav(phrase_audio, wav_path, sr=16000)
            print(f"Сохранено: {wav_path}")
