from pathlib import Path
import sys
from core.utils import convert_m4a_to_wav

sys.path.append(r"C:\Users\Dmitriy\PycharmProjects\ShipAssistant")

def rename_and_convert(src_folder: str, dst_folder: str, base_name: str = "другие слова"):
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)
    dst_path.mkdir(parents=True, exist_ok=True)

    files = sorted(src_path.glob("*.mp3"))
    for i, f in enumerate(files, start=1):
        out_file = dst_path / f"{base_name}_{i}.wav"
        convert_m4a_to_wav(f, out_file, sr=16000)
        print(f"Сохранено: {out_file}")

if __name__ == "__main__":
    SRC_FOLDER = r"C:\Users\Dmitriy\PycharmProjects\ShipAssistant\clf_dset\train_val\group=drug genwords\scr"
    DST_FOLDER = r"C:\Users\Dmitriy\PycharmProjects\ShipAssistant\clf_dset\train_val\group=drug genwords\samples"

    rename_and_convert(SRC_FOLDER, DST_FOLDER)
