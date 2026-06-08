from pathlib import Path
from itertools import product
from tqdm import tqdm
try:
    from pydub import AudioSegment
    PYDUB = True
except Exception:
    PYDUB = False

import soundfile as sf
import numpy as np

SOURCE_BASE = Path(r"C:\Users\Dmitriy\PycharmProjects\ShipAssistant\data_mozilla")

TARGET_MOZZILLA_SAMPLES = Path(r"C:\Users\Dmitriy\PycharmProjects\ShipAssistant\clf_dset\train_val\group=mozzilla\samples")
TARGET_DRUG_SAMPLES = Path(r"C:\Users\Dmitriy\PycharmProjects\ShipAssistant\clf_dset\train_val\group=drug mozzilla\samples")

PHRASES = {
    "машина": ["машина"],
    "приготовить машину": ["приготовить", "машину"],
    "самый малый вперед": ["самый", "малый", "вперед"]
}

COMBINED_FOLDER_MAP = {
    "приготовить машину": "приготовить_машину",
    "самый малый вперед": "самый_малый_вперед",
    "машина": "машина"
}

# Parameters
MAX_PER_PHRASE = 200
SILENCE_MS = 40
OUT_SR = 16000
OUT_CH = 1

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def ensure_dirs():
    TARGET_MOZZILLA_SAMPLES.mkdir(parents=True, exist_ok=True)
    TARGET_DRUG_SAMPLES.mkdir(parents=True, exist_ok=True)
    # create phrase subfolders
    for phrase in PHRASES.keys():
        (TARGET_MOZZILLA_SAMPLES / phrase).mkdir(parents=True, exist_ok=True)

def list_audio_files(folder: Path):
    if not folder.exists():
        return []
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return files

def concat_with_pydub(paths):
    silence_seg = AudioSegment.silent(duration=SILENCE_MS)
    seg = None
    for p in paths:
        seg_part = AudioSegment.from_file(str(p))
        seg_part = seg_part.set_frame_rate(OUT_SR).set_channels(OUT_CH)
        if seg is None:
            seg = seg_part
        else:
            seg = seg + silence_seg + seg_part
    return seg

def concat_with_numpy(paths):
    arrays = []
    for p in paths:
        data, sr = sf.read(str(p))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != OUT_SR:
            import math
            ratio = OUT_SR / sr
            new_len = int(math.ceil(len(data) * ratio))
            data = np.interp(
                np.linspace(0, len(data), new_len, endpoint=False),
                np.arange(len(data)),
                data
            ).astype(np.float32)
        arrays.append(data)
        silence_len = int(OUT_SR * (SILENCE_MS / 1000.0))
        if silence_len > 0:
            arrays.append(np.zeros(silence_len, dtype=np.float32))
    if not arrays:
        return None
    return np.concatenate(arrays).astype(np.float32)

def export_pydub(seg, out_path: Path):
    seg.export(str(out_path), format="wav", parameters=["-ar", str(OUT_SR), "-ac", str(OUT_CH)])

def export_numpy(arr, out_path: Path):
    sf.write(str(out_path), arr, OUT_SR)

def build_phrases():
    comp_files = {}
    for comp_folder in set(sum([v for v in PHRASES.values()], [])):
        comp_files[comp_folder] = list_audio_files(SOURCE_BASE / comp_folder)
    combined_files = {}
    for phrase, comb_name in COMBINED_FOLDER_MAP.items():
        combined_files[phrase] = list_audio_files(SOURCE_BASE / comb_name) if comb_name else []

    summary = {}
    for phrase, components in PHRASES.items():
        out_dir = TARGET_MOZZILLA_SAMPLES / phrase
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        lists = [comp_files.get(c, []) for c in components]
        combos = []
        if all(len(lst) > 0 for lst in lists):
            combos = list(product(*lists))
        # iterate combos first
        if combos:
            for combo in tqdm(combos, desc=f"Combos {phrase}", unit="combo"):
                if count >= MAX_PER_PHRASE:
                    break
                parts = [Path(p) for p in combo]
                out_name = f"{phrase} {count}.wav"
                out_path = out_dir / out_name
                try:
                    if PYDUB:
                        seg = concat_with_pydub(parts)
                        export_pydub(seg, out_path)
                    else:
                        arr = concat_with_numpy(parts)
                        if arr is None:
                            continue
                        export_numpy(arr, out_path)
                    count += 1
                except Exception as e:
                    print(" Error building combo:", e)
                    continue
        combined = combined_files.get(phrase, [])
        if combined:
            for src in combined:
                if count >= MAX_PER_PHRASE:
                    break
                out_name = f"{phrase} {count}.wav"
                out_path = out_dir / out_name
                try:
                    if PYDUB:
                        seg = AudioSegment.from_file(str(src)).set_frame_rate(OUT_SR).set_channels(OUT_CH)
                        export_pydub(seg, out_path)
                    else:
                        data, sr = sf.read(str(src))
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)
                        if sr != OUT_SR:
                            import math
                            data = np.interp(
                                np.linspace(0, len(data), int(len(data) * OUT_SR / sr), endpoint=False),
                                np.arange(len(data)),
                                data
                            ).astype(np.float32)
                        export_numpy(data, out_path)
                    count += 1
                except Exception as e:
                    print(" Error copying combined:", e)
                    continue

        summary[phrase] = count
    return summary

def process_other_words():
    used_folders = set()
    for comps in PHRASES.values():
        used_folders.update(comps)
    used_folders.update([v for v in COMBINED_FOLDER_MAP.values() if v])

    idx = 0
    for folder in sorted(SOURCE_BASE.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if name in used_folders:
            continue
        files = list_audio_files(folder)
        if not files:
            continue
        for f in tqdm(files, desc=f"other {name}", unit="file"):
            out_name = f"другие слова {idx}.wav"
            out_path = TARGET_DRUG_SAMPLES / out_name
            try:
                if PYDUB:
                    seg = AudioSegment.from_file(str(f)).set_frame_rate(OUT_SR).set_channels(OUT_CH)
                    export_pydub(seg, out_path)
                else:
                    data, sr = sf.read(str(f))
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if sr != OUT_SR:
                        data = np.interp(
                            np.linspace(0, len(data), int(len(data) * OUT_SR / sr), endpoint=False),
                            np.arange(len(data)),
                            data
                        ).astype(np.float32)
                    export_numpy(data, out_path)
                idx += 1
            except Exception as e:
                print(" Error processing other file:", e)
                continue
    return idx

def main():
    ensure_dirs()
    summary = build_phrases()
    others_count = process_other_words()
    print("\nSummary:")
    for ph, cnt in summary.items():
        print(f"  phrase '{ph}': {cnt} files -> {TARGET_MOZZILLA_SAMPLES / ph}")
    print(f"  other words saved: {others_count} -> {TARGET_DRUG_SAMPLES}")
    print("Done.")

if __name__ == "__main__":
    main()

