"""
Split clf_dset recordings into per-speaker folders.

Usage:
    python scripts/split_by_speaker.py

Output structure:
    clf_dset/train_val/
        group=new user 13/
            commands/<cmd>/
            negatives/
        group=new user 14/
            ...
        group=new user 15/
            ...
"""

import shutil
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_COMMANDS = PROJECT_ROOT / "scripts" / "clf_dset" / "commands"
SRC_NEGATIVES = PROJECT_ROOT / "scripts" / "clf_dset" / "negatives"
DST_BASE = PROJECT_ROOT / "clf_dset" / "train_val"

SPEAKERS = [13, 14, 15]
SPEAKER_POSITIONS = {13: [0, 1], 14: [2, 3], 15: [4, 5]}  # by position in sorted group
COMMANDS_RANGES = {13: (1, 12), 14: (13, 24), 15: (25, 36)}  # by absolute number


def get_speaker_by_range(num: int) -> int | None:
    for sp, (lo, hi) in COMMANDS_RANGES.items():
        if lo <= num <= hi:
            return sp
    return None


def split_commands() -> int:
    total = 0
    for cmd_dir in sorted(SRC_COMMANDS.iterdir()):
        if not cmd_dir.is_dir():
            continue
        cmd_name = cmd_dir.name
        for wav in sorted(cmd_dir.iterdir()):
            if wav.suffix != ".wav":
                continue
            num_str = wav.stem.split("_")[-1]
            try:
                num = int(num_str)
            except ValueError:
                print(f"  [WARN] Cannot parse number from {wav.name}")
                continue
            sp = get_speaker_by_range(num)
            if sp is None:
                print(f"  [WARN] No speaker for num {num}: {wav.name}")
                continue
            dst = DST_BASE / f"group=new user {sp}" / "commands" / cmd_name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav, dst / wav.name)
            total += 1
    return total


def split_negatives() -> int:
    pattern = re.compile(r"^(.+?)_(\d{3})\.wav$")
    word_files: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for wav in sorted(SRC_NEGATIVES.iterdir()):
        m = pattern.match(wav.name)
        if not m:
            continue
        word = m.group(1)
        num = int(m.group(2))
        word_files[word].append((num, wav))

    total = 0
    for word, items in word_files.items():
        items.sort(key=lambda x: x[0])
        if len(items) != 6:
            print(f"  [WARN] Unexpected file count for '{word}': {len(items)}")
            continue
        for sp, positions in SPEAKER_POSITIONS.items():
            dst = DST_BASE / f"group=new user {sp}" / "negatives"
            dst.mkdir(parents=True, exist_ok=True)
            for pos in positions:
                _, wav = items[pos]
                shutil.copy2(wav, dst / wav.name)
                total += 1
    return total


def main() -> None:
    print("Cleaning destination...")
    if DST_BASE.exists():
        shutil.rmtree(DST_BASE)
    DST_BASE.mkdir(parents=True)

    print("Splitting commands...")
    n_cmd = split_commands()
    print(f"  Copied {n_cmd} command files.")

    print("Splitting negatives...")
    n_neg = split_negatives()
    print(f"  Copied {n_neg} negative files.")

    print("\nResult:")
    for sp in SPEAKERS:
        for subdir in ["commands", "negatives"]:
            d = DST_BASE / f"group=new user {sp}" / subdir
            if d.exists():
                if subdir == "commands":
                    count = sum(1 for f in d.rglob("*.wav"))
                else:
                    count = sum(1 for f in d.glob("*.wav"))
                print(f"  sp{sp}/{subdir}: {count} files")

    print("\nDone.")


if __name__ == "__main__":
    main()
