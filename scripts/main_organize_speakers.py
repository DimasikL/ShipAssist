"""
organize_speakers.py

Distributes speech command dataset files into per-speaker folders.

Speakers:
  - sp13 (group=new user 13): commands 001-012, negatives positions 0,1
  - sp14 (group=new user 14): commands 013-024, negatives positions 2,3
  - sp15 (group=new user 15): commands 025-036, negatives positions 4,5
"""

import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_COMMANDS = PROJECT_ROOT / "scripts" / "clf_dset" / "commands"
SRC_NEGATIVES = PROJECT_ROOT / "scripts" / "clf_dset" / "negatives"
DST_ROOT = PROJECT_ROOT / "clf_dset" / "train_val"

# ---------------------------------------------------------------------------
# Speaker config
# ---------------------------------------------------------------------------
SPEAKERS: dict[str, dict] = {
    "new user 13": {
        "cmd_range": (1, 12),       # inclusive file numbers
        "neg_positions": (0, 1),    # indices within each word's 6-file group
    },
    "new user 14": {
        "cmd_range": (13, 24),
        "neg_positions": (2, 3),
    },
    "new user 15": {
        "cmd_range": (25, 36),
        "neg_positions": (4, 5),
    },
}


def copy_commands(
    speaker_name: str,
    dst_speaker: Path,
    cmd_range: tuple[int, int],
) -> int:
    """Copy command wav files for the given file-number range.

    Args:
        speaker_name: Human-readable speaker label (for logging).
        dst_speaker:  Destination speaker root directory.
        cmd_range:    Inclusive (start, end) file numbers to copy.

    Returns:
        Total number of files copied.
    """
    copied = 0
    start, end = cmd_range

    for cmd_dir in sorted(SRC_COMMANDS.iterdir()):
        if not cmd_dir.is_dir():
            continue

        dst_cmd = dst_speaker / "commands" / cmd_dir.name
        dst_cmd.mkdir(parents=True, exist_ok=True)

        # Collect all wav files, sort by name so numbering is stable
        wav_files = sorted(cmd_dir.glob("*.wav"))

        for wav in wav_files:
            # Extract the numeric stem (e.g. "001" → 1)
            try:
                file_num = int(wav.stem)
            except ValueError:
                print(f"  [WARN] Skipping non-numeric filename: {wav.name}")
                continue

            if start <= file_num <= end:
                shutil.copy2(wav, dst_cmd / wav.name)
                copied += 1

    return copied


def copy_negatives(
    speaker_name: str,
    dst_speaker: Path,
    neg_positions: tuple[int, int],
) -> int:
    """Copy negative wav files using positional selection within each word group.

    Each word sub-folder is expected to have exactly 6 files. Files are sorted
    by name and the two positions specified are copied.

    Args:
        speaker_name:   Human-readable speaker label (for logging).
        dst_speaker:    Destination speaker root directory.
        neg_positions:  Two 0-based indices into the sorted 6-file group.

    Returns:
        Total number of files copied.
    """
    copied = 0
    dst_neg = dst_speaker / "negatives"
    dst_neg.mkdir(parents=True, exist_ok=True)

    pos_a, pos_b = neg_positions

    for word_dir in sorted(SRC_NEGATIVES.iterdir()):
        if not word_dir.is_dir():
            continue

        wav_files = sorted(word_dir.glob("*.wav"))

        if len(wav_files) != 6:
            print(
                f"  [WARN] {word_dir.name}: expected 6 files, "
                f"found {len(wav_files)} — skipping"
            )
            continue

        for pos in (pos_a, pos_b):
            src = wav_files[pos]
            # Prefix destination filename with word name to avoid collisions
            dst_name = f"{word_dir.name}__{src.name}"
            shutil.copy2(src, dst_neg / dst_name)
            copied += 1

    return copied


def main() -> None:
    """Entry point: recreate speaker folders and populate them."""
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source cmds  : {SRC_COMMANDS}")
    print(f"Source negs  : {SRC_NEGATIVES}")
    print(f"Destination  : {DST_ROOT}\n")

    for speaker_name, cfg in SPEAKERS.items():
        dst_speaker = DST_ROOT / f"group={speaker_name}"

        # --- Wipe and recreate destination folder ---
        if dst_speaker.exists():
            shutil.rmtree(dst_speaker)
            print(f"[{speaker_name}] Removed existing folder.")
        dst_speaker.mkdir(parents=True)

        # --- Copy commands ---
        n_cmd = copy_commands(speaker_name, dst_speaker, cfg["cmd_range"])

        # --- Copy negatives ---
        n_neg = copy_negatives(speaker_name, dst_speaker, cfg["neg_positions"])

        total = n_cmd + n_neg
        print(
            f"[{speaker_name}]  commands={n_cmd}  negatives={n_neg}  total={total}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
