"""
Script to replace broken files in clf_dset/train_val/group=new user 13/negatives
with matching files from scripts/clf_dset1/negatives.
Only copies files whose names (without " - Copy") match exactly.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

SRC_DIR = PROJECT_ROOT / "scripts" / "clf_dset1" / "negatives"
DST_DIR = PROJECT_ROOT / "clf_dset" / "train_val" / "group=new user 13" / "negatives"


def main() -> None:
    if not SRC_DIR.exists():
        print(f"[ERROR] Source directory not found: {SRC_DIR}")
        return
    if not DST_DIR.exists():
        print(f"[ERROR] Destination directory not found: {DST_DIR}")
        return

    # Collect source files (exclude " - Copy" variants)
    src_files = {
        f.name: f
        for f in SRC_DIR.iterdir()
        if f.suffix == ".wav" and " - Copy" not in f.name
    }

    # Collect destination files
    dst_files = {f.name for f in DST_DIR.iterdir() if f.suffix == ".wav"}

    matched = sorted(set(src_files.keys()) & dst_files)
    only_in_src = sorted(set(src_files.keys()) - dst_files)
    only_in_dst = sorted(dst_files - set(src_files.keys()))

    print(f"Source files (no Copy): {len(src_files)}")
    print(f"Destination files:      {len(dst_files)}")
    print(f"Matched (will replace): {len(matched)}")
    print(f"Only in source:         {len(only_in_src)}")
    print(f"Only in destination:    {len(only_in_dst)}")

    if only_in_src:
        print("\n[INFO] Files in source but NOT in destination (skipped):")
        for name in only_in_src:
            print(f"  {name}")

    if only_in_dst:
        print("\n[INFO] Files in destination but NOT in source (untouched):")
        for name in only_in_dst:
            print(f"  {name}")

    if not matched:
        print("\n[WARNING] No matching files found. Nothing to replace.")
        return

    print(f"\nReplacing {len(matched)} files...")
    replaced = 0
    for name in matched:
        src_path = src_files[name]
        dst_path = DST_DIR / name
        shutil.copy2(src_path, dst_path)
        replaced += 1

    print(f"Done. Replaced {replaced} files.")


if __name__ == "__main__":
    main()
