import os
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from core.audio_aug import augment_audio

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}


def aug_input_d(input_d):
    import logging
    input_file = input_d['input_file']
    res_file = input_d['res_file']
    res_file.parent.mkdir(exist_ok=True, parents=True)
    assert input_file.exists(), input_file
    try:
        augment_audio(
            input_path=input_file,
            output_path=res_file,
            aug_n=30
        )
    except Exception as e:
        logging.warning(f"Skipping {input_file}: {type(e).__name__}: {e}")


def get_group_aug_dir(group_dir: Path) -> Path:
    """Return the augmentation output root for a given group directory."""
    assert 'group=' in str(group_dir)
    return Path(str(group_dir) + '-aug')


def main():
    base_dirs = [
        # 'clf_dset/train_val/group=drug slova1',
        # # 'clf_dset/train_val/group=drug slova2',
        # # 'clf_dset/train_val/group=drug slova2-new',
        # 'clf_dset/train_val/group=drug slova3',
        # 'clf_dset/train_val/group=drug slova4',
        'clf_dset/train_val/group=gtts',
        'clf_dset/train_val/group=gtts-drug',
        # # 'clf_dset/train_val/group=train user 2',
        # # 'clf_dset/train_val/group=train user 2 new',
        # 'clf_dset/train_val/group=train user 3',
        # 'clf_dset/train_val/group=train user 1',
        # # 'clf_dset/train_val/group=train user 4',
        # 'clf_dset/train_val/group=silero',
        # 'clf_dset/train_val/group=silero-drug',
        # 'clf_dset/train_val/group=genwords',
        # 'clf_dset/train_val/group=drug genwords',
        # # 'clf_dset/train_val/group=mozzilla',
        # # 'clf_dset/train_val/group=drug mozzilla',
        # 'clf_dset/train_val/group= =drug slova2',
        # 'clf_dset/train_val/group= =drug slova3',
        # 'clf_dset/train_val/group=drug slova',
        # 'clf_dset/train_val/group=test user 1',
        # 'clf_dset/train_val/group=test user 2',
        # 'clf_dset/train_val/group=test user 3',
        # 'clf_dset/train_val/group=train user 5',
        # 'clf_dset/train_val/group=train user 6',
        # 'clf_dset/train_val/group=train user 7',
        # "clf_dset/train_val/group=drug slova-hardneg1",
        # "clf_dset/train_val/group=drug slova-hardneg2",
        # "clf_dset/train_val/group=new user 8",
        # "clf_dset/train_val/group=new user 9",
        #"clf_dset/train_val/group=new user 10",
        #"clf_dset/train_val/group=new user 11",
        #"clf_dset/train_val/group=new user 12",
        #"clf_dset/train_val/group=new user 13",
        "clf_dset/train_val/group=new user 14",
        #"clf_dset/train_val/group=new user 15"
    ]

    # Sub-folders to augment within each group directory.
    # New-style groups use 'commands' and 'negatives' instead of 'samples'.
    sub_dirs = ['samples', 'commands', 'negatives']

    input_files = []
    for sample_dir in base_dirs:
        group_dir = Path(sample_dir)
        aug_base = get_group_aug_dir(group_dir)

        for sub in sub_dirs:
            src_root = group_dir / sub
            if not src_root.exists():
                continue

            for dirpath, _, files in os.walk(src_root):
                dirpath = Path(dirpath)
                for file in files:
                    input_file = dirpath / file
                    if input_file.suffix.lower() not in AUDIO_EXTENSIONS:
                        continue
                    rel_path = input_file.relative_to(src_root)
                    res_file = aug_base / sub / rel_path

                    input_files.append(dict(
                        input_file=input_file,
                        res_file=res_file
                    ))

    Parallel(n_jobs=4)(
        delayed(aug_input_d)(input_d=input_d)
        for input_d in tqdm(input_files)
    )


if __name__ == "__main__":
    main()

