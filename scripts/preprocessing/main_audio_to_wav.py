import os
import shutil
from pathlib import Path
from core.utils import convert_m4a_to_wav

files = [
    Path(f"{dir}/{file}")
    for dir, subdirs, files in os.walk('clf_dset/train_val/group=drug slova-hardneg2')
    for file in files
    #if '.m4a' in file
]

for file in files:
    target_path = file.parent.joinpath(file.stem + '.wav')
    if not target_path.exists():
        print(file)
        convert_m4a_to_wav(
            m4a_path=file,
            wav_path=target_path,
            sr=16000
        )
