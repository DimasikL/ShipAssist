import os
import re
import shutil
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm


def split_file(in_file: Path):
    res_dir = in_file.parent.parent.joinpath('samples').joinpath(in_file.stem)
    if res_dir.exists(): shutil.rmtree(res_dir)
    # if res_dir.exists(): return
    res_dir.mkdir(exist_ok=True, parents=True)
    audio = AudioSegment.from_wav(in_file)

    n_repeats = int(re.search('x[0-9]+', res_dir.stem).group(0).replace('x', ''))

    split_combs = [
        (min_silence_len, silence_thresh, min_phrase_len, max_phrase_len)
        for min_silence_len in [200, 400, 800]
        for silence_thresh in [-100, -80,-60, -40]
        for min_phrase_len in [400, 600, 800, 1000, 1500]
        for max_phrase_len in [1500, 2000, 3000, 4000]
    ]
    best_chunks = None
    best_abs_dist = 1e5
    best_dist = None

    for min_silence_len, silence_thresh, min_phrase_len, max_phrase_len in tqdm(split_combs):
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=200
        )

        chunks = [
            chunk
            for chunk in chunks
            if min_phrase_len < len(chunk) < max_phrase_len
        ]

        dist = len(chunks) - n_repeats
        abs_dist = abs(dist)
        if abs_dist < best_abs_dist:
            best_chunks = chunks
            best_abs_dist = abs_dist
            best_dist = dist

        if len(chunks) == n_repeats:
            break

    assert len(best_chunks) == n_repeats, \
        f'good split params not found not found for {in_file}, best dist = {best_dist}, n_repeats ={n_repeats}'

    for i, chunk in enumerate(chunks):
        chunk.export(f"{res_dir}/{in_file.stem}_{i}.wav", format="wav")


def main():
    files_to_split = [
        Path(f'{dir}/{file}')
        for dir, subdir, files in os.walk('clf_dset/train_val/group=drug slova-hardneg1')
        for file in files

    ]
    for file in tqdm(files_to_split):
        split_file(file)


if __name__ == '__main__':
    main()
