import dataclasses
import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

# Anchor to project root regardless of the CWD set by the IDE run config.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass
class AudioSample:
    audio_path: str
    y_class: Optional[str] = None
    y_angle: Optional[float] = None

    def __post_init__(self):
        assert Path(self.audio_path).exists(), self.audio_path
        normalized_path = self.audio_path.replace('\\', '/').replace('_', ' ')

        # New-style layout: files inside 'negatives/' folder → "другие слова"
        if '/negatives/' in normalized_path:
            self.y_class = 'другие слова'
        else:
            # Old-style: class name is encoded in the path (folder or filename).
            # Normalize underscores to spaces so that folder names like
            # "приготовить_машину_x5" still match the canonical class label.
            # Note: "самый малый вперёд" (ё) and "самый малый вперед" (е) are both accepted.
            for class_name in [
                'машина', 'приготовить машину',
                'самый малый вперед', 'самый малый вперёд',
                'другие слова',
            ]:
                if class_name in normalized_path:
                    self.y_class = 'самый малый вперед' if 'вперёд' in class_name else class_name
                    break

        assert self.y_class is not None, self.audio_path

        file_name = Path(self.audio_path).name
        for n_pref in ['поворот влево на', 'поворот вправо на']:
            if n_pref in file_name:
                self.y_angle = float(
                    re.search(string=file_name, pattern=f'{n_pref} [0-9]+')
                    .group().replace(n_pref, '')
                )


def build_audio_dataset(sample_audios) -> pd.DataFrame:
    data = []

    for sample_audio in sample_audios:
        try:
            group = re.search(
                r'group=([^/]+)',
                sample_audio.audio_path.replace('\\', '/')
            ).group(1)
        except:
            group = None

        data.append({
            'audio_path': sample_audio.audio_path,
            'audio_group': group,
            'class': sample_audio.y_class,
        })

    return pd.DataFrame(data)


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ds_path = PROJECT_ROOT / f"dset_meta_only_{timestamp}.csv"

    clf_dset_root = PROJECT_ROOT / 'clf_dset'

    def _is_included(dir_path: str) -> bool:
        """Accept old-style 'samples/' dirs and new-style 'commands/' + 'negatives/' dirs."""
        parts = dir_path.replace('\\', '/').split('/')
        return 'samples' in parts or 'commands' in parts or 'negatives' in parts

    sample_audios = [
        AudioSample(f"{dir}/{file}")
        for dir, _, files in os.walk(clf_dset_root)
        for file in files
        if file.endswith('.wav') and _is_included(dir)
    ]

    df = build_audio_dataset(sample_audios)
    df.to_csv(ds_path)

    print(f"Сохранено {len(df)} записей в файл: {ds_path.name}")


if __name__ == '__main__':
    main()
