import dataclasses
import os
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict
from pathlib import Path

from core.embedders import CommonEmbedder
from core.models import CommonSearchModel


@dataclasses.dataclass
class DistSearchModel(CommonSearchModel):
    sr: int
    examples_dirs: Dict[str, Path]
    th: float
    dist_mode: str
    embedder: CommonEmbedder

    @staticmethod
    def get_wav_files_from_dir(dir_name):
        return [
            Path(f"{dir}/{file}")
            for dir, subdirs, files in os.walk(dir_name)
            for file in files if '.wav' in file
        ]

    def __post_init__(self):
        self.examples_db: Dict[str, List[Path]] = {
            cmd: self.get_wav_files_from_dir(dir)
            for cmd, dir in
            self.examples_dirs.items()
        }

        for search_command in self.search_commands:
            assert search_command in self.examples_db.keys(), search_command
            for path in self.examples_db[search_command]:
                assert path.exists(), path

    def get_dist(self, emb1, emb2):
        if self.dist_mode == 'dtw':
            return fastdtw([emb1], [emb2], dist=cosine)[0]
        elif self.dist_mode == 'cosine':
            return cosine_similarity([emb1], [emb2])[0][0]
        else:
            raise ValueError(f"Undefined dist={self.dist_mode}")

    def search_keywords(self, wav_path) -> Optional[str]:
        self.verbose_print(wav_path)
        search_emb = self.embedder.get_emb(wav_path=wav_path)
        if search_emb is None:  # empty audio
            return None

        res_df = pd.DataFrame(
            [
                dict(
                    search_command=search_command,
                    sim=self.get_dist(emb1=self.embedder.get_emb(path), emb2=search_emb)
                )
                for search_command in self.search_commands
                for path in self.examples_db[search_command]
            ]
        ).groupby('search_command').mean()  # TODO min as option instead of mean
        self.verbose_print(res_df)
        res_df = res_df[res_df['sim'] > self.th].sort_values('sim', ascending=False)

        if len(res_df):
            return res_df.index[0]
        else:
            return None
