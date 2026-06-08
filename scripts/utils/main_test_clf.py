import dataclasses
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from core.audio_detecting import AudioDetector
from core.embedders import WTVEmbedder
from core.model_triplet import TorchClfBase
from core.preproc import PreprocClfTorch
from experiments.search.search_asr import Wav2VecAsrSearchModel
from experiments.search.search_clf import ClfSearchModel


@dataclasses.dataclass
class TestAudio:
    path: Path

    def __post_init__(self):
        assert self.path.exists()
        self.class_name = self.path
        self.n_reps = int(re.search('x[0-9]+', self.path.parent.name).group(0)[1:])
        self.class_name = re.search('.+ x', self.path.parent.stem).group(0).replace(' x', '')


TEST_FILES = [
    TestAudio(
        path=Path(f"{dir}/{file}")
    )
    for dir, _, files in os.walk('clf_dset/test')
    for file in files
    if 'group' in dir and 'samples' in dir
]
print(f'test classes = {set([test_file.class_name for test_file in TEST_FILES])}')


def get_class_f1_scores(y_true, y_pred):
    return {
        cls: float(f1)
        for cls, f1 in zip(sorted(set(y_true)), f1_score(y_true, y_pred, average=None, zero_division=0))
    }


def get_test_metrics(model):
    y_df = pd.DataFrame(
        [
            dict(
                y_pred=model.search_keywords(wav_path=test.path),
                y_true=test.class_name
            )
            for test in TEST_FILES
        ],
    )
    y_df['y_pred'] = y_df['y_pred'].fillna('другие слова')
    # assert y_df['y_pred'].isna().sum() == 0, y_df['y_pred'].isna().sum()

    return pd.Series(
        {
            "f1_macro": f1_score(y_df['y_true'], y_df['y_pred'], average='macro'),
            **get_class_f1_scores(y_true=y_df['y_true'], y_pred=y_df['y_pred'])
        }
    ).round(2)


if __name__ == '__main__':
    # print(
    #     get_test_metrics(
    #         model=Wav2VecAsrSearchModel(
    #             model_name='jonatasgrosman/wav2vec2-large-xlsr-53-russian',
    #             search_commands=['машина', 'приготовить машину', 'самый малый вперед'],
    #             # prep=PreprocClfTorch(sr=16000, norm_duration=True),
    #             verbose=False,
    #             fuzzy_th=22.
    #         )
    #     )
    # )
    clf_exp_dir = 'z_exp/problem_mode=clf/split=us2 to us2-new/embedder_name=w2v2,do_prep=True,norm_x=True,norm_rows=True,use_tts=True,use_aug=True'
    print(AudioDetector.get_clf_or_reg_model_from_exp_dir(
        exp_dir=Path(clf_exp_dir),
        clf_model='sk_linear'
    ))
    # clf_exp_dir = 'z_exp/problem_mode=clf/split=alexf to darya/embedder_name=w2v2,do_prep=True,norm_x=True,norm_rows=True,use_tts=True,use_aug=False'
    # clf_exp_dir = 'z_exp/problem_mode=clf/split=us2 to us2-new/embedder_name=w2v2,do_prep=True,norm_x=True,norm_rows=True,use_tts=True,use_aug=True'
    print(
        get_test_metrics(
            model=ClfSearchModel(
                search_commands=['машина', 'приготовить машину', 'самый малый вперед'],
                embedder=WTVEmbedder(
                    sr=16000,
                    preproc=PreprocClfTorch(sr=16000, norm_duration=True),  # TODO get prep from dset.csv file
                    output_hidden_states=True,
                    emb_model="jonatasgrosman/wav2vec2-large-xlsr-53-russian"
                ),
                clf=AudioDetector.get_clf_or_reg_model_from_exp_dir(
                    exp_dir=Path(clf_exp_dir),
                    clf_model='sk_linear'
                ),
                th=0.1,
                verbose=False,
                clf_num=None,
                reg_classes=None,
            )
        )
    )
