import dataclasses
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from experiments.search.search_dist import DistSearchModel


def get_dist_model_analysis(model: DistSearchModel):
    file_to_class = {file: key for key, l in model.examples_db.items() for file in l}

    db_pw_dists = pd.DataFrame(
        [
            dict(
                key1=key1,
                file1=file1,
                key2=key2,
                file2=file2,
                sim=cosine_similarity([model.get_emb(file1)], [model.get_emb(file2)])[0][0]
            )
            for key1, files1 in model.examples_db.items()
            for file1 in files1
            for key2, files2 in model.examples_db.items()
            for file2 in files2
            if file1 != file2
        ]
    )

    db_pw_dists_files_to_groups = db_pw_dists[['file1', 'key2', 'sim']].groupby(
        ['file1', 'key2']).mean().reset_index().pivot(index='file1', columns='key2', values='sim')

    db_preds_df = db_pw_dists_files_to_groups.idxmax(axis=1).reset_index().rename(columns={0: 'y_pred'})
    db_preds_df['y_true'] = db_preds_df['file1'].apply(lambda fpath: file_to_class[fpath])

    db_pw_dists_mean = db_pw_dists[['key1', 'key2', 'sim']].groupby(
        ['key1', 'key2']).mean().reset_index().pivot(index='key1', columns='key2', values='sim')

    db_pw_dists_min = db_pw_dists[['key1', 'key2', 'sim']].groupby(
        ['key1', 'key2']).min().reset_index().pivot(index='key1', columns='key2', values='sim')

    db_pw_dists_max = db_pw_dists[['key1', 'key2', 'sim']].groupby(
        ['key1', 'key2']).max().reset_index().pivot(index='key1', columns='key2', values='sim')

    return db_preds_df


@dataclasses.dataclass
class ModelExp:
    name: str
    claz: type
    args: dict


def main():
    search_commands = ['машина', 'приготовить машину', 'самый малый вперед']
    common_args = dict(
        search_commands=search_commands,
        sr=16000,
        examples_dirs={cmd: Path(f'mfcc_db/{cmd}') for cmd in search_commands},
        verbose=False
    )
    dist_modes = [
        'dtw',
        # 'cosine'
    ]

    model_exps: List[ModelExp] = [
        # *[
        #     ModelExp(name=f'mfcc_{dist_mode}', claz=MFCCModel, args=dict(dist_mode=dist_mode))
        #     for dist_mode in dist_modes
        # ],
        *[
            ModelExp(
                name=f'wtv_{dist_mode}_{emb_model[:8]}',
                claz=DistSearchModel, args=dict(dist_mode=dist_mode, emb_model=emb_model)
            )
            for dist_mode in dist_modes
            for emb_model in [
                # "facebook/wav2vec2-base",
                "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
                "IlyaGusev/wav2vec2-large-xlsr-ru",
                "speechbrain/wav2vec2-large-commonvoice-ru"
            ]

        ]
    ]

    for model_exp in model_exps:
        model = model_exp.claz(**common_args, th=0.99, **model_exp.args)
        print(model_exp.name)
        le = LabelEncoder()
        le.fit(model.search_commands)
        db_preds_df = get_dist_model_analysis(model=model)
        y_true_encoded = le.transform(db_preds_df['y_true'])
        y_pred_encoded = le.transform(db_preds_df['y_pred'])

        f1_scores = f1_score(y_true_encoded, y_pred_encoded, average=None)
        for label, score in zip(le.classes_, f1_scores):
            print(f"F1-score для класса '{label}': {score:.2f}")

        print()


if __name__ == '__main__':
    main()
