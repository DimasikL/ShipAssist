import dataclasses
import os
import shutil
import logging
import uuid
from pathlib import Path
import joblib
import optuna
from joblib import Parallel, delayed
from optuna import Trial
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.emb_preproc import RowStandardScaler
from core.model_triplet import LinearContrastive, LinearOrMlpModel, LinearTriplet


@dataclasses.dataclass
class ContrastiveDset:
    x_train1: np.ndarray
    y_train1: np.ndarray
    x_train2: np.ndarray
    y_train2: np.ndarray
    x_val1: np.ndarray
    y_val1: np.ndarray
    x_val2: np.ndarray
    y_val2: np.ndarray

    x_test_all: np.ndarray
    y_test_all: np.ndarray

    def __post_init__(self):
        self.x_train_all = np.vstack([self.x_train1, self.x_train2])
        self.y_train_all = np.concatenate([self.y_train1, self.y_train2])

        self.x_val_all = np.vstack([self.x_val1, self.x_val2])
        self.y_val_all = np.concatenate([self.y_val1, self.y_val2])

        u_train = np.unique(self.y_train_all)
        u_val = np.unique(self.y_val_all)
        u_test = np.unique(self.y_test_all)

        diff = (set(u_val) | set(u_train)) - set(u_val)
        assert len(u_train) == len(u_val) and (u_train == u_val).all(), f'train & val classes difference = {diff}'

        diff = (set(u_val) | set(u_test)) - set(u_val)
        assert len(u_val) == len(u_test) and (u_test == u_val).all(), f'train & val classes difference = {diff}'


def get_reduced_groups(all_df, groups, part, strat_key):
    assert 0.2 < part <= 1.0, part
    ids = all_df[all_df['audio_group'].isin(groups)].index.to_numpy()

    if part == 1.0:
        return ids
    else:
        ids, _ = train_test_split(ids, train_size=part, random_state=42, stratify=all_df.loc[ids, strat_key])
        return ids


def get_prev_best_trial(study):
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value,
                           reverse=(study.direction == optuna.study.StudyDirection.MAXIMIZE))
    if len(sorted_trials) >= 2:
        return sorted_trials[1]
    else:
        raise ValueError("Unexpected state")


def get_studies_path(res_dir, model_name, trial):
    return res_dir.joinpath(f"clf_model={model_name},best_val={trial.value:.3f},"
                            f"best_test={trial.user_attrs['test_score']:.3f}.csv")


def save_trials(study, trial, res_dir, model_name):
    best_trial = study.best_trial
    new_df_path = get_studies_path(res_dir=res_dir, model_name=model_name, trial=best_trial)
    if not new_df_path.exists():
        if len(study.trials_dataframe()) > 1:
            prev_best_trial = get_prev_best_trial(study)
            prev_df_path = get_studies_path(res_dir=res_dir, model_name=model_name, trial=prev_best_trial)
            os.remove(prev_df_path)

        study.trials_dataframe().to_csv(new_df_path)


def get_data_splits_from_all_df(
        all_df, problem_mode: str,
        use_tts: bool, use_aug: bool,
        tts_part: float, aug_part: float,
        split: str, exclude_classes=None
) -> ContrastiveDset:
    exclude_classes = exclude_classes if exclude_classes is not None else []
    all_df = all_df[~all_df['class'].isin(exclude_classes)]
    all_df = all_df[all_df['angle'].isin(list(range(5, 31))) | all_df['angle'].isna()]

    all_df = all_df.copy().reset_index(drop=True)

    y_keys_to_drop = ['angle', 'class']
    if problem_mode == 'clf':
        y_keys = ['class']
    elif problem_mode == 'reg':
        all_df = all_df[~all_df['angle'].isna()]
        y_keys = ['angle']
    elif problem_mode == 'clf_w_num':
        all_df = all_df[~all_df['angle'].isna()]
        # all_df['class'] = all_df['class'] + all_df['angle'].astype(str)
        all_df['class'] = all_df['angle'].astype(str)
        y_hist = all_df['class'].value_counts()
        assert not len(y_hist[y_hist < 5])
        y_keys = ['class']
    elif problem_mode == 'clf_reg':
        all_df = all_df[~all_df['angle'].isna()]
        y_keys = all_df['class', 'angle']
    else:
        raise ValueError(f"Undefined problem_mode = {problem_mode}")

    # all_df['y_strat'] = all_df['audio_group'] + all_df['class']
    # y_strat_key = 'y_strat'
    y_strat_key = 'class'

    tts_groups = ['gtts', 'gtts-aug',  'silero', 'silero-aug'
                  ] if use_aug else ['gtts' , 'silero'
     ]

    if split == 'alexf to darya':
        train_groups = ['alexf', 'drug slova']
        val_groups = ['darya', 'darya_bad']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'darya to alexf':
        train_groups = ['darya', 'darya_bad']
        val_groups = ['alexf', 'drug slova']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'leak':
        train_groups = ['test user 1 leak', 'drug slova leak']
        val_groups = ['test user 1 leak', 'drug slova leak']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2 to us1':
        train_groups = ['train user 2', 'drug slova2']
        val_groups = ['train user 1', 'drug slova1']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us1 to us2':
        train_groups = ['train user 1', 'drug slova1']
        val_groups = ['train user 2', 'drug slova2']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2 to us3':
        train_groups = ['train user 2', 'drug slova2']
        val_groups = ['train user 3', 'drug slova3']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us3 to us2':
        train_groups = ['train user 3', 'drug slova3']
        val_groups = ['train user 2', 'drug slova2']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us1 to us3':
        train_groups = ['train user 1', 'drug slova1']
        val_groups = ['train user 3', 'drug slova3']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us3 to us1':
        train_groups = ['train user 3', 'drug slova3']
        val_groups = ['train user 1', 'drug slova1']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2 to us1':
        train_groups = ['train user 2', 'drug slova2']
        val_groups = ['train user 1', 'drug slova1']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2-new to us2':
        train_groups = ['train user 2 new', 'drug slova2-new']
        val_groups = ['train user 2', 'drug slova2']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2-new to us3':
        train_groups = ['train user 2 new', 'drug slova2-new']
        val_groups = ['train user 3', 'drug slova3']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2-new to us1':
        train_groups = ['train user 2 new', 'drug slova2-new']
        val_groups = ['train user 1', 'drug slova1']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2 to us2-new':
        train_groups = ['train user 2', 'drug slova2']
        val_groups = ['train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us1 to us2-new':
        train_groups = ['train user 1', 'drug slova1']
        val_groups = ['train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us3 to us2-new':
        train_groups = ['train user 3', 'drug slova3']
        val_groups = ['train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'on_tts':
        train_groups = ['gtts', 'gtts-drug']
        val_groups = ['silero', 'silero-drug']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1 to us3':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1']
        val_groups = ['train user 3', 'drug slova3']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us3 to us1':
        train_groups = ['train user 2', 'drug slova2', 'train user 3', 'drug slova3']
        val_groups = ['train user 1', 'drug slova1']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us1us3 to us2':
        train_groups = ['train user 1', 'drug slova1', 'train user 3', 'drug slova3']
        val_groups = ['train user 2', 'drug slova2']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1 to us3us2-new':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1']
        val_groups = ['train user 3', 'drug slova3', 'train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us2-new to us3':
        train_groups = ['train user 2', 'drug slova2','train user 2 new', 'drug slova2-new']
        val_groups = ['train user 3', 'drug slova3']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1 to us2-new':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1']
        val_groups = ['train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us4us2us1 to us3':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'train user 4', 'drug slova4']
        val_groups = ['train user 3', 'drug slova3']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us4us2us1 to us2-new':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'train user 4', 'drug slova4']
        val_groups = ['train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1 to us3us4':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1']
        val_groups = ['train user 3', 'drug slova3', 'train user 4', 'drug slova4']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1 to us2-newus4':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1']
        val_groups = ['train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1us4 to us2-newus3':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'train user 4', 'drug slova4']
        val_groups = ['train user 2 new', 'drug slova2-new', 'train user 3', 'drug slova3']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1us4us3 to us2-new':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'train user 4', 'drug slova4', 'train user 3', 'drug slova3']
        val_groups = ['train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1us2-newus3 to us4':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'train user 3', 'drug slova3', 'train user 2 new', 'drug slova2-new']
        val_groups = ['train user 4', 'drug slova4']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1gen to us3us2-new':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'drug genwords', 'genwords']
        val_groups = ['train user 3', 'drug slova3', 'train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1genmozz to us3us2-new':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'drug genwords', 'genwords','drug mozzilla', 'mozzilla']
        val_groups = ['train user 3', 'drug slova3', 'train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us2us1gen to us3us2-newmozz':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'drug genwords', 'genwords']
        val_groups = ['train user 3', 'drug slova3', 'train user 2 new', 'drug slova2-new', 'drug mozzilla', 'mozzilla']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us4us2us1mozz to us2-new':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'train user 4', 'drug slova4', 'drug mozzilla', 'mozzilla']
        val_groups = ['train user 2 new', 'drug slova2-new']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    elif split == 'us4us2us1mozz to us2-newus3':
        train_groups = ['train user 2', 'drug slova2', 'train user 1', 'drug slova1', 'train user 4', 'drug slova4', 'drug mozzilla', 'mozzilla']
        val_groups = ['train user 2 new', 'drug slova2-new', 'train user 3', 'drug slova3']
        val_aug = [f"{val_group}-aug" for val_group in val_groups] if use_aug else []
        val_groups = [*val_groups, *val_aug]
        aug_groups = [f'{train_group}-aug' for train_group in train_groups] if use_aug else []
    else:
        raise ValueError(f"Undefined split={split}")

    for group in [*train_groups, *val_groups, *aug_groups, *tts_groups]:
        assert group in all_df['audio_group'].unique(), \
            f"{problem_mode}: '{group}' not in {all_df['audio_group'].unique()}"

    # train_val_real_ids = all_df[all_df['audio_group'].isin(train_real_groups)].index.to_numpy()
    train_ids = all_df[all_df['audio_group'].isin(train_groups)].index.to_numpy()
    val_ids = all_df[all_df['audio_group'].isin(val_groups)].index.to_numpy()

    if use_tts:
        train_ids = np.concatenate([train_ids, get_reduced_groups(all_df, groups=tts_groups, part=tts_part,
                                                                 strat_key=y_strat_key)])

    if use_aug:
        train_ids = np.concatenate([train_ids, get_reduced_groups(all_df, groups=aug_groups, part=aug_part,
                                                             strat_key=y_strat_key)])

    # train_val_df = all_df.loc[[*train_val_real_ids]]
    # assert all(train_val_df[y_strat_key].value_counts() > 1), train_val_df[y_strat_key].value_counts()
    # train_ids, val_ids = train_test_split(
    #     train_val_real_ids,
    #     train_size=0.75,
    #     stratify=train_val_df[y_strat_key],
    #     random_state=42
    # )

    train_df = all_df.loc[train_ids]
    val_df = all_df.loc[val_ids]
    train_val_df = pd.concat([train_df, val_df])

    #test_df = all_df[all_df['audio_path'].apply(lambda p: Path(p).parts[1] == 'test')]
    # test_df = test_df[
    #     test_df['audio_group'].apply(
    #         lambda g:
    #         '-aug' not in g
    #         and 'gtts' not in g
    #         # and g not in ['danila1']
    #     )
    # ]
    test_df = all_df[all_df['audio_group'].isin([ #'test user 1', 'drug slova 1',
                                                  'test user 2', '=drug slova 2',
                                                  'test user 3', '=drug slova 3'
    ])]
    assert len(test_df)
    test_df = test_df[test_df['class'].isin(train_val_df['class'].unique())]
    test_ids = test_df.index


    # train_ids = train_df.groupby('class').head(train_df['class'].value_counts().min()).index
    # val_ids = val_df.groupby('class').head(val_df['class'].value_counts().min()).index
    train_1_ids, train_2_ids = train_ids, np.array([])
    val_1_ids, val_2_ids = val_ids, np.array([])

    def get_mask_from_ids(ids):
        mask = all_df.index.isin(ids)
        assert sum(mask) == len(ids)
        return mask

    train_mask1 = get_mask_from_ids(train_1_ids)
    train_mask2 = get_mask_from_ids(train_2_ids)
    val_mask1 = get_mask_from_ids(val_1_ids)
    val_mask2 = get_mask_from_ids(val_2_ids)
    test_mask = get_mask_from_ids(test_ids)

    groups_train = all_df[train_mask1 | train_mask2]['audio_group'].unique()
    groups_val = all_df[val_mask1 | val_mask2]['audio_group'].unique()
    groups_test = all_df[test_mask]['audio_group'].unique()

    for group1 in [groups_train, groups_val]:
        for group2 in [groups_test]:
            inters = set(group1) & set(group2)
            assert not len(inters), inters

    X = all_df.drop(columns=['audio_path', 'audio_group',
                             # 'y_strat',
                             *y_keys_to_drop])
    y = all_df[y_keys].to_numpy()

    assert len(train_df) == (sum(train_mask1) + sum(train_mask2))
    assert len(val_df) == (sum(val_mask1) + sum(val_mask2))

    # print('y train_val distribution')
    # print(
    #     pd.merge(
    #         pd.Series(y_train_all).value_counts().to_frame(), pd.Series(y_val_all).value_counts().to_frame(),
    #         left_index=True, right_index=True
    #     )
    #     .rename(columns={'count_x': 'train_val', 'count_y': 'val'})
    # )

    return ContrastiveDset(
        x_train1=X[train_mask1].to_numpy(), y_train1=y[train_mask1],
        x_train2=X[train_mask2].to_numpy(), y_train2=y[train_mask2],
        x_val1=X[val_mask1].to_numpy(), y_val1=y[val_mask1],
        x_val2=X[val_mask2].to_numpy(), y_val2=y[val_mask2],
        x_test_all=X[test_mask].to_numpy(), y_test_all=y[test_mask]
    )


def objective(
        trial: Trial,
        model_name: str,
        all_df: pd.DataFrame,
        res_dir: Path,
        norm_x: bool,
        norm_rows: bool,
        use_tts: bool,
        use_aug: bool,
        problem_mode: str,
        split: str
) -> float:
    ds: ContrastiveDset = get_data_splits_from_all_df(
        all_df,
        problem_mode=problem_mode, use_tts=use_tts, use_aug=use_aug,
        # tts_part=trial.suggest_float('tts_part', 0.2, 1.0),
        # aug_part=trial.suggest_float('aug_part', 0.2, 1.0),
        tts_part=1.0,
        aug_part=1.0,
        split=split,
        exclude_classes=['поворот влево', 'поворот вправо']
    )
    if problem_mode in ['clf', 'reg', 'clf_w_num']:
        ds.y_train_all = ds.y_train_all.flatten()
        ds.y_val_all = ds.y_val_all.flatten()
        ds.y_test_all = ds.y_test_all.flatten()
    else:
        pass

    if model_name == 'lgbm':
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1
        }

        if problem_mode in ['clf', 'clf_w_num']:
            model = LGBMClassifier(**param_grid)
        elif problem_mode == 'reg':
            model = LGBMRegressor(**param_grid)
        else:
            raise ValueError(f"Undefined problem mode={problem_mode}")

        if norm_x:
            model = Pipeline([('scaler', StandardScaler()), ('clf', model)])

        model.fit(X=ds.x_train_all, y=ds.y_train_all)
    elif model_name in ['sk_linear']:
        if problem_mode in ['clf', 'clf_w_num']:
            params = {
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "C": trial.suggest_float("C", 1e-4, 1e6, log=True),
                "solver": trial.suggest_categorical("solver", ["liblinear"]),
                "max_iter": trial.suggest_int("max_iter", 50, 10000),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced"]),
                "l1_ratio": None,
            }

            model = LogisticRegression(**params)
    # elif model_name in ['sk_linear']:
    #     if problem_mode in ['clf', 'clf_w_num']:
    #         solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
    #         penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    #
    #         # Пропустить недопустимые комбинации
    #         invalid_combinations = [
    #             (solver == "liblinear" and penalty == "elasticnet"),
    #             (solver == "liblinear" and penalty == "l1" and "l1" not in ["l1"]),  # по умолчанию поддерживается, но оставлено для примера
    #             (solver == "liblinear" and penalty == "none"),  # если добавишь None как вариант
    #         ]
    #         if any(invalid_combinations):
    #             raise optuna.TrialPruned()  # Обрезаем trial, он недопустим
    #
    #         # elasticnet требует l1_ratio
    #         l1_ratio = None
    #         if penalty == "elasticnet":
    #             l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    #
    #         params = {
    #             "penalty": penalty,
    #             "C": trial.suggest_float("C", 1e-4, 1e5, log=True),
    #             "solver": solver,
    #             "max_iter": trial.suggest_int("max_iter", 100, 3000),
    #             "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    #             "l1_ratio": l1_ratio,
    #         }
    #         model = LogisticRegression(**params)

        elif problem_mode == 'reg':
            params = dict(
                alpha=trial.suggest_float("alpha", 1e-4, 1e4, log=True)
            )

            model = Ridge(**params)

        else:
            raise ValueError(f"Undefined problem mode={problem_mode}")
        pipeline_l = []
        if norm_rows:
            pipeline_l.append(('scaler_rows', RowStandardScaler()))

        if norm_x:
            pipeline_l.append(('scaler_x', StandardScaler()))

        model = Pipeline([*pipeline_l, ('clf', model)])

        model.fit(X=ds.x_train_all, y=ds.y_train_all)

    else:
        if problem_mode in ['clf', 'clf_w_num']:
            n_out = len(np.unique(ds.y_train_all))
        elif problem_mode == 'reg':
            n_out = 1
        else:
            raise ValueError(f"{problem_mode}")

        device = 'cpu'
        nn_fit_params = dict(
            embedding_dim=ds.x_train_all.shape[1], n_out=n_out,
            device=device,
            epochs=200,
            batch_size=16,
            lr=trial.suggest_float('lr', low=1e-4, high=1e-1, log=True),
            weight_decay=trial.suggest_float('weight_decay', low=1e-5, high=1e1, log=True),
            save_best_val=True,
            problem_mode=problem_mode.replace('_w_num', ''),
            dropout_rate=trial.suggest_float('dropout_rate', low=0.0, high=0.5)
        )
        hidden_neurons = [32, 64]
        hidden_layers = [1, 2, 3]
        if model_name in ['linear', 'mlp']:
            if model_name == 'linear':
                hidden_neurons = []
                hidden_layers = []
            elif model_name == 'mlp':
                pass

            model = LinearOrMlpModel(
                norm_x=norm_x,
                norm_rows=norm_rows,
                x_val=ds.x_val_all,
                y_val=ds.y_val_all,
                **nn_fit_params,
                hidden_neurons=trial.suggest_categorical('hidden_neurons', hidden_neurons),
                hidden_layers=trial.suggest_categorical('hidden_layers', hidden_layers)
            )
            model.fit(ds.x_train_all, ds.y_train_all)
        elif model_name in ['contrastive', 'triplet']:
            if model_name == 'contrastive':
                claz = LinearContrastive
            elif model_name == 'triplet':
                claz = LinearTriplet
            else:
                raise ValueError(f"Undefined model_name = {model_name}")

            model = claz(
                norm_x=norm_x,
                norm_rows=norm_rows,
                x_val1=ds.x_val1, y_val1=ds.y_val1,
                x_val2=ds.x_val2, y_val2=ds.y_val2,
                **nn_fit_params,
                hidden_neurons=trial.suggest_categorical('hidden_neurons', hidden_neurons),
                hidden_layers=trial.suggest_categorical('hidden_layers', hidden_layers),
                margin=trial.suggest_float('margin', 0.2, 0.6),
                alpha=1.
            )
            model.fit(x_train1=ds.x_train1, y_train1=ds.y_train1, x_train2=ds.x_train2, y_train2=ds.y_train2)
        else:
            raise ValueError(f"Undefined model_name = {model_name}")

    y_train_all_pred = model.predict(ds.x_train_all)
    y_val_all_pred = model.predict(ds.x_val_all)
    y_test_all_pred = model.predict(ds.x_test_all)

    if problem_mode in ['clf', 'clf_w_num']:
        train_score = f1_score(y_true=y_train_all_pred, y_pred=ds.y_train_all, average='macro')
        val_score = f1_score(y_true=y_val_all_pred, y_pred=ds.y_val_all, average='macro')
        test_score = f1_score(y_true=y_test_all_pred, y_pred=ds.y_test_all, average='macro')
    elif problem_mode == 'reg':
        train_score = r2_score(y_true=y_train_all_pred, y_pred=ds.y_train_all)
        val_score = r2_score(y_true=y_val_all_pred, y_pred=ds.y_val_all)
        test_score = r2_score(y_true=y_test_all_pred, y_pred=ds.y_test_all)
    else:
        raise ValueError(f"Undefined problem = {problem_mode}")

    logging.info(f"Trial #{trial.number}: Train score = {train_score:.4f}, Val score = {val_score:.4f}, Test score = {test_score:.4f}")

    model_dir = res_dir.joinpath(model_name)
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir.joinpath(f'{uuid.uuid4().hex}.pkl')
    joblib.dump(value=model, filename=model_path)

    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("train_score", train_score)
    trial.set_user_attr("val_score", val_score)
    trial.set_user_attr("test_score", test_score)

    if val_score == 1.0:
        trial.study.stop()

    #return val_score
    return test_score


def do_study_from_args(
        all_df: pd.DataFrame,
        embedder_name: str,
        do_prep: bool,
        norm_duration: bool,
        output_hidden_states: bool,
        model_name: str,
        norm_x: bool,
        norm_rows: bool,
        problem_mode: str,
        use_tts: bool,
        use_aug: bool,
        split: str
):
    res_dir = (Path('z_exp')
    .joinpath(f"problem_mode={problem_mode}")
    .joinpath(f"split={split}")
    .joinpath(
        f'embedder_name={embedder_name},'
        f'do_prep={do_prep},'
        # f'norm_duration={norm_duration},'
        # f'output_hidden_states={output_hidden_states},'
        f'norm_x={norm_x},'
        f'norm_rows={norm_rows},'
        f'use_tts={use_tts},'
        f'use_aug={use_aug}'
    ))
    res_dir.mkdir(exist_ok=True, parents=True)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        func=lambda trial: objective(
            trial=trial,
            model_name=model_name, all_df=all_df,
            res_dir=res_dir,
            norm_x=norm_x,
            norm_rows=norm_rows,
            problem_mode=problem_mode,
            use_tts=use_tts,
            use_aug=use_aug,
            split=split
        ),
        n_trials=200,
        callbacks=[lambda study, trial: save_trials(study=study, trial=trial, res_dir=res_dir, model_name=model_name)]
    )

    # for i, row in study.trials_dataframe().sort_values('value', ascending=False)[5:].iterrows():
    #     os.remove(row['user_attrs_model_path'])


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='study.log',
        filemode='a'
    )
    embedder_name = 'w2v2'
    do_prep = True
    norm_duration = True
    output_hidden_states = True
    prep_name = 3

    ds_path = Path(
        f'dset_embedder_name={embedder_name},'
        f'do_prep={do_prep},'
        f'norm_duration={norm_duration},'
        f'output_hidden_states={output_hidden_states}'
        #f',prep_name={prep_name}'
        '.csv'
    )

    combs = [
        (problem_mode, model_name, split, use_tts, use_aug, norm_x, norm_rows)
        for model_name in [
            #'lgbm',
            #'linear',
            #'mlp',
            'sk_linear',
            #'contrastive',
            #'triplet'
        ]
        for split in [
            # 'alexf to darya',
            # 'darya to alexf',
            #'leak'
            #'on_tts',
            #'us1 to us3',
            #'us2 to us3',
            #'us1 to us2',
            #'us3 to us1',
            #'us3 to us2',
            #'us2 to us1',
            #'us2 to us2-new',
            #'us1 to us2-new',
            #'us3 to us2-new',
            #'us2-new to us2',
            #'us2-new to us1',
            #'us2-new to us3',
            #'us2us1 to us3',#
            #'us2us1 to us3us2-new',#
            #'us2us2-new to us3'##
            #'us2us3 to us1'
            #'us1us3 to us2'
            #'us2us1 to us2-new'#
            #'us4us2us1 to us3'##
            #'us4us2us1 to us2-new'##!
            #'us2us1 to us3us4'#!
            #'us2us1 to us2-newus4'#
            #'us2us1us4 to us2-new'
            #'us2us1us4 to us2-newus3'##
            #'us2us1us4us3 to us2-new'##!
            #'us2us1us2-newus3 to us4'#
            #'us2us1gen to us3us2-new'!
            #'us2us1genmozz to us3us2-new'!
            #'us2us1gen to us3us2-newmozz'
            #'us4us2us1mozz to us2-new',
            'us4us2us1mozz to us2-newus3'

        ]
        for use_tts in [
            #False,
            True
        ]
        for use_aug in [
            # False,
            True
        ]
        for norm_x in [
            #False,
            True
        ]
        for norm_rows in [
            #False,
            True
        ]
        for problem_mode in [
            'clf',
            #'clf_w_num'
        ]
    ]

    all_df = pd.read_csv(ds_path, index_col=0)

    (
        Parallel(n_jobs=15)
            (
            delayed(do_study_from_args)(
                all_df=all_df,
                embedder_name=embedder_name,
                do_prep=do_prep,
                norm_duration=norm_duration,
                output_hidden_states=output_hidden_states,
                problem_mode=problem_mode,
                model_name=model_name,
                split=split,
                use_tts=use_tts,
                use_aug=use_aug,
                norm_x=norm_x,
                norm_rows=norm_rows
            )
            for problem_mode, model_name, split, use_tts, use_aug, norm_x, norm_rows in combs
        )
    )


if __name__ == '__main__':
    main()
