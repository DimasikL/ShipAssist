import dataclasses
import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm

from core.embedders import WTVEmbedder, ResemblyzerEmbedder, SpeechBrainEmbedder, GigaamEmbedder, MFCCModel
from core.preproc import PreprocClfTorch


@dataclasses.dataclass
class AudioSample:
    audio_path: str
    y_class: Optional[str] = None
    y_angle: Optional[float] = None

    def __post_init__(self):
        assert Path(self.audio_path).exists(), self.audio_path
        for class_name in [
            #'поворот влево', 'поворот вправо',
            'машина', 'приготовить машину', 'самый малый вперед',
            'другие слова'
        ]:
            if class_name in self.audio_path:
                self.y_class = class_name
        assert self.y_class is not None, self.audio_path

        file_name = Path(self.audio_path).name
        for n_pref in ['поворот влево на', 'поворот вправо на']:
            if n_pref in file_name:
                self.y_angle = float(
                    re.search(string=file_name, pattern=f'{n_pref} [0-9]+')
                    .group().replace(n_pref, ''))


def get_clf_audio_dset(
        sample_audios,
        embedder_name: str,
        do_prep: bool, norm_duration: bool,
        output_hidden_states: bool
) -> pd.DataFrame:
    if do_prep:
        preproc = PreprocClfTorch(sr=16000, norm_duration=norm_duration)
    else:
        preproc = None

    common_args = dict(preproc=preproc, sr=16000)

    if embedder_name == 'w2v2':
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
        model = WTVEmbedder(emb_model=model_name, output_hidden_states=output_hidden_states, **common_args)
        # processor = Wav2Vec2Processor.from_pretrained(model_name)
        # model = Wav2Vec2Model.from_pretrained(model_name)
        # get_features_f = lambda wav_path: get_emb_nn(model=model, processor=processor, wav_path=wav_path)
    elif embedder_name == 'gigaam':
        model_name = 'ssl'  # Options: "ssl", "v1_ssl"
        model = GigaamEmbedder(**common_args, emb_model=model_name)
    elif embedder_name == 'resemblyzer':
        model = ResemblyzerEmbedder(**common_args)
    elif embedder_name == 'speechbrain':
        model = SpeechBrainEmbedder(**common_args)
    elif embedder_name == 'mfcc':
        model = MFCCModel(**common_args)
    else:
        raise ValueError(f"Undefined embedder_name = {embedder_name}")

    features_ds = [
        model.get_emb(sample_audio.audio_path)
        for sample_audio in tqdm(sample_audios, desc='loading samples')
    ]

    features_ds = [
        {f"{embedder_name}_emb{i + 1}": val for i, val in enumerate(features_d)}
        if not features_d is None and not isinstance(features_d, dict)
        else
        features_d
        for features_d in features_ds
    ]

    assert len(features_ds) == len(sample_audios), f"{len(features_ds)}!={len(sample_audios)}"

    for features_d, sample_audio in zip(features_ds, sample_audios):
        if features_d is None:
            print(f'Can not extract features from {sample_audio.audio_path}')

    return pd.DataFrame(
        [
            {
                'audio_path': sample_audio.audio_path,
                'audio_group': re.search(r'group=([^/]+)', sample_audio.audio_path.replace('\\', '/')).group(1),
                **features_d,
                'class': sample_audio.y_class,
                'angle': sample_audio.y_angle
            }
            for features_d, sample_audio in zip(features_ds, sample_audios)
            if features_d is not None
        ]
    )


def main():
    # embedder_name = 'mfcc'
    # embedder_name = 'resemblyzer'
    # embedder_name = 'speechbrain'
    # embedder_name = 'gigaam'
    get_dset_args_d = dict(
        embedder_name='w2v2',  # w2v2, mfcc, resemblyzer,speechbrain, gigaam
        do_prep=True,
        norm_duration=True,
        output_hidden_states=True
    )

    ds_path = Path(f"dset_{','.join([f'{key}={val}' for key, val in get_dset_args_d.items()])}.csv")

    sample_audios = [
        AudioSample(f"{dir}/{file}")
        for dir, _, files in os.walk('clf_dset')
        for file in files
        if 'samples' in dir
    ]
    if ds_path.exists():
        all_df = pd.read_csv(ds_path, index_col=0)
        ok_mask = all_df['audio_path'].isin([sample_audio.audio_path for sample_audio in sample_audios])
        all_df = all_df[ok_mask]
        print(f'drop {(~ok_mask).sum()} audios')

        sample_audios = [
            sample_audio
            for sample_audio in sample_audios
            if sample_audio.audio_path not in all_df['audio_path'].to_list()
        ]
        if len(sample_audios):
            print(f'updating {len(sample_audios)}/{len(all_df)}')
            new_df = get_clf_audio_dset(sample_audios=sample_audios, **get_dset_args_d)
            print(f'updated {len(new_df)}')
            all_df = pd.concat([all_df, new_df])
            all_df.to_csv(ds_path)
    else:
        all_df = get_clf_audio_dset(sample_audios=sample_audios, **get_dset_args_d)
        all_df.to_csv(ds_path)


if __name__ == '__main__':
    main()