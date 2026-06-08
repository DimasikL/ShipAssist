import dataclasses
import os
import queue
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from pydub import AudioSegment

from core.embedders import MFCCModel, WTVEmbedder
from core.models import CommonSearchModel
from core.preproc import Preproc, PreprocNo, Preproc1, PreprocPipeline, PreprocNormalize, PreprocTrimSilence, \
    PreprocNoiseReduce, PreprocBandpass, PreprocLoadMono
from experiments.search.search_asr import WhisperAsrSearchModel, VoskAsrSearchModel, GigaamAsrSearchModel, Wav2VecAsrSearchModel
from experiments.search.search_clf import ClfSearchModel, Wav2Vec2ClassifierSearchModel
from transformers import AutoModelForAudioClassification
from transformers import Wav2Vec2FeatureExtractor

# from core.models_america import AmericaDetector

# ---------------------------------------------------------------------------
# Config-derived defaults (evaluated once at import time so the dataclass
# field defaults stay simple scalar values, which Python requires).
# If core.config is unavailable (e.g. isolated test), safe literals are used.
# ---------------------------------------------------------------------------
try:
    from core.config import settings as _settings, PROJECT_ROOT as _PROJECT_ROOT
    _DEFAULT_SR: int = _settings.audio.sample_rate
    _DEFAULT_THRESHOLD_DB: float = _settings.audio.threshold_db
    _DEFAULT_EMB_MODEL: str = _settings.training.model_name
    _TMP_BASE: Path = _settings.paths.artifacts_dir / "tmp"
except Exception:
    _DEFAULT_SR = 16000
    _DEFAULT_THRESHOLD_DB = -60.0
    _DEFAULT_EMB_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    _TMP_BASE = Path("artifacts/tmp")


@dataclasses.dataclass
class AudioDetector:
    model_name: str
    prep_name: str
    verbose: bool
    search_commands: List[str]
    block_size: int
    overlapping: Optional[float]
    reset_if_found: bool
    model_args: Optional[dict] = None
    # Default SR sourced from settings.audio.sample_rate (configs/default.yaml).
    sr: int = dataclasses.field(default_factory=lambda: _DEFAULT_SR)
    # Default energy gate sourced from settings.audio.threshold_db.
    threshold_db: float = dataclasses.field(default_factory=lambda: _DEFAULT_THRESHOLD_DB)

    def get_preproc_from_name(self, name: str) -> Preproc:
        if name == 'no':
            return PreprocNo()
        elif name == '1':
            return Preproc1(sr=self.sr)
        elif name == '2':
            return PreprocPipeline(
                steps=[
                    claz(sr=self.sr)
                    for claz in
                    [PreprocNormalize, PreprocTrimSilence, PreprocNoiseReduce, PreprocBandpass, PreprocLoadMono]
                ]
            )
        elif name == '3':
            return PreprocPipeline(
                steps=[
                    claz(sr=self.sr)
                    for claz in
                    [PreprocTrimSilence, PreprocNoiseReduce, PreprocLoadMono, PreprocBandpass, PreprocNormalize]
                ]
            )
        else:
            raise ValueError(f"Undefined preproc = {name}")

    @staticmethod
    def get_clf_or_reg_model_from_exp_dir(exp_dir: Path, clf_model: str):
        assert Path(exp_dir).exists(), exp_dir
        exp_files = [
            f"{dir}/{file}"
            for dir, _, files in os.walk(exp_dir)
            for file in files
            if re.match(pattern=f'clf_model={clf_model},.+', string=file)
        ]
        assert len(exp_files), "No exp files found"
        exp_df = pd.concat([pd.read_csv(exp_file) for exp_file in exp_files])
        best_model_path = Path(exp_df.sort_values('value', ascending=False).iloc[0]['user_attrs_model_path'])
        return joblib.load(best_model_path)

    @staticmethod
    def get_clf_or_reg_model_from_params(
            feature_model: str,
            clf_model: str, problem: str, norm_x: bool, norm_rows: bool, split: str, output_hidden_states: bool,
            my_prep: bool,
            use_tts: bool, use_aug: bool,
    ):
        my_prep = 'myprep' if my_prep else ''
        oh = 'oh' if output_hidden_states else ''
        exp_dir = (f'z_exp/problem_mode={problem}/split={split}/'
                   f"feature_mode={feature_model}+{my_prep}+{oh},"
                   f'norm_x={norm_x},'
                   f'norm_rows={norm_rows},'
                   f'use_tts={use_tts},use_aug={use_aug}')
        return AudioDetector.get_clf_or_reg_model_from_exp_dir(exp_dir=exp_dir, clf_model=clf_model)

    def get_decoder_from_name(self, name: str) -> CommonSearchModel:
        if name == 'whisper':
            return WhisperAsrSearchModel(
                search_commands=self.search_commands,
                verbose=False,
                **self.model_args
            )
        elif name == 'gigaam':
            return GigaamAsrSearchModel(
                search_commands=self.search_commands,
                verbose=False,
                **self.model_args
            )

        elif name == 'vosk':
            # Vosk model path sourced from model_args (preferred) or
            # settings.paths.artifacts_dir / "vosk-model" as the convention.
            # Override via model_args={'model_path': '...'} at call site.
            _vosk_default = str(_TMP_BASE.parent / "vosk-model")
            return VoskAsrSearchModel(
                search_commands=self.search_commands,
                model_path=self.model_args.pop('model_path', _vosk_default),
                verbose=True,
                **self.model_args
            )
        elif name == 'w2v2':
            return Wav2VecAsrSearchModel(
                search_commands=self.search_commands,
                verbose=False,
                **self.model_args
            )
        elif name == 'ft w2v2':
            model_dir = Path(self.model_args['model_path'])
            model = AutoModelForAudioClassification.from_pretrained(model_dir)
            processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)

            return Wav2Vec2ClassifierSearchModel(
                search_commands=self.search_commands,
                verbose=self.verbose,
                model=model,
                processor=processor,
                th=self.model_args.get('th', 0.8)
            )

        elif '_clf' in name:
            # TODO refactor all
            common_args = dict(
                search_commands=self.search_commands,
                verbose=False,
                **self.model_args
            )
            my_prep = True
            common_clf_args = dict(
                sr=self.sr,
                stretch_prep=True  # TODO replace hard code
            )

            output_hidden_states = True
            if 'mfcc' in name:
                embedder = MFCCModel(**common_clf_args, my_prep=my_prep)
            elif 'wtv' in name:
                # HuggingFace model name sourced from settings.training.model_name
                # (configs/default.yaml → training.model_name).
                embedder = WTVEmbedder(
                    **common_clf_args,
                    my_prep=my_prep,
                    output_hidden_states=output_hidden_states,
                    emb_model=_DEFAULT_EMB_MODEL,
                )
            else:
                raise ValueError(f"Undefined model = {name}")
            common_nn_args = dict(
                norm_x=True,
                norm_rows=True,
                use_tts=True,
                use_aug=True,
                my_prep=my_prep,
                split='alexf to darya'
            )

            clf = self.get_clf_or_reg_model_from_params(
                **common_nn_args,
                feature_model='nn',
                clf_model='sk_linear',
                problem='clf',
                output_hidden_states=output_hidden_states
            )
            clf_num = self.get_clf_or_reg_model_from_params(
                **common_nn_args,
                feature_model='nn',
                clf_model='sk_linear',
                problem='clf_w_num',
                output_hidden_states=output_hidden_states
            )
            # clf_num = None
            return ClfSearchModel(
                **common_args,
                embedder=embedder,
                clf=clf,
                clf_num=clf_num,
                reg_classes=['поворот влево', 'поворот вправо']  # TODO hard code
            )

        elif name == 'america':
            # return AmericaDetector(
            #     search_commands=['машина', 'приготовить машину', 'самый малый вперед'],
            #     verbose=False,
            #     license_path=Path("America/licensekey.txt"),
            #     keyword_models=[
            #         {
            #             "model_path": "America/models/mashina_model_28_08042025_pyv2.onnx",
            #             "callback_function": lambda params: params,
            #             "threshold": 0.8,
            #             "buffer_cnt": 2
            #         },
            #         {
            #             "model_path":
            #                 "America/models/prigotovit_mashinu_model_28_08042025_pyv2.onnx"
            #             ,
            #             "callback_function": lambda params: params,
            #             "threshold": 0.99,
            #             "buffer_cnt": 5
            #         },
            #         {
            #             "model_path":
            #                 "America/models/samyj_malyj_vpered_model_28_08042025_pyv2.onnx"
            #             ,
            #             "callback_function": lambda params: params,
            #             "threshold": 0.99,
            #             "buffer_cnt": 5
            #         }
            #     ]
            # )
            raise NotImplemented()
        else:

            raise ValueError(f"Undefined method = {name}")

    def __post_init__(self):
        self.model_args = {} if self.model_args is None else self.model_args
        self.block_size = int(self.sr * self.block_size)
        self.audio_queue = queue.Queue()
        self.preproc = self.get_preproc_from_name(name=self.prep_name)
        self.decoder = self.get_decoder_from_name(name=self.model_name)
        self.commands_history: List[str] = []
        self.prev_batch: Optional[np.ndarray] = None
        if self.overlapping is not None:
            assert 0 < self.overlapping < 1, self.overlapping

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Проблемы со звуком:", status)
        self.audio_queue.put(indata.copy())

    def verbose_print(self, *args):
        if self.verbose:
            print(*args)

    def gate(self, arr):
        rms = np.sqrt(np.mean(arr ** 2))
        if rms == 0:
            dbfs = -float('inf')
        else:
            dbfs = 20 * np.log10(rms)
        self.verbose_print(f"dbfs={dbfs}")
        if dbfs > self.threshold_db:
            return arr
        else:
            return None

    def recognize_audio_np_batch(self, curr_batch: np.ndarray, wav_path: Path):
        audio_int16 = (curr_batch * 32767).astype(np.int16)
        wav_path.parent.mkdir(exist_ok=True)
        wavfile.write(wav_path, self.sr, audio_int16)
        # time.sleep(0.05)
        self.preproc.preproc(wav_path=wav_path)
        command = self.decoder.search_keywords(wav_path=str(wav_path))
        if command is None:
            return
        else:
            for search_command in self.search_commands:
                # if command in self.search_commands:
                if search_command in command:
                    self.verbose_print(f"КОМАНДА: {command}")
                    self.commands_history.append(command)
                    if self.reset_if_found:
                        self.prev_batch = None
            else:
                self.verbose_print(f"НЕИЗВЕСТНАЯ КОМАНДА: {command}")
                pass

    def recognize_stream(self, ):
        # Temp directory rooted under settings.paths.artifacts_dir / "tmp_stream"
        # to avoid scattering temporary files across the project root.
        tmp_dir = _TMP_BASE / f"tmp_stream/tmp_{random.randint(0, int(2e20))}"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        wav_path = Path(f'{tmp_dir}/batch.wav')
        print('...')
        while True:
            curr_batch = self.audio_queue.get().squeeze()
            curr_batch = self.gate(curr_batch)
            if self.prev_batch is not None and self.overlapping is not None and curr_batch is not None:
                overl_batch = np.concatenate(
                    [self.prev_batch[int(len(self.prev_batch) * self.overlapping):], curr_batch]
                )

            elif curr_batch is not None:
                overl_batch = curr_batch

            else:
                # self.verbose_print('skipping, curr_batch is None')
                continue
            self.recognize_audio_np_batch(curr_batch=overl_batch, wav_path=wav_path)
            self.prev_batch = curr_batch
            if wav_path.exists: os.remove(wav_path)

    def recognize_file_streaming(self, audio_path: str):
        # Temp directory rooted under settings.paths.artifacts_dir / "tmp_file".
        tmp_dir = _TMP_BASE / f"tmp_file/tmp_{random.randint(0, int(2e20))}"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        # Загружаем аудио и конвертируем в нужный формат: моно, 16kHz, 16-bit PCM
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(self.sr).set_sample_width(2)  # 2 bytes = 16-bit
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32767.0
        total_samples = len(samples)
        block_size = self.block_size
        idx = 0
        file_i = 0

        while idx < total_samples:
            block = samples[idx:idx + block_size]
            idx += block_size
            if block.size == 0:
                break
            curr_batch = np.squeeze(block)
            if self.prev_batch is not None and self.overlapping is not None:
                overl_batch = np.concatenate(
                    [self.prev_batch[int(len(self.prev_batch) * self.overlapping):], curr_batch]
                )
            else:
                overl_batch = curr_batch
            wav_path = Path(f'{tmp_dir}/{Path(audio_path).name}_{file_i}.wav')
            self.recognize_audio_np_batch(curr_batch=overl_batch, wav_path=wav_path)
            self.prev_batch = curr_batch
            if wav_path.exists: os.remove(wav_path)
            file_i += 1

    def search_hist(self, wav_path: str) -> dict:
        self.recognize_file_streaming(wav_path)
        return dict(Counter(self.commands_history))
