import dataclasses
from typing import List
import numpy as np
import os
import noisereduce as nr
import librosa
import soundfile as sf
import torch
import torchaudio
from core.preproc_lib import equalizer


class Preproc:
    def __call__(self, wav_path):
        raise ValueError("Implement me")


class PreprocNo(Preproc):
    def __call__(self, wav_path):
        return


@dataclasses.dataclass
class Preproc1(Preproc):
    sr: int
    def __call__(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr)
        y = librosa.util.normalize(y)
        y = nr.reduce_noise(y=y, sr=sr)
        y = equalizer(y, sr=sr)
        sf.write(wav_path, y, sr)


@dataclasses.dataclass
class PreprocLoadMono(Preproc):
    sr: int
    def __call__(self, wav_path):
        y, _ = librosa.load(wav_path, sr=self.sr, mono=True)
        sf.write(wav_path, y, self.sr)


@dataclasses.dataclass
class PreprocNormalize(Preproc):
    sr: int
    def __call__(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr)
        y = librosa.util.normalize(y)
        sf.write(wav_path, y, sr)


@dataclasses.dataclass
class PreprocNoiseReduce(Preproc):
    sr: int
    def __call__(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr)
        y = nr.reduce_noise(y=y, sr=sr)
        sf.write(wav_path, y, sr)


@dataclasses.dataclass
class PreprocBandpass(Preproc):
    sr: int
    lowcut: float = 50.0
    highcut: float = 7000.0
    def __call__(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr)
        nyquist = 0.5 * sr
        # защитный фильтр — highcut должен быть < nyquist
        high = min(self.highcut / nyquist, 0.99)
        low = max(self.lowcut / nyquist, 0.001)
        if not 0 < low < high < 1:
            raise ValueError(f"Invalid filter bounds: low={low}, high={high}, nyquist={nyquist}")
        b, a = signal.butter(4, [low, high], btype='band')
        y = signal.lfilter(b, a, y)
        sf.write(wav_path, y, sr)


@dataclasses.dataclass
class PreprocTrimSilence(Preproc):
    sr: int
    top_db: int = 30
    def __call__(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr)
        y, _ = librosa.effects.trim(y, top_db=self.top_db)
        sf.write(wav_path, y, sr)


@dataclasses.dataclass
class PreprocClfTorch(Preproc):
    sr: int
    norm_duration: bool
    duration: int = 3
    def __post_init__(self):
        self.vad = torchaudio.transforms.Vad(sample_rate=self.sr)

    def norm_duration_f(self, waveform):
        target_length = int(self.sr * self.duration)
        if waveform.shape[1] < target_length:
            pad_length = target_length - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            return waveform[:, :target_length]

    def __call__(self, wav_path):
        waveform, wv_sr = torchaudio.load(wav_path)
        assert len(waveform)
        resampler = torchaudio.transforms.Resample(wv_sr, self.sr)
        for prep_f in [
            lambda waveform: waveform / waveform.abs().max(),
            lambda waveform: self.vad(waveform),
            lambda waveform: resampler(waveform)
        ]:
            waveform = prep_f(waveform)
            if not waveform.shape[1]:
                return None
        # waveform = eq_stretch(waveform, sr)
        if self.norm_duration:
            waveform = self.norm_duration_f(waveform=waveform)
        return waveform


@dataclasses.dataclass
class PreprocPipeline(Preproc):
    steps: List[Preproc]
    def __call__(self, wav_path):
        for step in self.steps:
            step.preproc(wav_path)
