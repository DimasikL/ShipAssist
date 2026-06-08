import dataclasses
from typing import Tuple, Optional

import librosa
import numpy as np
import torch
import torchaudio

from core.preproc import Preproc


@dataclasses.dataclass
class CommonEmbedder:
    sr: int
    preproc: Optional[Preproc]

    def get_prep_waveform(self, wav_path):
        if self.preproc is not None:
            waveform = self.preproc(wav_path)
        else:
            waveform, wv_sr = torchaudio.load(wav_path)
        return waveform

    def get_emb(self, wav_path) -> np.ndarray:
        """
        :return: 1D embeddings averaged by time axis
        """
        raise NotImplementedError()

    def get_emb_by_times(self, wav_path) -> Tuple[torch.Tensor, np.ndarray]:
        """
        :return: 1D waveform and 2D embeddings
        """
        raise NotImplementedError()


class MFCCModel(CommonEmbedder):
    def get_emb(self, wav_path):
        y, sr = librosa.load(wav_path, sr=None)  # Загружаем аудио без ресемплинга

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

        # --- Энергия ---
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # --- Zero-Crossing Rate ---
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # --- Спектральные признаки ---
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_flat = librosa.feature.spectral_flatness(y=y)

        # Усредняем
        features = {
            'mfcc_{}'.format(i + 1): val for i, val in enumerate(mfcc_mean)
        }
        features.update({
            'delta_mfcc_{}'.format(i + 1): val for i, val in enumerate(mfcc_delta_mean)
        })
        features.update({
            'rms': rms_mean,
            'zcr': zcr_mean,
            'spectral_centroid': np.mean(spec_centroid),
            'spectral_bandwidth': np.mean(spec_bw),
            'spectral_flatness': np.mean(spec_flat),
        })

        return features


@dataclasses.dataclass
class WTVEmbedder(CommonEmbedder):
    emb_model: str
    output_hidden_states: bool = True

    def __post_init__(self):
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        self.processor = Wav2Vec2Processor.from_pretrained(self.emb_model)
        self.model = Wav2Vec2Model.from_pretrained(self.emb_model, output_hidden_states=self.output_hidden_states)

    def get_emb(self, wav_path):
        waveform = self.get_prep_waveform(wav_path=wav_path)

        if waveform is None:
            return waveform

        input_values = self.processor(waveform.squeeze(), return_tensors="pt", sampling_rate=self.sr).input_values
        with torch.no_grad():
            if self.output_hidden_states:
                hidden_states = self.model(input_values).hidden_states
                selected_layers = torch.stack(hidden_states[-4:])
                mean_hidden = selected_layers.mean(dim=0)
                return mean_hidden.mean(dim=1).squeeze().numpy()
            else:
                hidden_states = self.model(input_values).last_hidden_state
                return hidden_states.mean(dim=1).squeeze().numpy()

    def get_emb_by_times(self, wav_path, save_waveform_path=None) -> Tuple[torch.Tensor, np.ndarray]:
        waveform = self.get_prep_waveform(wav_path=wav_path)
        if waveform is None:
            return waveform

        if save_waveform_path is not None:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            torchaudio.save(save_waveform_path, waveform, sample_rate=self.sr)

        waveform = waveform.squeeze()

        input_values = self.processor(waveform, return_tensors="pt", sampling_rate=self.sr).input_values
        with torch.no_grad():
            if self.output_hidden_states:
                hidden_states = self.model(input_values).hidden_states
                selected_layers = torch.stack(hidden_states[-4:])
                mean_hidden = selected_layers.mean(dim=0).squeeze(0)
                return waveform, mean_hidden.cpu().numpy()
            else:
                hidden_states = self.model(input_values).last_hidden_state.squeeze(0)
                return waveform, hidden_states.cpu().numpy()


@dataclasses.dataclass
class GigaamEmbedder(CommonEmbedder):
    emb_model: str
    output_hidden_states: bool = True

    def __post_init__(self):
        import gigaam
        self.model = gigaam.load_model(self.emb_model)  # Options: "ssl", "v1_ssl"

    def get_emb(self, wav_path):
        waveform = self.get_prep_waveform(wav_path=wav_path)

        if waveform is None:
            return waveform

        with torch.no_grad():
            embedding, _ = self.model.embed_audio(wav_path)
            embedding = embedding.mean(dim=1).squeeze().numpy()[:76]

            if len(embedding) < 76:
                return None
            else:
                assert len(embedding) == 76, len(embedding)
                return embedding


@dataclasses.dataclass
class ResemblyzerEmbedder(CommonEmbedder):
    def __post_init__(self):
        import resemblyzer
        self.prep_f = lambda waveform: resemblyzer.preprocess_wav(waveform.flatten().detach().numpy())
        self.encoder = resemblyzer.VoiceEncoder()

    def get_emb(self, wav_path):
        waveform = self.get_prep_waveform(wav_path=wav_path)
        waveform = self.prep_f(waveform.flatten().detach().numpy())
        return self.encoder.embed_utterance(waveform)


@dataclasses.dataclass
class SpeechBrainEmbedder(CommonEmbedder):
    def __post_init__(self):
        from speechbrain.inference import SpeakerRecognition
        self.classifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def get_emb(self, wav_path):
        waveform = self.get_prep_waveform(wav_path=wav_path)
        return self.classifier.encode_batch(waveform).flatten().detach().numpy()
