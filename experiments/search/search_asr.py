import dataclasses
import json
import wave
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
from rapidfuzz import process, fuzz

from core.models import CommonSearchModel
from core.preproc import PreprocClfTorch


def get_prompt_from_key(prompt_key: str, search_commands: List[str]) -> str:
    search_commands_str = ','.join(search_commands[:3])
    search_commands_str = f"""{search_commands_str}
        А также команды поворота:
        "поворот влево на УГОЛ"
        "поворот вправо на УГОЛ"
        УГОЛ принимает значения 30, 45, 60, 90
    """
    prompt_map = {
        'без подсказки': '',
        'перечисление': f"{search_commands_str}",
        'нейтральный': f"""
            Команды, которые могут прозвучать: {search_commands_str}.
        """,
        'контекст управления': f"""
            Это голосовое управление судном. Ожидаются команды: {search_commands_str}.
        """,
        'шумоустойчивость': f"""
                Команды, которые могут прозвучать: {search_commands_str}.
                Другие команды стоит игнорировать.
            """
    }
    assert prompt_key in prompt_map.keys(), prompt_key
    return prompt_map[prompt_key]


@dataclasses.dataclass
class WhisperAsrSearchModel(CommonSearchModel):
    model_mode: str
    prompt_key: str
    transcribe_kwargs: Optional[dict] = None
    fuzzy_th: Optional[float] = None

    def __post_init__(self):
        import whisper
        self.model = whisper.load_model(self.model_mode, device="cuda")
        if self.transcribe_kwargs is None:
            self.transcribe_kwargs = dict(best_of=5, beam_size=5)

        if self.fuzzy_th is not None:
            assert 20 < self.fuzzy_th < 100, self.fuzzy_th

    def search_keywords(self, wav_path) -> Optional[str]:
        assert Path(wav_path).exists(), wav_path
        result = self.model.transcribe(
            wav_path,
            initial_prompt=get_prompt_from_key(self.prompt_key, search_commands=self.search_commands),
            temperature=0.0,
            language="ru",

            **self.transcribe_kwargs
        )
        text = result["text"].strip().lower()
        if not text:
            return None

        self.verbose_print(text)

        if self.fuzzy_th:
            best_match, score, _ = process.extractOne(query=text, choices=self.search_commands, scorer=fuzz.ratio)
            if score > self.fuzzy_th:
                return best_match
        else:
            for search_command in self.search_commands:
                if search_command in text:
                    return search_command

        return None


@dataclasses.dataclass
class Wav2VecAsrSearchModel(CommonSearchModel):
    model_name: str
    prep: Optional[PreprocClfTorch] = None
    fuzzy_th: Optional[float] = None

    def __post_init__(self):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.fuzzy_th is not None:
            assert 20 < self.fuzzy_th < 100, self.fuzzy_th

    def transcribe(self, wav_path: str) -> str:
        if self.prep:
            waveform = self.prep(wav_path)
        else:
            waveform, sr = torchaudio.load(wav_path)

        if waveform is None:
            return None

        inputs = self.processor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(predicted_ids)[0].strip().lower()
        return transcription

    def search_keywords(self, wav_path) -> Optional[str]:
        assert Path(wav_path).exists(), wav_path

        text = self.transcribe(wav_path)
        if not text:
            return None

        self.verbose_print(text)

        if self.fuzzy_th:
            best_match, score, _ = process.extractOne(query=text, choices=self.search_commands, scorer=fuzz.ratio)
            if score > self.fuzzy_th:
                return best_match
        else:
            for search_command in self.search_commands:
                if search_command in text:
                    return search_command

        return None


@dataclasses.dataclass
class GigaamAsrSearchModel(CommonSearchModel):
    model_mode: str
    prompt_key: str
    transcribe_kwargs: Optional[dict] = None
    fuzzy_th: Optional[float] = None

    def __post_init__(self):
        import gigaam
        self.model = gigaam.load_model(
            self.model_mode
        )  # Options: "v2_ctc" or "ctc", "v2_rnnt" or "rnnt", "v1_ctc", "v1_rnnt"

        if self.transcribe_kwargs is None:
            self.transcribe_kwargs = dict(best_of=5, beam_size=5)

        if self.fuzzy_th is not None:
            assert 20 < self.fuzzy_th < 100, self.fuzzy_th

    def search_keywords(self, wav_path) -> Optional[str]:
        assert Path(wav_path).exists(), wav_path
        result = self.model.transcribe(wav_path)
        text = result["text"].strip().lower()
        if not text:
            return None

        self.verbose_print(text)

        if self.fuzzy_th:
            best_match, score, _ = process.extractOne(query=text, choices=self.search_commands, scorer=fuzz.ratio)
            if score > self.fuzzy_th:
                return best_match
        else:
            for search_command in self.search_commands:
                if search_command in text:
                    return search_command

        return None


@dataclasses.dataclass
class VoskAsrSearchModel(CommonSearchModel):
    model_path: str
    verbose: bool

    def __post_init__(self):
        import vosk
        self.model = vosk.Model(self.model_path)

    def search_keywords(self, wav_path):
        import vosk
        wf = wave.open(wav_path, "rb")
        rec = vosk.KaldiRecognizer(
            self.model,
            wf.getframerate(),
            # '["машина", "приготовить машину", "самый малый вперед"]'
        )
        # rec = KaldiRecognizer(self.model, wf.getframerate(), json.dumps(self.search_commands))

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower()

                self.verbose_print(text)

                for search_command in self.search_commands:
                    if search_command in text:
                        return search_command
        return None
