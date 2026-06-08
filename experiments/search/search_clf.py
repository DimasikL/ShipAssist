import dataclasses
from typing import Optional, List, Any

import numpy as np
import pandas as pd

from core.embedders import CommonEmbedder
from core.model_triplet import TorchClfBase
from core.models import CommonSearchModel


@dataclasses.dataclass
class ClfSearchModel(CommonSearchModel):
    embedder: CommonEmbedder
    clf: TorchClfBase
    th: float
    clf_num: Optional[TorchClfBase] = None
    reg_classes: Optional[List[str]] = None

    def __post_init__(self):
        assert 0 < self.th <= 1., self.th

        for search_command in self.search_commands:
            assert search_command in self.clf.classes_, f"'{search_command}' not in {self.clf.classes_}"

    def search_keywords(self, wav_path) -> Optional[str]:
        self.verbose_print(wav_path)
        search_emb = self.embedder.get_emb(wav_path=wav_path)
        if search_emb is None:  # empty audio
            return None

        probs = self.clf.predict_proba(search_emb[None]).flatten()
        classes = self.clf.classes_
        assert len(probs) == len(classes), f"{len(probs)} != {len(classes)}"
        self.verbose_print(pd.Series(dict(zip(self.clf.classes_, probs))))
        max_prob_id = np.argmax(probs)
        if probs[max_prob_id] > self.th:
            pred_class = classes[max_prob_id]
            if self.clf_num is not None and pred_class in self.reg_classes:
                pred_reg = int(float(self.clf_num.predict(search_emb[None])[0]))
                return f"{pred_class} {int(pred_reg)}"
            else:
                return pred_class
        else:
            return None

@dataclasses.dataclass
class Wav2Vec2ClassifierSearchModel(CommonSearchModel):
    model: Any
    processor: Any
    th: float = 0.8

    def search_keywords(self, wav_path) -> Optional[str]:
        import torch
        import soundfile as sf

        speech, sr = sf.read(wav_path)
        inputs = self.processor(
            speech,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().numpy()

        classes = self.model.config.id2label
        max_id = probs.argmax()

        if probs[max_id] > self.th:
            return classes[max_id]
        return None
