import dataclasses
from typing import List, Optional

from core.embedders import CommonEmbedder
from core.model_triplet import TorchClfBase
from core.models import CommonSearchModel
from experiments.search.search_asr import Wav2VecAsrSearchModel


@dataclasses.dataclass
class AsrRegSearchModel(CommonSearchModel):
    embedder: CommonEmbedder
    clf: Wav2VecAsrSearchModel
    clf_num: TorchClfBase
    th: float
    reg_classes: List[str]

    def __post_init__(self):
        assert 0 < self.th <= 1., self.th

    def search_keywords(self, wav_path) -> Optional[str]:
        self.verbose_print(wav_path)
        pred_class = self.clf.search_keywords(wav_path)

        if pred_class in self.reg_classes and self.clf_num is not None:
            search_emb = self.embedder.get_emb(wav_path=wav_path)
            if search_emb is None:  # empty audio
                return None

            if self.clf_num is not None and pred_class in self.reg_classes:
                pred_reg = int(float(self.clf_num.predict(search_emb[None])[0]))
                return f"{pred_class} {int(pred_reg)}"
            else:
                return pred_class
        else:
            return pred_class
