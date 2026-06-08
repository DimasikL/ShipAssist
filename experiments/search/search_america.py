import dataclasses
from pathlib import Path
from typing import List, Optional, Dict

from matplotlib import pyplot as plt

from core.models import CommonSearchModel

from keyword_detection import KeywordDetection


@dataclasses.dataclass
class AmericaPred:
    detections: int
    maxPred: float
    minPred: float
    avgPred: float
    avgConsecutiveCnt: int
    maxNonPred: float
    minNonPred: float
    avgNonPred: float


@dataclasses.dataclass
class AmericaDetector(CommonSearchModel):
    keyword_models: List[dict]
    license_path: Path
    lang = 'russian'

    def __post_init__(self):
        self.keyword_model = KeywordDetection(keyword_models=self.keyword_models)
        with open(self.license_path, "r") as file:
            license_key = file.read().strip()

        self.keyword_model.set_keyword_detection_license(license_key)

    def search_keywords(self, wav_path) -> Optional[str]:
        res = self.keyword_model.start_keyword_detection_from_file(wav_path)
        plt.close('all')
        res: Dict[str, AmericaPred] = {key: AmericaPred(**val) for key, val in res.items()}
        n_det = sum([res_pred.detections for res_pred in res.values()])
        assert n_det in [0, 1], n_det
        for search_key, res_pred in zip(self.search_commands, res.values()):
            res_pred: AmericaPred = res_pred
            if res_pred.detections > 0:
                return search_key

        return None

    def search_hist(self, wav_path) -> dict:
        res = self.keyword_model.start_keyword_detection_from_file(wav_path)
        plt.close('all')
        res: Dict[str, AmericaPred] = {key: AmericaPred(**val) for key, val in res.items()}
        return {search_key: res_pred.detections for search_key, res_pred in zip(self.search_commands, res.values())}


def realtime_callback(params):
    print(params['phrase'])
    return params['phrase']


def main():
    model = AmericaDetector(
        search_commands=['машина', 'приготовить машину', 'самый малый вперед'],
        verbose=False,
        license_path=Path("../America/licensekey.txt"),
        keyword_models=[
            {
                "model_path": "../America/models/mashina_model_28_08042025_pyv2.onnx",
                "callback_function": realtime_callback,
                "threshold": 0.8,
                "buffer_cnt": 2
            },
            {
                "model_path":
                    "../America/models/prigotovit_mashinu_model_28_08042025_pyv2.onnx"
                ,
                "callback_function": realtime_callback,
                "threshold": 0.99,
                "buffer_cnt": 5
            },
            {
                "model_path":
                    "../America/models/samyj_malyj_vpered_model_28_08042025_pyv2.onnx"
                ,
                "callback_function": realtime_callback,
                "threshold": 0.99,
                "buffer_cnt": 5
            }
        ]
    )

    print(model.search_keywords('../mfcc_db/машина/машина с паузой.wav'))
    # print(model.search_keywords('../tests/лицом, 0.5 метра.wav'))


if __name__ == '__main__':
    main()
