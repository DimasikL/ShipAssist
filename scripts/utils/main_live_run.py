import os
from core.realtime_recognizer import RealTimeRecognizer
MODEL_DIR = r"lora_tune/models/run_2026-02-25_19-07-15/best_model"

# labels = None
# cfg_path = os.path.join(MODEL_DIR, "config.json")
# if os.path.exists(cfg_path):
#     import json
#     cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
#     if "id2label" in cfg:
#         id2label = cfg["id2label"]
#         labels = [id2label[str(i)] for i in range(len(id2label))]
#         print("Loaded labels from config.json:", labels)
# if labels is None:
#     labels = ['другие слова', 'машина', 'приготовить машину', 'самый малый вперед']
#     print("Using fallback labels:", labels)

labels = ['другие слова', 'машина', 'приготовить машину', 'самый малый вперед']
conf_th_per_label = {
    "машина": 0.925,
    "приготовить машину": 0.94,
    "самый малый вперед": 0.925,
    "другие слова": 0.93
}

rec = RealTimeRecognizer(
    model_dir=MODEL_DIR,
    labels=labels,
    window_s=3.0,
    stride_s=0.8,
    energy_th=8e-6,
    conf_th=0.6,
    conf_th_per_label=conf_th_per_label,
    debounce_s=3
)
print('start')
def on_detect(d):
    print(f"[DETECTED] {d['label']} (prob={d['prob']:.3f}) at {d['time']:.3f}")

rec.start_stream(callback_on_detection=on_detect)

