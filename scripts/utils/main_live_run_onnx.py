"""
live_run_onnx.py — minimal real-time demo for the ONNX model.

Run from project root:
    python live_run_onnx.py

No PYTHONPATH tricks needed — the script anchors sys.path to its own
directory (project root) before importing anything from core/.
"""

import os
import sys

# Anchor: add project root to sys.path so `core.*` imports work
# regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.realtime_recognizer import RealTimeRecognizer

# ── Model paths ───────────────────────────────────────────────────────────────
# ONNX bundle exported from lora_tune/models/run_2026-02-25_19-07-15
ONNX_DIR  = "onnx_model/run_2026-04-30"                              # logits + embedding, INT8
MODEL_DIR = "lora_tune/models/run_2026-04-30_23-34-27/best_model"   # for profile config only
OUTLIER   = "artifacts/models/outlier_detector.pkl"

# ── Thresholds (calibrated) ───────────────────────────────────────────────────
CONF_TH_PER_LABEL = {
    "машина":             0.925,
    "приготовить машину": 0.94,
    "самый малый вперед": 0.925,
    "другие слова":       0.93,
}

# ── Audio device ─────────────────────────────────────────────────────────────
# Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to list.
# None = system default. On Windows, WASAPI (device 15) is more reliable than MME.
AUDIO_DEVICE = 15   # "Набор микрофонов (Realtek), WASAPI"

# ── Recognizer ────────────────────────────────────────────────────────────────
rec = RealTimeRecognizer(
    model_dir=MODEL_DIR,
    window_s=3.0,          # overridden by onnx_config.json win_samples=48000
    stride_s=0.8,
    energy_th=8e-6,
    conf_th=0.6,
    conf_th_per_label=CONF_TH_PER_LABEL,
    debounce_s=3.0,
    onnx_dir=ONNX_DIR,
    onnx_use_int8=True,
    outlier_detector=OUTLIER,
)


def on_detect(d: dict) -> None:
    print(
        f"  >>> {d['label']}  "
        f"prob={d['prob']:.3f}  "
        f"t={d['time_relative']:.1f}s  "
        f"[{d['inference_ms']:.0f}ms ONNX]"
    )


rec.start_stream(callback_on_detection=on_detect, device=AUDIO_DEVICE)
