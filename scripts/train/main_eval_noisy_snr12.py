"""
eval_noisy_snr12.py — Evaluate ONNX INT8 model on test split with Gaussian noise at SNR=12dB.

Replicates the exact noise-injection method used in benchmark_mfcc_svm.py and
benchmark_whisper_tiny.py so that the proposed-method F1(noisy) is comparable
with all baselines.

Usage:
    cd C:\\Users\\Dmitriy\\PycharmProjects\\ShipAssistant
    python scripts/train/eval_noisy_snr12.py

Output:
    artifacts/benchmarks/lora_wav2vec2_noisy_results.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Project root — two levels up from scripts/train/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match eval_onnx_model.py and benchmark_mfcc_svm.py
# ---------------------------------------------------------------------------
SR: int = 16_000
MAX_SECONDS: float = 3.0
MAX_SAMPLES: int = int(MAX_SECONDS * SR)
SNR_DB: float = 12.0

TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]

CSV_PATH    = _PROJECT_ROOT / "dset_meta_only_2026-05-09_10-27-42.csv"
ONNX_PATH   = _PROJECT_ROOT / "onnx_model/quant_benchmark/model_int8.onnx"
CONFIG_PATH = _PROJECT_ROOT / "lora_tune/models/run_2026-04-30_23-34-27/best_model/config.json"
OUTPUT_PATH = _PROJECT_ROOT / "artifacts/benchmarks/lora_wav2vec2_noisy_results.json"


# ---------------------------------------------------------------------------
# Exact noise function from benchmark_mfcc_svm.py  (seed=42, Gaussian, SNR-controlled)
# ---------------------------------------------------------------------------

def add_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at a fixed SNR.  Seed=42 for reproducibility."""
    rng = np.random.default_rng(42)
    signal_power = np.mean(wav ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return wav + noise


# ---------------------------------------------------------------------------
# Path helper (Windows → project root, robust to mixed separators)
# ---------------------------------------------------------------------------

def _fix_path(p: str) -> Path:
    p = p.replace("\\", "/")
    for win_root in (
        "C:/Users/Dmitriy/PycharmProjects/ShipAssistant",
        "D:/Users/Dmitriy/PycharmProjects/ShipAssistant",
    ):
        if win_root in p:
            p = p.replace(win_root, str(_PROJECT_ROOT).replace("\\", "/"))
            break
    return Path(p)


# ---------------------------------------------------------------------------
# Preprocessing — canonical window via core.audio_utils (same as OnnxEngine)
# ---------------------------------------------------------------------------

def _load_and_prepare_noisy(audio_path: Path, snr_db: float) -> np.ndarray:
    from core.audio_utils import load_wav, prepare_window

    try:
        waveform, _ = load_wav(str(audio_path), target_sr=SR)
    except Exception as exc:
        logger.warning("Could not load %s: %s — using silence.", audio_path, exc)
        return np.zeros(MAX_SAMPLES, dtype=np.float32)

    # Canonical window (pad/truncate + per-window normalise)
    clean = prepare_window(waveform, target_samples=MAX_SAMPLES, do_normalize=True)
    # Inject Gaussian noise at SNR=12 dB
    noisy = add_noise(clean, snr_db)
    return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _build_session(onnx_path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = os.cpu_count() or 1
    opts.intra_op_num_threads = os.cpu_count() or 1
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate paths
    for p, label in [(CSV_PATH, "CSV"), (ONNX_PATH, "ONNX"), (CONFIG_PATH, "config.json")]:
        if not p.exists():
            logger.error("Not found: %s  (%s)", p, label)
            sys.exit(1)

    # Load label map
    with open(CONFIG_PATH, encoding="utf-8") as f:
        model_cfg = json.load(f)
    id2label: Dict[str, str] = model_cfg["id2label"]
    label2id: Dict[str, int] = {v: int(k) for k, v in id2label.items()}
    label_names: List[str] = [id2label[str(i)] for i in range(len(id2label))]
    logger.info("Classes (%d): %s", len(label_names), label_names)

    # Build test split
    df = pd.read_csv(CSV_PATH)
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)
    test_df["audio_path"] = test_df["audio_path"].apply(lambda p: str(_fix_path(p)))
    logger.info("Test samples: %d", len(test_df))
    logger.info("Class distribution:\n%s", test_df["class"].value_counts().to_string())

    # Load ONNX session
    logger.info("Loading ONNX INT8 model: %s", ONNX_PATH)
    session = _build_session(ONNX_PATH)
    input_name = session.get_inputs()[0].name

    # Evaluate
    all_preds: List[int] = []
    all_targets: List[int] = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating (SNR={SNR_DB}dB)"):
        audio_path = Path(row["audio_path"])
        true_id = label2id[row["class"]]

        sample = _load_and_prepare_noisy(audio_path, SNR_DB)
        batch = sample[np.newaxis, :]  # (1, max_samples)
        outputs = session.run(None, {input_name: batch})
        logits = outputs[0][0]
        probs = _softmax(logits.astype(np.float32))
        pred = int(np.argmax(probs))

        all_preds.append(pred)
        all_targets.append(true_id)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    accuracy  = accuracy_score(all_targets, all_preds)
    report    = classification_report(all_targets, all_preds, target_names=label_names, zero_division=0)

    logger.info("\n%s", report)
    logger.info("=" * 60)
    logger.info("Model:    LoRA-Wav2Vec2 + ONNX INT8  (noisy, SNR=%.0f dB)", SNR_DB)
    logger.info("Macro F1: %.4f", macro_f1)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Samples:  %d", len(all_targets))
    logger.info("=" * 60)

    result = {
        "model": "LoRA-Wav2Vec2 + ONNX INT8",
        "test_type": f"noisy_snr{int(SNR_DB)}dB",
        "snr_db": SNR_DB,
        "noise_type": "gaussian_seed42",
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "n_samples": len(all_targets),
        "classification_report": report,
        "onnx_path": str(ONNX_PATH),
        "csv_path": str(CSV_PATH),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Results saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
