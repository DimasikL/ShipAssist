"""
benchmark_whisper.py — Baseline: Whisper-tiny zero-shot classification.

Runs openai/whisper-tiny on the test split without any fine-tuning.
Transcribes each file, then matches the transcription to a command label
using fuzzy substring matching (same strategy a naive integration would use).

Usage
-----
    python scripts/train/benchmark_whisper.py \\
        --data_csv dset_meta_only_2026-05-09_10-27-42.csv

    # Also evaluate with noise:
    python scripts/train/benchmark_whisper.py \\
        --data_csv dset_meta_only_2026-05-09_10-27-42.csv \\
        --noisy_test --snr_db 12

Output
------
artifacts/benchmarks/whisper_tiny_results.json
artifacts/benchmarks/whisper_tiny_noisy_results.json  (if --noisy_test)

Notes
-----
- Requires: pip install openai-whisper
- Model is downloaded automatically on first run (~150 MB)
- Matching logic: for each transcription, the command whose keywords
  appear as a substring (case-insensitive) is selected. Ties broken by
  keyword specificity (longer match wins). If no match → 'другие слова'.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# cmd.exe ^ line continuation leaves empty string tokens in sys.argv — strip them
sys.argv = [a for a in sys.argv if a.strip()]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SR = 16_000
WIN_SAMPLES = 48_000

TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]

# Keyword mapping: label → list of keywords that must appear in the transcription.
# Ordered from most to least specific (longest match wins).
LABEL_KEYWORDS: Dict[str, List[str]] = {
    "приготовить машину": ["приготовить", "приготовь", "готовить"],
    "самый малый вперед": ["малый", "вперед", "вперёд"],
    "машина":             ["машина", "машин"],
    "другие слова":       [],   # fallback
}

REJECT_LABEL = "другие слова"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fix_path(p: str) -> Path:
    p = p.replace("\\", "/")
    win_root = "C:/Users/Dmitriy/PycharmProjects/ShipAssistant"
    p = p.replace(win_root, str(_PROJECT_ROOT))
    return Path(p)


def load_and_prepare(path: str) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(str(_fix_path(path)), sr=SR, mono=True)
    if len(wav) < WIN_SAMPLES:
        wav = np.pad(wav, (0, WIN_SAMPLES - len(wav)))
    else:
        wav = wav[:WIN_SAMPLES]
    return wav.astype(np.float32)


def add_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    signal_power = np.mean(wav ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return wav + noise


def match_label(transcription: str) -> str:
    """Map Whisper transcription to closest command label.

    Strategy:
      1. Lowercase the transcription.
      2. For each label (most-specific first), check if any keyword appears.
      3. First match wins.
      4. If no keyword matches → REJECT_LABEL.
    """
    text = transcription.lower().strip()
    for label, keywords in LABEL_KEYWORDS.items():
        if not keywords:
            continue
        if any(kw in text for kw in keywords):
            return label
    return REJECT_LABEL


def evaluate_whisper(
    model,
    df: pd.DataFrame,
    label_list: List[str],
    noisy: bool = False,
    snr_db: float = 12.0,
) -> Dict:
    """Run Whisper transcription + matching on *df* and return metrics."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from tqdm import tqdm
    import soundfile as sf

    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    y_true, y_pred = [], []
    latencies_ms = []
    errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "tmp.wav"

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Whisper-tiny inference"):
            try:
                wav = load_and_prepare(row["audio_path"])
                if noisy:
                    wav = add_noise(wav, snr_db)

                # Write to temp wav (whisper expects file path)
                sf.write(str(tmp_wav), wav, SR)

                t0 = time.perf_counter()
                result = model.transcribe(
                    str(tmp_wav),
                    language="ru",
                    fp16=False,
                    verbose=False,
                )
                latencies_ms.append((time.perf_counter() - t0) * 1000)

                transcript = result.get("text", "").strip()
                pred_label = match_label(transcript)

                y_true.append(label2id[row["class"]])
                y_pred.append(label2id[pred_label])

            except Exception as exc:
                errors += 1
                if errors <= 5:
                    logger.warning("Skip %s: %s", row["audio_path"], exc)
                # Count as wrong prediction
                y_true.append(label2id[row["class"]])
                y_pred.append(label2id[REJECT_LABEL])

    if errors:
        logger.warning("Total errors: %d / %d", errors, len(df))

    report = classification_report(y_true, y_pred, target_names=label_list, zero_division=0)
    logger.info("\n%s", report)

    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "macro_f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "p95_latency_ms":  float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "n_samples":       len(y_true),
        "classification_report": report,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper-tiny zero-shot baseline")
    parser.add_argument("--data_csv", default="dset_meta_only_2026-05-09_10-27-42.csv")
    parser.add_argument("--noisy_test", action="store_true")
    parser.add_argument("--snr_db", type=float, default=12.0)
    args = parser.parse_args()

    try:
        import whisper
        import soundfile  # noqa: F401
        import librosa    # noqa: F401
    except ImportError as exc:
        logger.error(
            "Missing dependency: %s\n"
            "Install with:  pip install openai-whisper soundfile librosa",
            exc
        )
        sys.exit(1)

    out_dir = _PROJECT_ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _PROJECT_ROOT / args.data_csv
    df = pd.read_csv(csv_path)
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)
    label_list = sorted(df["class"].unique())

    logger.info("Test samples: %d  |  Labels: %s", len(test_df), label_list)

    logger.info("Loading whisper-tiny …")
    model = whisper.load_model("tiny")
    logger.info("Model loaded.")

    # ── Clean eval ────────────────────────────────────────────────────────────
    logger.info("Evaluating on clean test …")
    metrics_clean = evaluate_whisper(model, test_df, label_list, noisy=False)
    logger.info(
        "Clean — Accuracy: %.4f  Macro F1: %.4f  Latency: %.1f ms/sample",
        metrics_clean["accuracy"], metrics_clean["macro_f1"], metrics_clean["mean_latency_ms"]
    )

    out_clean = {
        "method": "Whisper-tiny (zero-shot)",
        "test_type": "clean",
        **metrics_clean,
    }
    p = out_dir / "whisper_tiny_results.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out_clean, f, ensure_ascii=False, indent=2)
    logger.info("Saved → %s", p)

    # ── Noisy eval ────────────────────────────────────────────────────────────
    if args.noisy_test:
        logger.info("Evaluating on noisy test (SNR=%.1f dB) …", args.snr_db)
        metrics_noisy = evaluate_whisper(
            model, test_df, label_list, noisy=True, snr_db=args.snr_db
        )
        logger.info(
            "Noisy — Accuracy: %.4f  Macro F1: %.4f",
            metrics_noisy["accuracy"], metrics_noisy["macro_f1"]
        )

        out_noisy = {
            "method": "Whisper-tiny (zero-shot)",
            "test_type": f"noisy_snr{args.snr_db:.0f}dB",
            "snr_db": args.snr_db,
            **metrics_noisy,
        }
        p = out_dir / "whisper_tiny_noisy_results.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out_noisy, f, ensure_ascii=False, indent=2)
        logger.info("Saved → %s", p)

    logger.info("Done.")


if __name__ == "__main__":
    main()
