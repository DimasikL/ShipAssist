"""
benchmark_mfcc_svm.py -- Baseline: MFCC + SVM classifier.

Trains an SVM on MFCC features extracted from the train split,
evaluates on the same test split used by eval_onnx_model.py,
and saves results to artifacts/benchmarks/mfcc_svm_results.json.

Usage
-----
    python scripts/train/benchmark_mfcc_svm.py \
        --data_csv dset_meta_only_2026-05-09_10-27-42.csv

    # With noise augmentation (SNR ~12 dB) at test time:
    python scripts/train/benchmark_mfcc_svm.py \
        --data_csv dset_meta_only_2026-05-09_10-27-42.csv \
        --noisy_test

Output
------
artifacts/benchmarks/mfcc_svm_results.json  -- clean test metrics
artifacts/benchmarks/mfcc_svm_noisy_results.json  -- noisy test metrics (if --noisy_test)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]


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


def extract_mfcc(wav: np.ndarray) -> np.ndarray:
    """Extract 78-dim: MFCC mean+std + delta mean+std + delta2 mean+std."""
    import librosa
    mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([
        mfcc.mean(axis=1),   mfcc.std(axis=1),
        delta1.mean(axis=1), delta1.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1),
    ])
    return features.astype(np.float32)


def build_features(df: pd.DataFrame, noisy: bool = False, snr_db: float = 12.0,
                   desc: str = "") -> Tuple[np.ndarray, np.ndarray]:
    from tqdm import tqdm
    label_list = sorted(df["class"].unique())
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    X, y = [], []
    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc or "Extracting MFCC"):
        try:
            wav = load_and_prepare(row["audio_path"])
            if noisy:
                wav = add_noise(wav, snr_db)
            feat = extract_mfcc(wav)
            X.append(feat)
            y.append(label2id[row["class"]])
        except Exception as exc:
            errors += 1
            if errors <= 5:
                logger.warning("Skip %s: %s", row["audio_path"], exc)
    if errors:
        logger.warning("Total skipped: %d / %d", errors, len(df))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="MFCC + SVM baseline benchmark")
    parser.add_argument("--data_csv", default="dset_meta_only_2026-05-09_10-27-42.csv")
    parser.add_argument("--max_train", type=int, default=5000,
                        help="Max train samples per class (speed up training). 0 = all.")
    parser.add_argument("--noisy_test", action="store_true",
                        help="Also evaluate on noise-augmented test set (SNR ~12 dB)")
    parser.add_argument("--snr_db", type=float, default=12.0,
                        help="SNR for noisy test augmentation (dB)")
    parser.add_argument("--c", type=float, default=10.0, help="SVM regularisation C")
    args = parser.parse_args()

    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        import librosa  # noqa: F401
    except ImportError as exc:
        logger.error("Missing dependency: %s  ->  pip install scikit-learn librosa", exc)
        sys.exit(1)

    out_dir = _PROJECT_ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _PROJECT_ROOT / args.data_csv
    logger.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)
    train_df = df[~df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)

    logger.info("Train samples: %d  |  Test samples: %d", len(train_df), len(test_df))
    logger.info("Label distribution (test):\n%s", test_df["class"].value_counts().to_string())

    label_list = sorted(df["class"].unique())
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    logger.info("Labels: %s", label_list)

    if args.max_train > 0:
        train_df = (
            train_df
            .groupby("class", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), args.max_train), random_state=42))
            .reset_index(drop=True)
        )
        logger.info("Subsampled train to %d samples", len(train_df))

    logger.info("Extracting train features ...")
    X_train, y_train = build_features(train_df, noisy=False, desc="Train MFCC")

    logger.info("Extracting test features (clean) ...")
    X_test_clean, y_test = build_features(test_df, noisy=False, desc="Test MFCC (clean)")

    logger.info("Training SVM (C=%.1f, kernel=rbf, class_weight=balanced) ...", args.c)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test_clean)

    t0 = time.perf_counter()
    clf = SVC(kernel="rbf", C=args.c, gamma="scale", class_weight="balanced",
              probability=True, random_state=42)
    clf.fit(X_train_sc, y_train)
    train_time = time.perf_counter() - t0
    logger.info("Training done in %.1f s", train_time)

    logger.info("Evaluating on clean test ...")
    t0 = time.perf_counter()
    y_pred_clean = clf.predict(X_test_sc)
    latency_ms = (time.perf_counter() - t0) * 1000 / len(y_test)

    acc_clean = accuracy_score(y_test, y_pred_clean)
    f1_clean = f1_score(y_test, y_pred_clean, average="macro", zero_division=0)
    report_clean = classification_report(y_test, y_pred_clean, target_names=label_list, zero_division=0)
    logger.info("\n%s", report_clean)
    logger.info("Clean -- Accuracy: %.4f  Macro F1: %.4f  Latency: %.2f ms/sample",
                acc_clean, f1_clean, latency_ms)

    results_clean = {
        "method": "MFCC + SVM",
        "test_type": "clean",
        "accuracy": float(acc_clean),
        "macro_f1": float(f1_clean),
        "mean_latency_ms": float(latency_ms),
        "n_samples": int(len(y_test)),
        "n_train_samples": int(len(X_train)),
        "classification_report": report_clean,
        "svm_C": args.c,
        "n_mfcc": N_MFCC,
        "features_dim": int(X_train.shape[1]),
    }
    out_path = out_dir / "mfcc_svm_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_clean, f, ensure_ascii=False, indent=2)
    logger.info("Saved -> %s", out_path)

    if args.noisy_test:
        logger.info("Extracting test features (noisy, SNR=%.1f dB) ...", args.snr_db)
        X_test_noisy, _ = build_features(
            test_df, noisy=True, snr_db=args.snr_db, desc=f"Test MFCC (SNR {args.snr_db} dB)"
        )
        X_test_noisy_sc = scaler.transform(X_test_noisy)

        t0 = time.perf_counter()
        y_pred_noisy = clf.predict(X_test_noisy_sc)
        latency_ms_noisy = (time.perf_counter() - t0) * 1000 / len(y_test)

        acc_noisy = accuracy_score(y_test, y_pred_noisy)
        f1_noisy = f1_score(y_test, y_pred_noisy, average="macro", zero_division=0)
        report_noisy = classification_report(y_test, y_pred_noisy, target_names=label_list, zero_division=0)
        logger.info("\n%s", report_noisy)
        logger.info("Noisy -- Accuracy: %.4f  Macro F1: %.4f  SNR: %.1f dB",
                    acc_noisy, f1_noisy, args.snr_db)

        results_noisy = {
            "method": "MFCC + SVM",
            "test_type": f"noisy_snr{args.snr_db:.0f}dB",
            "snr_db": args.snr_db,
            "accuracy": float(acc_noisy),
            "macro_f1": float(f1_noisy),
            "mean_latency_ms": float(latency_ms_noisy),
            "n_samples": int(len(y_test)),
            "classification_report": report_noisy,
        }
        out_path_noisy = out_dir / "mfcc_svm_noisy_results.json"
        with open(out_path_noisy, "w", encoding="utf-8") as f:
            json.dump(results_noisy, f, ensure_ascii=False, indent=2)
        logger.info("Saved -> %s", out_path_noisy)

    logger.info("Done.")


if __name__ == "__main__":
    main()
