"""
scripts/recalibrate_outlier_gate.py — recalibrate OutlierDetector threshold.

Problem: detector was calibrated on train speakers; test/calibration speakers
produce higher Mahalanobis distances → commands incorrectly rejected.

Fix: compute distances for all WAVs in the calibration set, set threshold at
the chosen percentile, overwrite artifacts/models/outlier_detector.pkl.

Usage (from project root, venv activated):
    python scripts/recalibrate_outlier_gate.py ^
        --detector_path artifacts/models/outlier_detector.pkl ^
        --model_dir lora_tune/models/run_2026-04-30_23-34-27/best_model ^
        --calibration_dir clf_dset/calibration ^
        --percentile 99

Dry run (just print new threshold, do not save):
    python scripts/recalibrate_outlier_gate.py ... --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── model + extractor ─────────────────────────────────────────────────────────

def _load_extractor(model_dir: Path):
    """Same bypass as realtime_recognizer._load_wav2vec2_model."""
    import torch
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2FeatureExtractor,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.utils.outlier_detection import EmbeddingExtractor, OutlierConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Wav2Vec2Config.from_pretrained(str(model_dir))
    model = Wav2Vec2ForSequenceClassification(config)

    sf_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if sf_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(sf_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No weight file in {model_dir}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    fe = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir))
    cfg = OutlierConfig(embedding_layer="projector", max_audio_seconds=3.0)
    extractor = EmbeddingExtractor(model=model, feature_extractor=fe,
                                   device=device, config=cfg)
    logger.info("EmbeddingExtractor ready (device=%s)", device)
    return extractor


# ── distance computation ──────────────────────────────────────────────────────

def _mahal_distance(embedding: np.ndarray, state: dict) -> float:
    """Minimum Mahalanobis distance to any class centroid."""
    emb = np.asarray(embedding, dtype=np.float32)
    cov_inv = np.asarray(state["covariance_inv"], dtype=np.float32)
    best = float("inf")
    for centroid in state["class_centroids"].values():
        c = np.asarray(centroid, dtype=np.float32)
        diff = emb - c
        d = float(np.sqrt(np.dot(np.dot(diff, cov_inv), diff)))
        if d < best:
            best = d
    return best


def _collect_distances(
    extractor,
    wav_files: List[Path],
) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings and compute distances; return (distances, names)."""
    import tempfile
    import soundfile as sf

    distances, names = [], []
    for i, wav in enumerate(wav_files, 1):
        try:
            data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, data, sr)
                emb = extractor.extract_single(tmp.name)
            import os; os.unlink(tmp.name)
            distances.append(emb)
            names.append(wav.stem)
            if i % 10 == 0:
                logger.info("  %d / %d processed", i, len(wav_files))
        except Exception as e:
            logger.warning("  Skipped %s: %s", wav.name, e)
    return distances, names


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--detector_path", type=Path,
                   default=Path("artifacts/models/outlier_detector.pkl"))
    p.add_argument("--model_dir", type=Path,
                   default=Path("lora_tune/models/run_2026-04-30_23-34-27/best_model"))
    p.add_argument("--calibration_dir", type=Path,
                   default=Path("clf_dset/calibration"))
    p.add_argument("--percentile", type=float, default=99.0,
                   help="Set threshold at this percentile of calibration distances")
    p.add_argument("--extra_dirs", type=Path, nargs="*", default=[],
                   help="Additional WAV directories to include in calibration (e.g. clf_dset/test)")
    p.add_argument("--margin", type=float, default=0.0,
                   help="Add fixed margin to computed threshold (e.g. 100.0 for GPU non-determinism buffer)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print new threshold but do not overwrite pkl")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── load detector state ───────────────────────────────────────────────────
    with open(args.detector_path, "rb") as f:
        state = pickle.load(f)
    old_threshold = state["threshold"]
    logger.info("Loaded detector: threshold=%.2f  method=%s",
                old_threshold, state["config"]["method"])

    # ── collect calibration WAVs ──────────────────────────────────────────────
    wav_files = sorted(args.calibration_dir.rglob("*.wav"))
    for extra in (args.extra_dirs or []):
        extra_wavs = sorted(extra.rglob("*.wav"))
        wav_files += extra_wavs
        logger.info("Extra dir %s: +%d WAVs", extra, len(extra_wavs))
    if not wav_files:
        logger.error("No WAV files found in %s", args.calibration_dir)
        return 1
    logger.info("Total: %d calibration WAVs", len(wav_files))

    # ── load model + extract embeddings ───────────────────────────────────────
    extractor = _load_extractor(args.model_dir)
    raw_embeddings, names = _collect_distances(extractor, wav_files)

    if not raw_embeddings:
        logger.error("No embeddings extracted.")
        return 1

    # ── compute Mahalanobis distances ─────────────────────────────────────────
    distances = np.array([_mahal_distance(e, state) for e in raw_embeddings],
                         dtype=np.float32)

    # ── stats ─────────────────────────────────────────────────────────────────
    print("\nCalibration distance distribution:")
    for pct in [50, 75, 90, 95, 99, 99.5, 100]:
        print(f"  p{pct:5.1f}: {np.percentile(distances, pct):.2f}")
    print(f"  mean:   {distances.mean():.2f}  std: {distances.std():.2f}")
    print(f"  old threshold: {old_threshold:.2f}")

    new_threshold = float(np.percentile(distances, args.percentile)) + args.margin
    label = f"p{args.percentile}" + (f"+{args.margin}" if args.margin else "")
    print(f"  new threshold ({label}): {new_threshold:.2f}")

    n_would_pass = int((distances < new_threshold).sum())
    print(f"  commands that would pass: {n_would_pass}/{len(distances)} "
          f"= {n_would_pass/len(distances)*100:.1f}%")

    if args.dry_run:
        logger.info("Dry run — pkl NOT modified.")
        return 0

    # ── overwrite pkl ─────────────────────────────────────────────────────────
    state["threshold"] = new_threshold
    state["calibration_stats"] = {
        "n_samples": len(distances),
        "percentile_used": args.percentile,
        "p50": float(np.percentile(distances, 50)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "mean": float(distances.mean()),
        "std": float(distances.std()),
        "calibration_dirs": [str(args.calibration_dir)] + [str(d) for d in (args.extra_dirs or [])],
    }

    with open(args.detector_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    # also update info json if it exists
    info_path = args.detector_path.with_suffix("").with_name(
        args.detector_path.stem + "_info.json"
    )
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        info["threshold"] = new_threshold
        info["recalibration"] = state["calibration_stats"]
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.info("Updated %s", info_path)

    logger.info("Saved new threshold=%.2f to %s", new_threshold, args.detector_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
