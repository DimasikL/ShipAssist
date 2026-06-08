"""
scripts/recalibrate_cosine_outlier.py — recalibrate cosine OutlierDetector (.npy).

The detector saved by main_full_tune_lora_v5.py uses cosine distance and
stores state as a plain numpy dict (.npy). This script:
  1. Loads the .npy detector (class centroids, per-class thresholds).
  2. Extracts embeddings from calibration WAV files using the merged model.
  3. Computes cosine distance from each embedding to its nearest class centroid.
  4. Prints the full percentile table.
  5. Sets new per-class thresholds at the chosen percentile.
  6. Saves the updated .npy (unless --dry_run).

Usage (from project root, venv activated):
    # Dry run — only print new thresholds:
    python scripts/recalibrate_cosine_outlier.py ^
        --detector_path experiments/archive_training/lora_tune/models/run_2026-05-22_09-50-17/outlier_detector.npy ^
        --model_dir experiments/archive_training/lora_tune/models/run_2026-05-22_09-50-17/best_model ^
        --calibration_dir clf_dset/calibration ^
        --percentile 90 ^
        --dry_run

    # Apply:
    python scripts/recalibrate_cosine_outlier.py ^
        --detector_path experiments/archive_training/lora_tune/models/run_2026-05-22_09-50-17/outlier_detector.npy ^
        --model_dir experiments/archive_training/lora_tune/models/run_2026-05-22_09-50-17/best_model ^
        --calibration_dir clf_dset/calibration ^
        --percentile 90
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ID2LABEL = {0: "другие слова", 1: "машина", 2: "приготовить машину", 3: "самый малый вперед"}

# Maps folder-name fragments (lowercased, spaces/underscores normalised) → class ID.
# Checked against the parent folder of the WAV file and all ancestor folders.
# More specific patterns must come first (longest match wins).
_FOLDER_TO_CLASS: list[tuple[str, int]] = [
    ("приготовить_машину",  2),
    ("приготовить машину",  2),
    ("самый_малый_вперед",  3),
    ("самый малый вперед",  3),
    ("самый_малый_вперёд",  3),
    ("самый малый вперёд",  3),
    ("машина",              1),
    ("другие слова",        0),
    ("genwords",            0),  # synthetic other-words
    ("negatives",           0),  # new-user negative recordings
]

def _label_from_path(wav_path: Path) -> int | None:
    """Infer class ID from the WAV file's directory hierarchy.

    Walks from the immediate parent folder upward (skips the filename itself).
    Returns None if no pattern matches (caller should fall back to nearest-centroid).
    """
    # Skip wav_path.parts[-1] (the filename) — match only on directory names.
    dir_parts = [p.lower() for p in wav_path.parts[:-1]]
    for part in reversed(dir_parts):  # innermost folder first
        part_us = part.replace("_", " ")
        for pattern, cls_id in _FOLDER_TO_CLASS:
            if pattern in part or pattern in part_us:
                return cls_id
    return None


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model_and_fe(model_dir: Path):
    """Load merged Wav2Vec2 model + feature extractor from model_dir."""
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2FeatureExtractor,
    )
    from safetensors.torch import load_file

    logger.info("Loading model from %s", model_dir)
    config = Wav2Vec2Config.from_pretrained(str(model_dir))
    model = Wav2Vec2ForSequenceClassification(config)

    sf_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if sf_path.exists():
        state_dict = load_file(str(sf_path))
        logger.info("Loaded weights from model.safetensors")
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
        logger.info("Loaded weights from pytorch_model.bin")
    else:
        raise FileNotFoundError(f"No weight file found in {model_dir}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    fe = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model ready on %s", device)
    return model, fe, device


# ── embedding extraction ──────────────────────────────────────────────────────

def _extract_embedding(
    model,
    fe,
    device: torch.device,
    wav_path: str,
    max_seconds: float = 3.0,
) -> Optional[np.ndarray]:
    """Extract mean-pooled projector embedding for a single WAV file."""
    import soundfile as sf

    try:
        data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)

        target_sr = fe.sampling_rate  # typically 16000
        if sr != target_sr:
            import torchaudio
            data_t = torch.from_numpy(data).unsqueeze(0)
            data_t = torchaudio.functional.resample(data_t, sr, target_sr)
            data = data_t.squeeze(0).numpy()

        max_samples = int(max_seconds * target_sr)
        if len(data) > max_samples:
            data = data[:max_samples]

        inputs = fe(data, sampling_rate=target_sr, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            hidden = model.wav2vec2(input_values)[0]   # (1, T, D_model)
            projected = model.projector(hidden)         # (1, T, D_proj)
            embedding = projected.mean(dim=1)           # (1, D_proj)
            embedding = F.normalize(embedding, dim=-1)  # unit norm for cosine

        return embedding.squeeze(0).cpu().numpy()

    except Exception as e:
        logger.warning("  Skipped %s: %s", wav_path, e)
        return None


# ── cosine distance ───────────────────────────────────────────────────────────

def _cosine_dist_to_nearest(
    embedding: np.ndarray,
    centroids: Dict[int, np.ndarray],
) -> Tuple[float, int]:
    """Return (min_cosine_distance, nearest_class_id)."""
    best_dist = float("inf")
    best_cls = -1
    emb = embedding / (np.linalg.norm(embedding) + 1e-8)
    for cls_id, centroid in centroids.items():
        c = np.asarray(centroid, dtype=np.float32)
        c = c / (np.linalg.norm(c) + 1e-8)
        cos_sim = float(np.dot(emb, c))
        dist = 1.0 - cos_sim  # cosine distance ∈ [0, 2]
        if dist < best_dist:
            best_dist = dist
            best_cls = cls_id
    return best_dist, best_cls


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recalibrate cosine outlier detector (.npy) thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--detector_path", type=Path,
        required=True,
        help="Path to outlier_detector.npy produced by main_full_tune_lora_v5.py",
    )
    p.add_argument(
        "--model_dir", type=Path,
        required=True,
        help="Path to best_model directory (merged weights).",
    )
    p.add_argument(
        "--calibration_dir", type=Path,
        default=Path("clf_dset/calibration"),
        help="Directory tree with calibration WAV files.",
    )
    p.add_argument(
        "--extra_dirs", type=Path, nargs="*",
        default=[Path("clf_dset/train_val")],
        help=(
            "Additional WAV directories included in full (all subfolders). "
            "Default: clf_dset/train_val"
        ),
    )
    p.add_argument(
        "--negatives_dirs", type=Path, nargs="*",
        default=[Path("clf_dset/test")],
        help=(
            "Root directories from which ONLY negatives/ subfolders are collected. "
            "Use for splits where you want negative examples for outlier calibration "
            "but wish to keep commands/ as a clean holdout for classifier evaluation. "
            "Default: clf_dset/test"
        ),
    )
    p.add_argument(
        "--include_aug", action="store_true", default=False,
        help=(
            "Include augmented WAV files (groups containing '-aug' in their path). "
            "By default augmented samples are excluded — they produce artificially "
            "narrow distance distributions and should not influence outlier thresholds."
        ),
    )
    p.add_argument(
        "--percentile", type=float, default=90.0,
        help="Set per-class threshold at this percentile of calibration distances.",
    )
    p.add_argument(
        "--other_percentile", type=float, default=None,
        help=(
            "Override percentile for the 'другие слова' class only. "
            "Use a lower value (e.g. 85) to keep the noise-class gate tight "
            "while --percentile stays high for real commands. "
            "Defaults to --percentile when not set."
        ),
    )
    p.add_argument(
        "--global_percentile", type=float, default=None,
        help="Override percentile for global_threshold (defaults to --percentile).",
    )
    p.add_argument(
        "--max_seconds", type=float, default=3.0,
        help="Max audio length in seconds (must match training config).",
    )
    p.add_argument(
        "--min_samples", type=int, default=30,
        help=(
            "Minimum number of calibration samples required for a reliable per-class "
            "threshold. Classes with fewer samples fall back to the global threshold "
            "to avoid p99 being dominated by a single outlier embedding."
        ),
    )
    p.add_argument(
        "--debug_top_n", type=int, default=10,
        help=(
            "For each class, print the N samples with the largest cosine distance "
            "to their centroid. Useful for identifying mislabelled or noisy files "
            "that inflate the per-class threshold. Set to 0 to disable."
        ),
    )
    p.add_argument(
        "--exclude_groups", type=str, nargs="*", default=[],
        help=(
            "Exclude WAV files whose path contains any of these substrings "
            "(case-insensitive). Matched against the full path. "
            "Example: --exclude_groups 'train user 7' 'new user 12' "
            "Use to remove a specific user's recordings from calibration "
            "without deleting files from disk."
        ),
    )
    p.add_argument(
        "--include_groups", type=str, nargs="*", default=[],
        help=(
            "Keep ONLY WAV files whose path contains at least one of these substrings "
            "(case-insensitive). All other files are excluded. "
            "Example: --include_groups 'new user 10' 'new user 11' 'new user 12' "
            "Useful for calibrating purely on a specific cohort of users. "
            "Applied after --exclude_groups (exclude wins if both match)."
        ),
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print new thresholds but do NOT overwrite the .npy file.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global_pct = args.global_percentile if args.global_percentile is not None else args.percentile
    other_pct = args.other_percentile if args.other_percentile is not None else args.percentile

    # ── load detector ─────────────────────────────────────────────────────────
    if not args.detector_path.exists():
        logger.error("Detector not found: %s", args.detector_path)
        return 1

    state = np.load(str(args.detector_path), allow_pickle=True).item()
    logger.info(
        "Loaded detector: method=%s  per_class=%s  threshold_percentile=%s",
        state["method"], state["per_class"], state["threshold_percentile"],
    )
    logger.info("Current global_threshold: %.6f", state["global_threshold"])
    for cls_id, thr in state["class_thresholds"].items():
        logger.info("  class %d (%s): threshold=%.6f", cls_id, ID2LABEL.get(cls_id, cls_id), float(thr))

    # ── collect WAV files ─────────────────────────────────────────────────────
    def _wav_duration(path: Path) -> float:
        """Return WAV duration in seconds without decoding audio."""
        import wave
        try:
            with wave.open(str(path)) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            return 0.0

    # Pre-process group filter patterns once (case-insensitive substring match).
    exclude_patterns: List[str] = [g.lower() for g in (args.exclude_groups or [])]
    include_patterns: List[str] = [g.lower() for g in (args.include_groups or [])]

    def _is_excluded(path: Path) -> bool:
        """Return True if the file should be dropped by group filters.

        A file is dropped if:
          - its path matches any --exclude_groups pattern, OR
          - --include_groups is set and its path matches NONE of the patterns.
        Exclude wins over include when both match (edge case with overlapping patterns).
        """
        path_lower = str(path).lower()
        if exclude_patterns and any(pat in path_lower for pat in exclude_patterns):
            return True
        if include_patterns and not any(pat in path_lower for pat in include_patterns):
            return True
        return False

    def _collect_wavs(directory: Path) -> List[Path]:
        """Collect WAV files, skipping augmented groups, excluded groups, and
        files longer than max_seconds (truncated audio → bad embeddings)."""
        wavs = sorted(directory.rglob("*.wav"))
        if not args.include_aug:
            before = len(wavs)
            wavs = [p for p in wavs if not any("-aug" in part for part in p.parts)]
            skipped = before - len(wavs)
            if skipped:
                logger.info("  Skipped %d aug WAVs in %s", skipped, directory)
        # Drop files longer than the model's max_seconds window — their embeddings
        # are computed from a truncated signal and will be far from any centroid.
        before = len(wavs)
        wavs = [p for p in wavs if _wav_duration(p) <= args.max_seconds + 0.1]
        skipped = before - len(wavs)
        if skipped:
            logger.info("  Skipped %d WAVs longer than %.1fs in %s", skipped, args.max_seconds, directory)
        # Apply group filters (--exclude_groups / --include_groups).
        if exclude_patterns or include_patterns:
            before = len(wavs)
            wavs = [p for p in wavs if not _is_excluded(p)]
            skipped = before - len(wavs)
            if skipped:
                logger.info("  Skipped %d WAVs by group filters in %s", skipped, directory)
        return wavs

    wav_files: List[Path] = _collect_wavs(args.calibration_dir)
    for extra in (args.extra_dirs or []):
        extra_wavs = _collect_wavs(extra)
        wav_files += extra_wavs
        logger.info("Extra dir %s: +%d WAVs", extra, len(extra_wavs))

    for neg_root in (args.negatives_dirs or []):
        # Collect only negatives/ subfolders — skip commands/ entirely.
        neg_wavs: List[Path] = []
        for neg_subdir in sorted(neg_root.rglob("negatives")):
            if neg_subdir.is_dir():
                found = sorted(neg_subdir.rglob("*.wav"))
                if not args.include_aug:
                    found = [p for p in found if not any("-aug" in part for part in p.parts)]
                if exclude_patterns or include_patterns:
                    found = [p for p in found if not _is_excluded(p)]
                neg_wavs.extend(found)
        wav_files += neg_wavs
        logger.info("Negatives dir %s: +%d WAVs (negatives/ only)", neg_root, len(neg_wavs))

    if not wav_files:
        logger.error("No WAV files found in %s", args.calibration_dir)
        return 1
    if exclude_patterns:
        logger.info("Excluded groups: %s", args.exclude_groups)
    if include_patterns:
        logger.info("Include-only groups: %s", args.include_groups)
    logger.info("Total calibration WAVs: %d (include_aug=%s)", len(wav_files), args.include_aug)

    # ── load model ────────────────────────────────────────────────────────────
    model, fe, device = _load_model_and_fe(args.model_dir)
    centroids: Dict[int, np.ndarray] = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in state["class_centroids"].items()
    }

    # ── diagnose centroid norms ───────────────────────────────────────────────
    for cls_id, c in centroids.items():
        norm = float(np.linalg.norm(c))
        logger.info("  centroid %d norm=%.4f  shape=%s", cls_id, norm, c.shape)

    # ── extract embeddings + compute distances ────────────────────────────────
    all_distances: List[float] = []
    per_class_distances: Dict[int, List[float]] = {k: [] for k in centroids}
    # Track (dist, path) per class for debug output
    per_class_records: Dict[int, List[Tuple[float, Path]]] = {k: [] for k in centroids}
    fallback_to_centroid = 0
    _diag_printed = 0  # print first few assignments for sanity check

    logger.info("Extracting embeddings...")
    for i, wav_path in enumerate(wav_files, 1):
        emb = _extract_embedding(model, fe, device, str(wav_path), args.max_seconds)
        if emb is None:
            continue
        dist, nearest_cls = _cosine_dist_to_nearest(emb, centroids)

        # Prefer label inferred from directory structure (avoids misassignment
        # when a command embedding happens to be closer to a wrong centroid).
        label_cls = _label_from_path(wav_path)
        if label_cls is not None and label_cls in per_class_distances:
            assigned_cls = label_cls
            # Recompute distance to the assigned centroid (not the nearest one).
            c = centroids[assigned_cls]
            c = c / (np.linalg.norm(c) + 1e-8)
            emb_n = emb / (np.linalg.norm(emb) + 1e-8)
            dist = float(1.0 - np.dot(emb_n, c))
        else:
            assigned_cls = nearest_cls
            fallback_to_centroid += 1

        all_distances.append(dist)
        if assigned_cls in per_class_distances:
            per_class_distances[assigned_cls].append(dist)
            per_class_records[assigned_cls].append((dist, wav_path))
        if _diag_printed < 8:
            logger.info(
                "  [diag] %s → cls=%s  nearest=%s  dist=%.6f  label_cls=%s",
                wav_path.name, assigned_cls, nearest_cls, dist, label_cls,
            )
            _diag_printed += 1
        if i % 20 == 0:
            logger.info("  %d / %d processed", i, len(wav_files))

    if fallback_to_centroid:
        logger.warning(
            "  %d WAVs had no folder label — assigned by nearest centroid.",
            fallback_to_centroid,
        )

    if not all_distances:
        logger.error("No embeddings extracted — check model_dir and calibration_dir.")
        return 1

    all_distances_arr = np.array(all_distances, dtype=np.float32)

    # ── print full percentile table ───────────────────────────────────────────
    print("\n=== Global distance distribution (calibration set) ===")
    print(f"  n_samples : {len(all_distances_arr)}")
    print(f"  mean      : {all_distances_arr.mean():.6f}   std: {all_distances_arr.std():.6f}")
    for pct in [50, 75, 85, 90, 95, 99, 100]:
        marker = " ◄ chosen" if pct == int(global_pct) else ""
        print(f"  p{pct:3d}     : {np.percentile(all_distances_arr, pct):.6f}{marker}")
    print(f"  current global_threshold: {state['global_threshold']:.6f}")
    new_global = float(np.percentile(all_distances_arr, global_pct))
    print(f"  new     global_threshold (p{global_pct}): {new_global:.6f}")

    OTHER_LABEL = "другие слова"
    print("\n=== Per-class distance distributions ===")
    new_class_thresholds: Dict[int, float] = {}
    for cls_id in sorted(centroids.keys()):
        dists = np.array(per_class_distances.get(cls_id, []), dtype=np.float32)
        label = ID2LABEL.get(cls_id, str(cls_id))
        is_other = label == OTHER_LABEL
        pct_used = other_pct if is_other else args.percentile
        if len(dists) == 0:
            logger.warning("  class %d (%s): no calibration samples — keeping old threshold", cls_id, label)
            new_class_thresholds[cls_id] = float(state["class_thresholds"][cls_id])
            continue

        # Guard: fall back to global threshold when too few samples make
        # high percentiles unreliable (one atypical embedding skews p99+).
        use_global_fallback = len(dists) < args.min_samples
        if use_global_fallback:
            new_thr = new_global
            fallback_note = f"⚠ fallback to global (n={len(dists)} < min_samples={args.min_samples})"
        else:
            new_thr = float(np.percentile(dists, pct_used))
            fallback_note = f"p{pct_used:.0f} {'[other_percentile]' if is_other else '[percentile]'}"

        new_class_thresholds[cls_id] = new_thr
        old_thr = float(state["class_thresholds"][cls_id])
        print(f"\n  class {cls_id} — {label}  (n={len(dists)})")
        print(f"    old threshold : {old_thr:.6f}")
        for pct in [50, 75, 85, 90, 95, 99]:
            marker = " ◄ chosen" if (not use_global_fallback and pct == int(pct_used)) else ""
            print(f"    p{pct:3d}         : {np.percentile(dists, pct):.6f}{marker}")
        print(f"    new threshold : {new_thr:.6f}  ({fallback_note})")

        if args.debug_top_n > 0 and cls_id in per_class_records:
            records = per_class_records[cls_id]
            top_n = sorted(records, key=lambda x: x[0], reverse=True)[: args.debug_top_n]
            print(f"    top-{args.debug_top_n} farthest samples:")
            for rank, (d, p) in enumerate(top_n, 1):
                # Show path relative to cwd for readability; fall back to full path.
                try:
                    rel = p.relative_to(Path.cwd())
                except ValueError:
                    rel = p
                outlier_marker = "  ← OUTLIER" if d > new_thr else ""
                print(f"      {rank:2d}. dist={d:.6f}  {rel}{outlier_marker}")

    # ── outlier rate comparison ───────────────────────────────────────────────
    print("\n=== Outlier rate on calibration set ===")
    old_outliers = sum(
        1 for d in all_distances if d > float(state["global_threshold"])
    )
    new_outliers_global = sum(1 for d in all_distances if d > new_global)
    n = len(all_distances)
    print(f"  old global threshold ({state['global_threshold']:.6f}): "
          f"{old_outliers}/{n} = {old_outliers/n*100:.1f}% outliers")
    print(f"  new global threshold ({new_global:.6f}): "
          f"{new_outliers_global}/{n} = {new_outliers_global/n*100:.1f}% outliers")

    if args.dry_run:
        logger.info("\nDry run — .npy NOT modified.")
        return 0

    # ── update and save ───────────────────────────────────────────────────────
    state["global_threshold"] = new_global
    state["threshold_percentile"] = args.percentile
    state["other_percentile"] = other_pct
    state["class_thresholds"] = {k: np.float64(v) for k, v in new_class_thresholds.items()}
    state["calibration_stats"] = {
        "n_samples": n,
        "percentile_used": args.percentile,
        "global_percentile_used": global_pct,
        "min_samples_for_per_class": args.min_samples,
        "mean_distance": float(all_distances_arr.mean()),
        "std_distance": float(all_distances_arr.std()),
        "p50": float(np.percentile(all_distances_arr, 50)),
        "p90": float(np.percentile(all_distances_arr, 90)),
        "p95": float(np.percentile(all_distances_arr, 95)),
        "p99": float(np.percentile(all_distances_arr, 99)),
        "calibration_dirs": [str(args.calibration_dir)] + [str(d) for d in (args.extra_dirs or [])],
    }

    np.save(str(args.detector_path), state)
    logger.info("\nSaved updated detector → %s", args.detector_path)
    logger.info("  global_threshold : %.6f", new_global)
    for cls_id, thr in new_class_thresholds.items():
        logger.info("  class %d (%s): %.6f", cls_id, ID2LABEL.get(cls_id, cls_id), thr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
