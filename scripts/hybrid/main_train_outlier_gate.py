"""
scripts/hybrid/train_outlier_gate.py - Fit and save an OutlierGate or
EnsembleOutlierGate.

What this script does
---------------------
1. Reads a labelled dataset CSV (``path``, ``label`` columns).
2. Optionally collects additional WAVs from ``--extra_dirs`` and negatives
   from ``--negatives_dirs`` (labels inferred from folder hierarchy).
3. Extracts Wav2Vec2 embeddings for all samples via ONNX.
4. Fits an outlier gate using the full embedding set:
   - ``--method mahalanobis|cosine|l2``: fits a single-metric ``OutlierGate``.
   - ``--method ensemble``: fits an ``EnsembleOutlierGate`` (Scenario 1 -
     Mahalanobis + cosine + L2 combined via z-score normalization, with
     optional adaptive per-class tau). **Recommended for production.**
5. Calibrates the rejection threshold at the configured percentile.
6. Saves the fitted gate to ``artifacts/hybrid/outlier_gate.pkl``.

Usage
-----
    # Single-metric (original):
    python scripts/hybrid/train_outlier_gate.py \\
        --csv artifacts/data/dataset.csv \\
        --onnx_dir onnx_model/models/run_2026-02-25_19-07-15/best_model \\
        --out artifacts/hybrid/outlier_gate.pkl \\
        --method mahalanobis \\
        --percentile 95.0

    # Ensemble OOD + adaptive tau (Scenario 1, recommended):
    python scripts/hybrid/train_outlier_gate.py \\
        --csv artifacts/data/dataset.csv \\
        --onnx_dir onnx_model/models/run_2026-02-25_19-07-15/best_model \\
        --out artifacts/hybrid/outlier_gate.pkl \\
        --method ensemble \\
        --percentile 95.0 \\
        --use_adaptive_tau \\
        --ensemble_weights 2.0 1.0 1.0 \\
        --extra_dirs clf_dset/train_val \\
        --negatives_dirs clf_dset/test \\
        --include_groups "new user 10" "new user 11" "new user 12"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window
from core.hybrid.outlier_gate import EnsembleOutlierGate, OutlierGate
from core.logger import get_logger

logger = get_logger(__name__)

_SR = 16_000
_WIN_SAMPLES = 16_000

# ── Label inference from directory structure ───────────────────────────────────
# Maps folder-name fragments (lowercased, spaces/underscores normalised) -> label.
# More specific patterns must come first (longest match wins).
_FOLDER_TO_LABEL: List[Tuple[str, str]] = [
    ("приготовить_машину", "приготовить машину"),
    ("приготовить машину",  "приготовить машину"),
    ("самый_малый_вперед",  "самый малый вперед"),
    ("самый малый вперед",  "самый малый вперед"),
    ("самый_малый_вперёд",  "самый малый вперед"),
    ("самый малый вперёд",  "самый малый вперед"),
    ("машина",              "машина"),
    ("другие слова",        "другие слова"),
    ("genwords",            "другие слова"),
    ("negatives",           "другие слова"),
]


def _label_from_path(wav_path: Path) -> Optional[str]:
    """Infer class label from the WAV file's directory hierarchy.

    Walks from the immediate parent folder upward (skips the filename itself).
    Returns None if no pattern matches.
    """
    dir_parts = [p.lower() for p in wav_path.parts[:-1]]
    for part in reversed(dir_parts):
        part_us = part.replace("_", " ")
        for pattern, label in _FOLDER_TO_LABEL:
            if pattern in part or pattern in part_us:
                return label
    return None


def _collect_wavs_from_dir(
    directory: Path,
    include_aug: bool,
    exclude_patterns: List[str],
    include_patterns: List[str],
    max_seconds: float,
    negatives_only: bool = False,
) -> List[Tuple[Path, Optional[str]]]:
    """Collect WAV files from *directory*, applying all filters.

    Args:
        directory:        Root directory to search recursively.
        include_aug:      If False, skip files whose path contains '-aug'.
        exclude_patterns: Drop files whose lowercased path contains any of these.
        include_patterns: Keep only files whose path contains at least one of these
                          (applied after exclude; ignored when empty).
        max_seconds:      Skip WAV files longer than this (avoids truncated embeddings).
        negatives_only:   If True, collect only files inside ``negatives/`` subfolders.

    Returns:
        List of (wav_path, label_or_None) pairs.
    """
    import wave as wave_mod

    def _duration(p: Path) -> float:
        try:
            with wave_mod.open(str(p)) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            return 0.0

    def _is_excluded(p: Path) -> bool:
        path_lower = str(p).lower()
        if exclude_patterns and any(pat in path_lower for pat in exclude_patterns):
            return True
        if include_patterns and not any(pat in path_lower for pat in include_patterns):
            return True
        return False

    if negatives_only:
        wavs: List[Path] = []
        for neg_subdir in sorted(directory.rglob("negatives")):
            if neg_subdir.is_dir():
                wavs.extend(sorted(neg_subdir.rglob("*.wav")))
    else:
        wavs = sorted(directory.rglob("*.wav"))

    if not include_aug:
        before = len(wavs)
        wavs = [p for p in wavs if not any("-aug" in part for part in p.parts)]
        if before - len(wavs):
            logger.info("  Skipped %d aug WAVs in %s", before - len(wavs), directory)

    before = len(wavs)
    wavs = [p for p in wavs if _duration(p) <= max_seconds + 0.1]
    if before - len(wavs):
        logger.info(
            "  Skipped %d WAVs longer than %.1fs in %s",
            before - len(wavs), max_seconds, directory,
        )

    if exclude_patterns or include_patterns:
        before = len(wavs)
        wavs = [p for p in wavs if not _is_excluded(p)]
        if before - len(wavs):
            logger.info(
                "  Skipped %d WAVs by group filters in %s",
                before - len(wavs), directory,
            )

    return [(p, _label_from_path(p)) for p in wavs]


def _extract_embeddings_onnx(
    wav_paths: List[str], onnx_dir: str
) -> Tuple[np.ndarray, List[int]]:
    """Extract ONNX embeddings and return array + list of valid indices.

    Args:
        wav_paths: Audio file paths.
        onnx_dir:  ONNX bundle directory.

    Returns:
        Tuple of ``(embeddings_array, valid_indices)`` where valid_indices
        maps rows in the output array back to rows in the input list.
    """
    from core.onnx_engine import OnnxEngine

    engine = OnnxEngine(onnx_dir=onnx_dir, precision="int8")
    embeddings: List[np.ndarray] = []
    valid_indices: List[int] = []

    for i, path in enumerate(wav_paths):
        try:
            wav, _ = load_wav(path, target_sr=_SR)
            audio = prepare_window(wav, target_samples=_WIN_SAMPLES, do_normalize=True)
            _, emb, _frames = engine.predict_logits(audio)
            if emb is None:
                raise ValueError("ONNX model has no embedding output (outputs[1]).")
            embeddings.append(emb.astype(np.float32))
            valid_indices.append(i)
        except Exception as exc:
            logger.warning("Skipping file %s: %s", path, exc)

        if (i + 1) % 100 == 0:
            logger.info("  %d/%d done.", i + 1, len(wav_paths))

    if not embeddings:
        raise RuntimeError("No embeddings could be extracted.")

    return np.stack(embeddings, axis=0), valid_indices


def _extract_embeddings_torch(
    wav_paths: List[str], model_dir: str
) -> Tuple[np.ndarray, List[int]]:
    """Extract embeddings via PyTorch (supports LoRA checkpoints).

    Loads a LoRA or plain Wav2Vec2ForSequenceClassification checkpoint,
    merges LoRA weights if present, and runs mean-pool over projected
    hidden states — identical to the ONNX ExportWrapper.forward() path.

    Args:
        wav_paths: Audio file paths.
        model_dir: Path to the model directory (LoRA or plain checkpoint).

    Returns:
        Tuple of ``(embeddings_array, valid_indices)``.
    """
    import json as _json
    import torch
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, AutoConfig

    model_path = Path(model_dir)
    adapter_cfg = model_path / "adapter_config.json"
    is_lora = adapter_cfg.exists()

    if is_lora:
        try:
            from peft import PeftModel
        except ImportError:
            raise RuntimeError("peft not installed — run: pip install peft")
        with open(adapter_cfg) as _f:
            _acfg = _json.load(_f)
        base_name = _acfg.get("base_model_name_or_path")
        if not base_name:
            raise RuntimeError("adapter_config.json missing 'base_model_name_or_path'")
        logger.info("PyTorch: LoRA detected, base=%s", base_name)
        ft_config = AutoConfig.from_pretrained(str(model_path))
        base = Wav2Vec2ForSequenceClassification.from_pretrained(base_name, config=ft_config)
        lora_model = PeftModel.from_pretrained(base, str(model_path))
        model = lora_model.merge_and_unload()
        logger.info("PyTorch: LoRA weights merged.")
    else:
        logger.info("PyTorch: loading plain checkpoint from %s", model_dir)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(str(model_path))

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("PyTorch: model on %s", device)

    # Feature extractor: prefer one saved alongside the model, else use base.
    fe_path = str(model_path) if (model_path / "preprocessor_config.json").exists() else (
        _acfg.get("base_model_name_or_path") if is_lora else str(model_path)
    )
    fe = Wav2Vec2FeatureExtractor.from_pretrained(fe_path)

    embeddings: List[np.ndarray] = []
    valid_indices: List[int] = []

    with torch.no_grad():
        for i, path in enumerate(wav_paths):
            try:
                wav, _ = load_wav(path, target_sr=_SR)
                audio = prepare_window(wav, target_samples=_WIN_SAMPLES, do_normalize=True)
                # Feature extractor expects raw float32 waveform.
                inputs = fe(
                    audio,
                    sampling_rate=_SR,
                    return_tensors="pt",
                    padding=False,
                )
                input_values = inputs.input_values.to(device)
                outputs = model.wav2vec2(input_values)
                hidden = outputs.last_hidden_state          # (1, T, D_model)
                projected = model.projector(hidden)         # (1, T, D_proj)
                emb = projected.mean(dim=1).squeeze(0)      # (D_proj,)
                embeddings.append(emb.cpu().numpy().astype(np.float32))
                valid_indices.append(i)
            except Exception as exc:
                logger.warning("Skipping file %s: %s", path, exc)

            if (i + 1) % 100 == 0:
                logger.info("  %d/%d done.", i + 1, len(wav_paths))

    if not embeddings:
        raise RuntimeError("No embeddings could be extracted.")

    return np.stack(embeddings, axis=0), valid_indices


def main(args: argparse.Namespace) -> None:
    """Fit and save an OutlierGate.

    Args:
        args: Parsed CLI arguments.
    """
    # ── Pre-process group filter patterns ─────────────────────────────
    exclude_patterns: List[str] = [g.lower() for g in (args.exclude_groups or [])]
    include_patterns: List[str] = [g.lower() for g in (args.include_groups or [])]

    def _is_excluded_csv(path_str: str) -> bool:
        p = path_str.lower()
        if exclude_patterns and any(pat in p for pat in exclude_patterns):
            return True
        if include_patterns and not any(pat in p for pat in include_patterns):
            return True
        return False

    # ── Load CSV ───────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    path_col = args.path_col
    label_col = args.label_col

    df = pd.read_csv(csv_path).dropna(subset=[path_col, label_col])

    # Apply group filters to CSV rows (same logic as --include/exclude_groups).
    if exclude_patterns or include_patterns:
        before = len(df)
        df = df[~df[path_col].apply(_is_excluded_csv)]
        logger.info(
            "Group filters applied to CSV: %d -> %d rows (dropped %d)",
            before, len(df), before - len(df),
        )

    # Filter aug WAVs from CSV (mirrors --include_aug behaviour in extra_dirs).
    if not args.include_aug:
        before = len(df)
        df = df[~df[path_col].str.contains("-aug", case=False, na=False)]
        skipped = before - len(df)
        if skipped:
            logger.info("  Skipped %d aug WAVs in CSV", skipped)

    wav_paths: List[str] = df[path_col].tolist()
    all_labels: List[str] = df[label_col].tolist()

    logger.info(
        "CSV dataset: %d samples, %d unique labels.", len(df), df[label_col].nunique()
    )

    # ── Collect extra WAVs from --extra_dirs ──────────────────────────
    extra_wavs: List[Tuple[str, str]] = []
    for extra_dir in (args.extra_dirs or []):
        extra_dir = Path(extra_dir)
        if not extra_dir.exists():
            logger.warning("--extra_dirs: directory not found, skipping: %s", extra_dir)
            continue
        collected = _collect_wavs_from_dir(
            extra_dir, args.include_aug, exclude_patterns, include_patterns,
            args.max_seconds, negatives_only=False,
        )
        # Drop entries that have no inferrable label (would be noise in fitting).
        labelled = [(str(p), lbl) for p, lbl in collected if lbl is not None]
        unlabelled = len(collected) - len(labelled)
        if unlabelled:
            logger.warning(
                "  %d WAVs in %s had no inferrable label - skipped.",
                unlabelled, extra_dir,
            )
        extra_wavs.extend(labelled)
        logger.info("extra_dir %s: +%d labelled WAVs", extra_dir, len(labelled))

    # ── Collect negatives from --negatives_dirs ───────────────────────
    neg_wavs: List[Tuple[str, str]] = []
    for neg_dir in (args.negatives_dirs or []):
        neg_dir = Path(neg_dir)
        if not neg_dir.exists():
            logger.warning("--negatives_dirs: directory not found, skipping: %s", neg_dir)
            continue
        collected = _collect_wavs_from_dir(
            neg_dir, args.include_aug, exclude_patterns, include_patterns,
            args.max_seconds, negatives_only=True,
        )
        labelled = [(str(p), lbl) for p, lbl in collected if lbl is not None]
        neg_wavs.extend(labelled)
        logger.info("negatives_dir %s: +%d WAVs (negatives/ only)", neg_dir, len(labelled))

    # ── Merge all sources ─────────────────────────────────────────────
    all_paths = wav_paths + [p for p, _ in extra_wavs] + [p for p, _ in neg_wavs]
    all_labels_merged = (
        all_labels
        + [lbl for _, lbl in extra_wavs]
        + [lbl for _, lbl in neg_wavs]
    )

    if exclude_patterns:
        logger.info("Excluded groups: %s", args.exclude_groups)
    if include_patterns:
        logger.info("Include-only groups: %s", args.include_groups)
    logger.info(
        "Total samples: %d  (csv=%d  extra=%d  negatives=%d)",
        len(all_paths), len(wav_paths), len(extra_wavs), len(neg_wavs),
    )

    # ── Extract embeddings ─────────────────────────────────────────────
    if args.pt_model_dir:
        logger.info("Extracting embeddings (PyTorch): %s", args.pt_model_dir)
        embeddings, valid_idx = _extract_embeddings_torch(all_paths, args.pt_model_dir)
    else:
        logger.info("Extracting embeddings (ONNX): %s", args.onnx_dir)
        embeddings, valid_idx = _extract_embeddings_onnx(all_paths, args.onnx_dir)
    labels = [all_labels_merged[i] for i in valid_idx]

    logger.info(
        "Valid embeddings: %d / %d  shape=%s",
        len(valid_idx), len(all_paths), embeddings.shape,
    )

    # ── Fit gate ───────────────────────────────────────────────────────
    if args.method == "ensemble":
        w = args.ensemble_weights
        gate = EnsembleOutlierGate(
            weights=(w[0], w[1], w[2]),
            percentile=args.percentile,
            use_adaptive_tau=args.use_adaptive_tau,
            per_class_percentile=args.per_class_percentile,
            regularization_eps=args.regularization_eps,
            fallback_threshold=0.0,
        )
        gate.fit(embeddings, labels)
    else:
        gate = OutlierGate(
            method=args.method,
            percentile=args.percentile,
            use_adaptive_tau=args.use_adaptive_tau,
            per_class_percentile=args.per_class_percentile,
            regularization_eps=args.regularization_eps,
            fallback_threshold=args.fallback_threshold,
        )
        gate.fit(embeddings, labels)

    # ── Print calibration summary ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"OutlierGate calibration summary  [method={args.method}]")
    print("=" * 60)
    for k, v in gate.summary().items():
        print(f"  {k}: {v}")

    if args.method != "ensemble":
        label_arr = np.array(labels)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms

        print("\nPer-class distance statistics (to nearest centroid):")
        for lbl in sorted(set(labels)):
            mask = label_arr == lbl
            dists = np.array([gate._distance_to_nearest(e) for e in normed[mask]])
            print(
                f"  {lbl!r:40s}  n={mask.sum():4d}  "
                f"mean={dists.mean():.3f}  p95={np.percentile(dists, 95):.3f}  "
                f"max={dists.max():.3f}"
            )
        print(
            f"\nCalibrated threshold (@{gate.percentile:.0f}th pct): {gate._threshold:.4f}"
        )
    else:
        print("\nPer-class adaptive tau:")
        for lbl, tau in sorted(gate._per_class_thresholds.items()):
            print(f"  {lbl!r:40s}  tau={tau:.4f}")
        print(
            f"\nGlobal threshold (@{gate.percentile:.0f}th pct): {gate._global_threshold:.4f}"
        )

        # ── Debug: ensemble score distribution per class ───────────────
        print("\n" + "=" * 60)
        print("DEBUG: ensemble score distribution per class")
        print("=" * 60)
        label_arr = np.array(labels)
        scores_all = np.array([gate.score(e)[0] for e in embeddings])
        for lbl in sorted(set(labels)):
            mask = label_arr == lbl
            s = scores_all[mask]
            nearest_labels = [gate.score(e)[1] for e in embeddings[mask]]
            wrong = sum(1 for nl in nearest_labels if nl != lbl)
            print(
                f"  {lbl!r:40s}  n={mask.sum():4d}  "
                f"min={s.min():.3f}  p50={np.percentile(s,50):.3f}  "
                f"p95={np.percentile(s,95):.3f}  p99={np.percentile(s,99):.3f}  "
                f"max={s.max():.3f}  wrong_centroid={wrong}"
            )
        print("=" * 60)

    print("=" * 60)

    # ── Save gate ──────────────────────────────────────────────────────
    out_path = Path(args.out)
    gate.save(out_path)
    print(f"\nOutlierGate saved to: {out_path.resolve()}")

    # ── Save embeddings cache (for fast diagnose_centroid_confusion.py) ─
    # Always saved next to the gate as <gate_stem>_embeddings.npz
    # Override path with --save_embeddings if needed.
    emb_out_default = out_path.with_name(out_path.stem + "_embeddings.npz")
    emb_out = Path(args.save_embeddings) if getattr(args, "save_embeddings", None) else emb_out_default
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    valid_paths = [all_paths[i] for i in valid_idx]
    np.savez_compressed(
        str(emb_out),
        embeddings=embeddings,          # (N, D) float32
        labels=np.array(labels),        # (N,) str
        paths=np.array(valid_paths),    # (N,) str
    )
    logger.info("Embeddings cache saved to %s  (shape=%s)", emb_out, embeddings.shape)
    print(f"Embeddings cache saved to: {emb_out.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Fit an OutlierGate on Wav2Vec2 embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True,
                        help="Dataset CSV with 'path' and 'label' columns.")
    parser.add_argument("--onnx_dir",
                        default="onnx_model/models/run_2026-02-25_19-07-15/best_model",
                        help="ONNX bundle directory.")
    parser.add_argument("--pt_model_dir", default=None,
                        help="PyTorch model directory (LoRA or plain checkpoint). "
                             "If set, takes priority over --onnx_dir.")
    parser.add_argument("--out",
                        default="artifacts/hybrid/outlier_gate.pkl",
                        help="Output path for the fitted gate.")
    parser.add_argument("--method",
                        choices=["mahalanobis", "cosine", "l2", "ensemble"],
                        default="mahalanobis",
                        help="Distance metric. Use 'ensemble' for Scenario 1.")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Percentile of training distances for threshold calibration.")
    parser.add_argument("--regularization_eps", type=float, default=1e-4,
                        help="Covariance regularization (Mahalanobis only).")
    parser.add_argument("--fallback_threshold", type=float, default=8.0,
                        help="Fallback threshold before calibration (single-metric only).")
    parser.add_argument("--use_adaptive_tau", action="store_true", default=False,
                        help="Enable per-class adaptive thresholds. Recommended with --method=ensemble.")
    parser.add_argument("--per_class_percentile", type=float, default=None,
                        help="Percentile for per-class tau (default: same as --percentile).")
    parser.add_argument("--ensemble_weights", type=float, nargs=3,
                        default=[2.0, 1.0, 1.0], metavar=("W_MAHAL", "W_COS", "W_L2"),
                        help="Weights for [mahalanobis, cosine, l2] in ensemble score.")
    parser.add_argument("--path_col", default="path",
                        help="CSV column name for audio file paths.")
    parser.add_argument("--label_col", default="label",
                        help="CSV column name for class labels.")
    parser.add_argument("--extra_dirs", type=Path, nargs="*", default=[],
                        help="Additional WAV dirs (all subfolders, labels inferred from folder hierarchy).")
    parser.add_argument("--negatives_dirs", type=Path, nargs="*", default=[],
                        help="Dirs where ONLY negatives/ subfolders are collected.")
    parser.add_argument("--include_groups", type=str, nargs="*", default=[],
                        help="Keep ONLY files whose path contains one of these (case-insensitive).")
    parser.add_argument("--exclude_groups", type=str, nargs="*", default=[],
                        help="Exclude files whose path contains any of these (case-insensitive).")
    parser.add_argument("--include_aug", action="store_true", default=False,
                        help="Include augmented WAVs (paths with -aug). Excluded by default.")
    parser.add_argument("--max_seconds", type=float, default=3.0,
                        help="Max WAV duration for extra_dirs/negatives_dirs.")
    parser.add_argument("--save_embeddings", type=str, default=None,
                        metavar="PATH",
                        help="If set, save extracted embeddings+labels+paths to a .npz "
                             "cache file (e.g. artifacts/hybrid/embeddings_cache.npz). "
                             "Load later with diagnose_centroid_confusion.py --load_embeddings.")

    main(parser.parse_args())