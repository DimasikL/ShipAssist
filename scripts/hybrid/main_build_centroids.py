"""
scripts/hybrid/build_centroids.py — Extract Wav2Vec2 embeddings and build centroids.

What this script does
---------------------
1. Reads a dataset CSV (same format as the main training CSV: must have ``path``
   and ``label`` columns).
2. Extracts a Wav2Vec2 embedding for every audio file using the project's
   existing ONNX engine (``core.onnx_engine.OnnxEngine``).
3. Computes the per-class mean centroid (L2-normalised).
4. Saves ``centroids.npy`` and ``centroid_labels.json`` to the output directory.
5. (Optional) Appends new phrases to an existing centroid registry without
   disturbing existing phrases (``--append`` flag).

Usage
-----
    # Build from scratch:
    python scripts/hybrid/build_centroids.py \
        --csv artifacts/data/dataset.csv \
        --onnx_dir onnx_model/models/run_2026-02-25_19-07-15/best_model \
        --out artifacts/hybrid/

    # Append a new phrase only:
    python scripts/hybrid/build_centroids.py \
        --csv new_phrase_recordings.csv \
        --onnx_dir onnx_model/models/run_2026-02-25_19-07-15/best_model \
        --out artifacts/hybrid/ \
        --append

    # Use standalone WTVEmbedder instead of ONNX:
    python scripts/hybrid/build_centroids.py \
        --csv artifacts/data/dataset.csv \
        --embedder wtv \
        --hf_model jonatasgrosman/wav2vec2-large-xlsr-53-russian \
        --out artifacts/hybrid/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window
from core.hybrid.centroid_search import CentroidSearch
from core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_SR = 16_000
_WIN_SAMPLES = 16_000


# ── Embedding extractors ───────────────────────────────────────────────────────

def _extract_with_onnx(wav_paths: List[str], onnx_dir: str) -> Tuple[np.ndarray, List[int]]:
    """Extract embeddings using the existing ONNX engine (outputs[1]).

    Args:
        wav_paths: List of absolute paths to ``.wav`` files.
        onnx_dir:  Path to the ONNX bundle directory.

    Returns:
        Tuple of ``(embeddings, valid_indices)`` — valid_indices maps rows in
        the output array back to rows in the input list.
    """
    from core.onnx_engine import OnnxEngine

    engine = OnnxEngine(onnx_dir=onnx_dir, precision="int8")
    embeddings: List[np.ndarray] = []
    valid_indices: List[int] = []
    skipped = 0

    for i, path in enumerate(wav_paths):
        try:
            # Normalise mixed slashes (common in CSV paths on Windows)
            norm_path = str(Path(path))
            wav, _ = load_wav(norm_path, target_sr=_SR)
            audio = prepare_window(wav, target_samples=_WIN_SAMPLES, do_normalize=True)
            _logits, emb, _frames = engine.predict_logits(audio)
            if emb is None:
                raise ValueError("ONNX model did not return embedding (outputs[1] missing).")
            embeddings.append(emb.astype(np.float32))
            valid_indices.append(i)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            skipped += 1

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d files (skipped=%d)", i + 1, len(wav_paths), skipped)

    if skipped:
        logger.warning("Total skipped: %d / %d files.", skipped, len(wav_paths))

    if not embeddings:
        raise RuntimeError("No embeddings extracted — check ONNX model and audio files.")

    return np.stack(embeddings, axis=0), valid_indices


def _extract_with_wtv(wav_paths: List[str], hf_model: str) -> Tuple[np.ndarray, List[int]]:
    """Extract embeddings using the standalone WTVEmbedder (HuggingFace).

    Args:
        wav_paths: List of absolute paths to ``.wav`` files.
        hf_model:  HuggingFace model identifier.

    Returns:
        Tuple of ``(embeddings, valid_indices)``.
    """
    from core.embedders import WTVEmbedder
    from core.preproc import Preproc1

    embedder = WTVEmbedder(
        sr=_SR,
        preproc=Preproc1(sr=_SR),
        emb_model=hf_model,
        output_hidden_states=True,
    )
    embeddings: List[np.ndarray] = []
    valid_indices: List[int] = []
    skipped = 0

    for i, path in enumerate(wav_paths):
        try:
            norm_path = str(Path(path))
            emb = embedder.get_emb(wav_path=norm_path)
            if emb is None:
                raise ValueError("Embedder returned None.")
            embeddings.append(np.asarray(emb, dtype=np.float32).flatten())
            valid_indices.append(i)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            skipped += 1

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d files.", i + 1, len(wav_paths))

    if not embeddings:
        raise RuntimeError("No embeddings extracted.")

    return np.stack(embeddings, axis=0), valid_indices


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Main entry-point for building centroids.

    Args:
        args: Parsed command-line arguments.
    """
    # ── Load dataset CSV ──────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    path_col = args.path_col
    label_col = args.label_col

    df = pd.read_csv(csv_path)
    required_cols = {path_col, label_col}
    if not required_cols.issubset(df.columns):
        logger.error(
            "CSV must have columns %s, found: %s", required_cols, list(df.columns)
        )
        sys.exit(1)

    df = df.dropna(subset=[path_col, label_col])
    wav_paths = df[path_col].tolist()
    labels: List[str] = df[label_col].tolist()

    logger.info(
        "Loaded CSV: %d samples, %d unique labels: %s",
        len(df), df[label_col].nunique(), sorted(df[label_col].unique()),
    )

    # ── Extract embeddings ────────────────────────────────────────────
    logger.info("Extracting embeddings using '%s' embedder…", args.embedder)

    if args.embedder == "onnx":
        if not args.onnx_dir:
            logger.error("--onnx_dir is required when --embedder=onnx")
            sys.exit(1)
        embeddings, valid_indices = _extract_with_onnx(wav_paths, args.onnx_dir)
    elif args.embedder == "wtv":
        if not args.hf_model:
            logger.error("--hf_model is required when --embedder=wtv")
            sys.exit(1)
        embeddings, valid_indices = _extract_with_wtv(wav_paths, args.hf_model)
    else:
        logger.error("Unknown embedder: %s. Choose 'onnx' or 'wtv'.", args.embedder)
        sys.exit(1)

    # Filter labels to only those for which embeddings were successfully extracted
    labels_filtered: List[str] = [labels[i] for i in valid_indices]
    logger.info(
        "Embeddings extracted: %d / %d  shape=%s, dtype=%s",
        len(valid_indices), len(wav_paths), embeddings.shape, embeddings.dtype,
    )

    # ── Build (or append to) centroid registry ────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = out_dir / "centroids.npy"
    l_path = out_dir / "centroid_labels.json"

    if args.append and c_path.exists() and l_path.exists():
        logger.info("--append: loading existing registry from %s", out_dir)
        search = CentroidSearch.load_npz(c_path, l_path)
    else:
        search = CentroidSearch()

    search.build_from_embeddings(embeddings, labels_filtered)

    # ── Save ─────────────────────────────────────────────────────────
    search.save_npz(c_path, l_path)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Centroids saved to:      {c_path}")
    print(f"Labels saved to:         {l_path}")
    print(f"Registered labels ({search.n_labels}):")
    for lbl in search.labels:
        print(f"  - {lbl!r}")
    print(f"Embedding dimension:     {search.embedding_dim}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build CentroidSearch artefacts from a labelled dataset CSV."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to dataset CSV (columns: path, label).",
    )
    parser.add_argument(
        "--out", default="artifacts/hybrid/",
        help="Output directory for centroids.npy and centroid_labels.json.",
    )
    parser.add_argument(
        "--embedder", choices=["onnx", "wtv"], default="onnx",
        help="Embedding source: 'onnx' (fast, uses existing engine) or 'wtv' (HF model).",
    )
    parser.add_argument(
        "--onnx_dir",
        default="onnx_model/models/run_2026-02-25_19-07-15/best_model",
        help="ONNX bundle directory (required for --embedder=onnx).",
    )
    parser.add_argument(
        "--hf_model",
        default="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        help="HuggingFace model identifier (required for --embedder=wtv).",
    )
    parser.add_argument(
        "--append", action="store_true",
        help=(
            "Append new phrases to an existing registry instead of rebuilding "
            "from scratch. Existing centroids are preserved unchanged."
        ),
    )
    parser.add_argument(
        "--path_col", default="path",
        help="Name of the CSV column containing audio file paths (default: 'path').",
    )
    parser.add_argument(
        "--label_col", default="label",
        help="Name of the CSV column containing class labels (default: 'label').",
    )

    main(parser.parse_args())
