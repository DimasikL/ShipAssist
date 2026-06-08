"""
scripts/hybrid/train_regressor.py — Train a numeric slot-filling regressor.

What this script does
---------------------
1. Reads a dataset CSV with columns ``path``, ``label``, and ``value``
   (the numeric target, e.g. heading in degrees or speed in knots).
2. Filters rows to the target ``--intent`` label.
3. Extracts Wav2Vec2 embeddings for every sample.
4. Splits 80/20 train/val (stratified by value quantile).
5. Trains a ``NumberRegressor`` (MLP on top of embeddings).
6. Reports validation MAE.
7. Saves the fitted regressor to ``<out>/<intent_key>.pkl``.

CSV format
----------
The CSV must have at minimum:

    path,label,value
    artifacts/data/numbers/курс_001.wav,курс УГОЛ градусов,1
    artifacts/data/numbers/курс_030.wav,курс УГОЛ градусов,30
    ...

You can generate synthetic data with:
    python scripts/generation/main_audio_generate_tts.py \
        --phrase "курс {n} градусов" --range "1,360,15" \
        --out artifacts/data/numbers/

Usage
-----
    # Train a heading regressor:
    python scripts/hybrid/train_regressor.py \
        --csv artifacts/data/numbers.csv \
        --intent "курс УГОЛ градусов" \
        --min_val 1 --max_val 360 \
        --onnx_dir onnx_model/models/run_2026-02-25_19-07-15/best_model \
        --out artifacts/hybrid/regressors/

    # Train a speed regressor:
    python scripts/hybrid/train_regressor.py \
        --csv artifacts/data/numbers.csv \
        --intent "скорость УГОЛ узлов" \
        --min_val 0 --max_val 30 \
        --out artifacts/hybrid/regressors/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window
from core.hybrid.number_regressor import NumberRegressor
from core.logger import get_logger

logger = get_logger(__name__)

_SR = 16_000
_WIN_SAMPLES = 16_000


def _extract_embeddings(wav_paths: List[str], onnx_dir: str) -> tuple[np.ndarray, List[int]]:
    """Extract embeddings via OnnxEngine, return array and valid indices.

    Args:
        wav_paths: Paths to audio files.
        onnx_dir:  ONNX bundle directory.

    Returns:
        ``(embeddings, valid_indices)``
    """
    from core.onnx_engine import OnnxEngine

    engine = OnnxEngine(onnx_dir=onnx_dir, precision="int8")
    embs: List[np.ndarray] = []
    valid: List[int] = []

    for i, p in enumerate(wav_paths):
        try:
            wav, _ = load_wav(p, target_sr=_SR)
            audio = prepare_window(wav, target_samples=_WIN_SAMPLES, do_normalize=True)
            _, emb, _frames = engine.predict_logits(audio)
            if emb is None:
                raise ValueError("No embedding output from ONNX.")
            embs.append(emb.astype(np.float32))
            valid.append(i)
        except Exception as exc:
            logger.warning("Skipping %s: %s", p, exc)

    if not embs:
        raise RuntimeError("No embeddings extracted.")
    return np.stack(embs, axis=0), valid


def main(args: argparse.Namespace) -> None:
    """Train and save a NumberRegressor.

    Args:
        args: Parsed CLI arguments.
    """
    # ── Load and filter CSV ────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path).dropna(subset=["path", "label", "value"])
    df_intent = df[df["label"] == args.intent].copy()
    if len(df_intent) < 4:
        logger.error(
            "Too few samples for intent '%s': found %d (need ≥ 4).",
            args.intent, len(df_intent),
        )
        sys.exit(1)

    logger.info(
        "Intent '%s': %d samples, value range [%.1f, %.1f].",
        args.intent,
        len(df_intent),
        df_intent["value"].min(),
        df_intent["value"].max(),
    )

    # ── Extract embeddings ─────────────────────────────────────────────
    logger.info("Extracting embeddings…")
    wav_paths = df_intent["path"].tolist()
    all_values = df_intent["value"].values.astype(np.float32)

    embeddings, valid_idx = _extract_embeddings(wav_paths, args.onnx_dir)
    values = all_values[valid_idx]

    logger.info(
        "Valid embeddings: %d / %d  shape=%s",
        len(valid_idx), len(wav_paths), embeddings.shape,
    )

    # ── Train/val split (by value quantile for uniform coverage) ──────
    n = len(embeddings)
    val_size = max(1, int(n * 0.2))
    # Stratified by quantile: sort by value, take every 5th sample for val
    sort_idx = np.argsort(values)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[sort_idx[::max(1, n // val_size)]] = True
    val_mask = val_mask[:n]    # ensure correct length after argsort stride

    X_train = embeddings[~val_mask]
    y_train = values[~val_mask]
    X_val = embeddings[val_mask]
    y_val = values[val_mask]

    logger.info(
        "Split: train=%d, val=%d  value_range=[%.1f, %.1f]",
        len(X_train), len(X_val), args.min_val, args.max_val,
    )

    # ── Train ──────────────────────────────────────────────────────────
    reg = NumberRegressor(
        min_val=args.min_val,
        max_val=args.max_val,
        hidden_neurons=args.hidden_neurons,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    reg.fit(X_train, y_train, X_val, y_val)

    # ── Validation report ──────────────────────────────────────────────
    preds = np.array([reg.predict(e) for e in X_val], dtype=np.float32)
    mae = float(np.abs(preds - y_val).mean())
    max_err = float(np.abs(preds - y_val).max())

    print("\n" + "=" * 55)
    print(f"Intent:      {args.intent!r}")
    print(f"Val samples: {len(X_val)}")
    print(f"Val MAE:     {mae:.2f}  (Max error: {max_err:.2f})")
    print(f"Value range: [{args.min_val}, {args.max_val}]")
    print("=" * 55)

    # ── Save ───────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_key = args.intent.replace(" ", "_")
    out_path = out_dir / f"{safe_key}.pkl"
    reg.save(out_path)
    print(f"\nRegressor saved to: {out_path.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Train a NumberRegressor for a slot intent."
    )
    parser.add_argument("--csv", required=True, help="CSV with path, label, value columns.")
    parser.add_argument("--intent", required=True, help="Exact slot intent label string.")
    parser.add_argument("--min_val", type=float, required=True, help="Minimum numeric value.")
    parser.add_argument("--max_val", type=float, required=True, help="Maximum numeric value.")
    parser.add_argument(
        "--onnx_dir",
        default="onnx_model/models/run_2026-02-25_19-07-15/best_model",
        help="ONNX bundle directory.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/hybrid/regressors/",
        help="Output directory for the .pkl file.",
    )
    parser.add_argument("--hidden_neurons", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    main(parser.parse_args())
