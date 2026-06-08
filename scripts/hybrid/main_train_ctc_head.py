"""
scripts/hybrid/train_ctc_head.py — Train the CTC digit head (Variant B).

What this script does
---------------------
1. Reads a CSV with columns ``path``, ``label``, ``digits``.
2. Filters rows to the slot intents listed in ``--intents`` (or all slot intents
   from the routing config if not specified).
3. Extracts per-frame projected features from the **PyTorch** model
   (wav2vec2 + projector, both frozen) for each audio clip.
4. Trains a single ``Linear(frame_dim, 11)`` CTC head over all intents jointly,
   using ``torch.nn.CTCLoss``.
5. Reports validation CER (character error rate on digit sequences) per epoch.
6. Saves the trained head to ``--out`` (default ``artifacts/hybrid/ctc_digit_head.pt``).

CSV format
----------
The CSV must have at minimum:

    path,label,numerals
    artifacts/data/nums/kurs_100.wav,курс УГОЛ градусов,сто
    artifacts/data/nums/kurs_285.wav,курс УГОЛ градусов,двести восемьдесят пять
    artifacts/data/nums/speed_15.wav,скорость УГОЛ узлов,пятнадцать

``numerals`` is a space-separated sequence of Russian numeral word tokens
from the vocabulary in ``core/hybrid/ctc_digit_decoder.py``.  Only the
numeral portion of the command is listed — NOT the surrounding phrase
("курс", "градусов", etc.).

Decoding rule: SUM all token values.
    двести восемьдесят пять  →  200 + 80 + 5 = 285
    тридцать                 →  30
    двадцать один            →  21

Generate synthetic data with:
    python scripts/generation/main_audio_generate_tts.py \\
        --phrase "курс {numeral} градусов" \\
        --values "один,двадцать,сто,двести восемьдесят,триста пятьдесят девять" \\
        --out artifacts/data/nums/

Verbalization assumption
------------------------
This script assumes **full Russian compound numerals**:
    "курс двести восемьдесят пять градусов"   →  numerals = "двести восемьдесят пять"
    "скорость пятнадцать узлов"               →  numerals = "пятнадцать"
    "поворот влево на девяносто"              →  numerals = "девяносто"

Usage
-----
    # Train on all slot intents at once:
    python scripts/hybrid/train_ctc_head.py \\
        --csv artifacts/data/digits.csv \\
        --model_dir best_model \\
        --out artifacts/hybrid/ctc_digit_head.pt

    # Train only on heading commands:
    python scripts/hybrid/train_ctc_head.py \\
        --csv artifacts/data/digits.csv \\
        --model_dir best_model \\
        --intents "курс УГОЛ градусов" "скорость УГОЛ узлов" \\
        --out artifacts/hybrid/ctc_digit_head.pt

    # Control training:
    python scripts/hybrid/train_ctc_head.py \\
        --csv artifacts/data/digits.csv \\
        --model_dir best_model \\
        --epochs 100 --lr 3e-4 --batch_size 16 \\
        --out artifacts/hybrid/ctc_digit_head.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window
from core.hybrid.ctc_digit_decoder import (
    BLANK_IDX,
    VOCAB_SIZE,
    WORD_TO_TOKEN,
    CTCDigitDecoder,
    DigitCTCHead,
)
from core.logger import get_logger

logger = get_logger(__name__)

_SR = 16_000
_WIN_SAMPLES = 16_000


# ── Feature extraction ────────────────────────────────────────────────────────

def _load_frozen_encoder(model_dir: str, device: str):
    """Load wav2vec2 + projector from a PyTorch checkpoint, frozen.

    The CTC head sits on top of the projector output, so we need the full
    frame sequence — not the pooled embedding available from ONNX.

    Args:
        model_dir: Directory containing a saved
                   ``Wav2Vec2ForSequenceClassification`` checkpoint.
        device:    Torch device string.

    Returns:
        Tuple ``(wav2vec2_module, projector_module, frame_dim)`` where
        both modules are frozen and set to eval mode.
    """
    import json as _json
    import torch
    from transformers import Wav2Vec2ForSequenceClassification

    adapter_cfg_path = Path(model_dir) / "adapter_config.json"
    is_lora = adapter_cfg_path.exists()

    if is_lora:
        try:
            from peft import PeftModel
        except ImportError:
            logger.error("peft not installed — run: pip install peft")
            sys.exit(1)
        from transformers import AutoConfig
        with open(adapter_cfg_path) as f:
            acfg = _json.load(f)
        base_name = acfg.get("base_model_name_or_path")
        ft_config = AutoConfig.from_pretrained(model_dir)
        base = Wav2Vec2ForSequenceClassification.from_pretrained(
            base_name, config=ft_config
        )
        lora_model = PeftModel.from_pretrained(base, model_dir)
        model = lora_model.merge_and_unload()
        logger.info("LoRA weights merged from %s.", model_dir)
    else:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    wav2vec2 = model.wav2vec2
    projector = model.projector

    # Probe frame_dim with a dummy input
    import torch
    dummy = torch.zeros(1, _WIN_SAMPLES, device=device)
    with torch.no_grad():
        h = wav2vec2(dummy)[0]          # (1, T, D_model)
        proj = projector(h)             # (1, T, D_proj)
    frame_dim = proj.shape[-1]
    logger.info(
        "Encoder loaded: frame_dim=%d, T=%d (for %d-sample window).",
        frame_dim, proj.shape[1], _WIN_SAMPLES,
    )
    return wav2vec2, projector, frame_dim


def _extract_frames(
    wav_paths: List[str],
    wav2vec2,
    projector,
    device: str,
) -> Tuple[List[np.ndarray], List[int]]:
    """Extract per-frame projected features for a list of audio files.

    Args:
        wav_paths: Paths to ``.wav`` files (16 kHz mono).
        wav2vec2:  Frozen Wav2Vec2 encoder module.
        projector: Frozen linear projection module.
        device:    Torch device string.

    Returns:
        ``(frames_list, valid_indices)`` where ``frames_list[i]`` is a
        float32 array of shape ``(T_i, D_proj)`` and ``valid_indices``
        contains the original row indices of successfully processed files.
    """
    import torch

    frames_list: List[np.ndarray] = []
    valid: List[int] = []

    for i, p in enumerate(wav_paths):
        try:
            wav, _ = load_wav(p, target_sr=_SR)
            audio = prepare_window(wav, target_samples=_WIN_SAMPLES, do_normalize=True)
            t = torch.from_numpy(audio).unsqueeze(0).to(device)    # (1, T_raw)
            with torch.no_grad():
                h = wav2vec2(t)[0]          # (1, T, D_model)
                proj = projector(h)         # (1, T, D_proj)
            frames_list.append(proj.squeeze(0).cpu().numpy().astype(np.float32))
            valid.append(i)
        except Exception as exc:
            logger.warning("Skipping %s: %s", p, exc)

    if not frames_list:
        raise RuntimeError("No frames extracted — check audio paths.")
    return frames_list, valid


# ── Numeral sequence parsing ──────────────────────────────────────────────────

def _parse_numerals(numerals_str: str) -> Optional[List[int]]:
    """Parse a space-separated Russian numeral word string into token indices.

    Looks each word up in ``WORD_TO_TOKEN`` (case-insensitive).

    Args:
        numerals_str: E.g. ``"двести восемьдесят пять"`` for 285.

    Returns:
        List of token indices for use with ``CTCLoss``, or ``None`` if any
        word is not in the vocabulary or the string is empty.

    Examples:
        >>> _parse_numerals("двести восемьдесят пять")
        [32, 29, 8]
        >>> _parse_numerals("пятнадцать")
        [18]
        >>> _parse_numerals("unknown")
        None
    """
    try:
        parts = str(numerals_str).strip().lower().split()
        if not parts:
            return None
        tokens = []
        for w in parts:
            tok = WORD_TO_TOKEN.get(w)
            if tok is None:
                logger.warning(
                    "Numeral word '%s' not in vocabulary — row will be skipped. "
                    "Known words: %s",
                    w, sorted(WORD_TO_TOKEN.keys()),
                )
                return None
            tokens.append(tok)
        return tokens
    except (ValueError, TypeError):
        return None


# ── Training ──────────────────────────────────────────────────────────────────

def _cer(pred_tokens: List[int], target_tokens: List[int]) -> float:
    """Simple CER (character = digit token) using edit distance.

    Args:
        pred_tokens:   Greedy-decoded token list.
        target_tokens: Ground-truth token list.

    Returns:
        CER in [0, 1].
    """
    n = len(target_tokens)
    if n == 0:
        return 0.0 if not pred_tokens else 1.0

    # Wagner-Fischer DP
    prev = list(range(n + 1))
    for pt in pred_tokens:
        curr = [prev[0] + 1]
        for j, tt in enumerate(target_tokens):
            curr.append(min(
                curr[-1] + 1,
                prev[j + 1] + 1,
                prev[j] + (0 if pt == tt else 1),
            ))
        prev = curr
    return prev[n] / n


def train(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Parsed CLI arguments.
    """
    import torch
    import torch.nn.functional as F
    from core.hybrid.ctc_digit_decoder import _greedy_ctc_decode

    device = args.device

    # ── Load CSV ───────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path).dropna(subset=["path", "label", "numerals"])

    if args.intents:
        df = df[df["label"].isin(args.intents)]
    if len(df) < 4:
        logger.error(
            "Too few samples after filtering (%d). "
            "Check --intents and CSV labels.", len(df),
        )
        sys.exit(1)

    # Parse Russian numeral word targets → token index sequences
    df["tokens"] = df["numerals"].map(_parse_numerals)
    bad = df["tokens"].isna().sum()
    if bad:
        logger.warning(
            "Skipping %d rows with unparseable numerals "
            "(check vocabulary in core/hybrid/ctc_digit_decoder.py).", bad,
        )
    df = df.dropna(subset=["tokens"]).reset_index(drop=True)

    logger.info(
        "Dataset: %d samples, intents=%s",
        len(df), df["label"].unique().tolist(),
    )

    # ── Load frozen encoder ────────────────────────────────────────────
    logger.info("Loading frozen encoder from %s …", args.model_dir)
    wav2vec2, projector, frame_dim = _load_frozen_encoder(args.model_dir, device)

    # ── Extract frames ─────────────────────────────────────────────────
    logger.info("Extracting per-frame features …")
    all_frames, valid_idx = _extract_frames(
        df["path"].tolist(), wav2vec2, projector, device
    )
    df = df.iloc[valid_idx].reset_index(drop=True)
    all_tokens = [df["tokens"].iloc[i] for i in range(len(df))]

    logger.info("Valid samples: %d / %d", len(all_frames), len(valid_idx))

    # ── Train / val split ──────────────────────────────────────────────
    n = len(all_frames)
    val_size = max(1, int(n * 0.2))
    rng = np.random.default_rng(42)
    val_idx = set(rng.choice(n, size=val_size, replace=False).tolist())

    train_frames = [all_frames[i] for i in range(n) if i not in val_idx]
    train_tokens = [all_tokens[i] for i in range(n) if i not in val_idx]
    val_frames   = [all_frames[i] for i in val_idx]
    val_tokens   = [all_tokens[i] for i in val_idx]

    logger.info("Split: train=%d  val=%d", len(train_frames), len(val_frames))

    # ── Build head and optimiser ───────────────────────────────────────
    head = DigitCTCHead(frame_dim)
    head.to(device)
    head.train()
    optimiser = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    best_val_cer = float("inf")
    best_state: Optional[Dict] = None

    # ── Training loop ──────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        head.train()
        # Shuffle training indices
        perm = rng.permutation(len(train_frames)).tolist()

        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(perm), args.batch_size):
            batch_idx = perm[batch_start: batch_start + args.batch_size]
            batch_frames = [train_frames[i] for i in batch_idx]
            batch_tgts   = [train_tokens[i] for i in batch_idx]

            # ── Pad frame sequences to same T ──
            max_t = max(f.shape[0] for f in batch_frames)
            padded = np.zeros(
                (len(batch_frames), max_t, frame_dim), dtype=np.float32
            )
            input_lengths = []
            for k, f in enumerate(batch_frames):
                t = f.shape[0]
                padded[k, :t] = f
                input_lengths.append(t)

            frames_t = torch.from_numpy(padded).to(device)   # (B, T, D)
            log_probs = head.forward(frames_t)                # (B, T, V)
            # CTCLoss expects (T, B, V)
            log_probs_ctc = log_probs.permute(1, 0, 2)       # (T, B, V)

            # Flatten targets for CTCLoss
            target_flat = torch.tensor(
                [tok for tgt in batch_tgts for tok in tgt],
                dtype=torch.long, device=device,
            )
            target_lengths = torch.tensor(
                [len(tgt) for tgt in batch_tgts],
                dtype=torch.long, device=device,
            )
            input_lengths_t = torch.tensor(
                input_lengths, dtype=torch.long, device=device,
            )

            loss = ctc_loss_fn(
                log_probs_ctc, target_flat,
                input_lengths_t, target_lengths,
            )

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ─────────────────────────────────────────────────
        if epoch % args.val_every == 0 or epoch == args.epochs:
            head.eval()
            cers: List[float] = []
            with torch.no_grad():
                for vf, vt in zip(val_frames, val_tokens):
                    t = torch.from_numpy(vf).unsqueeze(0).to(device)
                    lp = head.forward(t).squeeze(0).cpu().numpy()
                    pred = _greedy_ctc_decode(lp)
                    cers.append(_cer(pred, vt))
            val_cer = float(np.mean(cers))

            marker = ""
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                marker = " ← best"

            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_CER=%.4f%s",
                epoch, args.epochs, avg_loss, val_cer, marker,
            )
        else:
            logger.info("Epoch %3d/%d  loss=%.4f", epoch, args.epochs, avg_loss)

    # ── Restore best and save ──────────────────────────────────────────
    if best_state is not None:
        head.load_state_dict(best_state)
        logger.info("Best val CER: %.4f — restoring best checkpoint.", best_val_cer)

    head.eval()
    decoder = CTCDigitDecoder(frame_dim=frame_dim)
    decoder._head = head
    decoder._is_loaded = True
    decoder.save(args.out)

    print("\n" + "=" * 55)
    print(f"CTC head trained: frame_dim={frame_dim}")
    print(f"Vocab:            {VOCAB_SIZE} tokens (Russian compound numerals)")
    print(f"Best val CER:     {best_val_cer:.4f}")
    print(f"Saved to:         {Path(args.out).resolve()}")
    print("=" * 55)
    print(
        "\nNext step: re-export the ONNX model with projected_frames output:\n"
        "  python scripts/train/main_export_to_onnx.py "
        "--model_dir best_model --output_dir onnx_model/... --quantize"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Train the CTC digit head for slot-fill (Variant B)."
    )
    parser.add_argument(
        "--csv", required=True,
        help=(
            "CSV with columns: path, label, numerals. "
            "'numerals' is a space-separated sequence of Russian numeral words "
            "from the vocabulary in core/hybrid/ctc_digit_decoder.py. "
            "E.g. 'двести восемьдесят пять' for heading 285."
        ),
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Directory of the saved Wav2Vec2ForSequenceClassification checkpoint.",
    )
    parser.add_argument(
        "--intents", nargs="+", default=None,
        help="Slot intent labels to include. Defaults to all in CSV.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/hybrid/ctc_digit_head.pt",
        help="Output path for the trained head checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_every", type=int, default=5,
                        help="Validate every N epochs.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    train(parser.parse_args())
