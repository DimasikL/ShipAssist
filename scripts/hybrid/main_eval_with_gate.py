"""
scripts/hybrid/eval_with_gate.py — Evaluate HybridAudioEngine (ONNX logits + OOD gate)
on the standard test split.

Metrics reported
----------------
- Per-class: TP, FP (другие слова → target), FN (target → rejected/другие слова), Acc
- Overall accuracy (excluding gate-rejected samples counted as "другие слова")
- Gate stats: how many samples rejected per class, false rejection rate on targets

Usage
-----
    python scripts/hybrid/eval_with_gate.py ^
        --csv dset_meta_only_2026-04-30_15-46-30.csv ^
        --hybrid_cfg configs/hybrid/model.yaml ^
        --path_col audio_path ^
        --label_col class

    # Test without gate (ablation):
    python scripts/hybrid/eval_with_gate.py ^
        --csv dset_meta_only_2026-04-30_15-46-30.csv ^
        --hybrid_cfg configs/hybrid/model.yaml ^
        --no_gate

TEST_GROUPS (must stay in sync with eval_onnx_model.py)
--------------------------------------------------------
    train_user_2, drug slova2, train_user_2_new, drug slova2-new, train user 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window
from core.hybrid.config import HybridConfig
from core.hybrid.factory import create_hybrid_engine
from core.logger import get_logger

logger = get_logger(__name__)

TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]

_SR = 16_000
_WIN_SAMPLES = 48_000  # 3s — must match run_2026-04-30


def _load_audio(path: str) -> np.ndarray:
    """Load and prepare a 3-second window."""
    wav, _ = load_wav(path, target_sr=_SR)
    return prepare_window(wav, target_samples=_WIN_SAMPLES, do_normalize=True)


def main(args: argparse.Namespace) -> None:
    """Run evaluation with gate."""
    # ── Load dataset ──────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path).dropna(subset=[args.path_col, args.label_col])
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)
    logger.info("Test split: %d samples across %d groups", len(test_df), test_df["audio_group"].nunique())

    if len(test_df) == 0:
        logger.error("No test samples found. Check --path_col / --label_col and TEST_GROUPS.")
        sys.exit(1)

    # ── Load engine ───────────────────────────────────────────────────────────
    cfg = HybridConfig.from_yaml(args.hybrid_cfg, args.thresholds_yaml)

    if args.no_gate:
        # Patch config to disable gate
        cfg.outlier_gate.enabled = False
        logger.info("Gate DISABLED (ablation mode)")
    else:
        logger.info("Gate ENABLED — loading from %s", cfg.paths.outlier_gate)

    engine = create_hybrid_engine(cfg)

    # ── Run inference ─────────────────────────────────────────────────────────
    labels = sorted(test_df[args.label_col].unique())
    REJECTED = "__rejected__"

    # per_class[true_label][pred_label] = count
    confusion: Dict[str, Dict[str, int]] = {
        lbl: defaultdict(int) for lbl in labels
    }
    gate_rejected: Dict[str, int] = defaultdict(int)

    n_total = len(test_df)
    _debug_shown = 0  # print first 5 raw results for sanity check
    for i, row in test_df.iterrows():
        true_label = row[args.label_col]
        audio_path = row[args.path_col]

        try:
            audio = _load_audio(audio_path)
            result = engine.predict(audio)
        except Exception as exc:
            logger.warning("Error on %s: %s", audio_path, exc)
            confusion[true_label][REJECTED] += 1
            continue

        if _debug_shown < 5:
            logger.info(
                "DEBUG sample %d | true=%-25s | label=%-25s | rejected=%s | score=%.3f | path=...%s",
                _debug_shown, true_label, result.get("label"), result.get("outlier_rejected"),
                result.get("outlier_score", -1), audio_path[-50:],
            )
            _debug_shown += 1

        if result["outlier_rejected"]:
            gate_rejected[true_label] += 1
            pred = "другие слова"
        else:
            pred = result["label"] if result["label"] else "другие слова"

        confusion[true_label][pred] += 1

        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            logger.info("  %d / %d done", i + 1, n_total)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    gate_mode = "DISABLED" if args.no_gate else "ENABLED"
    print(f"  Eval with HybridAudioEngine  [gate={gate_mode}]")
    print("=" * 70)

    # Per-class accuracy
    header = f"  {'Class':<30} {'Correct':>7} {'Total':>7} {'Acc':>7} {'Rejected':>9}"
    print(header)
    print("  " + "-" * 66)

    total_correct = 0
    total_samples = 0

    for lbl in labels:
        row_counts = confusion[lbl]
        n = sum(row_counts.values())
        correct = row_counts.get(lbl, 0)
        rejected = gate_rejected[lbl]
        acc = correct / n if n > 0 else 0.0
        total_correct += correct
        total_samples += n
        print(f"  {lbl:<30} {correct:>7} {n:>7} {acc:>7.1%} {rejected:>9}")

    print("  " + "-" * 66)
    overall = total_correct / total_samples if total_samples > 0 else 0.0
    total_rejected = sum(gate_rejected.values())
    print(f"  {'OVERALL':<30} {total_correct:>7} {total_samples:>7} {overall:>7.1%} {total_rejected:>9}")

    # Confusion matrix
    print("\n  Confusion matrix (rows=true, cols=pred)")
    col_labels = labels + ([REJECTED] if any(
        confusion[l].get(REJECTED, 0) for l in labels
    ) else [])
    col_w = 16
    _hdr_label = "True \\ Pred"
    header_row = "  " + f"{_hdr_label:<30}" + "".join(f"{c[:col_w]:>{col_w}}" for c in col_labels)
    print(header_row)
    for true_lbl in labels:
        row_str = f"  {true_lbl:<30}"
        for pred_lbl in col_labels:
            cnt = confusion[true_lbl].get(pred_lbl, 0)
            cell = f"[{cnt}]" if pred_lbl == true_lbl else str(cnt)
            row_str += f"{cell:>{col_w}}"
        print(row_str)

    # Gate summary
    print("\n  Gate rejection summary (per true class):")
    for lbl in labels:
        n = sum(confusion[lbl].values())
        rej = gate_rejected[lbl]
        rate = rej / n if n > 0 else 0.0
        print(f"    {lbl:<30}  rejected={rej:>4}  rate={rate:.1%}")

    # False positives: другие слова → any target class
    if "другие слова" in confusion:
        fp_row = confusion["другие слова"]
        target_classes = [l for l in labels if l != "другие слова"]
        fp_total = sum(fp_row.get(t, 0) for t in target_classes)
        fp_details = {t: fp_row.get(t, 0) for t in target_classes if fp_row.get(t, 0) > 0}
        n_other = sum(fp_row.values())
        print(f"\n  FALSE POSITIVES (другие слова → target): {fp_total} / {n_other}  ({fp_total/n_other:.1%})")
        for t, cnt in sorted(fp_details.items(), key=lambda x: -x[1]):
            print(f"    → {t:<30} {cnt}")

    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate HybridAudioEngine with OOD gate on test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default="dset_meta_only_2026-04-30_15-46-30.csv",
                        help="Dataset CSV.")
    parser.add_argument("--hybrid_cfg", default="configs/hybrid/model.yaml",
                        help="Path to configs/hybrid/model.yaml.")
    parser.add_argument("--thresholds_yaml", default="configs/hybrid/thresholds.yaml",
                        help="Path to configs/hybrid/thresholds.yaml.")
    parser.add_argument("--path_col", default="audio_path",
                        help="CSV column for audio paths.")
    parser.add_argument("--label_col", default="class",
                        help="CSV column for class labels.")
    parser.add_argument("--no_gate", action="store_true",
                        help="Disable OOD gate (ablation).")

    main(parser.parse_args())
