"""
scripts/hybrid/diagnose_centroid_confusion.py - Deep-dive into centroid
confusion for a fitted EnsembleOutlierGate.

What this script does
---------------------
Loads a saved outlier gate (.pkl) and the same embeddings used to train it,
then for every sample that is assigned to the WRONG centroid (nearest centroid
≠ true label), prints:
  - the true label
  - which centroid it was pulled towards
  - all per-centroid distances (mahalanobis / cosine / l2) for that sample
  - the source file path

Additionally it prints a confusion matrix (true_label → nearest_centroid) and
a per-class centroid purity table, so you can see at a glance which classes
bleed into which.

Useful for debugging the 'машина' tau=711 anomaly (90% of samples pulled
to wrong centroid), or any other class with high wrong_centroid count.

Usage
-----
    python scripts/hybrid/diagnose_centroid_confusion.py \\
        --gate artifacts/hybrid/outlier_gate.pkl \\
        --csv dset_meta_only_2026-05-21_13-22-25.csv \\
        --pt_model_dir experiments/archive_training/lora_tune/models/run_2026-05-22_09-50-17/best_model \\
        --extra_dirs clf_dset/train_val \\
        --negatives_dirs clf_dset/test \\
        --include_groups "new user 10" "new user 11" "new user 12" "new user 13" "new user 14" "new user 15" \\
        --path_col audio_path \\
        --label_col class \\
        --focus_label "машина" \\
        --show_wrong_only

    # To see ALL samples (not just wrong ones), omit --show_wrong_only.
    # To examine all classes, omit --focus_label.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# ── Label inference (mirrors train_outlier_gate.py) ────────────────────────────
_FOLDER_TO_LABEL: List[Tuple[str, str]] = [
    ("приготовить_машину",  "приготовить машину"),
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
    dir_parts = [p.lower() for p in wav_path.parts[:-1]]
    for part in reversed(dir_parts):
        part_us = part.replace("_", " ")
        for pattern, label in _FOLDER_TO_LABEL:
            if pattern in part or pattern in part_us:
                return label
    return None


# ── Data collection (mirrors train_outlier_gate.py) ───────────────────────────

def _collect_from_csv(
    csv_path: Path,
    path_col: str,
    label_col: str,
    include_groups: Optional[List[str]],
) -> List[Tuple[Path, str]]:
    """Return (wav_path, label) pairs from CSV, skipping aug files."""
    df = pd.read_csv(csv_path)

    if include_groups:
        group_col = next(
            (c for c in df.columns if "group" in c.lower()), None
        )
        if group_col:
            before = len(df)
            df = df[df[group_col].isin(include_groups)]
            logger.info("Group filter on CSV: %d -> %d rows", before, len(df))

    pairs: List[Tuple[Path, str]] = []
    for _, row in df.iterrows():
        p = Path(str(row[path_col]))
        lbl = str(row[label_col])
        name = p.stem.lower()
        if "_aug" in name or name.endswith("aug"):
            continue
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        pairs.append((p, lbl))
    return pairs


def _collect_from_dir(
    directory: Path,
    include_groups: Optional[List[str]],
    negatives_only: bool = False,
    max_duration_s: float = 3.0,
) -> List[Tuple[Path, str]]:
    """Collect WAV files from a directory tree, inferring labels from structure."""
    pairs: List[Tuple[Path, str]] = []
    for wav in sorted(directory.rglob("*.wav")):
        name = wav.stem.lower()
        if "_aug" in name or name.endswith("aug"):
            continue
        # Duration guard
        try:
            import soundfile as sf
            info = sf.info(str(wav))
            if info.duration > max_duration_s:
                continue
        except Exception:
            pass
        # Group filter
        if include_groups:
            parts_str = " ".join(p.lower() for p in wav.parts)
            if not any(g.lower() in parts_str for g in include_groups):
                continue
        # Label from path
        lbl = _label_from_path(wav)
        if lbl is None:
            continue
        if negatives_only and lbl != "другие слова":
            continue
        pairs.append((wav, lbl))
    return pairs


def _load_embedding(
    wav_path: Path,
    model,
    tokenizer,
    device: str,
) -> Optional[np.ndarray]:
    """Load WAV and extract a 256-D embedding using the PyTorch model."""
    import torch

    if not wav_path.exists():
        # Try forward-slash -> backslash fix
        fixed = Path(str(wav_path).replace("/", "\\"))
        if not fixed.exists():
            return None
        wav_path = fixed

    try:
        audio = load_wav(wav_path, target_sr=_SR)
        window = prepare_window(audio, win_samples=_WIN_SAMPLES)
        inputs = tokenizer(
            window, sampling_rate=_SR, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # Use projector output (256-D) if available, else last hidden state mean
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            # Wav2Vec2 classification head: projector maps to 256-D
            hidden = out.hidden_states[-1]           # (1, T, 1024)
            emb = hidden.mean(dim=1).squeeze(0)      # (1024,)
        else:
            emb = out.logits.squeeze(0)
        return emb.cpu().float().numpy()
    except Exception as exc:
        logger.debug("Embedding failed for %s: %s", wav_path, exc)
        return None


def _load_pytorch_model(pt_model_dir: Path):
    """Load LoRA-merged Wav2Vec2 model. Returns (model, tokenizer, device).

    Mirrors the loading logic in train_outlier_gate.py: reads num_labels and
    id2label from the checkpoint config so the classifier head size matches.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    logger.info("Loading PyTorch model from %s", pt_model_dir)
    base_id = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    tokenizer = AutoFeatureExtractor.from_pretrained(base_id)
    # Load checkpoint config so num_labels matches (e.g. 4, not default 2)
    ft_config = AutoConfig.from_pretrained(str(pt_model_dir))
    base_model = Wav2Vec2ForSequenceClassification.from_pretrained(base_id, config=ft_config)
    model = PeftModel.from_pretrained(base_model, str(pt_model_dir))
    model = model.merge_and_unload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info("Model loaded on %s (num_labels=%d)", device, ft_config.num_labels)
    return model, tokenizer, device


# ── Distance helpers ───────────────────────────────────────────────────────────

def _all_distances(
    gate: EnsembleOutlierGate,
    embedding: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Return raw distances from embedding to ALL class centroids.

    Returns:
        {
          'mahalanobis': {'другие слова': 3.1, 'машина': 0.2, ...},
          'cosine':      {'другие слова': 0.05, ...},
          'l2':          {'другие слова': 0.08, ...},
        }
    """
    assert gate._gate_mahal is not None
    assert gate._gate_cos is not None
    assert gate._gate_l2 is not None

    result: Dict[str, Dict[str, float]] = {
        "mahalanobis": {},
        "cosine": {},
        "l2": {},
    }

    emb = embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    normed = emb / max(norm, 1e-12)

    labels = gate._gate_mahal._labels
    assert labels is not None

    # Mahalanobis to ALL centroids
    mahal_dists = gate._gate_mahal._mahalanobis_to_all(normed)
    for i, lbl in enumerate(labels):
        result["mahalanobis"][lbl] = float(mahal_dists[i])

    # Cosine to ALL centroids
    centroids_cos = gate._gate_cos._centroids
    assert centroids_cos is not None
    sims = centroids_cos @ normed
    for i, lbl in enumerate(labels):
        result["cosine"][lbl] = float(1.0 - sims[i])

    # L2 to ALL centroids
    centroids_l2 = gate._gate_l2._centroids
    assert centroids_l2 is not None
    diffs = centroids_l2 - normed[None, :]
    l2_dists = np.linalg.norm(diffs, axis=1)
    for i, lbl in enumerate(labels):
        result["l2"][lbl] = float(l2_dists[i])

    return result


def _nearest_per_metric(
    gate: EnsembleOutlierGate,
    embedding: np.ndarray,
) -> Dict[str, str]:
    """Return which label is nearest for each sub-gate metric."""
    dists = _all_distances(gate, embedding)
    return {
        metric: min(per_class, key=per_class.get)  # type: ignore[arg-type]
        for metric, per_class in dists.items()
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose centroid confusion in a fitted EnsembleOutlierGate."
    )
    parser.add_argument("--gate", required=True, help="Path to outlier_gate.pkl")
    parser.add_argument("--csv", default=None, help="Labelled dataset CSV")
    parser.add_argument("--pt_model_dir", default=None,
                        help="LoRA model dir for embedding extraction. "
                             "Not required when --load_embeddings is used.")
    parser.add_argument("--extra_dirs", nargs="*", default=[], help="Extra WAV dirs")
    parser.add_argument("--negatives_dirs", nargs="*", default=[], help="Negatives-only WAV dirs")
    parser.add_argument("--include_groups", nargs="*", default=None, help="Group filter (same as train script)")
    parser.add_argument("--path_col", default="audio_path", help="CSV column for file paths")
    parser.add_argument("--label_col", default="class", help="CSV column for labels")
    parser.add_argument("--focus_label", default=None,
                        help="Only show confusion for this true label (e.g. 'машина')")
    parser.add_argument("--show_wrong_only", action="store_true",
                        help="Print per-sample detail only for wrong-centroid samples")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (for fast smoke-tests)")
    parser.add_argument("--load_embeddings", type=str, default=None,
                        metavar="PATH",
                        help="Load embeddings+labels+paths from a .npz cache produced by "
                             "train_outlier_gate.py --save_embeddings. Skips model loading "
                             "and audio processing entirely (much faster).")
    args = parser.parse_args()

    if not args.load_embeddings and not args.pt_model_dir:
        parser.error("--pt_model_dir is required when --load_embeddings is not provided.")

    # ── Load gate ─────────────────────────────────────────────────────────
    gate_path = Path(args.gate)
    if not gate_path.is_absolute():
        gate_path = _PROJECT_ROOT / gate_path
    logger.info("Loading gate from %s", gate_path)
    gate = EnsembleOutlierGate.load(gate_path)
    assert isinstance(gate, EnsembleOutlierGate), "Only EnsembleOutlierGate supported"
    known_labels: List[str] = gate._gate_mahal._labels  # type: ignore[index]
    logger.info("Gate labels: %s", known_labels)

    # ── Collect samples ───────────────────────────────────────────────────
    pairs: List[Tuple[Path, str]] = []
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = _PROJECT_ROOT / csv_path
        pairs.extend(_collect_from_csv(csv_path, args.path_col, args.label_col, args.include_groups))

    for d in args.extra_dirs:
        p = Path(d) if Path(d).is_absolute() else _PROJECT_ROOT / d
        pairs.extend(_collect_from_dir(p, args.include_groups, negatives_only=False))

    for d in args.negatives_dirs:
        p = Path(d) if Path(d).is_absolute() else _PROJECT_ROOT / d
        pairs.extend(_collect_from_dir(p, args.include_groups, negatives_only=True))

    if args.focus_label:
        pairs = [(p, l) for p, l in pairs if l == args.focus_label]
        logger.info("Focus label '%s': %d samples", args.focus_label, len(pairs))

    if args.max_samples and len(pairs) > args.max_samples:
        pairs = pairs[: args.max_samples]
        logger.info("Capped to %d samples", args.max_samples)

    logger.info("Total samples to process: %d", len(pairs))

    # ── Load embeddings: from cache (.npz) or extract fresh ──────────────
    if args.load_embeddings:
        cache_path = Path(args.load_embeddings)
        if not cache_path.is_absolute():
            cache_path = _PROJECT_ROOT / cache_path
        logger.info("Loading embeddings from cache: %s", cache_path)
        data = np.load(str(cache_path), allow_pickle=True)
        all_embs: np.ndarray = data["embeddings"]         # (N, D)
        all_labels_cache: np.ndarray = data["labels"]     # (N,) str
        all_paths_cache: np.ndarray = data["paths"]       # (N,) str

        # Apply focus_label filter on cached data
        embeddings_list: List[np.ndarray] = []
        valid_pairs: List[Tuple[Path, str]] = []
        for emb, lbl, pth in zip(all_embs, all_labels_cache, all_paths_cache):
            lbl_str = str(lbl)
            if args.focus_label and lbl_str != args.focus_label:
                continue
            embeddings_list.append(emb.astype(np.float32))
            valid_pairs.append((Path(str(pth)), lbl_str))

        if args.max_samples and len(embeddings_list) > args.max_samples:
            embeddings_list = embeddings_list[: args.max_samples]
            valid_pairs = valid_pairs[: args.max_samples]

        embeddings_arr = np.stack(embeddings_list, axis=0)
        logger.info(
            "Loaded %d embeddings from cache (focus=%s)",
            len(embeddings_list), args.focus_label or "all",
        )
    else:
        # Extract fresh embeddings from audio files
        model, tokenizer, device = _load_pytorch_model(Path(args.pt_model_dir))

        raw_embeddings: List[np.ndarray] = []
        valid_pairs = []
        for i, (wav_path, lbl) in enumerate(pairs):
            emb = _load_embedding(wav_path, model, tokenizer, device)
            if emb is None:
                logger.warning("Skipping %s (embedding failed)", wav_path)
                continue
            raw_embeddings.append(emb)
            valid_pairs.append((wav_path, lbl))
            if (i + 1) % 100 == 0:
                logger.info("  %d/%d processed", i + 1, len(pairs))

        if args.max_samples and len(raw_embeddings) > args.max_samples:
            raw_embeddings = raw_embeddings[: args.max_samples]
            valid_pairs = valid_pairs[: args.max_samples]

        embeddings_arr = np.stack(raw_embeddings, axis=0)
        logger.info("Valid embeddings: %d / %d", len(raw_embeddings), len(pairs))

    embeddings = embeddings_arr  # alias used below

    # ── Analyse confusion ─────────────────────────────────────────────────
    # confusion[true_label][nearest_label] = count
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # For per-sample breakdown: store rows
    wrong_rows: List[dict] = []

    for (wav_path, true_label), emb in zip(valid_pairs, embeddings):
        ensemble_score, nearest_global = gate.score(emb)
        nearest_per_m = _nearest_per_metric(gate, emb)
        all_dists = _all_distances(gate, emb)

        confusion[true_label][nearest_global] += 1

        is_wrong = nearest_global != true_label
        if is_wrong or not args.show_wrong_only:
            wrong_rows.append({
                "path": wav_path,
                "true_label": true_label,
                "nearest_global": nearest_global,
                "ensemble_score": ensemble_score,
                "nearest_mahal": nearest_per_m["mahalanobis"],
                "nearest_cos": nearest_per_m["cosine"],
                "nearest_l2": nearest_per_m["l2"],
                "dists_mahal": all_dists["mahalanobis"],
                "dists_cos": all_dists["cosine"],
                "dists_l2": all_dists["l2"],
                "is_wrong": is_wrong,
            })

    # ── Print confusion matrix ─────────────────────────────────────────────
    all_true = sorted(confusion.keys())
    print("\n" + "=" * 80)
    print("CENTROID CONFUSION MATRIX   (rows=true label, cols=nearest centroid)")
    print("=" * 80)
    col_labels = sorted(set(lbl for row in confusion.values() for lbl in row))
    header = f"{'TRUE LABEL':35s}" + "".join(f"{c:25s}" for c in col_labels)
    print(header)
    print("-" * len(header))
    for true_lbl in all_true:
        row_total = sum(confusion[true_lbl].values())
        row = f"{true_lbl:35s}"
        for col_lbl in col_labels:
            cnt = confusion[true_lbl].get(col_lbl, 0)
            pct = cnt / row_total * 100 if row_total else 0.0
            row += f"{cnt:5d} ({pct:5.1f}%)          "
        print(row)
    print("=" * 80)

    # ── Per-class purity summary ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-CLASS PURITY  (% samples correctly assigned to own centroid)")
    print("=" * 80)
    for true_lbl in all_true:
        total = sum(confusion[true_lbl].values())
        correct = confusion[true_lbl].get(true_lbl, 0)
        wrong_cnt = total - correct
        purity = correct / total * 100 if total else 0.0
        tau = gate._per_class_thresholds.get(true_lbl, gate._global_threshold)
        print(
            f"  {true_lbl:35s}  n={total:4d}  "
            f"correct={correct:4d} ({purity:5.1f}%)  "
            f"wrong={wrong_cnt:4d}  tau={tau:.4f}"
        )
    print("=" * 80)

    # ── Per-sample breakdown for wrong-centroid samples ───────────────────
    wrong_only = [r for r in wrong_rows if r["is_wrong"]]
    focus_str = f" (focus: '{args.focus_label}')" if args.focus_label else ""
    print(f"\n{'=' * 80}")
    print(f"WRONG-CENTROID SAMPLES{focus_str}  [{len(wrong_only)} total]")
    print("=" * 80)

    # Group by true_label for readability
    by_true: Dict[str, List[dict]] = defaultdict(list)
    for r in wrong_only:
        by_true[r["true_label"]].append(r)

    for true_lbl in sorted(by_true.keys()):
        rows = by_true[true_lbl]
        print(f"\n── True: '{true_lbl}'  ({len(rows)} wrong samples) ──")

        # Show which centroid attracted them and distribution of pull
        pull_counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            pull_counts[r["nearest_global"]] += 1
        for pulled_to, cnt in sorted(pull_counts.items(), key=lambda x: -x[1]):
            print(f"   → pulled to '{pulled_to}': {cnt} samples")

        # Per-sample detail: distances to ALL centroids
        print(f"\n   {'FILE':60s} {'→ NEAREST':25s} {'SCORE':8s}  MAHAL[all]  COS[all]  L2[all]")
        print("   " + "-" * 140)
        for r in sorted(rows, key=lambda x: abs(x["ensemble_score"]), reverse=True):
            # Format distances to all centroids compactly
            mahal_str = "  ".join(
                f"{lbl[:6]}:{v:.2f}"
                for lbl, v in sorted(r["dists_mahal"].items())
            )
            cos_str = "  ".join(
                f"{lbl[:6]}:{v:.4f}"
                for lbl, v in sorted(r["dists_cos"].items())
            )
            l2_str = "  ".join(
                f"{lbl[:6]}:{v:.4f}"
                for lbl, v in sorted(r["dists_l2"].items())
            )
            path_str = str(r["path"])[-58:]
            nearest = r["nearest_global"]
            score = r["ensemble_score"]
            print(f"   {path_str:60s} → {nearest:23s}  {score:8.3f}")
            # Show per-metric nearest
            print(
                f"   {'':60s}   mahal_near={r['nearest_mahal']:20s} "
                f"cos_near={r['nearest_cos']:20s} "
                f"l2_near={r['nearest_l2']:20s}"
            )
            print(f"   {'':60s}   MAHAL: {mahal_str}")
            print(f"   {'':60s}   COS:   {cos_str}")
            print(f"   {'':60s}   L2:    {l2_str}")
            print()

    # ── Quick actionable diagnosis ─────────────────────────────────────────
    print("=" * 80)
    print("DIAGNOSIS NOTES")
    print("=" * 80)
    for true_lbl in all_true:
        total = sum(confusion[true_lbl].values())
        correct = confusion[true_lbl].get(true_lbl, 0)
        purity = correct / total * 100 if total else 0.0
        if purity < 70.0:
            main_pull = max(
                ((k, v) for k, v in confusion[true_lbl].items() if k != true_lbl),
                key=lambda x: x[1],
                default=(None, 0),
            )
            print(
                f"  ⚠  '{true_lbl}' purity={purity:.1f}%  "
                f"→ mostly pulled to '{main_pull[0]}' ({main_pull[1]} samples)"
            )
            print(
                f"     Possible causes:\n"
                f"     1. Label leakage: '{true_lbl}' audio appears under a '{main_pull[0]}' folder\n"
                f"     2. Acoustic overlap: the two commands sound similar in embedding space\n"
                f"     3. Class imbalance: '{main_pull[0]}' dominates centroid space (n>>)\n"
                f"     4. Wrong label in CSV for these samples\n"
                f"     5. The embedding model conflates these two classes\n"
            )
    print("=" * 80)
    print("\nDone.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
