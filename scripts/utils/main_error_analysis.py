"""
error_analysis.py — Per-sample error analysis for a saved LoRA checkpoint.

Shows exactly which files were misclassified, with predicted label,
true label, confidence, and audio path.

Usage:
    python scripts/utils/error_analysis.py \
        --run_dir lora_tune/models/run_2026-05-08_16-28-04 \
        --data_csv dset_meta_only_2026-05-09_10-27-42.csv
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message="You should probably TRAIN")

import transformers
transformers.logging.set_verbosity_error()

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]


# ── Dataset ──────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, df, feature_extractor, label2id, max_seconds=3.0):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.max_samples = int(max_seconds * 16_000)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import librosa
        row = self.df.iloc[idx]
        try:
            waveform, _ = librosa.load(row["audio_path"], sr=16_000, mono=True)
            waveform, _ = librosa.effects.trim(waveform, top_db=20)
        except Exception as exc:
            logger.warning(f"Cannot load {row['audio_path']}: {exc}")
            waveform = np.zeros(self.max_samples, dtype=np.float32)

        if len(waveform) > self.max_samples:
            waveform = waveform[: self.max_samples]

        inputs = self.feature_extractor(
            waveform, sampling_rate=16_000, return_tensors="pt", padding=False
        )
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs.get(
                "attention_mask", torch.ones(len(waveform))
            ).squeeze(0),
            "label": self.label2id[row["class"]],
            "audio_path": row["audio_path"],
            "audio_group": row.get("audio_group", ""),
        }


def _collate(batch):
    max_len = max(x["input_values"].shape[0] for x in batch)
    input_values = torch.zeros(len(batch), max_len)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    paths = [x["audio_path"] for x in batch]
    groups = [x["audio_group"] for x in batch]

    for i, x in enumerate(batch):
        n = x["input_values"].shape[0]
        input_values[i, :n] = x["input_values"]
        attention_mask[i, :n] = x["attention_mask"][:n]

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "paths": paths,
        "groups": groups,
    }


# ── Model loader ─────────────────────────────────────────────────────────────

def load_model(model_path, label2id, id2label, device):
    model_path = str(Path(model_path).resolve())
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        model.to(device).eval()
        return model
    except Exception as exc:
        logger.warning(f"from_pretrained failed: {exc} — falling back to state_dict.")

    from transformers import Wav2Vec2Config
    config = Wav2Vec2Config.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification(config)

    sf = Path(model_path) / "model.safetensors"
    bn = Path(model_path) / "pytorch_model.bin"
    if sf.exists():
        from safetensors.torch import load_file
        sd = load_file(str(sf))
    elif bn.exists():
        sd = torch.load(str(bn), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No weights in {model_path}")

    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seconds", type=float, default=3.0)
    parser.add_argument("--group_filter", type=str, default=None,
                        help="Показать ошибки только для конкретной группы")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    best_model = run_dir / "best_model"

    with open(best_model / "config.json") as f:
        cfg = json.load(f)
    id2label = cfg["id2label"]
    label2id = {v: int(k) for k, v in id2label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(best_model))
    model = load_model(str(best_model), label2id, id2label, device)

    df = pd.read_csv(args.data_csv)
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)
    logger.info(f"Test samples: {len(test_df)}")

    dataset = EvalDataset(test_df, feature_extractor, label2id, args.max_seconds)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=_collate, num_workers=0)

    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            iv = batch["input_values"].to(device)
            am = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(input_values=iv, attention_mask=am)
            probs = torch.softmax(out.logits, dim=-1).cpu()
            confs, preds = probs.max(dim=-1)

            for i in range(len(batch["labels"])):
                true_id = batch["labels"][i].item()
                pred_id = preds[i].item()
                records.append({
                    "audio_path": batch["paths"][i],
                    "audio_group": batch["groups"][i],
                    "true_label": id2label[str(true_id)],
                    "pred_label": id2label[str(pred_id)],
                    "confidence": round(confs[i].item(), 4),
                    "correct": true_id == pred_id,
                    "all_probs": {
                        id2label[str(j)]: round(probs[i][j].item(), 4)
                        for j in range(len(id2label))
                    },
                })

    results_df = pd.DataFrame(records)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    total = len(results_df)
    correct = results_df["correct"].sum()
    logger.info(f"\n{'='*60}")
    logger.info(f"Overall: {correct}/{total} correct ({100*correct/total:.1f}%)")

    # Per-group accuracy
    logger.info("\nPer-group accuracy:")
    for grp, grp_df in results_df.groupby("audio_group"):
        acc = grp_df["correct"].mean()
        logger.info(f"  {grp:30s}: {grp_df['correct'].sum():3d}/{len(grp_df):3d} ({100*acc:.1f}%)")

    # ── Error details ─────────────────────────────────────────────────────────
    errors = results_df[~results_df["correct"]]
    if args.group_filter:
        errors = errors[errors["audio_group"] == args.group_filter]

    logger.info(f"\n{'='*60}")
    logger.info(f"ERRORS ({len(errors)} total):")
    logger.info(f"{'='*60}")

    for _, row in errors.iterrows():
        fname = Path(row["audio_path"]).name
        logger.info(
            f"  [{row['audio_group']}] {fname}\n"
            f"    TRUE: {row['true_label']:25s}  PRED: {row['pred_label']:25s}  conf={row['confidence']:.4f}\n"
            f"    probs: {row['all_probs']}"
        )

    # ── Confusion summary ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Confusion pairs (true → pred):")
    confusion = errors.groupby(["true_label", "pred_label"]).size().reset_index(name="count")
    confusion = confusion.sort_values("count", ascending=False)
    for _, row in confusion.iterrows():
        logger.info(f"  {row['true_label']:25s} → {row['pred_label']:25s}: {row['count']} errors")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = run_dir / "error_analysis.csv"
    errors[["audio_group", "true_label", "pred_label", "confidence", "audio_path"]].to_csv(
        out_path, index=False, encoding="utf-8"
    )
    logger.info(f"\nErrors saved → {out_path}")

    full_out = run_dir / "predictions_full.csv"
    results_df[["audio_group", "true_label", "pred_label", "confidence", "correct", "audio_path"]].to_csv(
        full_out, index=False, encoding="utf-8"
    )
    logger.info(f"Full predictions → {full_out}")


if __name__ == "__main__":
    main()
