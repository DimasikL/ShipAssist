"""
eval_lora_model.py — Standalone evaluation of a saved LoRA checkpoint.

Usage
-----
    python scripts/train/eval_lora_model.py \
        --run_dir lora_tune/models/run_2026-04-30_23-34-27 \
        --data_csv dset_meta_only_2026-04-30_15-46-30.csv

The script re-creates the correct 4-class test split from the dataset CSV,
loads the saved model weights, and prints a full classification report.
Results are saved next to the checkpoint as  eval_results.json.
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Silence HuggingFace "uninitialized weights" noise — this fires before
# fine-tuned weights are applied and would be misleading in an eval context.
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message="You should probably TRAIN")

import transformers  # noqa: E402  (after env var set)
transformers.logging.set_verbosity_error()

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test-split definition  (fixed — all 4 classes covered)
# ---------------------------------------------------------------------------
TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    """Minimal dataset for evaluation — no augmentation, no online processing."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: Wav2Vec2FeatureExtractor,
        label2id: dict,
        max_seconds: float = 3.0,
        path_col: str = "audio_path",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.max_samples = int(max_seconds * 16_000)
        self.path_col = path_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        import librosa

        row = self.df.iloc[idx]
        audio_path = row[self.path_col]
        label = self.label2id[row["class"]]

        try:
            waveform, _ = librosa.load(audio_path, sr=16_000, mono=True)
            # Remove leading/trailing silence before truncation so that
            # long recordings with silence at the start are not clipped mid-word.
            waveform, _ = librosa.effects.trim(waveform, top_db=20)
        except Exception as exc:
            logger.warning(f"Could not load {audio_path}: {exc} — using silence.")
            waveform = np.zeros(self.max_samples, dtype=np.float32)

        if len(waveform) > self.max_samples:
            waveform = waveform[: self.max_samples]

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=False,
        )

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs.get("attention_mask", torch.ones(len(waveform))).squeeze(0),
            "label": label,
        }


def _collate(batch: list) -> dict:
    max_len = max(x["input_values"].shape[0] for x in batch)
    input_values = torch.zeros(len(batch), max_len)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    for i, x in enumerate(batch):
        n = x["input_values"].shape[0]
        input_values[i, :n] = x["input_values"]
        attention_mask[i, :n] = x["attention_mask"][:n]

    return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(
    model_path: str,
    base_name: str,
    label2id: dict,
    id2label: dict,
    device: torch.device,
) -> Wav2Vec2ForSequenceClassification:
    """Load merged or unmerged LoRA checkpoint for inference."""
    safetensors_path = os.path.join(model_path, "model.safetensors")
    bin_path         = os.path.join(model_path, "pytorch_model.bin")
    config_path      = os.path.join(model_path, "config.json")
    lora_info_path   = os.path.join(model_path, "lora_info.json")

    # Decide loading strategy from lora_info.json
    is_merged = True  # assume merged / plain HF if no marker
    if os.path.exists(lora_info_path):
        with open(lora_info_path) as f:
            is_merged = bool(json.load(f).get("merged", False))

    # Route 1: merged/plain HF save
    if os.path.exists(config_path) and is_merged:
        try:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(label2id),
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            )
            model.to(device).eval()
            logger.info("Loaded via from_pretrained (merged checkpoint).")
            return model
        except Exception as exc:
            logger.warning(f"from_pretrained(local) failed: {exc} — falling back to state_dict load.")

    # Route 2: unmerged adapter save — base model + state_dict overlay
    logger.info(f"Loading base architecture from: {base_name}")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        base_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    else:
        raise FileNotFoundError(f"No weights found in {model_path}.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.to(device).eval()
    logger.info(
        f"Loaded via state_dict overlay "
        f"({sum(p.numel() for p in model.parameters()):,} params)."
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def run_eval(
    model: Wav2Vec2ForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    id2label: dict,
) -> dict:
    all_preds, all_targets, all_confs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_values  = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["labels"]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(input_values=input_values, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=-1).cpu()
            confs, preds = probs.max(dim=-1)

            all_preds.extend(preds.tolist())
            all_targets.extend(labels.tolist())
            all_confs.extend(confs.tolist())

    label_names = [id2label[str(i)] for i in range(len(id2label))]

    report_str = classification_report(all_targets, all_preds, target_names=label_names)
    logger.info(f"\n{report_str}")

    return {
        "accuracy":       accuracy_score(all_targets, all_preds),
        "macro_f1":       f1_score(all_targets, all_preds, average="macro"),
        "weighted_f1":    f1_score(all_targets, all_preds, average="weighted"),
        "macro_precision": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "macro_recall":   recall_score(all_targets, all_preds, average="macro"),
        "mean_confidence": float(np.mean(all_confs)),
        "n_samples":      len(all_targets),
        "n_classes":      len(set(all_targets)),
        "per_class": {
            label_names[i]: {
                "accuracy": float(np.mean([p == t for p, t in zip(all_preds, all_targets) if t == i])),
                "support": int(sum(1 for t in all_targets if t == i)),
            }
            for i in range(len(label_names))
        },
        "classification_report": report_str,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved LoRA checkpoint.")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the run directory, e.g. lora_tune/models/run_2026-04-30_23-34-27",
    )
    parser.add_argument(
        "--data_csv",
        required=True,
        help="Path to the dataset metadata CSV (audio_path, audio_group, class columns required).",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seconds", type=float, default=3.0)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    run_dir      = Path(args.run_dir)
    best_model   = run_dir / "best_model"
    results_path = run_dir / "eval_results.json"

    if not best_model.exists():
        logger.error(f"best_model not found at {best_model}")
        sys.exit(1)

    # --- Load checkpoint metadata ---
    with open(best_model / "config.json") as f:
        model_cfg = json.load(f)

    id2label = model_cfg.get("id2label", {})
    if not id2label:
        logger.error("id2label not found in config.json.")
        sys.exit(1)

    label2id   = {v: int(k) for k, v in id2label.items()}
    base_name  = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"

    # --- Build test split ---
    df = pd.read_csv(args.data_csv)
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)

    logger.info(f"Test split: {len(test_df)} samples")
    logger.info(f"Class distribution:\n{test_df['class'].value_counts().to_string()}")

    missing_classes = set(label2id.keys()) - set(test_df["class"].unique())
    if missing_classes:
        logger.warning(
            f"Test split is still missing classes: {sorted(missing_classes)}. "
            f"Consider expanding TEST_GROUPS."
        )

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(best_model))
    model = load_model(str(best_model), base_name, label2id, id2label, device)

    # --- DataLoader ---
    dataset = EvalDataset(
        test_df, feature_extractor, label2id,
        max_seconds=args.max_seconds,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Evaluate ---
    results = run_eval(model, loader, device, id2label)

    logger.info("=" * 60)
    logger.info(f"Accuracy:        {results['accuracy']:.4f}")
    logger.info(f"Macro F1:        {results['macro_f1']:.4f}")
    logger.info(f"Weighted F1:     {results['weighted_f1']:.4f}")
    logger.info(f"Mean confidence: {results['mean_confidence']:.4f}")
    logger.info(f"Samples:         {results['n_samples']}  |  Classes: {results['n_classes']}")
    logger.info("=" * 60)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
