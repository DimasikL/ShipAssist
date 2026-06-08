"""Find the one remaining misclassified sample (машина, recall=0.96)."""
import json
import warnings
import os
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Config

RUN_DIR   = Path("lora_tune/models/run_2026-05-08_16-28-04")
BEST_MODEL = RUN_DIR / "best_model"
DATA_CSV  = Path("dset_meta_only_2026-05-09_10-27-42.csv")
MAX_SEC   = 3.0
SR        = 16_000

TEST_GROUPS = ["train_user_2", "drug slova2", "train_user_2_new", "drug slova2-new", "train user 4"]

# Load model
with open(BEST_MODEL / "config.json") as f:
    cfg = json.load(f)
id2label = cfg["id2label"]
label2id = {v: int(k) for k, v in id2label.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fe = Wav2Vec2FeatureExtractor.from_pretrained(str(BEST_MODEL))

try:
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        str(BEST_MODEL), num_labels=len(label2id), ignore_mismatched_sizes=True)
except Exception:
    config = Wav2Vec2Config.from_pretrained(str(BEST_MODEL))
    model = Wav2Vec2ForSequenceClassification(config)
    sf = BEST_MODEL / "model.safetensors"
    bn = BEST_MODEL / "pytorch_model.bin"
    if sf.exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(sf)), strict=False)
    else:
        model.load_state_dict(torch.load(str(bn), map_location="cpu", weights_only=True), strict=False)

model.to(device).eval()

# Run inference only on "машина" class
df = pd.read_csv(DATA_CSV)
test_df = df[df["audio_group"].isin(TEST_GROUPS) & (df["class"] == "машина")].reset_index(drop=True)
print(f"Checking {len(test_df)} 'машина' samples...\n")

max_samples = int(MAX_SEC * SR)
errors = []

with torch.no_grad():
    for _, row in test_df.iterrows():
        try:
            y, _ = librosa.load(row["audio_path"], sr=SR, mono=True)
            y, _ = librosa.effects.trim(y, top_db=20)
        except Exception as e:
            print(f"LOAD ERROR: {row['audio_path']} — {e}")
            continue

        if len(y) > max_samples:
            y = y[:max_samples]

        inputs = fe(y, sampling_rate=SR, return_tensors="pt", padding=False)
        iv = inputs["input_values"].to(device)
        am = torch.ones_like(iv, dtype=torch.long)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(input_values=iv, attention_mask=am)

        probs = torch.softmax(out.logits, dim=-1).squeeze().cpu().numpy()
        pred_id = int(np.argmax(probs))
        pred_label = id2label[str(pred_id)]
        conf = float(probs[pred_id])

        duration_raw = librosa.get_duration(path=row["audio_path"])
        duration_trimmed = len(y) / SR

        status = "✓" if pred_label == "машина" else "✗ ERROR"
        print(f"{status}  {Path(row['audio_path']).name:30s}  "
              f"pred={pred_label:25s}  conf={conf:.4f}  "
              f"dur_raw={duration_raw:.2f}s  dur_trim={duration_trimmed:.2f}s  "
              f"group={row['audio_group']}")

        if pred_label != "машина":
            errors.append({
                "file": row["audio_path"],
                "group": row["audio_group"],
                "pred": pred_label,
                "conf": conf,
                "probs": {id2label[str(i)]: round(float(probs[i]), 4) for i in range(len(id2label))},
                "dur_raw": round(duration_raw, 2),
                "dur_trim": round(duration_trimmed, 2),
            })

print(f"\n{'='*60}")
if errors:
    print(f"Found {len(errors)} error(s):")
    for e in errors:
        print(f"\n  File:   {e['file']}")
        print(f"  Group:  {e['group']}")
        print(f"  Pred:   {e['pred']} (conf={e['conf']:.4f})")
        print(f"  Probs:  {e['probs']}")
        print(f"  Raw duration:     {e['dur_raw']}s")
        print(f"  Trimmed duration: {e['dur_trim']}s")
else:
    print("No errors — all 'машина' samples classified correctly!")
