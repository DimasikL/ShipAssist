import os
import pandas as pd
import torch
import torchaudio
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, WeightedRandomSampler
from core.config import settings
from core.logger import get_logger

logger = get_logger("DataUtils")

class CommandDataset(Dataset):
    """CHANGED: Универсальный датасет с поддержкой обрезки по времени."""
    def __init__(self, df: pd.DataFrame, feature_extractor, label2id: Dict[str, int]):
        self.df = df
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.max_samples = int(settings.audio.max_duration * settings.audio.sample_rate)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["audio_path"]
        label = self.label2id[row["class"]]

        waveform, sr = torchaudio.load(wav_path)
        if sr != settings.audio.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, settings.audio.sample_rate)
            waveform = resampler(waveform)

        # CHANGED: Обрезка/Паддинг до фиксированной длины (из v5)
        waveform = waveform.squeeze()
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=settings.audio.sample_rate,
            return_tensors="pt"
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "attention_mask": inputs.attention_mask.squeeze(0)
        }

def make_balanced_sampler(df: pd.DataFrame, label2id: Dict[str, int]) -> WeightedRandomSampler:
    """CHANGED: Создание сэмплера для борьбы с дисбалансом классов (из v3)."""
    class_counts = df["class"].value_counts().to_dict()
    weights = [1.0 / class_counts[c] for c in df["class"]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def parse_metadata(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """Парсинг CSV и создание словарей меток."""
    df = pd.read_csv(csv_path)
    labels = sorted(df["class"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return df, label2id, id2label