import os
import torch
import torch.nn as nn
from typing import Dict, Any, List
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    get_scheduler
)
from torch.optim import AdamW

# CHANGED: Импорты из нашей новой структуры
from src.data_utils import CommandDataset, make_balanced_sampler, parse_metadata
from core.config import settings
from core.logger import get_logger
from src.trainer_utils import EMA, FocalLoss, mixup_data, mixup_criterion

logger = get_logger("Trainer")

class Trainer:
    """
    CHANGED: Универсальный класс Trainer.
    Реализует SOTA методы обучения (v1-v5) в едином интерфейсе.
    """
    def __init__(self, label2id: Dict[str, int], id2label: Dict[int, str]):
        self.cfg = settings.training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label2id = label2id
        self.id2label = id2label

        # 1. Модель и экстрактор
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.model_name)
        self.model = self._init_model()

        # 2. Регуляризация (EMA из v2)
        self.ema = EMA(self.model, decay=self.cfg.ema_decay) if self.cfg.use_ema else None

        # 3. Функция потерь (Focal Loss из v3)
        if self.cfg.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        else:
            self.criterion = FocalLoss()

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

    def _init_model(self) -> nn.Module:
        """Настройка Wav2Vec2 с поддержкой LoRA адаптеров."""
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=len(self.label2id),
            label2id=self.label2id,
            id2label=self.id2label
        )

        if self.cfg.use_lora:
            from peft import LoraConfig, get_peft_model
            logger.info(f"Активация LoRA: r={self.cfg.lora_r}")
            lora_config = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                modules_to_save=["classifier", "projector"]
            )
            model = get_peft_model(model, lora_config)

        return model.to(self.device)

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Any) -> float:
        """Цикл обучения за одну эпоху."""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Batch", leave=False):
            x = batch["input_values"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            y = batch["labels"].to(self.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                # CHANGED: Поддержка Mixup (v4)
                if hasattr(self.cfg, 'use_mixup') and self.cfg.use_mixup:
                    x_mix, y_a, y_b, lam = mixup_data(x, y)
                    outputs = self.model(input_values=x_mix, attention_mask=mask)
                    loss = mixup_criterion(self.criterion, outputs.logits, y_a, y_b, lam)
                else:
                    outputs = self.model(input_values=x, attention_mask=mask, labels=y)
                    loss = outputs.loss

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            if self.ema: self.ema.update()
            total_loss += loss.item()

        return total_loss / len(loader)

    def save_checkpoint(self, path: str):
        """
        CHANGED: Безопасное сохранение LoRA.
        Делает merge адаптеров в основную модель для инференса.
        """
        os.makedirs(path, exist_ok=True)
        if self.cfg.use_lora:
            # Чтобы не ломать текущие веса, работаем с копией
            model_copy = deepcopy(self.model).cpu()
            merged_model = model_copy.merge_and_unload()
            merged_model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

        self.feature_extractor.save_pretrained(path)
        logger.info(f"Чекпоинт сохранен: {path}")

def run_full_training():
    """Основная функция запуска обучения."""
    # 1. Данные
    df, l2id, id2l = parse_metadata(str(settings.paths.dataset_csv))

    # 2. Инициализация тренера
    trainer = Trainer(label2id=l2id, id2label=id2l)

    # 3. Загрузчики
    # Разделение 80/20
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_ds, val_ds = build_datasets(train_df, val_df, trainer.feature_extractor, l2id, settings)

    sampler = make_balanced_sampler(train_df, l2id) if settings.training.use_balanced_sampler else None

    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, settings.training.batch_size, settings, trainer.device, sampler=sampler
    )

    # 4. Оптимизатор
    optimizer = AdamW(trainer.model.parameters(), lr=settings.training.learning_rate)
    scheduler, _, _ = build_scheduler(optimizer, settings.training.epochs, len(train_loader), settings.training.warmup_ratio)

    # 5. Цикл
    best_f1 = 0.0
    for epoch in range(settings.training.epochs):
        loss = trainer.train_epoch(train_loader, optimizer, scheduler)
        metrics = trainer.validate(val_loader)

        logger.info(f"Epoch {epoch+1} | Loss: {loss:.4f} | Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            trainer.save_checkpoint(str(settings.paths.checkpoints / "best_model"))

if __name__ == "__main__":
    run_full_training()