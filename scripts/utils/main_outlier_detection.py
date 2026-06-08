"""
outlier_detection.py

Outlier Detection для обученной Wav2Vec2 модели.

Идея:
  1. Извлекаем эмбеддинги из обученной модели для всей обучающей выборки
  2. Считаем статистики (центроиды по классам, ковариационную матрицу)
  3. На инференсе: если эмбеддинг нового аудио далеко от всех центроидов → outlier

Поддерживаемые методы:
  - mahalanobis: расстояние Махаланобиса (учитывает ковариацию) — лучший
  - cosine: косинусное расстояние до ближайшего центроида
  - l2: евклидово расстояние до ближайшего центроида

Использование:
  # 1. Построить детектор на обучающих данных
  python outlier_detection.py fit --model_path best_model --csv_path data.csv --save_path detector.pkl

  # 2. Калибровать порог на валидации
  python outlier_detection.py calibrate --detector_path detector.pkl --model_path best_model --csv_path data.csv

  # 3. Инференс с outlier detection
  python outlier_detection.py predict --detector_path detector.pkl --model_path best_model --audio_path test.wav
"""

import os
import sys
import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# =====================================================================
#  Конфигурация
# =====================================================================
@dataclass
class OutlierConfig:
    """Конфигурация outlier-детектора."""

    # --- Метод детекции ---
    #   mahalanobis: учитывает корреляции между измерениями (рекомендуется)
    #   cosine: угловое расстояние, инвариантно к масштабу
    #   l2: евклидово расстояние
    method: str = "mahalanobis"

    # --- Режим: per-class или global ---
    #   per_class: отдельный центроид для каждого класса (рекомендуется)
    #   global: один общий центроид
    mode: str = "per_class"

    # --- Порог ---
    #   Процентиль расстояний на обучающей/валидационной выборке.
    #   95 означает: если расстояние > 95% обучающих примеров → outlier.
    threshold_percentile: float = 95.0

    # --- Явный порог (если задан, перебивает percentile) ---
    threshold: Optional[float] = None

    # --- Регуляризация ковариационной матрицы (для Mahalanobis) ---
    #   Добавляем epsilon * I к ковариационной матрице для стабильности.
    cov_regularization: float = 1e-6

    # --- Параметры извлечения эмбеддингов ---
    #   Какой слой использовать для эмбеддингов:
    #   "projector" — выход проектора (перед классификатором)
    #   "last_hidden" — последний скрытый слой wav2vec2 (средний пулинг)
    embedding_layer: str = "projector"

    # --- Аудио ---
    max_audio_seconds: float = 3.0
    target_sample_rate: int = 16000

    # --- Батч ---
    batch_size: int = 16
    num_workers: int = 4


# =====================================================================
#  Извлечение эмбеддингов из модели
# =====================================================================
class EmbeddingExtractor:
    """
    Извлекает эмбеддинги из обученной Wav2Vec2ForSequenceClassification.

    Архитектура Wav2Vec2ForSequenceClassification:
        wav2vec2:  audio → (batch, seq_len, 768/1024)   # encoder
        projector: Linear → (batch, seq_len, 256)        # проекция ПО-ТАЙМСТЕПНО
        [mean pooling] → (batch, 256)                    # модель делает сама
        classifier: Linear → (batch, num_classes)        # голова

    Hook перехватывает выход projector ДО mean pooling,
    поэтому получаем 3D тензор. Мы делаем pooling сами.
    """

    def __init__(
            self,
            model: nn.Module,
            feature_extractor: Wav2Vec2FeatureExtractor,
            device: torch.device,
            config: OutlierConfig,
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.config = config

        self.model.eval()
        self.model.to(device)

        self._captured_embedding = None
        self._hook_handle = None

        self._register_hook()

    def _get_base_model(self):
        """
        Получаем Wav2Vec2ForSequenceClassification.
        Проверяем именно isinstance(PeftModel), НЕ hasattr("base_model").
        """
        if PEFT_AVAILABLE and isinstance(self.model, PeftModel):
            base = self.model.base_model
            if hasattr(base, "model"):
                return base.model
            return base
        return self.model

    def _register_hook(self):
        """Регистрируем forward hook для перехвата эмбеддингов."""
        base_model = self._get_base_model()

        model_type = type(base_model).__name__
        available_attrs = [
            attr for attr in ["projector", "classifier", "wav2vec2"]
            if hasattr(base_model, attr)
        ]
        logging.info(
            f"Model type: {model_type}, "
            f"available modules: {available_attrs}"
        )

        if self.config.embedding_layer == "projector":
            if not hasattr(base_model, "projector"):
                logging.warning(
                    f"Model {model_type} has no 'projector'! "
                    f"Available: {available_attrs}. "
                    f"Falling back to classifier_input."
                )
                self.config.embedding_layer = "classifier_input"
                self._register_hook()
                return

            target_module = base_model.projector
            logging.info(
                f"Hook registered on: projector "
                f"({type(target_module).__name__}) "
                f"→ output is 3D, will apply mean pooling"
            )

        elif self.config.embedding_layer == "classifier_input":
            if not hasattr(base_model, "classifier"):
                raise ValueError(
                    f"Model {model_type} has no 'classifier'! "
                    f"Available: {available_attrs}"
                )
            target_module = base_model.classifier
            logging.info(
                f"Hook registered on: classifier INPUT "
                f"({type(target_module).__name__}) "
                f"→ output is 2D (already pooled)"
            )

            def pre_hook_fn(module, input):
                if isinstance(input, tuple) and len(input) > 0:
                    self._captured_embedding = input[0]
                else:
                    self._captured_embedding = input

            self._hook_handle = target_module.register_forward_pre_hook(
                pre_hook_fn
            )
            return

        elif self.config.embedding_layer == "last_hidden":
            if not hasattr(base_model, "wav2vec2"):
                raise ValueError(
                    f"Model {model_type} has no 'wav2vec2'! "
                    f"Available: {available_attrs}"
                )
            target_module = base_model.wav2vec2
            logging.info(
                f"Hook registered on: wav2vec2 encoder "
                f"({type(target_module).__name__}) "
                f"→ output is 3D, will apply mean pooling"
            )
        else:
            raise ValueError(
                f"Unknown embedding_layer: {self.config.embedding_layer}. "
                f"Use 'projector', 'classifier_input', or 'last_hidden'."
            )

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._captured_embedding = output[0]
            else:
                self._captured_embedding = output

        self._hook_handle = target_module.register_forward_hook(hook_fn)

    def remove_hook(self):
        """Убираем hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _pool_embedding(
            self,
            embedding: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Пулинг 3D → 2D.

        Wav2Vec2 CNN сжимает временную ось (примерно в 320 раз),
        поэтому seq_len_embedding ≠ seq_len_input.
        Создаём маску нужной длины.

        Args:
            embedding: (batch, seq_len, dim) или (batch, dim)
            attention_mask: (batch, seq_len_input) — из batch

        Returns:
            (batch, dim)
        """
        if embedding.dim() == 2:
            return embedding

        if embedding.dim() != 3:
            raise ValueError(
                f"Expected 2D or 3D embedding, got {embedding.dim()}D "
                f"shape={embedding.shape}"
            )

        batch_size, seq_len_emb, dim = embedding.shape

        if attention_mask is not None:
            # attention_mask для input_values: (batch, seq_len_input)
            # Wav2Vec2 CNN сжимает: seq_len_emb ≈ seq_len_input / 320
            # Нужно создать маску для embedding-пространства

            seq_len_input = attention_mask.size(1)

            if seq_len_input == seq_len_emb:
                # Совпадает — используем напрямую
                mask = attention_mask
            elif seq_len_input > seq_len_emb:
                # Input длиннее — адаптируем маску
                # Простой подход: для каждого embedding-фрейма
                # проверяем, есть ли хотя бы один валидный input-фрейм
                # в соответствующем окне
                ratio = seq_len_input / seq_len_emb
                mask = torch.ones(
                    batch_size, seq_len_emb,
                    dtype=torch.long, device=embedding.device,
                )
                for b in range(batch_size):
                    # Находим длину валидной части
                    valid_len = attention_mask[b].sum().item()
                    valid_emb_len = int(valid_len / ratio) + 1
                    valid_emb_len = min(valid_emb_len, seq_len_emb)
                    mask[b, valid_emb_len:] = 0
            else:
                # Embedding длиннее input (не должно быть, но на всякий случай)
                mask = torch.ones(
                    batch_size, seq_len_emb,
                    dtype=torch.long, device=embedding.device,
                )
        else:
            # Нет маски — все фреймы валидны
            mask = torch.ones(
                batch_size, seq_len_emb,
                dtype=torch.long, device=embedding.device,
            )

        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (
                (embedding * mask_expanded).sum(dim=1)
                / mask_expanded.sum(dim=1).clamp(min=1)
        )  # (batch, dim)

        return pooled

    @torch.no_grad()
    def extract_single(self, audio_path: str) -> np.ndarray:
        """
        Извлекает эмбеддинг из одного аудиофайла.

        Returns:
            np.ndarray: вектор эмбеддинга [embedding_dim]
        """
        speech_array, sr = torchaudio.load(audio_path)

        if sr != self.config.target_sample_rate:
            speech_array = torchaudio.functional.resample(
                speech_array, sr, self.config.target_sample_rate
            )

        if speech_array.ndim > 1 and speech_array.size(0) > 1:
            speech_array = speech_array.mean(dim=0)
        speech_array = speech_array.squeeze().float()

        max_samples = int(
            self.config.max_audio_seconds * self.config.target_sample_rate
        )
        if speech_array.shape[-1] > max_samples:
            total_len = speech_array.shape[-1]
            start = (total_len - max_samples) // 2
            speech_array = speech_array[start : start + max_samples]

        inputs = self.feature_extractor(
            speech_array.numpy(),
            sampling_rate=self.config.target_sample_rate,
            return_tensors="pt",
            padding=False,
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = torch.ones_like(input_values, dtype=torch.long)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            _ = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
            )

        embedding = self._captured_embedding
        embedding = self._pool_embedding(embedding, attention_mask)
        embedding = embedding.squeeze().cpu().numpy()
        return embedding

    @torch.no_grad()
    def extract_batch(self, batch: dict) -> np.ndarray:
        """
        Извлекает эмбеддинги из батча.

        Returns:
            np.ndarray: матрица эмбеддингов [batch_size, embedding_dim]
        """
        input_values = batch["input_values"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            _ = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
            )

        embedding = self._captured_embedding
        embedding = self._pool_embedding(embedding, attention_mask)
        return embedding.cpu().numpy()

    @torch.no_grad()
    def extract_dataset(
            self,
            dataloader: DataLoader,
            return_labels: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Извлекает эмбеддинги из всего датасета.

        Returns:
            embeddings: np.ndarray [N, embedding_dim]
            labels: np.ndarray [N]
        """
        all_embeddings = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            emb = self.extract_batch(batch)

            # Проверка: должен быть 2D после пулинга
            if emb.ndim != 2:
                raise RuntimeError(
                    f"Expected 2D embeddings after pooling, "
                    f"got {emb.ndim}D shape={emb.shape}"
                )

            all_embeddings.append(emb)

            if return_labels and "labels" in batch:
                all_labels.append(batch["labels"].numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)

        logging.info(
            f"Final embeddings shape: {embeddings.shape} "
            f"(should be [N, dim])"
        )

        if return_labels and all_labels:
            labels = np.concatenate(all_labels, axis=0)
            return embeddings, labels
        return embeddings, None

# =====================================================================
#  Outlier Detector
# =====================================================================
class OutlierDetector:
    """
    Детектор аномальных аудио-паттернов.

    Принцип работы:
    1. fit(): считает центроиды (средние эмбеддинги) для каждого класса
       и ковариационную матрицу
    2. score(): для нового эмбеддинга считает расстояние до ближайшего центроида
    3. predict(): сравнивает расстояние с порогом

    Если эмбеддинг нового аудио далеко от всех обученных паттернов,
    скорее всего это:
    - Посторонний шум
    - Неизвестная команда
    - Сильно искажённая речь
    - Другой язык
    """

    def __init__(self, config: Optional[OutlierConfig] = None):
        self.config = config or OutlierConfig()
        self.fitted = False

        # --- Статистики (заполняются в fit()) ---
        self.global_centroid: Optional[np.ndarray] = None  # [D]
        self.class_centroids: Dict[int, np.ndarray] = {}   # {class_id: [D]}
        self.class_counts: Dict[int, int] = {}
        self.covariance_inv: Optional[np.ndarray] = None   # [D, D]

        # --- Пороги (заполняются в calibrate()) ---
        self.threshold: Optional[float] = None
        self.per_class_thresholds: Dict[int, float] = {}

        # --- Статистики расстояний обучающей выборки ---
        self.train_distances: Optional[np.ndarray] = None
        self.train_distance_stats: Dict[str, float] = {}

        # --- Мета-информация ---
        self.embedding_dim: Optional[int] = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

    def fit(
            self,
            embeddings: np.ndarray,
            labels: np.ndarray,
            id2label: Optional[Dict[int, str]] = None,
    ) -> "OutlierDetector":
        """
        Обучает детектор на эмбеддингах обучающей выборки.

        Args:
            embeddings: [N, D] — эмбеддинги обучающих примеров
            labels: [N] — метки классов
            id2label: маппинг id → имя класса

        Returns:
            self (для chaining)
        """
        N, D = embeddings.shape
        self.embedding_dim = D

        if id2label:
            self.id2label = id2label
            self.label2id = {v: k for k, v in id2label.items()}

        logging.info(f"Fitting outlier detector: {N} samples, {D}-dim embeddings")

        # --- Глобальный центроид ---
        self.global_centroid = embeddings.mean(axis=0)

        # --- Per-class центроиды ---
        unique_labels = np.unique(labels)
        self.class_centroids = {}
        self.class_counts = {}

        for cls_id in unique_labels:
            mask = labels == cls_id
            cls_embeddings = embeddings[mask]
            self.class_centroids[int(cls_id)] = cls_embeddings.mean(axis=0)
            self.class_counts[int(cls_id)] = int(mask.sum())

            cls_name = self.id2label.get(int(cls_id), str(cls_id))
            logging.info(
                f"  Class '{cls_name}' (id={cls_id}): "
                f"{self.class_counts[int(cls_id)]} samples"
            )

        # --- Ковариационная матрица (для Mahalanobis) ---
        if self.config.method == "mahalanobis":
            self._fit_covariance(embeddings, labels)

        # --- Считаем расстояния на обучающих данных ---
        self.train_distances = np.array([
            self.score(embeddings[i]) for i in range(N)
        ])

        self.train_distance_stats = {
            "min": float(self.train_distances.min()),
            "mean": float(self.train_distances.mean()),
            "std": float(self.train_distances.std()),
            "median": float(np.median(self.train_distances)),
            "p90": float(np.percentile(self.train_distances, 90)),
            "p95": float(np.percentile(self.train_distances, 95)),
            "p99": float(np.percentile(self.train_distances, 99)),
            "max": float(self.train_distances.max()),
        }

        logging.info("Training distance statistics:")
        for k, v in self.train_distance_stats.items():
            logging.info(f"  {k}: {v:.4f}")

        # --- Порог по умолчанию ---
        if self.config.threshold is not None:
            self.threshold = self.config.threshold
        else:
            self.threshold = float(
                np.percentile(
                    self.train_distances,
                    self.config.threshold_percentile,
                )
            )

        logging.info(
            f"Default threshold ({self.config.threshold_percentile}th percentile): "
            f"{self.threshold:.4f}"
        )

        self.fitted = True
        return self

    def _fit_covariance(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Считает общую ковариационную матрицу (shared across classes).

        Используем class-centered embeddings:
        вычитаем из каждого эмбеддинга центроид его класса,
        потом считаем ковариацию. Это стандартный подход
        (как в Gaussian Discriminant Analysis).
        """
        N, D = embeddings.shape
        centered = np.zeros_like(embeddings)

        for cls_id, centroid in self.class_centroids.items():
            mask = labels == cls_id
            centered[mask] = embeddings[mask] - centroid

        # Ковариационная матрица
        cov = np.cov(centered, rowvar=False)  # [D, D]

        # Регуляризация: cov + eps * I
        cov += self.config.cov_regularization * np.eye(D)

        # Обратная матрица
        try:
            self.covariance_inv = np.linalg.inv(cov)
            logging.info(
                f"Covariance matrix: {D}x{D}, "
                f"condition number: {np.linalg.cond(cov):.1f}"
            )
        except np.linalg.LinAlgError:
            logging.warning(
                "Covariance matrix is singular! "
                "Using pseudo-inverse."
            )
            self.covariance_inv = np.linalg.pinv(cov)

    def score(self, embedding: np.ndarray) -> float:
        """
        Считает outlier score для одного эмбеддинга.

        Чем БОЛЬШЕ score, тем больше вероятность что это outlier.

        Returns:
            float: расстояние до ближайшего центроида
        """
        if not self.fitted and not self.class_centroids:
            raise RuntimeError("Detector not fitted! Call .fit() first.")

        if self.config.mode == "per_class":
            # Расстояние до БЛИЖАЙШЕГО центроида
            distances = []
            for cls_id, centroid in self.class_centroids.items():
                d = self._distance(embedding, centroid)
                distances.append(d)
            return float(min(distances))

        else:  # global
            return float(
                self._distance(embedding, self.global_centroid)
            )

    def score_with_details(
            self, embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Считает расстояния до ВСЕХ центроидов.

        Returns:
            dict: {
                "min_distance": float,
                "nearest_class": str,
                "nearest_class_id": int,
                "distances": {class_name: distance, ...}
            }
        """
        distances = {}
        for cls_id, centroid in self.class_centroids.items():
            d = self._distance(embedding, centroid)
            cls_name = self.id2label.get(cls_id, str(cls_id))
            distances[cls_name] = float(d)

        nearest_cls = min(distances, key=distances.get)
        nearest_cls_id = self.label2id.get(nearest_cls, -1)

        return {
            "min_distance": distances[nearest_cls],
            "nearest_class": nearest_cls,
            "nearest_class_id": nearest_cls_id,
            "distances": distances,
            "is_outlier": distances[nearest_cls] > (self.threshold or float("inf")),
            "threshold": self.threshold,
        }

    def _distance(self, x: np.ndarray, centroid: np.ndarray) -> float:
        """Считает расстояние между эмбеддингом и центроидом."""
        diff = x - centroid

        if self.config.method == "mahalanobis":
            if self.covariance_inv is not None:
                # d = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
                return float(
                    np.sqrt(np.dot(np.dot(diff, self.covariance_inv), diff))
                )
            else:
                # Fallback to L2
                return float(np.linalg.norm(diff))

        elif self.config.method == "cosine":
            # 1 - cos_sim (0 = идентичный, 2 = противоположный)
            norm_x = np.linalg.norm(x)
            norm_c = np.linalg.norm(centroid)
            if norm_x < 1e-10 or norm_c < 1e-10:
                return 1.0
            cos_sim = np.dot(x, centroid) / (norm_x * norm_c)
            return float(1.0 - cos_sim)

        elif self.config.method == "l2":
            return float(np.linalg.norm(diff))

        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def predict(
            self,
            embedding: np.ndarray,
            threshold: Optional[float] = None,
    ) -> bool:
        """
        Предсказывает: является ли эмбеддинг outlier-ом.

        Returns:
            True если outlier (далеко от обучающих данных)
        """
        th = threshold or self.threshold
        if th is None:
            raise ValueError(
                "Threshold not set! Call calibrate() or set manually."
            )

        distance = self.score(embedding)
        return distance > th

    def predict_batch(
            self,
            embeddings: np.ndarray,
            threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает outlier для батча эмбеддингов.

        Returns:
            is_outlier: np.ndarray[bool] — True если outlier
            distances: np.ndarray[float] — расстояния
        """
        th = threshold or self.threshold
        distances = np.array([self.score(emb) for emb in embeddings])
        is_outlier = distances > th
        return is_outlier, distances

    def calibrate(
            self,
            val_embeddings: np.ndarray,
            val_labels: np.ndarray,
            percentile: Optional[float] = None,
            outlier_embeddings: Optional[np.ndarray] = None,
    ) -> float:
        """
        Калибрует порог на валидационных данных.

        Стратегия:
        1. Если есть outlier_embeddings (примеры аномалий):
           ищем порог, максимизирующий разделение
        2. Иначе: используем percentile от валидационных расстояний

        Args:
            val_embeddings: эмбеддинги валидационной выборки (нормальные)
            val_labels: метки валидационной выборки
            percentile: процентиль для порога
            outlier_embeddings: (опционально) эмбеддинги заведомых аномалий

        Returns:
            threshold: откалиброванный порог
        """
        p = percentile or self.config.threshold_percentile

        val_distances = np.array([
            self.score(emb) for emb in val_embeddings
        ])

        logging.info(f"Validation distances ({len(val_distances)} samples):")
        logging.info(f"  Mean: {val_distances.mean():.4f}")
        logging.info(f"  Std:  {val_distances.std():.4f}")
        logging.info(f"  95%:  {np.percentile(val_distances, 95):.4f}")
        logging.info(f"  99%:  {np.percentile(val_distances, 99):.4f}")
        logging.info(f"  Max:  {val_distances.max():.4f}")

        if outlier_embeddings is not None and len(outlier_embeddings) > 0:
            # Если есть примеры аномалий — ищем оптимальный порог
            outlier_distances = np.array([
                self.score(emb) for emb in outlier_embeddings
            ])

            logging.info(
                f"Outlier distances ({len(outlier_distances)} samples):"
            )
            logging.info(f"  Mean: {outlier_distances.mean():.4f}")
            logging.info(f"  Min:  {outlier_distances.min():.4f}")

            # Ищем порог, максимизирующий separation
            best_threshold = self._find_optimal_threshold(
                val_distances, outlier_distances
            )
            self.threshold = best_threshold
        else:
            self.threshold = float(np.percentile(val_distances, p))

        logging.info(f"Calibrated threshold: {self.threshold:.4f}")

        # --- Per-class thresholds ---
        unique_labels = np.unique(val_labels)
        for cls_id in unique_labels:
            mask = val_labels == cls_id
            cls_distances = val_distances[mask]
            cls_threshold = float(np.percentile(cls_distances, p))
            self.per_class_thresholds[int(cls_id)] = cls_threshold

            cls_name = self.id2label.get(int(cls_id), str(cls_id))
            logging.info(
                f"  Class '{cls_name}': threshold={cls_threshold:.4f} "
                f"(mean={cls_distances.mean():.4f})"
            )

        return self.threshold

    def _find_optimal_threshold(
            self,
            normal_distances: np.ndarray,
            outlier_distances: np.ndarray,
    ) -> float:
        """
        Находит порог, максимизирующий F1 между normal и outlier.
        """
        all_distances = np.concatenate([normal_distances, outlier_distances])
        all_labels = np.concatenate([
            np.zeros(len(normal_distances)),  # 0 = normal
            np.ones(len(outlier_distances)),  # 1 = outlier
        ])

        best_f1 = 0
        best_threshold = float(np.percentile(normal_distances, 95))

        # Перебираем кандидатов
        candidates = np.percentile(
            all_distances,
            np.linspace(50, 99.9, 500),
        )

        for th in candidates:
            preds = (all_distances > th).astype(int)
            tp = ((preds == 1) & (all_labels == 1)).sum()
            fp = ((preds == 1) & (all_labels == 0)).sum()
            fn = ((preds == 0) & (all_labels == 1)).sum()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(th)

        logging.info(
            f"Optimal threshold: {best_threshold:.4f} "
            f"(F1={best_f1:.4f})"
        )
        return best_threshold

    # =====================================================================
    #  Сохранение / Загрузка
    # =====================================================================
    def save(self, path: str):
        """Сохраняет детектор в файл."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        state = {
            "config": asdict(self.config),
            "global_centroid": self.global_centroid,
            "class_centroids": self.class_centroids,
            "class_counts": self.class_counts,
            "covariance_inv": self.covariance_inv,
            "threshold": self.threshold,
            "per_class_thresholds": self.per_class_thresholds,
            "train_distance_stats": self.train_distance_stats,
            "embedding_dim": self.embedding_dim,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "fitted": self.fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logging.info(f"Outlier detector saved to: {path}")

        # Также сохраняем человекочитаемый JSON
        # Также сохраняем человекочитаемый JSON (UTF‑8, чтобы не падать на кириллице в Windows)
        json_path = path.replace(".pkl", "_info.json")
        json_state = {
            "config": asdict(self.config),
            "threshold": self.threshold,
            "per_class_thresholds": {
                self.id2label.get(k, str(k)): v
                for k, v in self.per_class_thresholds.items()
            },
            "train_distance_stats": self.train_distance_stats,
            "class_counts": {
                self.id2label.get(k, str(k)): v
                for k, v in self.class_counts.items()
            },
            "embedding_dim": self.embedding_dim,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_state, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "OutlierDetector":
        """Загружает детектор из файла."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        config = OutlierConfig(**state["config"])
        detector = cls(config)

        detector.global_centroid = state["global_centroid"]
        detector.class_centroids = state["class_centroids"]
        detector.class_counts = state["class_counts"]
        detector.covariance_inv = state["covariance_inv"]
        detector.threshold = state["threshold"]
        detector.per_class_thresholds = state.get("per_class_thresholds", {})
        detector.train_distance_stats = state.get("train_distance_stats", {})
        detector.embedding_dim = state["embedding_dim"]
        detector.id2label = state["id2label"]
        detector.label2id = state["label2id"]
        detector.fitted = state["fitted"]

        logging.info(f"Outlier detector loaded from: {path}")
        logging.info(
            f"  Method: {config.method}, "
            f"Threshold: {detector.threshold:.4f}, "
            f"Embedding dim: {detector.embedding_dim}"
        )

        return detector


# =====================================================================
#  Production Pipeline — полный инференс с outlier detection
# =====================================================================
class ProductionPipeline:
    """
    Production-ready пайплайн: классификация + outlier detection.

    Использование:
        pipeline = ProductionPipeline.from_saved(
            model_path="best_model/",
            detector_path="detector.pkl",
            device="cuda",
        )

        result = pipeline.predict("audio.wav")
        # {
        #     "predicted_class": "включи свет",
        #     "confidence": 0.95,
        #     "is_outlier": False,
        #     "outlier_score": 3.2,
        #     "outlier_threshold": 8.5,
        #     "all_probabilities": {"включи свет": 0.95, ...},
        #     "processing_time_ms": 45.3,
        # }
    """

    def __init__(
            self,
            model: nn.Module,
            feature_extractor: Wav2Vec2FeatureExtractor,
            detector: OutlierDetector,
            id2label: Dict[int, str],
            device: torch.device,
            config: Optional[OutlierConfig] = None,
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.detector = detector
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.device = device
        self.config = config or detector.config

        self.model.eval()
        self.model.to(device)

        # --- Extractor для получения эмбеддингов ---
        self.extractor = EmbeddingExtractor(
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            config=self.config,
        )

    @classmethod
    def from_saved(
            cls,
            model_path: str,
            detector_path: str,
            device: str = "cuda",
            label2id: Optional[dict] = None,
            id2label: Optional[dict] = None,
    ) -> "ProductionPipeline":
        """
        Создаёт пайплайн из сохранённых артефактов.

        Args:
            model_path: путь к сохранённой модели (merged или с LoRA)
            detector_path: путь к outlier detector (.pkl)
            device: "cuda" или "cpu"
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        # --- Загружаем модель ---
        model, feature_extractor = _load_model_robust(model_path, device)

        # --- id2label из модели ---
        if id2label is None:
            id2label = model.config.id2label
            # Конвертируем ключи в int
            id2label = {int(k): v for k, v in id2label.items()}

        # --- Загружаем детектор ---
        detector = OutlierDetector.load(detector_path)

        logging.info(
            f"Pipeline loaded: model from '{model_path}', "
            f"detector from '{detector_path}'"
        )

        return cls(
            model=model,
            feature_extractor=feature_extractor,
            detector=detector,
            id2label=id2label,
            device=device,
            config=detector.config,
        )

    @torch.no_grad()
    def predict(
            self,
            audio_path: str,
            threshold: Optional[float] = None,
            return_embedding: bool = False,
    ) -> dict:
        """
        Полный инференс: классификация + outlier detection.

        Args:
            audio_path: путь к аудиофайлу
            threshold: порог outlier (None = использовать калиброванный)
            return_embedding: вернуть ли эмбеддинг в результате

        Returns:
            dict с результатами
        """
        t0 = time.time()

        # --- Загрузка аудио ---
        speech_array, sr = torchaudio.load(audio_path)

        if sr != self.config.target_sample_rate:
            speech_array = torchaudio.functional.resample(
                speech_array, sr, self.config.target_sample_rate
            )

        if speech_array.ndim > 1 and speech_array.size(0) > 1:
            speech_array = speech_array.mean(dim=0)
        speech_array = speech_array.squeeze().float()

        # Обрезка
        max_samples = int(
            self.config.max_audio_seconds * self.config.target_sample_rate
        )
        if speech_array.shape[-1] > max_samples:
            total_len = speech_array.shape[-1]
            start = (total_len - max_samples) // 2
            speech_array = speech_array[start : start + max_samples]

        # --- Feature extraction ---
        inputs = self.feature_extractor(
            speech_array.numpy(),
            sampling_rate=self.config.target_sample_rate,
            return_tensors="pt",
            padding=False,
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = torch.ones_like(input_values, dtype=torch.long)

        # --- Forward pass ---
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
            )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_class_id = int(np.argmax(probs))
        pred_class = self.id2label[pred_class_id]
        confidence = float(probs[pred_class_id])

        # В ProductionPipeline.predict(), замените блок получения эмбеддинга:

        # --- Получаем эмбеддинг (уже перехвачен hook-ом) ---
        embedding = self.extractor._captured_embedding

        # Пулинг 3D → 2D (projector выдаёт per-timestep)
        if embedding.dim() == 3:
            embedding = embedding.mean(dim=1)

        embedding = embedding.squeeze().cpu().numpy()

        # --- Outlier detection ---
        outlier_details = self.detector.score_with_details(embedding)

        th = threshold or self.detector.threshold
        is_outlier = outlier_details["min_distance"] > th

        processing_time = (time.time() - t0) * 1000  # ms

        result = {
            "predicted_class": pred_class,
            "predicted_class_id": pred_class_id,
            "confidence": confidence,
            "is_outlier": is_outlier,
            "outlier_score": outlier_details["min_distance"],
            "outlier_threshold": th,
            "nearest_class": outlier_details["nearest_class"],
            "class_distances": outlier_details["distances"],
            "all_probabilities": {
                self.id2label[i]: float(probs[i])
                for i in range(len(probs))
            },
            "processing_time_ms": round(processing_time, 1),
        }

        if return_embedding:
            result["embedding"] = embedding

        return result

    def predict_batch_files(
            self,
            audio_paths: List[str],
            threshold: Optional[float] = None,
    ) -> List[dict]:
        """Инференс для списка файлов."""
        results = []
        for path in tqdm(audio_paths, desc="Predicting"):
            try:
                result = self.predict(path, threshold=threshold)
            except Exception as e:
                result = {
                    "predicted_class": "ERROR",
                    "confidence": 0.0,
                    "is_outlier": True,
                    "outlier_score": float("inf"),
                    "error": str(e),
                    "audio_path": path,
                }
            results.append(result)
        return results


# =====================================================================
#  Визуализация
# =====================================================================
def plot_outlier_analysis(
        detector: OutlierDetector,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        outlier_embeddings: Optional[np.ndarray] = None,
        save_dir: str = ".",
):
    """
    Визуализирует outlier detection:
    1. Распределение расстояний по классам
    2. t-SNE / PCA визуализация эмбеддингов
    3. Гистограмма расстояний с порогом
    """
    if not PLOT_AVAILABLE:
        logging.warning("matplotlib/seaborn not available, skipping plots.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # === 1. Гистограмма расстояний ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-class distances
    unique_labels = np.unique(train_labels)
    for cls_id in unique_labels:
        mask = train_labels == cls_id
        cls_embs = train_embeddings[mask]
        cls_distances = np.array([
            detector.score(emb) for emb in cls_embs
        ])
        cls_name = detector.id2label.get(int(cls_id), str(cls_id))
        axes[0].hist(
            cls_distances, bins=50, alpha=0.5, label=cls_name, density=True
        )

    if detector.threshold:
        axes[0].axvline(
            detector.threshold, color="red", linestyle="--",
            linewidth=2, label=f"Threshold={detector.threshold:.2f}",
        )

    axes[0].set_title("Distance Distribution by Class (Train)")
    axes[0].set_xlabel("Distance to nearest centroid")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Overall + val + outliers
    all_train_dist = np.array([
        detector.score(emb) for emb in train_embeddings
    ])
    axes[1].hist(
        all_train_dist, bins=50, alpha=0.5, label="Train", density=True
    )

    if val_embeddings is not None:
        val_dist = np.array([
            detector.score(emb) for emb in val_embeddings
        ])
        axes[1].hist(
            val_dist, bins=50, alpha=0.5, label="Validation", density=True
        )

    if outlier_embeddings is not None:
        out_dist = np.array([
            detector.score(emb) for emb in outlier_embeddings
        ])
        axes[1].hist(
            out_dist, bins=50, alpha=0.5, label="Outliers", density=True,
            color="red",
        )

    if detector.threshold:
        axes[1].axvline(
            detector.threshold, color="red", linestyle="--",
            linewidth=2, label=f"Threshold={detector.threshold:.2f}",
        )

    axes[1].set_title("Distance Distribution Overview")
    axes[1].set_xlabel("Distance to nearest centroid")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "outlier_distances.png"), dpi=150
    )
    plt.close()

    # === 2. t-SNE визуализация ===
    try:
        from sklearn.manifold import TSNE

        all_embs = [train_embeddings]
        all_types = ["train"] * len(train_embeddings)
        all_cls = list(train_labels)

        if val_embeddings is not None:
            all_embs.append(val_embeddings)
            all_types.extend(["val"] * len(val_embeddings))
            all_cls.extend(
                list(val_labels) if val_labels is not None
                else [-1] * len(val_embeddings)
            )

        if outlier_embeddings is not None:
            all_embs.append(outlier_embeddings)
            all_types.extend(["outlier"] * len(outlier_embeddings))
            all_cls.extend([-2] * len(outlier_embeddings))

        combined = np.concatenate(all_embs, axis=0)

        # Ограничиваем для скорости
        max_points = 3000
        if len(combined) > max_points:
            indices = np.random.choice(len(combined), max_points, replace=False)
            combined = combined[indices]
            all_types = [all_types[i] for i in indices]
            all_cls = [all_cls[i] for i in indices]

        logging.info(
            f"Computing t-SNE for {len(combined)} points..."
        )
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(combined)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # По типу (train/val/outlier)
        for t, color in [("train", "blue"), ("val", "green"), ("outlier", "red")]:
            mask = [i for i, x in enumerate(all_types) if x == t]
            if mask:
                axes[0].scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, alpha=0.3, s=10, label=t,
                )
        axes[0].set_title("t-SNE: Train / Val / Outlier")
        axes[0].legend()

        # По классу
        unique_cls = sorted(set(all_cls))
        cmap = plt.cm.get_cmap("tab10", len(unique_cls))
        for i, cls_id in enumerate(unique_cls):
            mask = [j for j, x in enumerate(all_cls) if x == cls_id]
            cls_name = detector.id2label.get(cls_id, f"id={cls_id}")
            if cls_id == -1:
                cls_name = "val (no label)"
            elif cls_id == -2:
                cls_name = "outlier"
            axes[1].scatter(
                coords[mask, 0], coords[mask, 1],
                c=[cmap(i)], alpha=0.4, s=10, label=cls_name,
            )

        # Центроиды на t-SNE
        for cls_id, centroid in detector.class_centroids.items():
            # Проецируем центроид (ближайшая точка)
            dists = np.linalg.norm(
                combined - centroid, axis=1
            )
            nearest = np.argmin(dists)
            axes[1].scatter(
                coords[nearest, 0], coords[nearest, 1],
                c="black", marker="X", s=200, zorder=5,
            )

        axes[1].set_title("t-SNE: By Class")
        axes[1].legend(markerscale=3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "outlier_tsne.png"), dpi=150
        )
        plt.close()
        logging.info("t-SNE plot saved.")

    except ImportError:
        logging.warning("sklearn not available, skipping t-SNE plot.")


# =====================================================================
#  Utility: Dataset и DataLoader (переиспользуем из основного кода)
# =====================================================================
class SimpleAudioDataset(Dataset):
    """Простой датасет для извлечения эмбеддингов."""

    def __init__(
            self,
            df: pd.DataFrame,
            feature_extractor: Wav2Vec2FeatureExtractor,
            label2id: dict,
            max_seconds: float = 3.0,
            path_col: str = "audio_path",
    ):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.target_sr = 16000
        self.max_samples = int(max_seconds * self.target_sr)
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.label2id[row["class"]]

        try:
            speech_array, sr = torchaudio.load(row[self.path_col])
        except Exception:
            speech_array = torch.zeros(1, self.max_samples)
            sr = self.target_sr

        if sr != self.target_sr:
            speech_array = torchaudio.functional.resample(
                speech_array, sr, self.target_sr
            )

        if speech_array.ndim > 1 and speech_array.size(0) > 1:
            speech_array = speech_array.mean(dim=0)
        speech_array = speech_array.squeeze().float()

        if speech_array.shape[-1] > self.max_samples:
            total_len = speech_array.shape[-1]
            start = (total_len - self.max_samples) // 2
            speech_array = speech_array[start : start + self.max_samples]

        inputs = self.feature_extractor(
            speech_array,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=False,
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": label,
        }


def simple_collator(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    lengths = [len(x) for x in input_values]

    padded = torch.nn.utils.rnn.pad_sequence(
        input_values, batch_first=True, padding_value=0.0
    )
    mask = torch.zeros(len(batch), padded.size(1), dtype=torch.long)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    return {
        "input_values": padded,
        "attention_mask": mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# =====================================================================
#  Helper: robust model loader (handles merged LoRA checkpoints)
# =====================================================================
def _load_model_robust(
    model_path: str,
    device: torch.device,
) -> tuple:
    """
    Load Wav2Vec2ForSequenceClassification from a local checkpoint robustly.

    Tries from_pretrained first; if it fails (e.g. adapter_attn_dim error
    from merged LoRA saves), falls back to loading config + state_dict
    overlay so the fine-tuned weights are applied correctly.

    Returns:
        (model, feature_extractor) both ready for inference on ``device``.
    """
    import json as _json

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    # Read id2label from config so we can initialise num_labels correctly
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = _json.load(f)
    id2label = cfg_dict.get("id2label", {})
    num_labels = len(id2label) if id2label else 4

    # --- Route 1: standard from_pretrained (works for clean HF saves) ---
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        model.to(device).eval()
        logging.info("Model loaded via from_pretrained.")
        return model, feature_extractor
    except Exception as exc:
        logging.warning(f"from_pretrained failed: {exc}  →  falling back to state_dict load.")

    # --- Route 2: base architecture + state_dict overlay ---
    from transformers import Wav2Vec2Config
    config = Wav2Vec2Config.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification(config)

    sf_path = os.path.join(model_path, "model.safetensors")
    bin_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(sf_path):
        from safetensors.torch import load_file
        state_dict = load_file(sf_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No weight file found in {model_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logging.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.to(device).eval()
    logging.info(f"Model loaded via state_dict overlay ({sum(p.numel() for p in model.parameters()):,} params).")
    return model, feature_extractor


# =====================================================================
#  CLI: fit / calibrate / predict / analyze
# =====================================================================
def cmd_fit(args):
    """Обучает outlier detector на обучающих данных."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OutlierConfig(
        method=args.method,
        mode=args.mode,
        threshold_percentile=args.percentile,
        max_audio_seconds=args.max_seconds,
        embedding_layer=args.embedding_layer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --- Загружаем модель ---
    model_path = str(Path(args.model_path).resolve())
    logging.info(f"Loading model from: {model_path}")
    model, feature_extractor = _load_model_robust(model_path, device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in model.config.id2label.items()}

    # --- Загружаем данные ---
    logging.info(f"Loading data from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    # Фильтруем по группам
    if args.train_groups:
        groups = [g.strip() for g in args.train_groups.split(",")]
        df = df[df["audio_group"].isin(groups)]
        logging.info(f"Filtered to groups: {groups} → {len(df)} samples")

    logging.info(f"Dataset: {len(df)} samples, classes: {df['class'].value_counts().to_dict()}")

    # --- Датасет ---
    dataset = SimpleAudioDataset(
        df, feature_extractor, label2id,
        max_seconds=config.max_audio_seconds,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=simple_collator,
        num_workers=config.num_workers,
    )

    # --- Извлекаем эмбеддинги ---
    extractor = EmbeddingExtractor(model, feature_extractor, device, config)
    embeddings, labels = extractor.extract_dataset(dataloader, return_labels=True)
    extractor.remove_hook()

    logging.info(f"Extracted embeddings: shape={embeddings.shape}")

    # --- Fit detector ---
    detector = OutlierDetector(config)
    detector.fit(embeddings, labels, id2label=id2label)

    # --- Сохраняем ---
    detector.save(args.save_path)
    logging.info(f"Detector saved to: {args.save_path}")

    # --- Визуализация ---
    if args.plot:
        plot_dir = os.path.dirname(args.save_path) or "."
        plot_outlier_analysis(
            detector, embeddings, labels, save_dir=plot_dir
        )

    return detector


def cmd_calibrate(args):
    """Калибрует порог на валидационных данных."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Загружаем детектор ---
    detector = OutlierDetector.load(args.detector_path)
    config = detector.config

    # --- Загружаем модель ---
    model_path = str(Path(args.model_path).resolve())
    model, feature_extractor = _load_model_robust(model_path, device)

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in model.config.id2label.items()}

    # --- Валидационные данные ---
    df = pd.read_csv(args.csv_path)
    if args.val_groups:
        groups = [g.strip() for g in args.val_groups.split(",")]
        df = df[df["audio_group"].isin(groups)]
    logging.info(f"Validation data: {len(df)} samples")

    dataset = SimpleAudioDataset(
        df, feature_extractor, label2id,
        max_seconds=config.max_audio_seconds,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=simple_collator,
        num_workers=config.num_workers,
    )

    extractor = EmbeddingExtractor(model, feature_extractor, device, config)
    val_embeddings, val_labels = extractor.extract_dataset(
        dataloader, return_labels=True
    )
    extractor.remove_hook()

    # --- Outlier данные (если есть) ---
    outlier_embeddings = None
    if args.outlier_csv:
        outlier_df = pd.read_csv(args.outlier_csv)
        logging.info(f"Outlier data: {len(outlier_df)} samples")

        # Для outlier-ов нужен хотя бы фиктивный label
        if "class" not in outlier_df.columns:
            # Берём первый класс как фиктивный
            outlier_df["class"] = list(label2id.keys())[0]

        outlier_dataset = SimpleAudioDataset(
            outlier_df, feature_extractor, label2id,
            max_seconds=config.max_audio_seconds,
        )
        outlier_loader = DataLoader(
            outlier_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=simple_collator,
            num_workers=config.num_workers,
        )

        extractor2 = EmbeddingExtractor(model, feature_extractor, device, config)
        outlier_embeddings, _ = extractor2.extract_dataset(
            outlier_loader, return_labels=False
        )
        extractor2.remove_hook()

    # --- Калибровка ---
    threshold = detector.calibrate(
        val_embeddings, val_labels,
        percentile=args.percentile,
        outlier_embeddings=outlier_embeddings,
    )

    # --- Сохраняем обновлённый детектор ---
    detector.save(args.detector_path)
    logging.info(f"Updated detector saved with threshold={threshold:.4f}")

    return threshold


def cmd_predict(args):
    """Инференс с outlier detection."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device_str = args.device if hasattr(args, "device") else "cuda"
    pipeline = ProductionPipeline.from_saved(
        model_path=str(Path(args.model_path).resolve()),
        detector_path=str(Path(args.detector_path).resolve()),
        device=device_str,
    )

    if os.path.isfile(args.audio_path):
        # Один файл
        result = pipeline.predict(args.audio_path)
        _print_prediction(result, args.audio_path)

    elif os.path.isdir(args.audio_path):
        # Директория с файлами
        audio_files = []
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            import glob
            audio_files.extend(
                glob.glob(os.path.join(args.audio_path, f"**/*{ext}"), recursive=True)
            )
        logging.info(f"Found {len(audio_files)} audio files")

        results = pipeline.predict_batch_files(audio_files)

        outliers = [r for r in results if r.get("is_outlier", False)]
        logging.info(
            f"\nResults: {len(results)} total, "
            f"{len(outliers)} outliers ({100*len(outliers)/len(results):.1f}%)"
        )

        for r, path in zip(results, audio_files):
            status = "⚠ OUTLIER" if r.get("is_outlier") else "✓ OK"
            logging.info(
                f"  {status} | {r.get('predicted_class', '?'):20s} "
                f"| conf={r.get('confidence', 0):.3f} "
                f"| dist={r.get('outlier_score', 0):.3f} "
                f"| {os.path.basename(path)}"
            )

    else:
        logging.error(f"Path not found: {args.audio_path}")


def _print_prediction(result, audio_path):
    """Красиво выводит результат предсказания."""
    status = "⚠ OUTLIER" if result["is_outlier"] else "✓ NORMAL"

    print(f"\n{'='*60}")
    print(f"Audio: {audio_path}")
    print(f"Status: {status}")
    print(f"{'='*60}")
    print(f"Predicted class:   {result['predicted_class']}")
    print(f"Confidence:        {result['confidence']:.4f}")
    print(f"Outlier score:     {result['outlier_score']:.4f}")
    print(f"Threshold:         {result['outlier_threshold']:.4f}")
    print(f"Nearest class:     {result['nearest_class']}")
    print(f"Processing time:   {result['processing_time_ms']:.1f} ms")

    print(f"\nClass probabilities:")
    for cls, prob in sorted(
            result["all_probabilities"].items(),
            key=lambda x: -x[1],
    ):
        bar = "█" * int(prob * 30)
        print(f"  {cls:20s} {prob:.4f} {bar}")

    print(f"\nDistances to centroids:")
    for cls, dist in sorted(
            result["class_distances"].items(),
            key=lambda x: x[1],
    ):
        marker = " ← nearest" if cls == result["nearest_class"] else ""
        print(f"  {cls:20s} {dist:.4f}{marker}")

    print(f"{'='*60}\n")


# =====================================================================
#  Интеграция с основным обучающим скриптом
# =====================================================================
def build_outlier_detector_after_training(
        cfg: dict,
        model_path: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        label2id: dict,
        id2label: dict,
        device: torch.device,
        save_dir: str,
        method: str = "mahalanobis",
        percentile: float = 95.0,
) -> OutlierDetector:
    """
    Вызывается после обучения модели для автоматического
    построения outlier detector.

    Встраивается в main() обучающего скрипта.
    """
    logging.info("=" * 60)
    logging.info("Building Outlier Detector...")
    logging.info("=" * 60)

    config = OutlierConfig(
        method=method,
        mode="per_class",
        threshold_percentile=percentile,
        max_audio_seconds=cfg.get("max_audio_seconds", 3.0),
        embedding_layer="projector",
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 4),
    )

    # --- Загружаем обученную модель ---
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # --- Извлекаем train эмбеддинги ---
    logging.info("Extracting train embeddings...")
    train_dataset = SimpleAudioDataset(
        train_df, feature_extractor, label2id,
        max_seconds=config.max_audio_seconds,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=simple_collator,
        num_workers=config.num_workers,
    )

    extractor = EmbeddingExtractor(model, feature_extractor, device, config)
    train_embeddings, train_labels = extractor.extract_dataset(
        train_loader, return_labels=True
    )

    logging.info(
        f"Train embeddings: {train_embeddings.shape} "
        f"({train_embeddings.shape[1]}-dim)"
    )

    # --- Fit detector ---
    detector = OutlierDetector(config)
    detector.fit(train_embeddings, train_labels, id2label=id2label)

    # --- Калибровка на валидации ---
    logging.info("Extracting val embeddings for calibration...")
    val_dataset = SimpleAudioDataset(
        val_df, feature_extractor, label2id,
        max_seconds=config.max_audio_seconds,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=simple_collator,
        num_workers=config.num_workers,
    )

    val_embeddings, val_labels = extractor.extract_dataset(
        val_loader, return_labels=True
    )
    extractor.remove_hook()

    detector.calibrate(val_embeddings, val_labels, percentile=percentile)

    # --- Сохраняем ---
    detector_path = os.path.join(save_dir, "outlier_detector.pkl")
    detector.save(detector_path)

    # --- Визуализация ---
    plot_outlier_analysis(
        detector, train_embeddings, train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        save_dir=save_dir,
    )

    # --- Cleanup ---
    del model, extractor
    del train_embeddings, val_embeddings
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Outlier detector ready!")
    return detector


# =====================================================================
#  MAIN: CLI interface
# =====================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Outlier Detection для Wav2Vec2 модели"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- fit ---
    p_fit = subparsers.add_parser("fit", help="Обучить детектор")
    p_fit.add_argument("--model_path", required=True, help="Путь к обученной модели")
    p_fit.add_argument("--csv_path", required=True, help="Путь к CSV с данными")
    p_fit.add_argument("--save_path", default="outlier_detector.pkl")
    p_fit.add_argument("--method", default="mahalanobis", choices=["mahalanobis", "cosine", "l2"])
    p_fit.add_argument("--mode", default="per_class", choices=["per_class", "global"])
    p_fit.add_argument("--percentile", type=float, default=95.0)
    p_fit.add_argument("--max_seconds", type=float, default=3.0)
    p_fit.add_argument("--embedding_layer", default="projector", choices=["projector", "last_hidden"])
    p_fit.add_argument("--batch_size", type=int, default=16)
    p_fit.add_argument("--num_workers", type=int, default=4)
    p_fit.add_argument("--train_groups", type=str, default=None, help="Группы через запятую")
    p_fit.add_argument("--plot", action="store_true")

    # --- calibrate ---
    p_cal = subparsers.add_parser("calibrate", help="Калибровать порог")
    p_cal.add_argument("--detector_path", required=True)
    p_cal.add_argument("--model_path", required=True)
    p_cal.add_argument("--csv_path", required=True, help="Валидационные данные")
    p_cal.add_argument("--val_groups", type=str, default=None)
    p_cal.add_argument("--outlier_csv", type=str, default=None, help="CSV с outlier-ами")
    p_cal.add_argument("--percentile", type=float, default=95.0)

    # --- predict ---
    p_pred = subparsers.add_parser("predict", help="Инференс")
    p_pred.add_argument("--model_path", required=True)
    p_pred.add_argument("--detector_path", required=True)
    p_pred.add_argument("--audio_path", required=True, help="Файл или директория")
    p_pred.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if args.command == "fit":
        cmd_fit(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()