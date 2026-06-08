#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Единый скрипт для:
  - калибровки порогов по классам и энергетического порога шума
  - сохранения профилей (default / bridge / engine / ...)
  - логирования результатов (JSON, CSV, PNG-графики pos/neg)
  - запуска RealTimeRecognizer с выбранным профилем
  - офлайн-тестов с микрофона (подкоманда test)
"""

import os
import glob
import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample_poly

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from datetime import datetime

# ---------------------------------------------------------------------------
# Runtime defaults — sourced from core.config.settings (YAML-driven).
# Absolute Windows paths have been removed: all defaults are now relative to
# PROJECT_ROOT or resolved via settings.paths.* at import time.
# To override at the CLI level, pass the relevant argparse flags explicitly.
# ---------------------------------------------------------------------------
try:
    from core.config import get_settings as _get_settings, PROJECT_ROOT as _ROOT
    _cfg = _get_settings()
    BASE_MODEL_NAME     = _cfg.training.model_name
    TARGET_SR           = _cfg.audio.sample_rate
    WINDOW_S            = _cfg.audio.window_seconds
    STRIDE_S            = _cfg.audio.stride_seconds
    DEFAULT_MODEL_DIR   = str(_cfg.paths.best_model)
    DEFAULT_DATASET_DIR = str(_ROOT / "artifacts" / "data" / "calibration")
    del _cfg, _get_settings, _ROOT
except Exception:
    # Fallback literals — update configs/default.yaml instead of editing here.
    BASE_MODEL_NAME     = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    TARGET_SR           = 16000
    WINDOW_S            = 1.0
    STRIDE_S            = 0.5
    DEFAULT_MODEL_DIR   = "artifacts/models/best_model"
    DEFAULT_DATASET_DIR = "artifacts/data/calibration"

# Параметры порогов
SAFETY_MARGIN     = 0.01
MIN_CONF          = 0.60
MAX_CONF          = 0.99
OTHER_CLASS       = "другие слова"
DEFAULT_OTHER_TH  = 0.95

# Параметры энерго-порога
ENERGY_DURATION_S = 10.0
ENERGY_FACTOR     = 1.5
MIN_ENERGY_TH     = 5e-6
MAX_ENERGY_TH     = 1e-2

# Логи калибровки
LOG_DIR_NAME      = "calibration_logs"

# Базовый порог, если нет индивидуального
DEFAULT_BASE_CONF_TH = 0.6


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Ресемплинг к целевой частоте, если нужно."""
    if sr == target_sr:
        return audio.astype(np.float32)
    gcd = np.gcd(sr, target_sr)
    up = target_sr // gcd
    down = sr // gcd
    audio_rs = resample_poly(audio, up, down).astype(np.float32)
    return audio_rs


# ======================================================================
# ОБЕРТКА НАД МОДЕЛЬЮ (совместима с RealTimeRecognizer)
# ======================================================================

class ModelWrapper:
    """Обертка над Wav2Vec2ForSequenceClassification для удобного вызова."""

    def __init__(self, model_dir: str, base_model_name: str, device: Optional[str] = None):
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.sr = TARGET_SR

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MODEL] Загрузка модели из {model_dir} на {self.device}...")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.base_model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Лейблы из config модели (id2label)
        id2label = self.model.config.id2label
        keys = list(id2label.keys())
        # HuggingFace иногда хранит ключи как строки
        if isinstance(keys[0], str):
            self.labels = [id2label[str(i)] for i in range(len(id2label))]
        else:
            self.labels = [id2label[i] for i in range(len(id2label))]
        print("[MODEL] Лейблы:", self.labels)

    def predict_window_proba(self, audio: np.ndarray) -> np.ndarray:
        """
        audio: 1D np.array float32, mono, sr == self.sr
        Возвращает: probs shape = (num_labels,)
        """
        assert audio.ndim == 1

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(input_values=inputs["input_values"])
            logits = outputs.logits[0]

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs


# ======================================================================
# ОКОННАЯ ОБРАБОТКА ФАЙЛА
# ======================================================================

def get_max_probs_for_file(
        model: ModelWrapper,
        filepath: str,
        window_s: float,
        stride_s: float
) -> np.ndarray:
    """
    Разбивает аудио на окна [window_s] с шагом [stride_s] и
    возвращает максимум вероятности по каждому классу для файла.
    """
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # моно

    audio = resample_if_needed(audio, sr, model.sr)

    win_samples = int(window_s * model.sr)
    stride_samples = int(stride_s * model.sr)

    if len(audio) < win_samples:
        pad = win_samples - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")

    num_labels = len(model.labels)
    max_probs = np.zeros(num_labels, dtype=np.float32)
    any_window = False

    i = 0
    while i + win_samples <= len(audio):
        window = audio[i:i + win_samples]
        probs = model.predict_window_proba(window)
        if not any_window:
            max_probs[:] = probs
            any_window = True
        else:
            max_probs = np.maximum(max_probs, probs)
        i += stride_samples

    return max_probs


# ======================================================================
# СБОР ПОЗИТИВНЫХ/НЕГАТИВНЫХ СКOРОВ
# ======================================================================

def collect_pos_neg_scores(
        model: ModelWrapper,
        dataset_dir: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Для каждого лейбла:
      pos_scores[label] = [max_prob(label) на файлах этого класса]
      neg_scores[label] = [max_prob(label) на файлах других классов]
    """
    labels = model.labels
    pos = {l: [] for l in labels}
    neg = {l: [] for l in labels}

    for folder_label in labels:
        label_dir = os.path.join(dataset_dir, folder_label)
        if not os.path.isdir(label_dir):
            print(f"[WARN] Папка для лейбла '{folder_label}' не найдена: {label_dir}")
            continue

        files: List[str] = []
        for ext in ("*.wav", "*.flac", "*.mp3"):
            files.extend(glob.glob(os.path.join(label_dir, ext)))

        if not files:
            print(f"[WARN] Нет аудиофайлов в {label_dir}")
            continue

        print(f"[DATA] Лейбл '{folder_label}', файлов: {len(files)}")

        for fp in files:
            max_probs = get_max_probs_for_file(model, fp, WINDOW_S, STRIDE_S)

            for i, lab in enumerate(labels):
                score = float(max_probs[i])
                if lab == folder_label:
                    pos[lab].append(score)
                else:
                    neg[lab].append(score)

    pos = {k: np.array(v, dtype=np.float32) for k, v in pos.items()}
    neg = {k: np.array(v, dtype=np.float32) for k, v in neg.items()}
    return pos, neg, labels


# ======================================================================
# ПОДБОР ПОРОГОВ — ДВЕ СТРАТЕГИИ
# ======================================================================

def find_thresholds_precision_first(
        pos_scores: Dict[str, np.ndarray],
        neg_scores: Dict[str, np.ndarray],
        labels: List[str],
        existing_conf: Optional[Dict[str, float]] = None,
        target_precision: float = 0.995,
        grid_min: float = 0.5,
        grid_max: float = 0.99,
        grid_steps: int = 100,
) -> Dict[str, float]:
    """
    Стратегия 1: precision-first
    - Для каждого порога ищем первый th, при котором precision >= target_precision
    - Если не нашли — fallback: max_neg + SAFETY_MARGIN
    - OTHER_CLASS теперь тоже калибруется по данным
    """
    if existing_conf is None:
        existing_conf = {}

    thresholds: Dict[str, float] = {}

    print("\n[THRESH] Подбор порогов (precision-first):")

    for label in labels:
        pos = pos_scores.get(label, np.array([]))
        neg = neg_scores.get(label, np.array([]))

        if len(pos) == 0 or len(neg) == 0:
            # Нет данных — используем старое значение или разумный дефолт
            if label == OTHER_CLASS:
                fallback = existing_conf.get(label, DEFAULT_OTHER_TH)
            else:
                fallback = existing_conf.get(label, max(MIN_CONF, 0.8))
            thresholds[label] = float(fallback)
            print(f"  {label}: [WARN] нет данных, используем существующий/дефолт {fallback:.3f}")
            continue

        best_th = 0.95
        strategy = "strict_precision"

        for th in np.linspace(grid_min, grid_max, grid_steps):
            fp = np.sum(neg >= th)
            tp = np.sum(pos >= th)
            if (tp + fp) == 0:
                continue
            precision = tp / (tp + fp)
            if precision >= target_precision:
                best_th = th
                break
        else:
            # fallback: как в safety-first
            max_noise = np.percentile(neg, 99)
            best_th = min(max_noise + SAFETY_MARGIN, MAX_CONF)
            strategy = "noise_margin_fallback"

        best_th = max(MIN_CONF, round(float(best_th), 3))
        thresholds[label] = best_th

        recall = float((pos >= best_th).sum() / (len(pos) + 1e-9))
        fp_count = int((neg >= best_th).sum())
        max_neg = float(neg.max()) if len(neg) > 0 else 0.0

        print(f"  {label}: TH={best_th:.3f} ({strategy}) | Recall={recall*100:.1f}% | FP_test={fp_count} | MaxNeg={max_neg:.4f}")

    return thresholds


def find_thresholds_safety_first(
        pos_scores: Dict[str, np.ndarray],
        neg_scores: Dict[str, np.ndarray],
        labels: List[str],
        existing_conf: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Стратегия 2: safety-first
    - Берём max(neg) + SAFETY_MARGIN, ограничиваем [MIN_CONF, MAX_CONF].
    - Стремимся к минимизации ложных срабатываний.
    - OTHER_CLASS также калибруется по данным (если есть pos/neg).
    """
    if existing_conf is None:
        existing_conf = {}

    thresholds: Dict[str, float] = {}

    print("\n[THRESH] Подбор порогов (safety-first):")

    for label in labels:
        pos = pos_scores.get(label, np.array([]))
        neg = neg_scores.get(label, np.array([]))

        if len(pos) == 0 or len(neg) == 0:
            if label == OTHER_CLASS:
                fallback = existing_conf.get(label, DEFAULT_OTHER_TH)
            else:
                fallback = existing_conf.get(label, max(MIN_CONF, 0.8))
            thresholds[label] = float(fallback)
            print(f"  {label}: [WARN] нет данных, используем существующий/дефолт {fallback:.3f}")
            continue

        max_fp = float(np.max(neg)) if len(neg) > 0 else 0.0
        suggested_th = max(max_fp + SAFETY_MARGIN, MIN_CONF)
        suggested_th = min(suggested_th, MAX_CONF)
        recall = float((pos >= suggested_th).sum() / (len(pos) + 1e-9))

        th_final = float(round(suggested_th, 3))
        thresholds[label] = th_final

        print(f"  {label}:")
        print(f"    max FP score: {max_fp:.4f}")
        print(f"    threshold:    {th_final:.3f}")
        print(f"    recall:       {recall*100:.1f}% (pos >= th)")

    return thresholds


# ======================================================================
# КАЛИБРОВКА ЭНЕРГИИ
# ======================================================================

def calibrate_energy_threshold(
        duration_s: float = ENERGY_DURATION_S,
        sr: int = TARGET_SR,
        window_s: float = WINDOW_S,
        factor: float = ENERGY_FACTOR
) -> float:
    """
    energy_th = factor * median(mean(x^2) по окнам длины window_s).
    """
    print(f"\n[ENERGY] Калибровка шума {duration_s} сек.")
    print("         Сохраняйте обычный рабочий фон (без команд).")

    try:
        recording = sd.rec(int(duration_s * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
    except Exception as e:
        print(f"[ENERGY] Ошибка записи звука: {e}. Ставлю дефолт 1e-4.")
        return 1e-4

    audio = recording[:, 0]

    win_samples = int(window_s * sr)
    if len(audio) < win_samples:
        pad = win_samples - len(audio)
        audio = np.pad(audio, (0, pad))

    energies = []
    step = win_samples // 2 if win_samples > 1 else 1
    for i in range(0, len(audio) - win_samples + 1, step):
        frame = audio[i:i+win_samples]
        e = float(np.mean(frame**2))
        energies.append(e)

    if not energies:
        print("[ENERGY] Не удалось посчитать энергию, ставлю дефолт 1e-4")
        return 1e-4

    energies = np.array(energies)
    median_e = float(np.median(energies))
    th = median_e * factor
    th = max(min(th, MAX_ENERGY_TH), MIN_ENERGY_TH)

    print(f"  median energy: {median_e:.2e}")
    print(f"  energy_th:     {th:.2e}")
    return th


# ======================================================================
# РАБОТА С CONFIG.JSON И ПРОФИЛЯМИ
# ======================================================================

def load_base_config(model_dir: str) -> Tuple[dict, str]:
    """
    Загружает базовый config.json (HF config модели + наши поля).
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    return cfg, cfg_path


def save_profile_config(
        base_cfg: dict,
        base_cfg_path: str,
        conf_th_per_label: Dict[str, float],
        energy_th: float,
        profile_name: str
) -> str:
    """
    Если profile_name == "default":
        обновляем базовый config.json (HF config + наши поля).
    Иначе:
        создаём/перезаписываем config_<profile>.json с порогами профиля.
    """
    if profile_name == "default":
        cfg_to_save = base_cfg
        cfg_to_save["conf_th_per_label"] = conf_th_per_label
        cfg_to_save["energy_th"] = float(energy_th)
        out_path = base_cfg_path
    else:
        cfg_to_save = {
            "profile": profile_name,
            "conf_th_per_label": conf_th_per_label,
            "energy_th": float(energy_th),
            "window_s": WINDOW_S,
            "stride_s": STRIDE_S
        }
        out_path = os.path.join(os.path.dirname(base_cfg_path),
                                f"config_{profile_name}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg_to_save, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Профиль '{profile_name}' сохранён в {out_path}")
    return out_path


def load_profile_settings(
        model_dir: str,
        profile: str
) -> Tuple[List[str], Optional[Dict[str, float]], float]:
    """
    Загружает:
      - labels (из config.json, либо дефолтный список)
      - conf_th_per_label для профиля
      - energy_th для профиля
    """
    base_cfg_path = os.path.join(model_dir, "config.json")
    base_cfg = {}
    labels: Optional[List[str]] = None

    if os.path.exists(base_cfg_path):
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
        if "id2label" in base_cfg:
            id2label = base_cfg["id2label"]
            keys = list(id2label.keys())
            if isinstance(keys[0], str):
                labels = [id2label[str(i)] for i in range(len(id2label))]
            else:
                labels = [id2label[i] for i in range(len(id2label))]

    if labels is None:
        # запасной вариант, если id2label нет в config.json
        labels = ['другие слова', 'машина', 'приготовить машину', 'самый малый вперед']

    conf_th_per_label: Optional[Dict[str, float]] = None
    energy_th = 1e-4

    if profile == "default":
        if base_cfg:
            conf_th_per_label = base_cfg.get("conf_th_per_label", None)
            energy_th = base_cfg.get("energy_th", energy_th)
            print(f"[PROFILE] Используется default из {base_cfg_path}")
        else:
            print(f"[WARN] base config.json не найден: {base_cfg_path}")
    else:
        profile_cfg_path = os.path.join(model_dir, f"config_{profile}.json")
        if os.path.exists(profile_cfg_path):
            with open(profile_cfg_path, "r", encoding="utf-8") as f:
                p_cfg = json.load(f)
            conf_th_per_label = p_cfg.get("conf_th_per_label", None)
            energy_th = p_cfg.get("energy_th", energy_th)
            print(f"[PROFILE] Загружен профиль {profile}: {profile_cfg_path}")
        else:
            print(f"[WARN] Профильный конфиг не найден: {profile_cfg_path}, используем дефолтные пороги.")

    return labels, conf_th_per_label, energy_th


# ======================================================================
# ЛОГИРОВАНИЕ РЕЗУЛЬТАТОВ КАЛИБРОВКИ + ГРАФИКИ
# ======================================================================

def save_calibration_report(
        model_dir: str,
        profile_name: str,
        labels: List[str],
        pos_scores: Dict[str, np.ndarray],
        neg_scores: Dict[str, np.ndarray],
        thresholds: Dict[str, float],
        energy_th: float,
        strategy: str
) -> Tuple[str, str, str, str]:
    """
    Сохраняет JSON-отчёт и CSV с сырыми скорами.
    Возвращает:
      profile_dir, base_name, json_path, csv_path
    """
    log_root = os.path.join(model_dir, LOG_DIR_NAME)
    os.makedirs(log_root, exist_ok=True)

    profile_dir = os.path.join(log_root, profile_name)
    os.makedirs(profile_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"calib_{profile_name}_{ts}"

    json_path = os.path.join(profile_dir, base_name + ".json")
    csv_path  = os.path.join(profile_dir, base_name + "_scores.csv")

    report = {
        "timestamp": ts,
        "profile": profile_name,
        "model_dir": model_dir,
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "energy_th": float(energy_th),
        "threshold_strategy": strategy,
        "classes": {}
    }

    bins = np.linspace(0.0, 1.0, 21)  # 20 бинов

    for label in labels:
        pos = pos_scores.get(label, np.array([]))
        neg = neg_scores.get(label, np.array([]))

        def stats(arr: np.ndarray) -> dict:
            if len(arr) == 0:
                return {
                    "count": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            return {
                "count": int(len(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }

        hist_pos, _ = np.histogram(pos, bins=bins) if len(pos) > 0 else (np.zeros(len(bins)-1, dtype=int), bins)
        hist_neg, _ = np.histogram(neg, bins=bins) if len(neg) > 0 else (np.zeros(len(bins)-1, dtype=int), bins)

        report["classes"][label] = {
            "threshold": float(thresholds.get(label, 0.0)),
            "pos_stats": stats(pos),
            "neg_stats": stats(neg),
            "hist_bins": bins.tolist(),
            "hist_pos_counts": hist_pos.astype(int).tolist(),
            "hist_neg_counts": hist_neg.astype(int).tolist(),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["label", "kind", "score"])
        for label in labels:
            for v in pos_scores.get(label, []):
                writer.writerow([label, "pos", f"{float(v):.6f}"])
            for v in neg_scores.get(label, []):
                writer.writerow([label, "neg", f"{float(v):.6f}"])

    print(f"[LOG] Отчёт сохранён: {json_path}")
    print(f"[LOG] Сырые скоры сохранены: {csv_path}")

    return profile_dir, base_name, json_path, csv_path


def save_calibration_plots(
        profile_dir: str,
        base_name: str,
        labels: List[str],
        pos_scores: Dict[str, np.ndarray],
        neg_scores: Dict[str, np.ndarray],
        thresholds: Dict[str, float]
) -> None:
    """
    Генерирует PNG-графики распределений pos/neg для каждого класса.
    Сохраняет в ту же папку, что и отчёты (profile_dir).
    """
    try:
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            use_sns = True
        except ImportError:
            sns = None
            use_sns = False
    except ImportError:
        print("[PLOT] matplotlib не установлена, графики не будут сгенерированы.")
        return

    for label in labels:
        pos = pos_scores.get(label, np.array([]))
        neg = neg_scores.get(label, np.array([]))

        if len(pos) == 0 and len(neg) == 0:
            continue

        plt.figure(figsize=(6, 4))

        if use_sns:
            if len(pos) > 0:
                sns.histplot(pos, bins=20, color="green", alpha=0.5, stat="count", label="pos")
            if len(neg) > 0:
                sns.histplot(neg, bins=20, color="red", alpha=0.5, stat="count", label="neg")
        else:
            if len(pos) > 0:
                plt.hist(pos, bins=20, color="green", alpha=0.5, label="pos")
            if len(neg) > 0:
                plt.hist(neg, bins=20, color="red", alpha=0.5, label="neg")

        th = thresholds.get(label, None)
        if th is not None:
            plt.axvline(th, color="blue", linestyle="--", label=f"th={th:.3f}")

        plt.xlim(0.0, 1.0)
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.title(label)
        plt.legend()

        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        fig_path = os.path.join(profile_dir, f"{base_name}_{safe_label}.png")

        plt.tight_layout()
        plt.savefig(fig_path, dpi=120)
        plt.close()

    print(f"[PLOT] Гистограммы pos/neg сохранены в {profile_dir}")


# ======================================================================
# ЗАПУСК REALTIMERECOGNIZER
# ======================================================================

def run_realtime(
        model_dir: str,
        profile: str = "default",
        base_conf_th: float = DEFAULT_BASE_CONF_TH,
        debounce_s: float = 1.0
):
    """
    Запуск RealTimeRecognizer c указанным профилем (онлайн-режим).
    Профиль:
      - "default": берём пороги из config.json
      - любой другой: config_<profile>.json
    """
    from core.realtime_recognizer import RealTimeRecognizer

    labels, conf_th_per_label, energy_th = load_profile_settings(model_dir, profile)

    rec = RealTimeRecognizer(
        model_dir=model_dir,
        labels=labels,
        sr=TARGET_SR,
        window_s=WINDOW_S,
        stride_s=STRIDE_S,
        energy_th=energy_th,
        conf_th=base_conf_th,
        conf_th_per_label=conf_th_per_label,
        debounce_s=debounce_s
    )

    print("\n[RUN] Запуск стрима...")
    print(f"      Профиль: {profile}")
    print(f"      Energy Threshold: {energy_th:.2e}")
    print(f"      Base conf_th: {base_conf_th:.2f}")

    def on_detect(d):
        print(f"[{profile}] {d['label']} ({d['prob']:.2f})")

    rec.start_stream(callback_on_detection=on_detect)


# ======================================================================
# ОФЛАЙН-ТЕСТ С МИКРОФОНА (ИНТЕРАКТИВНЫЙ)
# ======================================================================

def offline_mic_test(
        model_dir: str,
        profile: str = "default",
        duration_s: float = 15.0,
        base_conf_th: float = DEFAULT_BASE_CONF_TH,
        debounce_s: float = 1.0,
        show_other: bool = False
):
    """
    Офлайн-тест: записывает звук с микрофона и прогоняет через модель
    с использованием порогов и energy_th текущего профиля.
    Выводит детекции с отметкой времени.
    """
    print(f"\n[TEST] Профиль: {profile}")
    print(f"[TEST] Длительность записи: {duration_s} сек")

    labels, conf_th_per_label, energy_th = load_profile_settings(model_dir, profile)
    model = ModelWrapper(model_dir, BASE_MODEL_NAME)

    try:
        print("\n[TEST] Подготовьтесь, через мгновение начнётся запись.")
        sd.sleep(1000)
        print("[TEST] Говорите команды...")
        recording = sd.rec(int(duration_s * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype='float32')
        sd.wait()
        print("[TEST] Запись окончена, анализирую...")
    except Exception as e:
        print(f"[TEST] Ошибка записи звука: {e}")
        return []

    audio = recording[:, 0]
    sr = TARGET_SR

    win_samples = int(WINDOW_S * sr)
    stride_samples = int(STRIDE_S * sr)

    if len(audio) < win_samples:
        pad = win_samples - len(audio)
        audio = np.pad(audio, (0, pad))

    def get_threshold(label: str) -> float:
        if conf_th_per_label and label in conf_th_per_label:
            return conf_th_per_label[label]
        return base_conf_th

    last_label: Optional[str] = None
    last_time: float = -1e9
    detections = []

    for i in range(0, len(audio) - win_samples + 1, stride_samples):
        frame = audio[i:i + win_samples]
        energy = float(np.mean(frame ** 2))
        if energy < energy_th:
            continue

        probs = model.predict_window_proba(frame)
        best_idx = int(np.argmax(probs))
        best_label = model.labels[best_idx]
        best_prob = float(probs[best_idx])

        if best_label == OTHER_CLASS and not show_other:
            # OTHER_CLASS — обычно неинтересен для пользователя
            continue

        th = get_threshold(best_label)
        if best_prob < th:
            continue

        t = i / sr
        if last_label == best_label and (t - last_time) < debounce_s:
            # простая реализация debounce
            continue

        last_label = best_label
        last_time = t

        detections.append((t, best_label, best_prob))
        print(f"  t={t:5.2f}s: {best_label} ({best_prob:.2f})")

    if not detections:
        print("  За время теста команд не обнаружено (или все ниже порогов).")
    else:
        print(f"\n[TEST] Всего детекций: {len(detections)}")

    return detections


# ======================================================================
# MAIN: CLI С ТРЕМЯ ПОДКОМАНДАМИ
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Калибровка и запуск голосового ассистента (Wav2Vec2 + RealTimeRecognizer)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- calibrate ---
    calib_p = subparsers.add_parser(
        "calibrate", help="Калибровка порогов по датасету и энерго-порога"
    )
    calib_p.add_argument("--profile", default="default", help="Имя профиля (default, bridge, engine, ...)")
    calib_p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Директория модели")
    calib_p.add_argument("--data_dir", default=DEFAULT_DATASET_DIR, help="Директория калибровочного датасета")
    calib_p.add_argument(
        "--strategy",
        choices=["precision", "safety"],
        default="safety",
        help="Стратегия подбора порогов: precision (precision-first) или safety (safety-first)"
    )
    calib_p.add_argument(
        "--skip_energy",
        action="store_true",
        help="Не калибровать энергию, оставить как есть/дефолтную"
    )

    # --- run (онлайн) ---
    run_p = subparsers.add_parser(
        "run", help="Запуск RealTimeRecognizer с выбранным профилем (онлайн-режим)"
    )
    run_p.add_argument("--profile", default="default", help="Имя профиля (default, bridge, engine, ...)")
    run_p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Директория модели")
    run_p.add_argument("--base_conf_th", type=float, default=DEFAULT_BASE_CONF_TH, help="Базовый порог (если нет per-label)")
    run_p.add_argument("--debounce_s", type=float, default=1.0, help="Debounce (сек) между детекциями")

    # --- test (офлайн) ---
    test_p = subparsers.add_parser(
        "test", help="Быстрый офлайн-тест с микрофона для проверки профиля"
    )
    test_p.add_argument("--profile", default="default", help="Имя профиля (default, bridge, engine, ...)")
    test_p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Директория модели")
    test_p.add_argument("--duration", type=float, default=15.0, help="Длительность записи (сек)")
    test_p.add_argument("--base_conf_th", type=float, default=DEFAULT_BASE_CONF_TH, help="Базовый порог (если нет per-label)")
    test_p.add_argument("--debounce_s", type=float, default=1.0, help="Debounce (сек) между детекциями")
    test_p.add_argument(
        "--show_other",
        action="store_true",
        help=f"Показывать детекции класса '{OTHER_CLASS}' (по умолчанию скрыт)"
    )

    args = parser.parse_args()

    if args.command == "calibrate":
        profile_name = args.profile
        model_dir = args.model_dir
        data_dir = args.data_dir

        print(f"[INFO] Калибровка профиля: {profile_name}")
        print(f"[INFO] MODEL_DIR: {model_dir}")
        print(f"[INFO] DATASET_DIR: {data_dir}")
        print(f"[INFO] Стратегия порогов: {args.strategy}")

        # 1) Модель
        model = ModelWrapper(model_dir, BASE_MODEL_NAME)

        # 2) Базовый config.json (для id2label и старых порогов)
        base_cfg, base_cfg_path = load_base_config(model_dir)
        existing_conf = base_cfg.get("conf_th_per_label", {})

        # 3) Сбор статистики
        pos_scores, neg_scores, labels = collect_pos_neg_scores(model, data_dir)

        # 4) Пороги
        if args.strategy == "precision":
            conf_th_per_label = find_thresholds_precision_first(
                pos_scores, neg_scores, labels, existing_conf
            )
        else:
            conf_th_per_label = find_thresholds_safety_first(
                pos_scores, neg_scores, labels, existing_conf
            )

        # 5) Энергия
        if args.skip_energy:
            energy_th = base_cfg.get("energy_th", 1e-4)
            print(f"[ENERGY] Пропускаем калибровку энергии. Текущее/дефолтное значение: {energy_th:.2e}")
        else:
            energy_th = calibrate_energy_threshold()

        print("\n[RESULT] Итоговые пороги:")
        for k, v in conf_th_per_label.items():
            print(f"  {k}: {v:.3f}")
        print(f"  energy_th: {energy_th:.2e}")

        # 6) Сохранение профиля
        out_cfg_path = save_profile_config(
            base_cfg, base_cfg_path, conf_th_per_label, energy_th, profile_name
        )

        # 7) Логирование (JSON + CSV)
        profile_dir, base_name, json_path, csv_path = save_calibration_report(
            model_dir, profile_name, labels,
            pos_scores, neg_scores,
            conf_th_per_label, energy_th,
            strategy=args.strategy
        )

        # 8) Графики pos/neg
        save_calibration_plots(
            profile_dir, base_name, labels,
            pos_scores, neg_scores,
            conf_th_per_label
        )

        print("\nГотово.")
        print(f"Профиль '{profile_name}' записан в: {out_cfg_path}")
        print("Для быстрого теста можно запустить:")
        print(f"  python main_calib_live_run.py test --profile {profile_name}")

    elif args.command == "run":
        run_realtime(
            model_dir=args.model_dir,
            profile=args.profile,
            base_conf_th=args.base_conf_th,
            debounce_s=args.debounce_s
        )

    elif args.command == "test":
        offline_mic_test(
            model_dir=args.model_dir,
            profile=args.profile,
            duration_s=args.duration,
            base_conf_th=args.base_conf_th,
            debounce_s=args.debounce_s,
            show_other=args.show_other
        )


if __name__ == "__main__":
    main()

'''# 1) Калибровка
python main_calib_live_run.py calibrate --profile home --model_dir ... --data_dir ... --strategy safety/precision --skip_energy ...

# 2) Быстрый офлайн-тест с микрофона
python main_calib_live_run.py test --profile home --model_dir ... --duration 15 --base_conf_th ... --debounce_s ...

# 3) Онлайн-режим (RealTimeRecognizer)
python main_calib_live_run.py run --profile home --model_dir ... --base_conf_th ... --debounce_s ... --show_other ...'''