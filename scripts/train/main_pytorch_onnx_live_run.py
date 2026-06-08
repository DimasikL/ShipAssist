"""
Единый скрипт для:
  - калибровки порогов по классам и энергетического порога шума
  - сохранения профилей (default / home / out / ...)
  - логирования результатов (JSON, CSV, графики pos/neg)
  - запуска RealTimeRecognizer с выбранным профилем (PyTorch или ONNX)
  - офлайн-тестов с микрофона (подкоманда test)
  - экспорта модели в ONNX + INT8 квантизация (подкоманда export_onnx)
  - опциональная загрузка outlier_detector.pkl

Примеры запуска:

  # Калибровка:
  python main_calib_live_outdet_run.py calibrate \\
      --profile home --model_dir best_model --data_dir clf_dset/calibration/home

  # Онлайн-режим (PyTorch, как раньше):
  python main_calib_live_outdet_run.py run \\
      --profile home --model_dir best_model --base_conf_th 0.6 \\
      --debounce_s 1.0 --outlier_detector outlier_detector.pkl

  # Онлайн-режим (ONNX, быстрее в 2-4×):
  python main_calib_live_outdet_run.py run \\
      --profile home --model_dir best_model --onnx_dir onnx_model \\
      --base_conf_th 0.6 --debounce_s 1.0 --outlier_detector outlier_detector.pkl

  # Экспорт в ONNX (один раз):
  python main_calib_live_outdet_run.py export_onnx \\
      --model_dir best_model --output_dir onnx_model --quantize --benchmark

  # Офлайн-тест:
  python main_calib_live_outdet_run.py test \\
      --profile home --model_dir best_model --duration 15
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
from scripts.utils.outlier_detection import OutlierDetector, EmbeddingExtractor

# ═══════════════════════════════════════════════════════════
#  Константы и дефолты — sourced from core.config.settings
# ═══════════════════════════════════════════════════════════
# All model-name, sample-rate, and window parameters are loaded from the
# centralised YAML config (configs/base.yaml + model.yaml + inference.yaml).
# Hard-coded literals here serve only as last-resort fallbacks when the
# config stack cannot be initialised (isolated test / missing YAML).
# ---------------------------------------------------------------------------
try:
    from core.config import get_settings as _get_settings, PROJECT_ROOT as _ROOT
    _cfg = _get_settings()
    BASE_MODEL_NAME = _cfg.training.model_name
    TARGET_SR       = _cfg.audio.sample_rate
    WINDOW_S        = _cfg.audio.window_seconds
    STRIDE_S        = _cfg.audio.stride_seconds
    # Prefer the config-resolved best_model path; fall back to a relative hint.
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

SAFETY_MARGIN = 0.01
MIN_CONF = 0.60
MAX_CONF = 0.99
OTHER_CLASS = "другие слова"
DEFAULT_OTHER_TH = 0.95

ENERGY_DURATION_S = 10.0
ENERGY_FACTOR = 1.5
MIN_ENERGY_TH = 5e-6
MAX_ENERGY_TH = 1e-2

LOG_DIR_NAME = "calibration_logs"

DEFAULT_BASE_CONF_TH = 0.6


# ═══════════════════════════════════════════════════════════
#  Утилиты
# ═══════════════════════════════════════════════════════════

def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)
    gcd = np.gcd(sr, target_sr)
    up = target_sr // gcd
    down = sr // gcd
    audio_rs = resample_poly(audio, up, down).astype(np.float32)
    return audio_rs


# ═══════════════════════════════════════════════════════════
#  ModelWrapper (для калибровки и офлайн-теста)
# ═══════════════════════════════════════════════════════════

class ModelWrapper:
    def __init__(
            self,
            model_dir: str,
            base_model_name: str,
            device: Optional[str] = None,
            detector_path: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.sr = TARGET_SR

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MODEL] Загрузка модели из {model_dir} на {self.device}...")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.base_model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        keys = list(id2label.keys())
        if isinstance(keys[0], str):
            self.labels = [id2label[str(i)] for i in range(len(id2label))]
        else:
            self.labels = [id2label[i] for i in range(len(id2label))]
        print("[MODEL] Лейблы:", self.labels)

        self.detector: Optional[OutlierDetector] = None
        self.embedding_extractor: Optional[EmbeddingExtractor] = None

        if detector_path is not None and os.path.exists(detector_path):
            print(f"[MODEL][OUTLIER] Загрузка детектора выбросов из {detector_path}")
            self.detector = OutlierDetector.load(detector_path)
            self.embedding_extractor = EmbeddingExtractor(
                model=self.model,
                feature_extractor=self.feature_extractor,
                device=torch.device(self.device),
                config=self.detector.config,
            )
        else:
            if detector_path is not None:
                print(f"[MODEL][OUTLIER] Файл детектора не найден: {detector_path} — работаем без outlier detection.")
            else:
                print("[MODEL][OUTLIER] Детектор выбросов не указан — работаем без outlier detection.")

    def predict_window_proba(self, audio: np.ndarray) -> np.ndarray:
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

    def predict_window_proba_with_outlier(self, audio: np.ndarray):
        probs = self.predict_window_proba(audio)
        if self.detector is None or self.embedding_extractor is None:
            return probs, None, None
        emb = self.embedding_extractor._captured_embedding
        emb = self.embedding_extractor._pool_embedding(emb, attention_mask=None)
        emb_np = emb.squeeze(0).cpu().numpy()

        out_info = self.detector.score_with_details(emb_np)
        is_outlier = out_info["is_outlier"]

        return probs, is_outlier, out_info


# ═══════════════════════════════════════════════════════════
#  Калибровка: сбор скоров, подбор порогов
# ═══════════════════════════════════════════════════════════

def get_max_probs_for_file(
        model: ModelWrapper,
        filepath: str,
        window_s: float,
        stride_s: float
) -> np.ndarray:
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

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


def collect_pos_neg_scores(
        model: ModelWrapper,
        dataset_dir: str
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    List[str],
    Dict[str, List[dict]],
    Dict[str, List[dict]]
]:
    labels = model.labels
    pos: Dict[str, List[float]] = {l: [] for l in labels}
    neg: Dict[str, List[float]] = {l: [] for l in labels}
    pos_details: Dict[str, List[dict]] = {l: [] for l in labels}
    neg_details: Dict[str, List[dict]] = {l: [] for l in labels}

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
            rel_fp = os.path.relpath(fp, dataset_dir)

            for i, lab in enumerate(labels):
                score = float(max_probs[i])
                rec = {
                    "file": rel_fp,
                    "true_label": folder_label,
                    "score": score
                }
                if lab == folder_label:
                    pos[lab].append(score)
                    pos_details[lab].append(rec)
                else:
                    neg[lab].append(score)
                    neg_details[lab].append(rec)

    pos_scores = {k: np.array(v, dtype=np.float32) for k, v in pos.items()}
    neg_scores = {k: np.array(v, dtype=np.float32) for k, v in neg.items()}

    return pos_scores, neg_scores, labels, pos_details, neg_details


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
    if existing_conf is None:
        existing_conf = {}

    thresholds: Dict[str, float] = {}
    print("\n[THRESH] Подбор порогов (precision-first):")

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
            max_noise = np.percentile(neg, 99)
            best_th = min(max_noise + SAFETY_MARGIN, MAX_CONF)
            strategy = "noise_margin_fallback"

        best_th = float(max(MIN_CONF, min(best_th, MAX_CONF)))
        best_th = float(round(best_th, 3))

        tp = int((pos >= best_th).sum())
        fn = int((pos < best_th).sum())
        fp = int((neg >= best_th).sum())
        recall = tp / max(1, tp + fn)
        max_neg = float(neg.max()) if len(neg) > 0 else 0.0

        if recall == 0.0:
            q = 0.7
            q_th = float(np.quantile(pos, q))
            q_th = float(max(MIN_CONF, min(q_th, MAX_CONF)))
            q_th = float(round(q_th, 3))

            tp_q = int((pos >= q_th).sum())
            fn_q = int((pos < q_th).sum())
            fp_q = int((neg >= q_th).sum())
            recall_q = tp_q / max(1, tp_q + fn_q)

            if recall_q > 0.0:
                best_th = q_th
                tp, fn, fp, recall = tp_q, fn_q, fp_q, recall_q
                strategy = strategy + "+recall_fallback"

        thresholds[label] = best_th

        print(f"  {label}: TH={best_th:.3f} ({strategy}) | "
              f"Recall={recall * 100:.1f}% | FP_test={fp} | MaxNeg={max_neg:.4f}")

    return thresholds


def find_thresholds_safety_first(
        pos_scores: Dict[str, np.ndarray],
        neg_scores: Dict[str, np.ndarray],
        labels: List[str],
        existing_conf: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
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
        suggested_th = float(round(suggested_th, 3))

        tp = int((pos >= suggested_th).sum())
        fn = int((pos < suggested_th).sum())
        recall = tp / max(1, tp + fn)

        th_final = suggested_th
        strategy = "max_fp_plus_margin"

        if recall == 0.0:
            q = 0.7
            q_th = float(np.quantile(pos, q))
            q_th = float(max(MIN_CONF, min(q_th, MAX_CONF)))
            q_th = float(round(q_th, 3))

            tp_q = int((pos >= q_th).sum())
            fn_q = int((pos < q_th).sum())
            recall_q = tp_q / max(1, tp_q + fn_q)

            if recall_q > 0.0:
                th_final = q_th
                recall = recall_q
                strategy = strategy + "+recall_fallback"

        thresholds[label] = th_final

        print(f"  {label}:")
        print(f"    max FP score: {max_fp:.4f}")
        print(f"    threshold:    {th_final:.3f}")
        print(f"    recall:       {recall * 100:.1f}% (pos >= th) [{strategy}]")

    return thresholds


# ═══════════════════════════════════════════════════════════
#  Калибровка энергии
# ═══════════════════════════════════════════════════════════

def calibrate_energy_threshold(
        duration_s: float = ENERGY_DURATION_S,
        sr: int = TARGET_SR,
        window_s: float = WINDOW_S,
        factor: float = ENERGY_FACTOR
) -> float:
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
        frame = audio[i:i + win_samples]
        e = float(np.mean(frame ** 2))
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


# ═══════════════════════════════════════════════════════════
#  Конфиги и профили
# ═══════════════════════════════════════════════════════════

def load_base_config(model_dir: str) -> Tuple[dict, str]:
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


# ═══════════════════════════════════════════════════════════
#  Логирование и графики
# ═══════════════════════════════════════════════════════════

def save_calibration_report(
        model_dir: str,
        profile_name: str,
        labels: List[str],
        pos_scores: Dict[str, np.ndarray],
        neg_scores: Dict[str, np.ndarray],
        thresholds: Dict[str, float],
        energy_th: float,
        strategy: str,
        pos_details: Optional[Dict[str, List[dict]]] = None,
        neg_details: Optional[Dict[str, List[dict]]] = None,
) -> Tuple[str, str, str, str]:
    log_root = os.path.join(model_dir, LOG_DIR_NAME)
    os.makedirs(log_root, exist_ok=True)

    profile_dir = os.path.join(log_root, profile_name)
    os.makedirs(profile_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"calib_{profile_name}_{ts}"

    json_path = os.path.join(profile_dir, base_name + ".json")
    csv_path = os.path.join(profile_dir, base_name + "_scores.csv")

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

    bins = np.linspace(0.0, 1.0, 21)

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

        hist_pos, _ = np.histogram(pos, bins=bins) if len(pos) > 0 else (np.zeros(len(bins) - 1, dtype=int), bins)
        hist_neg, _ = np.histogram(neg, bins=bins) if len(neg) > 0 else (np.zeros(len(bins) - 1, dtype=int), bins)

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

        if pos_details is not None and neg_details is not None:
            writer.writerow(["label", "kind", "score", "file", "true_label"])
            for label in labels:
                for d in pos_details.get(label, []):
                    writer.writerow([
                        label,
                        "pos",
                        f"{float(d['score']):.6f}",
                        d["file"],
                        d["true_label"],
                    ])
                for d in neg_details.get(label, []):
                    writer.writerow([
                        label,
                        "neg",
                        f"{float(d['score']):.6f}",
                        d["file"],
                        d["true_label"],
                    ])
        else:
            writer.writerow(["label", "kind", "score"])
            for label in labels:
                for v in pos_scores.get(label, []):
                    writer.writerow([label, "pos", f"{float(v):.6f}"])
                for v in neg_scores.get(label, []):
                    writer.writerow([label, "neg", f"{float(v):.6f}"])

    print(f"[LOG] Отчёт сохранён: {json_path}")
    print(f"[LOG] Сырые скоры сохранены: {csv_path}")

    if pos_details is not None and neg_details is not None:
        errors_path = os.path.join(profile_dir, base_name + "_errors.csv")
        with open(errors_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["err_type", "label", "score", "threshold",
                             "delta", "file", "true_label"])

            for label in labels:
                th = float(thresholds.get(label, 0.0))

                for d in pos_details.get(label, []):
                    s = float(d["score"])
                    if s < th:
                        writer.writerow([
                            "FN",
                            label,
                            f"{s:.6f}",
                            f"{th:.6f}",
                            f"{s - th:.6f}",
                            d["file"],
                            d["true_label"],
                        ])

                for d in neg_details.get(label, []):
                    s = float(d["score"])
                    if s >= th:
                        writer.writerow([
                            "FP",
                            label,
                            f"{s:.6f}",
                            f"{th:.6f}",
                            f"{s - th:.6f}",
                            d["file"],
                            d["true_label"],
                        ])

        print(f"[LOG] Ошибки (FP/FN) сохранены: {errors_path}")

    return profile_dir, base_name, json_path, csv_path


def save_calibration_plots(
        profile_dir: str,
        base_name: str,
        labels: List[str],
        pos_scores: Dict[str, np.ndarray],
        neg_scores: Dict[str, np.ndarray],
        thresholds: Dict[str, float]
) -> None:
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


# ═══════════════════════════════════════════════════════════
#  run — онлайн-стриминг (PyTorch или ONNX)
# ═══════════════════════════════════════════════════════════

def run_realtime(
        model_dir: str,
        profile: str = "default",
        base_conf_th: float = DEFAULT_BASE_CONF_TH,
        debounce_s: float = 1.0,
        outlier_detector: Optional[str] = None,
        onnx_dir: Optional[str] = None,
):
    """
    Запуск RealTimeRecognizer.

    Бэкенд выбирается автоматически:
      - onnx_dir=None  → PyTorch (как раньше)
      - onnx_dir="..."  → ONNX Runtime (быстрее в 2-4× на CPU)
      - Если ONNX не удалось загрузить → fallback на PyTorch
    """
    from core.realtime_recognizer import RealTimeRecognizer

    labels, conf_th_per_label, energy_th = load_profile_settings(
        model_dir, profile
    )

    rec = RealTimeRecognizer(
        model_dir=model_dir,
        labels=labels,
        sr=TARGET_SR,
        window_s=WINDOW_S,
        stride_s=STRIDE_S,
        energy_th=energy_th,
        conf_th=base_conf_th,
        conf_th_per_label=conf_th_per_label,
        debounce_s=debounce_s,
        outlier_detector=outlier_detector,
        report_other=False,
        warmup_s=1.5,
        sd_blocksize=8000,
        # ONNX: передаём если указан, иначе None → PyTorch
        onnx_dir=onnx_dir,
        onnx_use_int8=True,
    )

    backend = "ONNX" if rec._use_onnx else "PyTorch"

    print(f"\n[RUN] ══════════════════════════════════════")
    print(f"[RUN] Backend:          {backend}")
    print(f"[RUN] Profile:          {profile}")
    print(f"[RUN] Energy Threshold: {energy_th:.2e}")
    print(f"[RUN] Base conf_th:     {base_conf_th:.2f}")
    print(f"[RUN] Debounce:         {debounce_s}s")
    if conf_th_per_label:
        for lbl, th in conf_th_per_label.items():
            print(f"[RUN]   {lbl}: {th:.3f}")
    print(f"[RUN] ══════════════════════════════════════")

    def on_detect(d):
        label = d["label"]
        prob = d["prob"]
        ms = d.get("inference_ms", 0)
        bk = d.get("backend", backend)
        print(f"  >>> [{profile}] {label} ({prob:.3f}) "
              f"[{ms:.0f}ms {bk}]")

    rec.start_stream(callback_on_detection=on_detect)


# ═══════════════════════════════════════════════════════════
#  export_onnx — экспорт модели в ONNX + INT8
# ═══════════════════════════════════════════════════════════

def cmd_export_onnx(args):
    """
    Вызывает export_to_onnx.py как подпроцесс.

    Для корректной работы export_to_onnx.py должен лежать
    в корне проекта рядом с этим скриптом.
    """
    import subprocess

    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_script = os.path.join(script_dir, "main_export_to_onnx.py")

    if not os.path.exists(export_script):
        print(f"[ERROR] Скрипт экспорта не найден: {export_script}")
        print(f"        Положите export_to_onnx.py рядом с этим файлом.")
        return

    cmd = [
        "python", export_script,
        "--model_dir", args.model_dir,
        "--output_dir", args.output_dir,
    ]
    if args.quantize:
        cmd.append("--quantize")
    if args.benchmark:
        cmd.append("--benchmark")

    print(f"[EXPORT] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ═══════════════════════════════════════════════════════════
#  test — офлайн-тест с микрофона
# ═══════════════════════════════════════════════════════════

def offline_mic_test(
        model_dir: str,
        profile: str = "default",
        duration_s: float = 15.0,
        base_conf_th: float = DEFAULT_BASE_CONF_TH,
        debounce_s: float = 1.0,
        show_other: bool = False,
        outlier_detector: Optional[str] = None,
):
    print(f"\n[TEST] Профиль: {profile}")
    print(f"[TEST] Длительность записи: {duration_s} сек")

    labels, conf_th_per_label, energy_th = load_profile_settings(model_dir, profile)
    model = ModelWrapper(
        model_dir,
        BASE_MODEL_NAME,
        detector_path=outlier_detector,
    )

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

        probs, is_outlier, out_info = model.predict_window_proba_with_outlier(frame)

        if is_outlier:
            continue

        best_idx = int(np.argmax(probs))
        best_label = model.labels[best_idx]
        best_prob = float(probs[best_idx])

        if best_label == OTHER_CLASS and not show_other:
            continue

        th = get_threshold(best_label)
        if best_prob < th:
            continue

        t = i / sr
        if last_label == best_label and (t - last_time) < debounce_s:
            continue

        last_label = best_label
        last_time = t

        detections.append((t, best_label, best_prob))
        print(f"  t={t:5.2f}s: {best_label} ({best_prob:.2f})")

    if not detections:
        print("  За время теста команд не обнаружено (или все ниже порогов / отфильтрованы как outlier).")
    else:
        print(f"\n[TEST] Всего детекций: {len(detections)}")

    return detections


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Калибровка и запуск голосового ассистента "
                    "(Wav2Vec2 + RealTimeRecognizer + OutlierDetection + ONNX Runtime)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── calibrate ──
    calib_p = subparsers.add_parser(
        "calibrate", help="Калибровка порогов по датасету и энерго-порога"
    )
    calib_p.add_argument("--profile", default="default",
                         help="Имя профиля (default, bridge, engine, ...)")
    calib_p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                         help="Директория модели")
    calib_p.add_argument("--data_dir", default=DEFAULT_DATASET_DIR,
                         help="Директория калибровочного датасета")
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

    # ── run (онлайн) ──
    run_p = subparsers.add_parser(
        "run", help="Запуск RealTimeRecognizer с выбранным профилем (онлайн-режим)"
    )
    run_p.add_argument("--profile", default="default",
                       help="Имя профиля (default, bridge, engine, ...)")
    run_p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                       help="Директория модели")
    run_p.add_argument("--base_conf_th", type=float, default=DEFAULT_BASE_CONF_TH,
                       help="Базовый порог (если нет per-label)")
    run_p.add_argument("--debounce_s", type=float, default=1.0,
                       help="Debounce (сек) между детекциями")
    run_p.add_argument(
        "--outlier_detector",
        type=str,
        default=None,
        help="Путь к outlier_detector.pkl"
    )
    run_p.add_argument(
        "--onnx_dir",
        type=str,
        default=None,
        help="Путь к onnx_model/ для ускорения (None = PyTorch)"
    )

    # ── export_onnx ──
    export_p = subparsers.add_parser(
        "export_onnx", help="Экспорт модели в ONNX + INT8 квантизация"
    )
    export_p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                          help="Директория модели")
    export_p.add_argument("--output_dir", default="onnx_model",
                          help="Выходная папка для ONNX файлов")
    export_p.add_argument("--quantize", action="store_true",
                          help="Квантизация в INT8")
    export_p.add_argument("--benchmark", action="store_true",
                          help="Запустить бенчмарк PyTorch vs ONNX")

    # ── test (офлайн) ──
    test_p = subparsers.add_parser(
        "test", help="Быстрый офлайн-тест с микрофона для проверки профиля"
    )
    test_p.add_argument("--profile", default="default",
                        help="Имя профиля (default, bridge, engine, ...)")
    test_p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                        help="Директория модели")
    test_p.add_argument("--duration", type=float, default=15.0,
                        help="Длительность записи (сек)")
    test_p.add_argument("--base_conf_th", type=float, default=DEFAULT_BASE_CONF_TH,
                        help="Базовый порог (если нет per-label)")
    test_p.add_argument("--debounce_s", type=float, default=1.0,
                        help="Debounce (сек) между детекциями")
    test_p.add_argument(
        "--show_other",
        action="store_true",
        help=f"Показывать детекции класса '{OTHER_CLASS}' (по умолчанию скрыт)"
    )
    test_p.add_argument(
        "--outlier_detector",
        type=str,
        default=None,
        help="Путь к outlier_detector.pkl"
    )

    args = parser.parse_args()

    # ══════════════════════════════════════════════
    #  calibrate
    # ══════════════════════════════════════════════
    if args.command == "calibrate":
        profile_name = args.profile
        model_dir = args.model_dir
        data_dir = args.data_dir

        print(f"[INFO] Калибровка профиля: {profile_name}")
        print(f"[INFO] MODEL_DIR: {model_dir}")
        print(f"[INFO] DATASET_DIR: {data_dir}")
        print(f"[INFO] Стратегия порогов: {args.strategy}")
        model = ModelWrapper(model_dir, BASE_MODEL_NAME)

        base_cfg, base_cfg_path = load_base_config(model_dir)
        existing_conf = base_cfg.get("conf_th_per_label", {})

        pos_scores, neg_scores, labels, pos_details, neg_details = collect_pos_neg_scores(model, data_dir)
        if args.strategy == "precision":
            conf_th_per_label = find_thresholds_precision_first(
                pos_scores, neg_scores, labels, existing_conf
            )
        else:
            conf_th_per_label = find_thresholds_safety_first(
                pos_scores, neg_scores, labels, existing_conf
            )

        if args.skip_energy:
            energy_th = base_cfg.get("energy_th", 1e-4)
            print(f"[ENERGY] Пропускаем калибровку энергии. Текущее/дефолтное значение: {energy_th:.2e}")
        else:
            energy_th = calibrate_energy_threshold()

        print("\n[RESULT] Итоговые пороги:")
        for k, v in conf_th_per_label.items():
            print(f"  {k}: {v:.3f}")
        print(f"  energy_th: {energy_th:.2e}")

        out_cfg_path = save_profile_config(
            base_cfg, base_cfg_path, conf_th_per_label, energy_th, profile_name
        )

        profile_dir, base_name, json_path, csv_path = save_calibration_report(
            model_dir, profile_name, labels,
            pos_scores, neg_scores,
            conf_th_per_label, energy_th,
            strategy=args.strategy,
            pos_details=pos_details,
            neg_details=neg_details
        )

        save_calibration_plots(
            profile_dir, base_name, labels,
            pos_scores, neg_scores,
            conf_th_per_label
        )

        print("\nГотово.")
        print(f"Профиль '{profile_name}' записан в: {out_cfg_path}")
        print("Для быстрого теста можно запустить:")
        print(f"  python {os.path.basename(__file__)} test --profile {profile_name}")

    # ══════════════════════════════════════════════
    #  run (онлайн, PyTorch или ONNX)
    # ══════════════════════════════════════════════
    elif args.command == "run":
        run_realtime(
            model_dir=args.model_dir,
            profile=args.profile,
            base_conf_th=args.base_conf_th,
            debounce_s=args.debounce_s,
            outlier_detector=args.outlier_detector,
            onnx_dir=args.onnx_dir,
        )

    # ══════════════════════════════════════════════
    #  export_onnx
    # ══════════════════════════════════════════════
    elif args.command == "export_onnx":
        cmd_export_onnx(args)

    # ══════════════════════════════════════════════
    #  test (офлайн)
    # ══════════════════════════════════════════════
    elif args.command == "test":
        offline_mic_test(
            model_dir=args.model_dir,
            profile=args.profile,
            duration_s=args.duration,
            base_conf_th=args.base_conf_th,
            debounce_s=args.debounce_s,
            show_other=args.show_other,
            outlier_detector=args.outlier_detector,
        )


if __name__ == "__main__":
    main()
