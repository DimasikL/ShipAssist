"""
scripts/vkr/experiment_f1_vs_snr.py — Эксперимент 6: деградация F1 при снижении SNR.

Измеряет macro-F1 трёх методов на тестовой выборке (123 записи, 5 дикторов,
speaker-disjoint, §4.1) при уровнях ОСШ от чистого сигнала до −2 дБ.
Шум — гауссовский с контролем SNR (воспроизводимо, seed=42), аналогично
eval_noisy_snr12.py и benchmark_mfcc_svm.py.

Методы:
    • LoRA-Wav2Vec2 + ONNX INT8  (предложенный)
    • MFCC + SVM                  (базовая линия 1)
    • Whisper-tiny zero-shot      (базовая линия 2, опционально)

Usage:
    cd <PROJECT_ROOT>
    python scripts/vkr/experiment_f1_vs_snr.py [--skip-whisper] [--dry-run]

    --skip-whisper  Пропустить Whisper (экономит ~5–10 мин CPU).
    --dry-run       Проверить пути и модели без запуска инференса.

Output:
    artifacts/benchmarks/f1_vs_snr.csv
    artifacts/plots/vkr_figures/fig_4_5_f1_vs_snr.png  (если установлен matplotlib)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Bootstrap: PROJECT_ROOT независимо от cwd
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]   # scripts/vkr/ → scripts/ → root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы — должны совпадать с eval_noisy_snr12.py и benchmark_mfcc_svm.py
# ---------------------------------------------------------------------------
SR: int = 16_000
WIN_SAMPLES: int = 48_000            # 3 секунды при 16 кГц
NOISE_SEED: int = 42                 # воспроизводимость

# Тестовые группы (speaker-disjoint §4.1)
TEST_GROUPS: List[str] = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]

# Уровни ОСШ для эксперимента (None = чистый сигнал)
SNR_LEVELS: List[Optional[float]] = [None, 20, 15, 12, 10, 8, 5, 2, 0, -2]

# Пути (все относительно PROJECT_ROOT)
CSV_PATH    = PROJECT_ROOT / "dset_meta_only_2026-05-09_10-27-42.csv"
ONNX_DIR    = PROJECT_ROOT / "onnx_model" / "quant_benchmark"
OUTPUT_CSV  = PROJECT_ROOT / "artifacts" / "benchmarks" / "f1_vs_snr.csv"
PLOT_PATH   = PROJECT_ROOT / "artifacts" / "plots" / "vkr_figures" / "fig_4_5_f1_vs_snr.png"

# MFCC-параметры (идентично benchmark_mfcc_svm.py)
N_MFCC: int = 13
HOP_LENGTH: int = 512
N_FFT: int = 2048


# ---------------------------------------------------------------------------
# Утилиты: путь и шум
# ---------------------------------------------------------------------------

def _fix_path(p: str) -> Path:
    """Перевести Windows-путь из CSV в актуальный путь на хосте.

    Args:
        p: Строка пути (может содержать обратные слэши и старый Windows root).

    Returns:
        Объект Path с актуальным расположением файла.
    """
    p = p.replace("\\", "/")
    for win_root in (
        "C:/Users/Dmitriy/PycharmProjects/ShipAssistant",
        "D:/Users/Dmitriy/PycharmProjects/ShipAssistant",
    ):
        if win_root in p:
            p = p.replace(win_root, str(PROJECT_ROOT).replace("\\", "/"))
            break
    return Path(p)


def add_noise(wav: np.ndarray, snr_db: float, seed: int = NOISE_SEED) -> np.ndarray:
    """Добавить гауссовский шум с заданным ОСШ (воспроизводимо).

    Реализация идентична eval_noisy_snr12.py и benchmark_mfcc_svm.py
    для сопоставимости результатов.

    Args:
        wav:    Входной сигнал float32, форма (N,).
        snr_db: Целевое отношение сигнал/шум в дБ.
        seed:   Seed генератора случайных чисел.

    Returns:
        Зашумлённый сигнал float32 той же формы.
    """
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(wav.astype(np.float32) ** 2)) + 1e-10
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return wav.astype(np.float32) + noise


def _load_wav_trimmed(audio_path: Path) -> np.ndarray:
    """Загрузить WAV и обрезать/дополнить до WIN_SAMPLES.

    Args:
        audio_path: Путь к WAV-файлу.

    Returns:
        Float32-массив длиной WIN_SAMPLES.
    """
    import librosa
    try:
        wav, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception as exc:
        logger.warning("Не удалось загрузить %s: %s — используется тишина.", audio_path.name, exc)
        return np.zeros(WIN_SAMPLES, dtype=np.float32)
    if len(wav) < WIN_SAMPLES:
        wav = np.pad(wav, (0, WIN_SAMPLES - len(wav)))
    else:
        wav = wav[:WIN_SAMPLES]
    return wav.astype(np.float32)


def _prepare_for_onnx(audio_path: Path, snr_db: Optional[float]) -> np.ndarray:
    """Загрузка + подготовка окна через core.audio_utils + инъекция шума.

    Args:
        audio_path: Путь к WAV-файлу.
        snr_db:     ОСШ в дБ; None → чистый сигнал.

    Returns:
        Float32-массив формы (WIN_SAMPLES,).
    """
    from core.audio_utils import load_wav, prepare_window
    try:
        waveform, _ = load_wav(str(audio_path), target_sr=SR)
    except Exception as exc:
        logger.warning("ONNX load failed %s: %s", audio_path.name, exc)
        return np.zeros(WIN_SAMPLES, dtype=np.float32)
    clean = prepare_window(waveform, target_samples=WIN_SAMPLES, do_normalize=True)
    if snr_db is not None:
        return add_noise(clean, snr_db)
    return clean.astype(np.float32)


# ---------------------------------------------------------------------------
# Тестовая выборка
# ---------------------------------------------------------------------------

def build_test_df() -> pd.DataFrame:
    """Загрузить тестовый сплит из CSV и разрешить пути.

    Returns:
        DataFrame с колонками: audio_path (Path), label (str).

    Raises:
        FileNotFoundError: если CSV не найден.
        ValueError: если ни одна запись не попала в TEST_GROUPS.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV не найден: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)
    if test_df.empty:
        raise ValueError(f"Тестовая выборка пуста — проверьте TEST_GROUPS: {TEST_GROUPS}")
    test_df["audio_path"] = test_df["audio_path"].apply(lambda p: _fix_path(str(p)))
    logger.info("Тестовая выборка: %d записей, классы: %s",
                len(test_df), dict(test_df["class"].value_counts()))
    return test_df[["audio_path", "class"]].rename(columns={"class": "label"})


# ---------------------------------------------------------------------------
# Метод 1: LoRA-Wav2Vec2 + ONNX INT8
# ---------------------------------------------------------------------------

def _build_onnx_session(onnx_dir: Path):
    """Создать InferenceSession для INT8-модели.

    Args:
        onnx_dir: Директория с onnx_config.json и весами.

    Returns:
        Кортеж (session, label_names, input_name).
    """
    import onnxruntime as ort
    config = json.loads((onnx_dir / "onnx_config.json").read_text(encoding="utf-8"))
    model_file = onnx_dir / config["model_int8"]
    if not model_file.exists():
        raise FileNotFoundError(f"INT8-модель не найдена: {model_file}")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = os.cpu_count() or 1
    opts.intra_op_num_threads = os.cpu_count() or 1
    session = ort.InferenceSession(
        str(model_file),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    label_names: List[str] = config["labels"]
    input_name: str = session.get_inputs()[0].name
    logger.info("ONNX INT8 загружен: %s  (%d классов)", model_file.name, len(label_names))
    return session, label_names, input_name


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def eval_onnx(
    test_df: pd.DataFrame,
    session,
    label_names: List[str],
    input_name: str,
    snr_db: Optional[float],
) -> float:
    """Вычислить macro-F1 для ONNX INT8 при заданном ОСШ.

    Args:
        test_df:    DataFrame с колонками audio_path, label.
        session:    ONNX InferenceSession.
        label_names: Список меток в порядке логитов.
        input_name: Имя входного тензора модели.
        snr_db:     ОСШ в дБ; None → чистый сигнал.

    Returns:
        Macro-F1 (float в [0, 1]).
    """
    label2id = {lbl: i for i, lbl in enumerate(label_names)}
    preds, targets = [], []
    for _, row in test_df.iterrows():
        sample = _prepare_for_onnx(row["audio_path"], snr_db)
        batch = sample[np.newaxis, :]
        logits = session.run(None, {input_name: batch})[0][0]
        preds.append(int(np.argmax(_softmax(logits.astype(np.float32)))))
        targets.append(label2id[row["label"]])
    return float(f1_score(targets, preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Метод 2: MFCC + SVM
# ---------------------------------------------------------------------------

def _extract_mfcc(wav: np.ndarray) -> np.ndarray:
    """78-мерный MFCC: mean+std × (MFCC, Δ, ΔΔ).

    Args:
        wav: Float32-массив длиной WIN_SAMPLES.

    Returns:
        Вектор признаков формы (78,).
    """
    import librosa
    mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        d1.mean(axis=1),   d1.std(axis=1),
        d2.mean(axis=1),   d2.std(axis=1),
    ])


def _build_train_groups(df_full: pd.DataFrame) -> pd.DataFrame:
    """Выбрать обучающий сплит (все группы, кроме тестовых).

    Args:
        df_full: Полный DataFrame из CSV с разрешёнными путями.

    Returns:
        DataFrame обучающего сплита.
    """
    train_df = df_full[~df_full["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)
    train_df["audio_path"] = train_df["audio_path"].apply(lambda p: _fix_path(str(p)))
    return train_df


def train_svm(df_full: pd.DataFrame) -> Tuple:
    """Обучить MFCC+SVM на обучающем сплите (чистый сигнал).

    Args:
        df_full: Полный DataFrame из CSV (до фильтрации тестовых групп).

    Returns:
        Кортеж (svm_pipeline, label2id, label_names).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    import librosa  # noqa: F401 — проверка импорта

    train_df = _build_train_groups(df_full)
    label_names = sorted(train_df["class"].unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(label_names)}

    logger.info("Обучаю MFCC+SVM на %d записях (%d классов)…", len(train_df), len(label_names))
    X, y = [], []
    missing = 0
    for _, row in train_df.iterrows():
        path = _fix_path(str(row["audio_path"]))
        if not path.exists():
            missing += 1
            continue
        wav = _load_wav_trimmed(path)
        X.append(_extract_mfcc(wav))
        y.append(label2id[row["class"]])

    if missing:
        logger.warning("MFCC+SVM: пропущено %d файлов (не найдены).", missing)
    if not X:
        raise RuntimeError("Нет доступных обучающих файлов для MFCC+SVM.")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", decision_function_shape="ovr")),
    ])
    pipe.fit(np.array(X), y)
    logger.info("MFCC+SVM обучен: %d обучающих примеров.", len(X))
    return pipe, label2id, label_names


def eval_mfcc_svm(
    test_df: pd.DataFrame,
    pipe,
    label2id: Dict[str, int],
    snr_db: Optional[float],
) -> float:
    """Вычислить macro-F1 для MFCC+SVM при заданном ОСШ.

    Args:
        test_df:  DataFrame с колонками audio_path, label.
        pipe:     Обученный sklearn-пайплайн.
        label2id: Маппинг метка → индекс.
        snr_db:   ОСШ в дБ; None → чистый сигнал.

    Returns:
        Macro-F1.
    """
    X, targets = [], []
    for _, row in test_df.iterrows():
        wav = _load_wav_trimmed(row["audio_path"])
        if snr_db is not None:
            wav = add_noise(wav, snr_db)
        X.append(_extract_mfcc(wav))
        targets.append(label2id[row["label"]])
    preds = pipe.predict(np.array(X)).tolist()
    return float(f1_score(targets, preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Метод 3: Whisper-tiny (опционально)
# ---------------------------------------------------------------------------

WHISPER_KEYWORDS: Dict[str, List[str]] = {
    "машина":                ["машина", "машин"],
    "приготовить машину":   ["приготовить", "приготовь"],
    "самый малый вперед":   ["малый", "вперед", "вперёд"],
    "другие слова":          [],
}


def _transcribe_batch(model, audio_paths: List[Path], snr_db: Optional[float]) -> List[str]:
    """Транскрибировать список файлов через Whisper-tiny.

    Args:
        model:       Загруженная Whisper-модель.
        audio_paths: Список путей к WAV.
        snr_db:      ОСШ в дБ; None → чистый сигнал.

    Returns:
        Список транскрипций (строки в нижнем регистре).
    """
    import tempfile
    import soundfile as sf

    transcriptions = []
    for path in audio_paths:
        wav = _load_wav_trimmed(path)
        if snr_db is not None:
            wav = add_noise(wav, snr_db)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, wav, SR)
            result = model.transcribe(tmp_path, language="ru", fp16=False)
            transcriptions.append(result.get("text", "").strip().lower())
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return transcriptions


def _match_label(text: str) -> str:
    """Сопоставить транскрипцию с командой по ключевым словам.

    Args:
        text: Транскрипция в нижнем регистре.

    Returns:
        Название класса.
    """
    # Проверяем составные команды сначала (более специфичные)
    order = ["приготовить машину", "самый малый вперед", "машина", "другие слова"]
    for cmd in order:
        keywords = WHISPER_KEYWORDS[cmd]
        if any(kw in text for kw in keywords):
            return cmd
    return "другие слова"


def eval_whisper(
    test_df: pd.DataFrame,
    label2id: Dict[str, int],
    snr_db: Optional[float],
) -> float:
    """Вычислить macro-F1 для Whisper-tiny при заданном ОСШ.

    Args:
        test_df:  DataFrame с колонками audio_path, label.
        label2id: Маппинг метка → индекс.
        snr_db:   ОСШ в дБ; None → чистый сигнал.

    Returns:
        Macro-F1, или -1.0 если Whisper недоступен.
    """
    try:
        import whisper
    except ImportError:
        logger.warning("whisper не установлен — Whisper-tiny пропущен.")
        return -1.0
    logger.info("  Загружаю Whisper-tiny…")
    model = whisper.load_model("tiny")
    paths = test_df["audio_path"].tolist()
    transcriptions = _transcribe_batch(model, paths, snr_db)
    preds = [label2id[_match_label(t)] for t in transcriptions]
    targets = [label2id[row["label"]] for _, row in test_df.iterrows()]
    return float(f1_score(targets, preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def run_experiment(skip_whisper: bool = False, dry_run: bool = False) -> pd.DataFrame:
    """Запустить эксперимент F1 vs SNR по всем методам и уровням шума.

    Args:
        skip_whisper: Если True — пропустить Whisper-tiny.
        dry_run:      Если True — только проверить пути, не запускать инференс.

    Returns:
        DataFrame с результатами (колонки: snr_db, snr_label, method, macro_f1).
    """
    # -- Проверка наличия файлов ------------------------------------------------
    logger.info("=" * 60)
    logger.info("Проверка путей…")
    missing_paths = []
    for path, name in [
        (CSV_PATH,                        "CSV"),
        (ONNX_DIR / "onnx_config.json",   "ONNX config"),
        (ONNX_DIR / "model_int8.onnx",    "ONNX INT8"),
    ]:
        if path.exists():
            logger.info("  ✔  %s: %s", name, path)
        else:
            logger.error("  ✘  %s не найден: %s", name, path)
            missing_paths.append(str(path))
    if missing_paths:
        raise FileNotFoundError("Отсутствуют критические файлы:\n" + "\n".join(missing_paths))
    if dry_run:
        logger.info("dry-run: все пути найдены. Завершаю без запуска инференса.")
        return pd.DataFrame()

    # -- Тестовый сплит --------------------------------------------------------
    test_df = build_test_df()
    df_full = pd.read_csv(CSV_PATH)  # нужен для обучения SVM

    # -- Загрузка/обучение моделей ---------------------------------------------
    logger.info("=" * 60)
    logger.info("Инициализация моделей…")
    session, label_names, input_name = _build_onnx_session(ONNX_DIR)
    svm_pipe, svm_label2id, svm_label_names = train_svm(df_full)

    whisper_label2id = {lbl: i for i, lbl in enumerate(svm_label_names)}

    # -- Главный цикл по SNR ---------------------------------------------------
    rows = []
    logger.info("=" * 60)
    logger.info("Запуск оценки по %d уровням SNR…", len(SNR_LEVELS))

    for snr in SNR_LEVELS:
        snr_label = "clean" if snr is None else f"{int(snr):+d} dB"
        snr_val   = float("inf") if snr is None else snr
        logger.info("-" * 40)
        logger.info("SNR = %s", snr_label)

        # 1. ONNX INT8
        f1_onnx = eval_onnx(test_df, session, label_names, input_name, snr)
        logger.info("  LoRA-Wav2Vec2 ONNX INT8 : F1 = %.4f", f1_onnx)
        rows.append({
            "snr_db":    snr_val,
            "snr_label": snr_label,
            "method":    "LoRA-Wav2Vec2 ONNX INT8",
            "macro_f1":  f1_onnx,
        })

        # 2. MFCC+SVM
        f1_svm = eval_mfcc_svm(test_df, svm_pipe, svm_label2id, snr)
        logger.info("  MFCC + SVM               : F1 = %.4f", f1_svm)
        rows.append({
            "snr_db":    snr_val,
            "snr_label": snr_label,
            "method":    "MFCC + SVM",
            "macro_f1":  f1_svm,
        })

        # 3. Whisper-tiny
        if not skip_whisper:
            f1_whisper = eval_whisper(test_df, whisper_label2id, snr)
            if f1_whisper >= 0:
                logger.info("  Whisper-tiny             : F1 = %.4f", f1_whisper)
            else:
                logger.info("  Whisper-tiny             : ПРОПУЩЕН (нет пакета)")
            rows.append({
                "snr_db":    snr_val,
                "snr_label": snr_label,
                "method":    "Whisper-tiny",
                "macro_f1":  f1_whisper if f1_whisper >= 0 else float("nan"),
            })

    result_df = pd.DataFrame(rows)

    # -- Сохранение CSV --------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info("=" * 60)
    logger.info("Результаты сохранены → %s", OUTPUT_CSV)

    # -- Вывод сводной таблицы -------------------------------------------------
    pivot = result_df.pivot(index="snr_label", columns="method", values="macro_f1")
    # Сортируем по убыванию SNR: clean первый
    snr_order = ["clean"] + [f"{int(s):+d} dB" for s in [20, 15, 12, 10, 8, 5, 2, 0, -2]]
    pivot = pivot.reindex([s for s in snr_order if s in pivot.index])
    logger.info("\nСводная таблица macro-F1 (строки = SNR, столбцы = метод):\n%s",
                pivot.to_string(float_format="{:.4f}".format))

    return result_df


# ---------------------------------------------------------------------------
# Построение графика (вызывается из скрипта или напрямую)
# ---------------------------------------------------------------------------

def plot_f1_vs_snr(result_df: pd.DataFrame, save_path: Path = PLOT_PATH) -> None:
    """Построить кривую F1 vs SNR и сохранить в PNG (рис. 4.5).

    Args:
        result_df: DataFrame с колонками snr_db, method, macro_f1.
        save_path: Путь для сохранения PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        logger.warning("matplotlib не установлен — график не построен.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    style_map = {
        "LoRA-Wav2Vec2 ONNX INT8": dict(color="#1f77b4", lw=2.5, marker="o", ms=7, zorder=5),
        "MFCC + SVM":              dict(color="#ff7f0e", lw=1.8, marker="s", ms=6, ls="--"),
        "Whisper-tiny":            dict(color="#2ca02c", lw=1.8, marker="^", ms=6, ls="-."),
    }

    methods = result_df["method"].unique()
    for method in methods:
        sub = result_df[result_df["method"] == method].copy()
        sub = sub.sort_values("snr_db", ascending=False)  # clean → -2 dB
        # Заменяем inf на 22 для оси X
        x = sub["snr_db"].replace(float("inf"), 22).tolist()
        y = sub["macro_f1"].tolist()
        kw = style_map.get(method, dict(lw=1.8, marker="D", ms=6))
        ax.plot(x, y, label=method, **kw)

    # Горизонтальная линия F1 = 0.9
    ax.axhline(0.9, color="gray", lw=1.2, ls=":", label="F1 = 0,90")

    # Ось X: заменяем 22 на «чистый»
    xticks = [22, 20, 15, 12, 10, 8, 5, 2, 0, -2]
    xlabels = ["чистый", "20", "15", "12", "10", "8", "5", "2", "0", "−2"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.invert_xaxis()  # чистый → -2 дБ слева направо

    ax.set_xlabel("ОСШ, дБ", fontsize=13)
    ax.set_ylabel("Macro-F1", fontsize=13)
    ax.set_title(
        "Рисунок 4.5 — Зависимость macro-F1 от уровня ОСШ\n"
        "(тестовая выборка: 123 записи, 5 дикторов, шум: гауссовский seed=42)",
        fontsize=12,
        pad=12,
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("График сохранён → %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Эксперимент 6 ВКР: F1 vs SNR для предложенного метода и базовых линий.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        default=False,
        help="Пропустить Whisper-tiny (экономит ~5–10 мин на CPU).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Только проверить пути и модели, не запускать инференс.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Не строить график после расчёта.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа CLI."""
    args = _parse_args()
    logger.info("ShipAssistant — Эксперимент 6: F1 vs SNR")
    logger.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logger.info("skip_whisper=%s  dry_run=%s", args.skip_whisper, args.dry_run)

    result_df = run_experiment(
        skip_whisper=args.skip_whisper,
        dry_run=args.dry_run,
    )

    if not result_df.empty and not args.no_plot:
        plot_f1_vs_snr(result_df)

    if not result_df.empty:
        logger.info("Готово. CSV: %s", OUTPUT_CSV)
        logger.info("График: %s", PLOT_PATH)


if __name__ == "__main__":
    main()
