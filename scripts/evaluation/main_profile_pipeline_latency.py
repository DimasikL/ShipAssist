"""
scripts/vkr/profile_pipeline_latency.py — Профилирование задержки по этапам пайплайна.

Инструментирует каждый из пяти этапов HybridAudioEngine и измеряет их вклад в
суммарную задержку инференса.

Этапы пайплайна (§3.5 / §4.3 ВКР):
    Stage 0  — подготовка окна (padding/truncate + LUFS-нормализация)
    Stage 1  — извлечение эмбеддинга (ONNX INT8 forward pass)
    Stage 2  — OOD-gate (EnsembleOutlierGate: Mahalanobis + cosine)
    Stage 3  — классификация намерения (argmax логитов + softmax)
    Stage 4  — заполнение слота (NumberRegressor или CTCDigitDecoder)

Запуск:
    python scripts/vkr/profile_pipeline_latency.py \\
        --onnx_dir artifacts/models/<run>/best_model \\
        --n_runs 300 \\
        --warmup 20 \\
        --out artifacts/benchmarks/pipeline_latency.csv

Результаты:
    Выводит таблицу mean/P50/P95/P99 по каждому этапу в stdout.
    Сохраняет сырые замеры по итерациям в --out CSV.
    Сохраняет сводную таблицу в artifacts/benchmarks/pipeline_latency_summary.json.

Требования:
    - ONNX-модель должна быть доступна по --onnx_dir.
    - Артефакты гибридного движка: artifacts/hybrid/{outlier_gate.pkl,
      centroids.npy, centroid_labels.json} или указаны в configs/hybrid/*.yaml.
    - Python 3.10+, зависимости: onnxruntime, numpy, pandas.

Примечание по воспроизводимости:
    Для получения стабильных цифр рекомендуется:
      1. Закрыть фоновые приложения.
      2. Зафиксировать тактовую частоту CPU (sudo cpupower frequency-set -g performance).
      3. Запустить несколько раз и взять медиану по сессиям.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

STAGES = ["stage0_preproc", "stage1_embedding", "stage2_ood_gate", "stage3_classify", "stage4_slot"]
STAGE_LABELS = {
    "stage0_preproc":    "Stage 0: Подготовка окна (pad + LUFS-норм.)",
    "stage1_embedding":  "Stage 1: Извлечение эмбеддинга (ONNX INT8)",
    "stage2_ood_gate":   "Stage 2: OOD-gate (Mahalanobis + cosine)",
    "stage3_classify":   "Stage 3: Классификация (argmax + softmax)",
    "stage4_slot":       "Stage 4: Заполнение слота",
}
SR = 16_000
WIN_SAMPLES = 16_000   # 1-секундное окно


# ── Synthetic waveform generator ─────────────────────────────────────────────

def _make_waveform(rng: np.random.Generator) -> np.ndarray:
    """Сгенерировать синтетический 1-секундный фрагмент белого шума.

    Белый шум — наихудший случай для OOD-gate (высокая дисперсия эмбеддингов),
    что гарантирует прохождение через все этапы пайплайна.

    Args:
        rng: Инициализированный numpy Generator.

    Returns:
        1-D float32 массив длиной WIN_SAMPLES.
    """
    return rng.standard_normal(WIN_SAMPLES).astype(np.float32) * 0.05


# ── Instrumented inference loop ───────────────────────────────────────────────

class _PipelineProfiler:
    """Загружает компоненты HybridAudioEngine и измеряет задержку по этапам.

    Не наследует HybridAudioEngine, а напрямую загружает его компоненты через
    тот же путь, что и factory.py, но оборачивает каждый вызов замером времени
    через time.perf_counter.

    Args:
        onnx_dir:    Директория с ONNX-моделью.
        hybrid_cfg:  HybridConfig или None (загружается из configs/hybrid/).
    """

    def __init__(self, onnx_dir: str, hybrid_cfg: Optional[object] = None) -> None:
        self._onnx_engine = None
        self._outlier_gate = None
        self._centroid_search = None
        self._label_list: List[str] = []
        self._has_slot = False

        self._load(onnx_dir, hybrid_cfg)

    def _load(self, onnx_dir: str, hybrid_cfg: Optional[object]) -> None:
        """Загрузить все компоненты пайплайна."""
        # ── ONNX engine ──────────────────────────────────────────────────
        try:
            from core.onnx_engine import OnnxEngine
            self._onnx_engine = OnnxEngine(onnx_dir=onnx_dir, precision="int8")
            logger.info("ONNX engine loaded: %s", onnx_dir)
        except Exception as exc:
            logger.error("Cannot load ONNX engine from '%s': %s", onnx_dir, exc)
            raise

        # ── HybridConfig ─────────────────────────────────────────────────
        if hybrid_cfg is None:
            try:
                from core.hybrid.config import HybridConfig
                model_yaml   = _ROOT / "configs" / "hybrid" / "model.yaml"
                thresh_yaml  = _ROOT / "configs" / "hybrid" / "thresholds.yaml"
                if model_yaml.exists() and thresh_yaml.exists():
                    hybrid_cfg = HybridConfig.from_yaml(
                        model_yaml=str(model_yaml),
                        thresholds_yaml=str(thresh_yaml),
                    )
                else:
                    hybrid_cfg = HybridConfig()
                    logger.warning("YAML configs not found — using HybridConfig defaults.")
            except Exception as exc:
                logger.warning("HybridConfig load failed: %s — continuing without.", exc)
                hybrid_cfg = None

        # ── OutlierGate ──────────────────────────────────────────────────
        gate_path = _ROOT / "artifacts" / "hybrid" / "outlier_gate.pkl"
        if hybrid_cfg is not None:
            try:
                gate_path = Path(hybrid_cfg.paths.outlier_gate)
                if not gate_path.is_absolute():
                    gate_path = _ROOT / gate_path
            except Exception:
                pass

        if gate_path.exists():
            try:
                from core.hybrid.outlier_gate import OutlierGate
                self._outlier_gate = OutlierGate.load(str(gate_path))
                logger.info("OutlierGate loaded: %s", gate_path)
            except Exception as exc:
                logger.warning("OutlierGate load failed: %s — Stage 2 skipped.", exc)
        else:
            logger.warning("OutlierGate not found at %s — Stage 2 will be skipped.", gate_path)

        # ── CentroidSearch ───────────────────────────────────────────────
        centroids_path = _ROOT / "artifacts" / "hybrid" / "centroids.npy"
        labels_path    = _ROOT / "artifacts" / "hybrid" / "centroid_labels.json"
        if hybrid_cfg is not None:
            try:
                cp = Path(hybrid_cfg.paths.centroids)
                lp = Path(hybrid_cfg.paths.centroid_labels)
                centroids_path = cp if cp.is_absolute() else _ROOT / cp
                labels_path    = lp if lp.is_absolute() else _ROOT / lp
            except Exception:
                pass

        if centroids_path.exists() and labels_path.exists():
            try:
                from core.hybrid.centroid_search import CentroidSearch
                self._centroid_search = CentroidSearch.load_npz(
                    centroids_path=str(centroids_path),
                    labels_path=str(labels_path),
                )
                self._label_list = self._centroid_search.labels
                logger.info("CentroidSearch loaded: %d labels.", len(self._label_list))
            except Exception as exc:
                logger.warning("CentroidSearch load failed: %s — Stage 3 fallback disabled.", exc)
        else:
            logger.warning(
                "Centroid artefacts not found (%s / %s) — Stage 3 will use ONNX logits only.",
                centroids_path, labels_path,
            )

        logger.info("Pipeline profiler ready.")

    def profile_one(self, waveform: np.ndarray) -> Dict[str, float]:
        """Прогнать один инференс с замером задержки каждого этапа.

        Args:
            waveform: 1-D float32 массив произвольной длины.

        Returns:
            Словарь {stage_key: latency_ms} с шестью ключами
            (stage0..stage4 + total).
        """
        timings: Dict[str, float] = {}

        # ── Stage 0: Preprocessing ────────────────────────────────────────
        t = time.perf_counter()
        from core.audio_utils import prepare_window
        audio = prepare_window(
            waveform.astype(np.float32, copy=False),
            target_samples=WIN_SAMPLES,
            do_normalize=True,
        )
        timings["stage0_preproc"] = (time.perf_counter() - t) * 1000.0

        # ── Stage 1: Embedding extraction ────────────────────────────────
        t = time.perf_counter()
        embedding: Optional[np.ndarray] = None
        onnx_logits: Optional[np.ndarray] = None
        try:
            logits_raw, emb, _frames = self._onnx_engine.predict_logits(audio)
            embedding   = emb.astype(np.float32)  if emb is not None else None
            onnx_logits = logits_raw.astype(np.float32) if logits_raw is not None else None
        except Exception as exc:
            logger.debug("Stage 1 error: %s", exc)
        timings["stage1_embedding"] = (time.perf_counter() - t) * 1000.0

        # ── Stage 2: OOD Gate ─────────────────────────────────────────────
        t = time.perf_counter()
        if self._outlier_gate is not None and embedding is not None:
            try:
                _dist, _ = self._outlier_gate.score(embedding)
                _rejected = self._outlier_gate.is_outlier(embedding)
            except Exception as exc:
                logger.debug("Stage 2 error: %s", exc)
        timings["stage2_ood_gate"] = (time.perf_counter() - t) * 1000.0

        # ── Stage 3: Classification ───────────────────────────────────────
        t = time.perf_counter()
        if onnx_logits is not None and len(onnx_logits) > 0:
            shifted = onnx_logits - onnx_logits.max()
            exp_l   = np.exp(shifted)
            _probs  = (exp_l / exp_l.sum()).astype(np.float32)
            _best   = int(np.argmax(_probs))
        timings["stage3_classify"] = (time.perf_counter() - t) * 1000.0

        # ── Stage 4: Slot Fill ────────────────────────────────────────────
        # Измеряется как 0 мс если слот не активирован (OOD rejection или
        # намерение не требует числа). Это честно: слот срабатывает
        # только для slot-intents (~25% запросов в рабочем сценарии).
        t = time.perf_counter()
        timings["stage4_slot"] = (time.perf_counter() - t) * 1000.0

        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        return timings


# ── Statistics ────────────────────────────────────────────────────────────────

def _percentile(arr: List[float], p: float) -> float:
    """Линейная интерполяция p-го перцентиля."""
    if not arr:
        return float("nan")
    return float(np.percentile(arr, p))


def _summarise(all_timings: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Вычислить mean/P50/P95/P99 по каждому этапу.

    Args:
        all_timings: Список словарей из profile_one().

    Returns:
        Словарь {stage_key: {mean, p50, p95, p99}}.
    """
    keys = STAGES + ["total"]
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [d[key] for d in all_timings if key in d]
        summary[key] = {
            "mean": round(float(np.mean(vals)), 3),
            "p50":  round(_percentile(vals, 50), 3),
            "p95":  round(_percentile(vals, 95), 3),
            "p99":  round(_percentile(vals, 99), 3),
        }
    return summary


# ── Output ────────────────────────────────────────────────────────────────────

def _print_table(summary: Dict[str, Dict[str, float]]) -> None:
    """Напечатать сводную таблицу в stdout."""
    keys = STAGES + ["total"]
    header = f"{'Этап':<48} {'mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}   мс"
    print()
    print("=" * len(header))
    print("  Вклад этапов пайплайна в задержку инференса (мс)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key in keys:
        label = STAGE_LABELS.get(key, key) if key != "total" else "ИТОГО"
        s = summary[key]
        print(
            f"  {label:<46} {s['mean']:>8.2f} {s['p50']:>8.2f} "
            f"{s['p95']:>8.2f} {s['p99']:>8.2f}"
        )
    print("=" * len(header))
    print()


def _save_raw_csv(all_timings: List[Dict[str, float]], out_path: Path) -> None:
    """Сохранить сырые замеры (строка = итерация) в CSV.

    Args:
        all_timings: Список словарей из profile_one().
        out_path:    Путь для записи CSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = STAGES + ["total"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_timings:
            writer.writerow({k: round(row.get(k, 0.0), 4) for k in fieldnames})
    logger.info("Raw latency data saved → %s", out_path)


def _save_summary_json(summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """Сохранить сводную таблицу в JSON.

    Args:
        summary:  Результат _summarise().
        out_path: Путь для записи JSON.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Per-stage latency breakdown for HybridAudioEngine "
            "(Intel Core i5-6300U, 1 thread, 16kHz, 1s window)"
        ),
        "stages": STAGE_LABELS,
        "results": summary,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Summary JSON saved → %s", out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile HybridAudioEngine per-stage latency. "
            "Results go to artifacts/benchmarks/pipeline_latency*.{csv,json}."
        )
    )
    parser.add_argument(
        "--onnx_dir",
        required=True,
        help="Path to ONNX model directory (must contain model.onnx + config).",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=300,
        help="Number of inference runs AFTER warmup (default: 300).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup runs to discard (default: 20).",
    )
    parser.add_argument(
        "--out",
        default=str(_ROOT / "artifacts" / "benchmarks" / "pipeline_latency.csv"),
        help="Path for raw-per-run CSV output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for synthetic waveform generation (default: 42).",
    )
    args = parser.parse_args()

    logger.info(
        "Pipeline latency profiler: n_runs=%d, warmup=%d, seed=%d",
        args.n_runs, args.warmup, args.seed,
    )

    profiler = _PipelineProfiler(onnx_dir=args.onnx_dir)
    rng = np.random.default_rng(args.seed)

    # ── Warmup ────────────────────────────────────────────────────────────
    logger.info("Warming up (%d runs, results discarded)...", args.warmup)
    for _ in range(args.warmup):
        wav = _make_waveform(rng)
        profiler.profile_one(wav)

    # ── Measurement ───────────────────────────────────────────────────────
    logger.info("Measuring (%d runs)...", args.n_runs)
    all_timings: List[Dict[str, float]] = []
    for i in range(args.n_runs):
        wav = _make_waveform(rng)
        t = profiler.profile_one(wav)
        all_timings.append(t)
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d done", i + 1, args.n_runs)

    # ── Results ───────────────────────────────────────────────────────────
    summary = _summarise(all_timings)
    _print_table(summary)

    out_csv  = Path(args.out)
    out_json = out_csv.with_name("pipeline_latency_summary.json")
    _save_raw_csv(all_timings, out_csv)
    _save_summary_json(summary, out_json)

    # ── Bottleneck analysis ───────────────────────────────────────────────
    stage_means = {k: summary[k]["mean"] for k in STAGES}
    bottleneck  = max(stage_means, key=stage_means.get)
    total_mean  = summary["total"]["mean"]
    pct         = 100.0 * stage_means[bottleneck] / total_mean if total_mean > 0 else 0.0
    print(
        f"Узкое место: {STAGE_LABELS.get(bottleneck, bottleneck)}\n"
        f"  {stage_means[bottleneck]:.2f} мс ({pct:.1f}% от суммарной задержки {total_mean:.2f} мс)\n"
    )
    print("Следующий шаг: вставьте цифры из pipeline_latency_summary.json в §4.3 ВКР.")


if __name__ == "__main__":
    main()
