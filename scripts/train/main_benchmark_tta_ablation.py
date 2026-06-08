"""
scripts/train/benchmark_tta_ablation.py — TTA ablation: accuracy при k=1,2,3.

Запуск:
    python scripts/train/benchmark_tta_ablation.py \\
        --onnx_dir artifacts/models/<run>/best_model \\
        --test_csv artifacts/data/dset_meta_only_2026-02-24_17-36-00.csv \\
        --audio_root <корень для audio_path из CSV>

CSV должен содержать столбцы (поддерживаются оба варианта):
  - ``path`` + ``label``   (канонический формат скрипта)
  - ``audio_path`` + ``class``  (формат dset_meta_only_*.csv)

Результаты сохраняются в artifacts/benchmarks/tta_ablation.json.
Добавить таблицу «F1 и задержка при k=1,2,3» в §5.3 ВКР после получения цифр.

ИЗМЕНЕНИЯ (2026-05-25):
  - Импорт TTAWrapper исправлен: core.audio_tta (не experiments).
  - Функция загрузки аудио исправлена: core.audio_utils.load_wav.
  - CSV-столбцы: поддержка audio_path/class наряду с path/label.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_columns(df) -> tuple[str, str]:
    """Определить имена столбцов пути и метки в DataFrame.

    Поддерживает два варианта:
      - ``path`` + ``label``        (канонический)
      - ``audio_path`` + ``class``  (формат dset_meta_only_*.csv)

    Args:
        df: Загруженный pandas DataFrame.

    Returns:
        Кортеж ``(path_col, label_col)``.

    Raises:
        KeyError: Если ни один из поддерживаемых вариантов не найден.
    """
    cols = set(df.columns)
    if "path" in cols and "label" in cols:
        return "path", "label"
    if "audio_path" in cols and "class" in cols:
        logger.info("CSV: using columns 'audio_path'/'class' (dset_meta_only format).")
        return "audio_path", "class"
    raise KeyError(
        f"CSV must contain ('path','label') or ('audio_path','class'). "
        f"Found columns: {sorted(cols)}"
    )


def _label_from_logits(logits: np.ndarray, labels: list) -> str:
    """Derive predicted label from raw logit/score vector."""
    idx = int(np.argmax(logits))
    return labels[idx] if idx < len(labels) else ""


def run_ablation(
    onnx_dir: str,
    test_csv: str,
    audio_root: str,
    max_samples: int = 0,
    seed: int = 42,
) -> Dict:
    """Запустить TTA ablation для k in {1, 2, 3}.

    Оптимизация (2026-05-26 v2): 3 инференса на файл вместо 6.
    Оригинал + 2 аугментации прогоняются ровно один раз; метрики для k=1,2,3
    вычисляются из накопленных логитов без повторных вызовов engine.predict().

    Args:
        onnx_dir:    Путь к директории с ONNX-моделью.
        test_csv:    CSV с колонками ``path``/``label`` или ``audio_path``/``class``.
        audio_root:  Корневая директория аудио.  Пусто → пути абсолютные.
        max_samples: Если > 0 — стратифицированно выбирает не более max_samples
                     строк из CSV.  Удобно для быстрого ablation-прогона.
        seed:        Random seed для стратифицированной выборки.

    Returns:
        Словарь с ключами "k=1", "k=2", "k=3" и метриками accuracy + 95% CI.
    """
    import pandas as pd

    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm not installed — progress bar disabled. Run: pip install tqdm")
        def tqdm(it, **_kwargs):  # type: ignore[misc]
            return it

    from core.engine import OnnxAudioEngine
    from core.audio_utils import load_wav
    from core.audio_tta import augment_gaussian, augment_time_shift

    engine = OnnxAudioEngine(onnx_dir=onnx_dir, precision="int8")
    label_list: list = getattr(engine, "labels", [])

    df = pd.read_csv(test_csv)
    path_col, label_col = _resolve_columns(df)

    # ── Optional stratified subsample ────────────────────────────────────────
    if max_samples > 0 and max_samples < len(df):
        rng_pd = np.random.default_rng(seed)
        groups = df.groupby(label_col, group_keys=False)
        n_classes = df[label_col].nunique()
        per_class = max(1, max_samples // n_classes)

        def _sample(grp: "pd.DataFrame") -> "pd.DataFrame":
            n = min(len(grp), per_class)
            idx = rng_pd.choice(len(grp), size=n, replace=False)
            return grp.iloc[sorted(idx)]

        df = groups.apply(_sample).reset_index(drop=True)
        logger.info(
            "Stratified subsample: %d → %d samples (%d classes, ≤%d each, seed=%d)",
            max_samples, len(df), n_classes, per_class, seed,
        )

    paths_list = df[path_col].astype(str).tolist()
    labels_list = df[label_col].astype(str).tolist()

    ks = [1, 2, 3]
    correct = {k: 0 for k in ks}
    total   = {k: 0 for k in ks}
    skipped = 0

    # Augmentation params (mirror TTAWrapper defaults)
    sr = 16_000
    shift_ms = 10.0
    noise_sigma = 0.005
    shift_samples = int(round(shift_ms * sr / 1_000.0))
    aug_rng = np.random.default_rng(seed)

    logger.info(
        "3-inference-per-file ablation: %d samples × 3 passes (orig+aug1+aug2)",
        len(paths_list),
    )

    for rel_path, true_label in tqdm(
        zip(paths_list, labels_list),
        total=len(paths_list),
        desc="TTA ablation",
        unit="file",
        dynamic_ncols=True,
    ):
        audio_path = Path(audio_root) / rel_path if audio_root else Path(rel_path)

        if not audio_path.exists():
            skipped += 1
            if skipped <= 5:
                logger.warning("Audio not found, skipping: %s", audio_path)
            elif skipped == 6:
                logger.warning("Further missing-file warnings suppressed.")
            continue

        try:
            waveform, _ = load_wav(str(audio_path), target_sr=sr)
        except Exception as exc:
            logger.warning("Load error, skipping %s: %s", audio_path.name, exc)
            skipped += 1
            continue

        # ── 3 inferences: original, gaussian-noise copy, time-shift copy ─────
        result_orig = engine.predict(waveform)
        logits_orig: np.ndarray = result_orig.get("logits", np.array([], dtype=np.float32))

        if logits_orig.size == 0:
            # Engine returned no logits — fall back to label string comparison only
            pred_label = result_orig.get("label", "")
            for k in ks:
                total[k] += 1
                if pred_label == true_label:
                    correct[k] += 1
            continue

        aug1_wav = augment_gaussian(waveform, sigma=noise_sigma, rng=aug_rng)
        result_aug1 = engine.predict(aug1_wav)
        logits_aug1: np.ndarray = result_aug1.get("logits", np.array([], dtype=np.float32))

        aug2_wav = augment_time_shift(waveform, shift_samples)
        result_aug2 = engine.predict(aug2_wav)
        logits_aug2: np.ndarray = result_aug2.get("logits", np.array([], dtype=np.float32))

        # ── Accumulate per-k ─────────────────────────────────────────────────
        # k=1: original only
        # k=2: mean(orig, aug1)  — aug1 used only if not outlier-rejected & shape matches
        # k=3: mean(orig, aug1, aug2)
        logit_sets = {
            1: [logits_orig],
            2: [logits_orig],
            3: [logits_orig],
        }
        for aug_res, aug_logits in [(result_aug1, logits_aug1), (result_aug2, logits_aug2)]:
            if (
                not aug_res.get("outlier_rejected", False)
                and aug_logits.shape == logits_orig.shape
                and aug_logits.size > 0
            ):
                logit_sets[2].append(aug_logits) if len(logit_sets[2]) < 2 else None
                logit_sets[3].append(aug_logits)

        for k in ks:
            stacked = np.stack(logit_sets[k], axis=0)
            avg = np.mean(stacked, axis=0)
            pred_label = _label_from_logits(avg, label_list) if label_list else result_orig.get("label", "")
            total[k] += 1
            if pred_label == true_label:
                correct[k] += 1

    if skipped:
        logger.warning("Total skipped files: %d", skipped)

    results = {}
    for k in ks:
        acc = correct[k] / total[k] if total[k] > 0 else 0.0
        results[f"k={k}"] = {
            "accuracy": round(acc, 4),
            "correct": correct[k],
            "total": total[k],
        }
        logger.info("k=%d: accuracy=%.4f (%d/%d)", k, acc, correct[k], total[k])

    # Clopper-Pearson 95% confidence intervals
    from scipy.stats import beta as beta_dist

    for key, v in results.items():
        n_correct, n_total = v["correct"], v["total"]
        lo = float(beta_dist.ppf(0.025, n_correct, n_total - n_correct + 1)) if n_correct > 0 else 0.0
        hi = float(beta_dist.ppf(0.975, n_correct + 1, n_total - n_correct)) if n_correct < n_total else 1.0
        v["ci_95"] = [round(lo, 4), round(hi, 4)]

    return results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="TTA ablation: accuracy at k=1,2,3 on a held-out test set."
    )
    parser.add_argument("--onnx_dir", required=True, help="Path to ONNX model directory")
    parser.add_argument(
        "--test_csv",
        default="artifacts/data/test_set.csv",
        help="CSV with 'path' and 'label' columns (default: artifacts/data/test_set.csv)",
    )
    parser.add_argument(
        "--audio_root",
        default="",
        help=(
            "Root directory prepended to each audio path from CSV. "
            "Pass empty string '' if paths in CSV are already absolute. "
            "(default: '' — use paths as-is)"
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=(
            "Stratified subsample size for fast ablation runs. "
            "0 = use full CSV (default). "
            "Example: --max_samples 3000 uses ~3000 samples spread evenly across classes."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified subsampling (default: 42).",
    )
    args = parser.parse_args()

    results = run_ablation(args.onnx_dir, args.test_csv, args.audio_root, args.max_samples, args.seed)

    out_path = _ROOT / "artifacts" / "benchmarks" / "tta_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== TTA Ablation Results ===")
    print(f"{'k':>4} | {'Accuracy':>10} | {'95% CI':>22} | N")
    print("-" * 54)
    for key, v in results.items():
        ci = v.get("ci_95", [0.0, 0.0])
        print(f"{key:>4} | {v['accuracy']:>10.4f} | [{ci[0]:.4f}; {ci[1]:.4f}] | {v['total']}")

    print(f"\nSaved to: {out_path}")
    print("\nNext steps:")
    print("  1. Add results table to §5.3 in the thesis (patch_experiments_ab.py --fill-tta).")
    print("  2. TTAWrapper is already in core/audio_tta.py — no move needed.")
    print("  3. Update the docstring note in core/audio_tta.py (remove TODO).")


if __name__ == "__main__":
    main()
