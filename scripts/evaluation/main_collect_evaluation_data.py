"""
collect_vkr_data.py — VKR Data Auditor for ShipAssistant.

Scans the project for all artefacts needed to produce VKR figures and
tables, copies/exports them to artifacts/vkr_data/, and writes a
status report to artifacts/vkr_data/_README.md.

Collected datasets
------------------
a) training_curves.csv        — loss / F1 per epoch (train + val)
b) confusion_matrix.csv       — 4×4 confusion matrix
   per_class_metrics.csv      — precision / recall / F1 / support per class
c) mahalanobis_distances.csv  — in-distribution distance statistics
   (per-class centroids + global threshold τ)
d) memory_24h.csv             — RSS memory time-series from a 24-h load test
e) benchmarks_summary.csv     — latency / F1 / size for PyTorch FP32,
                                 ONNX FP32, ONNX INT8
f) corpus_by_class.csv        — number of samples per class in the corpus

Usage
-----
    python scripts/vkr/collect_vkr_data.py [--dry-run]

Google-style docstrings, no hardcoded absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap: find PROJECT_ROOT regardless of cwd
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]   # scripts/vkr/ → scripts/ → project/
VKR_DATA = PROJECT_ROOT / "artifacts" / "vkr_data"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"


def _ok(msg: str) -> str:
    return f"{_GREEN}✔  НАЙДЕНО{_RESET}  {msg}"


def _miss(msg: str) -> str:
    return f"{_RED}✘  НЕ НАЙДЕНО{_RESET}  {msg}"


def _warn(msg: str) -> str:
    return f"{_YELLOW}⚠  ПРЕДУПРЕЖДЕНИЕ{_RESET}  {msg}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, Any]], *, dry_run: bool = False) -> None:
    """Write a list of dicts as CSV."""
    if dry_run:
        log.info("  [dry-run] would write %s (%d rows)", path.name, len(rows))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("  → сохранено: %s (%d строк)", path.relative_to(PROJECT_ROOT), len(rows))


def _latest_lora_run() -> Optional[Path]:
    """Return the most recent lora_tune run directory that has training_history.csv."""
    base = PROJECT_ROOT / "lora_tune" / "models"
    if not base.exists():
        return None
    candidates = sorted(
        (d for d in base.iterdir()
         if d.is_dir() and (d / "training_history.csv").exists()),
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def _latest_dset_meta() -> Optional[Path]:
    """Return the most recent dset_meta_only_*.csv in project root."""
    candidates = sorted(PROJECT_ROOT.glob("dset_meta_only_*.csv"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Section collectors
# ---------------------------------------------------------------------------

def collect_training_curves(dry_run: bool) -> dict[str, str]:
    """
    (a) Copy training_history.csv → artifacts/vkr_data/training_curves.csv
    keeping only the columns needed for Fig 4.2.

    Returns:
        Status record for the README table.
    """
    key = "training_curves.csv"
    out = VKR_DATA / key
    section = "a) Кривые обучения LoRA (рис. 4.2)"
    needed_cols = ["epoch", "train_loss", "val_loss", "macro_f1"]

    run_dir = _latest_lora_run()
    if run_dir is None:
        log.info(_miss(f"{section} — lora_tune/models/ не найден или пуст"))
        return {"dataset": key, "status": "НЕ НАЙДЕНО", "source": "—",
                "action": "Запустить scripts/train/main_fit_clf_and_reg.py или lora trainer"}

    src = run_dir / "training_history.csv"
    with open(src, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Keep only essential columns (add others if present)
    available = set(rows[0].keys()) if rows else set()
    export_cols = [c for c in needed_cols if c in available]
    # Also keep val_acc / weighted_f1 if present
    for extra in ["val_acc", "weighted_f1"]:
        if extra in available:
            export_cols.append(extra)

    exported = [{c: r[c] for c in export_cols} for r in rows]
    _write_csv(out, exported, dry_run=dry_run)

    log.info(
        _ok(
            f"{section}\n"
            f"     src: {src.relative_to(PROJECT_ROOT)}\n"
            f"     эпох: {len(rows)}, колонки: {export_cols}"
        )
    )
    return {
        "dataset": key,
        "status": "НАЙДЕНО",
        "source": str(src.relative_to(PROJECT_ROOT)),
        "action": "—",
    }


def collect_confusion_and_per_class(dry_run: bool) -> dict[str, str]:
    """
    (b) Build confusion_matrix.csv and per_class_metrics.csv from
    predictions_full.csv + eval_results.json in the latest lora run.

    Returns:
        Status record for the README table.
    """
    run_dir = _latest_lora_run()
    section = "b) Матрица ошибок + per-class метрики (рис. 4.1 / табл.)"

    # ---- per-class metrics from eval_results.json ----
    per_class_out = VKR_DATA / "per_class_metrics.csv"
    cm_out = VKR_DATA / "confusion_matrix.csv"

    if run_dir is None:
        log.info(_miss(f"{section} — run_dir не найден"))
        return {"dataset": "confusion_matrix.csv + per_class_metrics.csv",
                "status": "НЕ НАЙДЕНО", "source": "—",
                "action": "Запустить scripts/train/eval_onnx_model.py"}

    eval_json = run_dir / "eval_results.json"
    pred_csv = run_dir / "predictions_full.csv"

    if not eval_json.exists():
        log.info(_miss(f"{section} — eval_results.json не найден в {run_dir.name}"))
        return {"dataset": "confusion_matrix.csv + per_class_metrics.csv",
                "status": "НЕ НАЙДЕНО", "source": "—",
                "action": "Запустить scripts/train/eval_onnx_model.py"}

    with open(eval_json, encoding="utf-8") as f:
        eval_data = json.load(f)

    # Parse classification_report string into per-class rows
    report_str: str = eval_data.get("classification_report", "")
    per_class_rows: list[dict] = []
    for line in report_str.splitlines():
        parts = line.split()
        # lines like: "   машина   1.00  0.96  0.98   27"
        if len(parts) >= 5:
            # last 4 tokens are precision recall f1 support
            try:
                support = int(parts[-1])
                f1 = float(parts[-2])
                recall = float(parts[-3])
                precision = float(parts[-4])
                label = " ".join(parts[:-4])
                if label and not label.startswith("accuracy") \
                        and not label.startswith("macro") \
                        and not label.startswith("weighted"):
                    per_class_rows.append({
                        "class": label,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "support": support,
                    })
            except ValueError:
                pass

    # Also try per_class block from eval_results
    if not per_class_rows and "per_class" in eval_data:
        for cls, vals in eval_data["per_class"].items():
            per_class_rows.append({
                "class": cls,
                "precision": "",
                "recall": "",
                "f1": "",
                "support": vals.get("support", ""),
                "accuracy": vals.get("accuracy", ""),
            })

    if per_class_rows:
        _write_csv(per_class_out, per_class_rows, dry_run=dry_run)

    # ---- confusion matrix from predictions_full.csv ----
    cm_rows: list[dict] = []
    if pred_csv.exists():
        with open(pred_csv, encoding="utf-8") as f:
            preds = list(csv.DictReader(f))

        # Collect unique labels in sorted order
        labels = sorted(set(r["true_label"] for r in preds))
        # Build matrix
        cm: dict[tuple[str, str], int] = {}
        for r in preds:
            key = (r["true_label"], r["pred_label"])
            cm[key] = cm.get(key, 0) + 1

        # Header row: true\pred | class1 | class2 ...
        for true_lbl in labels:
            row: dict[str, Any] = {"true \\ pred": true_lbl}
            for pred_lbl in labels:
                row[pred_lbl] = cm.get((true_lbl, pred_lbl), 0)
            cm_rows.append(row)

        _write_csv(cm_out, cm_rows, dry_run=dry_run)
    else:
        log.info(_warn(f"predictions_full.csv не найден — матрица из predictions не построена"))

    log.info(
        _ok(
            f"{section}\n"
            f"     src: {run_dir.relative_to(PROJECT_ROOT)}/\n"
            f"     per-class классов: {len(per_class_rows)}, "
            f"confusion matrix: {len(cm_rows)}×{len(cm_rows[0])-1 if cm_rows else 0}"
        )
    )
    return {
        "dataset": "confusion_matrix.csv + per_class_metrics.csv",
        "status": "НАЙДЕНО",
        "source": str(run_dir.relative_to(PROJECT_ROOT)),
        "action": "—",
    }


def collect_mahalanobis(dry_run: bool) -> dict[str, str]:
    """
    (c) Export Mahalanobis / cosine distance statistics + threshold τ
    from outlier_detector_info.json and outlier_detector.npy.

    Returns:
        Status record for the README table.
    """
    out = VKR_DATA / "mahalanobis_distances.csv"
    section = "c) OOD-калибровка — расстояния Махаланобиса / cosine (рис. 4.3)"

    # Prefer most recent lora run that has outlier_detector_info.json
    base = PROJECT_ROOT / "lora_tune" / "models"
    candidates = sorted(
        (d for d in base.iterdir()
         if d.is_dir() and (d / "outlier_detector_info.json").exists()),
        key=lambda d: d.name,
    ) if base.exists() else []

    # Also check artifacts/hybrid/
    hybrid_pkl = PROJECT_ROOT / "artifacts" / "hybrid" / "outlier_gate.pkl"

    if not candidates and not hybrid_pkl.exists():
        log.info(_miss(f"{section} — outlier_detector_info.json и outlier_gate.pkl не найдены"))
        return {"dataset": "mahalanobis_distances.csv", "status": "НЕ НАЙДЕНО",
                "source": "—",
                "action": "Запустить scripts/train/eval_onnx_model.py или "
                           "scripts/hybrid/train_outlier_gate.py"}

    rows: list[dict] = []

    if candidates:
        run_dir = candidates[-1]
        info_path = run_dir / "outlier_detector_info.json"
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        method = info.get("config", {}).get("method", "mahalanobis")
        global_threshold = info.get("threshold", None)
        stats = info.get("train_distance_stats", {})

        # Global summary row
        rows.append({
            "split": "in-distribution (global)",
            "method": method,
            "n_samples": info.get("class_counts", {}) and sum(info["class_counts"].values()),
            "mean": stats.get("mean", ""),
            "std": stats.get("std", ""),
            "median": stats.get("median", ""),
            "p90": stats.get("p90", ""),
            "p95": stats.get("p95", ""),
            "p99": stats.get("p99", ""),
            "min": stats.get("min", ""),
            "max": stats.get("max", ""),
            "threshold_tau": global_threshold,
            "source": str(info_path.relative_to(PROJECT_ROOT)),
        })

        # Also try to read per-class stats from .npy
        npy_path = run_dir / "outlier_detector.npy"
        if npy_path.exists():
            try:
                import numpy as np  # lazy import — may not be in sandbox
                arr = np.load(str(npy_path), allow_pickle=True)
                item = arr.item()
                if isinstance(item, dict) and "train_stats" in item:
                    ts = item["train_stats"]
                    for class_key, class_stats in ts.items():
                        if isinstance(class_stats, dict) and "mean_distance" in class_stats:
                            rows.append({
                                "split": f"in-distribution ({class_key})",
                                "method": method,
                                "n_samples": class_stats.get("n_samples", ""),
                                "mean": class_stats.get("mean_distance", ""),
                                "std": class_stats.get("std_distance", ""),
                                "median": "",
                                "p90": "",
                                "p95": "",
                                "p99": "",
                                "min": "",
                                "max": "",
                                "threshold_tau": class_stats.get("threshold", ""),
                                "source": str(npy_path.relative_to(PROJECT_ROOT)),
                            })
            except Exception as exc:  # noqa: BLE001
                log.info(_warn(f"Не удалось прочитать outlier_detector.npy: {exc}"))

        # OOD row from outlier_results.json
        res_path = run_dir / "outlier_results.json"
        if res_path.exists():
            with open(res_path, encoding="utf-8") as f:
                res = json.load(f)
            rows.append({
                "split": "OOD (outliers)",
                "method": method,
                "n_samples": res.get("outliers", {}).get("n_samples", ""),
                "mean": "",
                "std": "",
                "median": "",
                "p90": "",
                "p95": "",
                "p99": "",
                "min": "",
                "max": "",
                "threshold_tau": global_threshold,
                "source": str(res_path.relative_to(PROJECT_ROOT)),
            })

    if rows:
        _write_csv(out, rows, dry_run=dry_run)
        log.info(
            _ok(
                f"{section}\n"
                f"     src: {candidates[-1].relative_to(PROJECT_ROOT) if candidates else 'artifacts/hybrid/'}\n"
                f"     метод: {rows[0].get('method','?')}, "
                f"τ (глобальный): {rows[0].get('threshold_tau','?'):.4f}, "
                f"строк: {len(rows)}"
            )
        )
        return {"dataset": "mahalanobis_distances.csv", "status": "НАЙДЕНО",
                "source": str(candidates[-1].relative_to(PROJECT_ROOT)) if candidates else "—",
                "action": "—"}

    log.info(_miss(f"{section} — данные не удалось извлечь"))
    return {"dataset": "mahalanobis_distances.csv", "status": "НЕ НАЙДЕНО",
            "source": "—", "action": "Запустить scripts/hybrid/train_outlier_gate.py"}


def collect_memory_log(dry_run: bool) -> dict[str, str]:
    """
    (d) Export 24-h load test memory time-series → memory_24h.csv.
    Looks for JSON / CSV logs with RSS/memory fields under logs/.

    Returns:
        Status record for the README table.
    """
    out = VKR_DATA / "memory_24h.csv"
    section = "d) 24-часовой нагрузочный тест — RSS памяти (рис. 4.4)"

    # Search patterns
    search_dirs = [PROJECT_ROOT / "logs", PROJECT_ROOT / "src" / "logs"]
    memory_keywords = ["memory", "rss", "stress", "load", "24h", "stability"]

    found_files: list[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.is_file() and any(kw in f.name.lower() for kw in memory_keywords):
                found_files.append(f)

    # Also look for JSON log lines that contain memory_mb / rss fields
    candidate_logs: list[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.rglob("*.log"):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if "memory_mb" in text or "rss" in text.lower():
                    candidate_logs.append(f)
            except OSError:
                pass

    if not found_files and not candidate_logs:
        log.info(
            _miss(
                f"{section}\n"
                "     Лог 24-часового теста не найден.\n"
                "     Для получения: запустить нагрузочный тест с psutil-мониторингом\n"
                "     (например tests/stress_test.py или отдельный stress_runner.py),\n"
                "     записывающий timestamp + rss_mb в logs/memory_24h.csv."
            )
        )
        return {
            "dataset": "memory_24h.csv",
            "status": "НЕ НАЙДЕНО",
            "source": "—",
            "action": (
                "Создать/запустить stress runner: "
                "python tests/stress_test.py --duration 86400 --log logs/memory_24h.csv"
            ),
        }

    # --- Priority 1: try found CSV files directly (timestamp + rss_mb columns) ---
    MIN_DURATION_S = 3600  # at least 1 hour to count as a real load test
    for csv_file in found_files:
        if csv_file.suffix.lower() != ".csv":
            continue
        try:
            import csv as _csv
            with open(csv_file, newline="", encoding="utf-8", errors="ignore") as fh:
                reader = _csv.DictReader(fh)
                if reader.fieldnames is None:
                    continue
                cols = [c.strip().lower() for c in reader.fieldnames]
                if "timestamp" not in cols or "rss_mb" not in cols:
                    continue
                csv_rows = list(reader)
        except OSError:
            continue

        if not csv_rows:
            continue

        # Determine test duration from elapsed_s column (if present)
        duration_s = 0
        if "elapsed_s" in [c.strip().lower() for c in (reader.fieldnames or [])]:
            try:
                duration_s = int(float(csv_rows[-1].get("elapsed_s", 0)))
            except (ValueError, TypeError):
                duration_s = len(csv_rows) * 60  # rough estimate: 1 row/min

        is_full_24h = duration_s >= 86000  # allow a small margin for 24 h
        is_partial = duration_s >= MIN_DURATION_S

        if not is_partial and len(csv_rows) < 10:
            continue

        # Copy the CSV to vkr_data output
        if not dry_run:
            import shutil as _shutil
            _shutil.copy2(csv_file, out)
        else:
            # Still write a minimal output so downstream tools work
            _write_csv(out, [
                {"timestamp": r.get("timestamp", ""), "rss_mb": r.get("rss_mb", "")}
                for r in csv_rows[:5]
            ], dry_run=dry_run)

        status = "НАЙДЕНО" if is_full_24h else "НАЙДЕНО (частично)"
        dur_label = (
            f"{duration_s // 3600}ч {(duration_s % 3600) // 60}мин"
            if duration_s else f"{len(csv_rows)} строк"
        )
        log.info(
            _ok(
                f"{section}\n"
                f"     src: {csv_file.relative_to(PROJECT_ROOT)}\n"
                f"     точек: {len(csv_rows)}, длительность: {dur_label}"
            )
        )
        return {
            "dataset": "memory_24h.csv",
            "status": status,
            "source": str(csv_file.relative_to(PROJECT_ROOT)),
            "action": "—",
        }

    # --- Priority 2: try to parse memory_mb from JSON log lines ---
    rows: list[dict] = []
    for log_file in candidate_logs:
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "memory_mb" in entry:
                        rows.append({
                            "timestamp": entry.get("timestamp", i),
                            "rss_mb": entry["memory_mb"],
                            "source": str(log_file.relative_to(PROJECT_ROOT)),
                        })
                except (json.JSONDecodeError, TypeError):
                    pass

    if rows:
        _write_csv(out, rows, dry_run=dry_run)
        log.info(
            _ok(
                f"{section}\n"
                f"     src: {candidate_logs[0].relative_to(PROJECT_ROOT)}\n"
                f"     точек памяти: {len(rows)}"
            )
        )
        return {"dataset": "memory_24h.csv", "status": "НАЙДЕНО (частично)",
                "source": str(candidate_logs[0].relative_to(PROJECT_ROOT)), "action": "—"}

    # Found files but couldn't parse
    log.info(
        _warn(
            f"{section}\n"
            f"     Файлы найдены ({[f.name for f in found_files + candidate_logs]}),\n"
            "     но не содержат ожидаемых колонок (timestamp + rss_mb).\n"
            "     Нужен полноценный 24-часовой нагрузочный тест."
        )
    )
    return {
        "dataset": "memory_24h.csv",
        "status": "НЕ НАЙДЕНО (нет 24h теста)",
        "source": "logs/app.log (краткие записи)",
        "action": (
            "Запустить: python tests/stress_test.py --duration 86400 "
            "--output logs/memory_24h.csv"
        ),
    }


def collect_benchmarks(dry_run: bool) -> dict[str, str]:
    """
    (e) Consolidate latency/F1/size benchmarks → benchmarks_summary.csv.
    Sources: onnx_model/quant_benchmark/benchmark_results.json,
             artifacts/benchmarks/thesis_stats.json.

    Returns:
        Status record for the README table.
    """
    out = VKR_DATA / "benchmarks_summary.csv"
    section = "e) Бенчмарки PyTorch / ONNX FP32 / ONNX INT8 (табл. 4.2 / промпт 06)"

    quant_json = PROJECT_ROOT / "onnx_model" / "quant_benchmark" / "benchmark_results.json"
    thesis_json = PROJECT_ROOT / "artifacts" / "benchmarks" / "thesis_stats.json"

    if not quant_json.exists() and not thesis_json.exists():
        log.info(_miss(f"{section} — benchmark_results.json и thesis_stats.json не найдены"))
        return {"dataset": "benchmarks_summary.csv", "status": "НЕ НАЙДЕНО",
                "source": "—", "action": "Запустить scripts/train/benchmark_quantization.py"}

    rows: list[dict] = []

    # Primary: quant_benchmark/benchmark_results.json
    if quant_json.exists():
        with open(quant_json, encoding="utf-8") as f:
            qdata = json.load(f)

        raw_rows = qdata.get("rows", [])
        for r in raw_rows:
            fmt = r.get("Format", "")
            lat = r.get("Latency (ms)", "")
            size = r.get("Size (MB)", "")
            conf = r.get("Confidence", "")
            speedup = r.get("Speedup", "")
            rows.append({
                "model": fmt,
                "size_mb": size,
                "latency_ms": lat,
                "speedup": speedup,
                "confidence_sample": conf,
                "macro_f1": "",
                "source": str(quant_json.relative_to(PROJECT_ROOT)),
            })

    # Supplement with thesis_stats.json (has macro_f1 for LoRA ONNX INT8 and baselines)
    if thesis_json.exists():
        with open(thesis_json, encoding="utf-8") as f:
            tdata = json.load(f)

        lat_bench = tdata.get("latency_benchmark", {})
        table_rows = tdata.get("table_4_3", [])

        for tr in table_rows:
            method = tr.get("method", "")
            f1 = tr.get("macro_f1", "")
            lat = tr.get("lat_mean_ms", "")
            # Try to merge into existing rows
            merged = False
            for existing in rows:
                em = existing["model"].lower()
                if ("pytorch" in method.lower() and "pytorch" in em) \
                        or ("int8" in method.lower() and "int8" in em) \
                        or ("fp32" in method.lower() and "fp32" in em and "onnx" in em):
                    if not existing["macro_f1"]:
                        existing["macro_f1"] = f1
                    merged = True
                    break
            if not merged:
                rows.append({
                    "model": method,
                    "size_mb": "",
                    "latency_ms": lat,
                    "speedup": "",
                    "confidence_sample": "",
                    "macro_f1": f1,
                    "source": str(thesis_json.relative_to(PROJECT_ROOT)),
                })

    if not rows:
        log.info(_miss(f"{section} — не удалось распарсить данные"))
        return {"dataset": "benchmarks_summary.csv", "status": "НЕ НАЙДЕНО",
                "source": "—", "action": "Запустить scripts/train/benchmark_quantization.py"}

    _write_csv(out, rows, dry_run=dry_run)
    log.info(
        _ok(
            f"{section}\n"
            f"     src: {quant_json.relative_to(PROJECT_ROOT) if quant_json.exists() else thesis_json.relative_to(PROJECT_ROOT)}\n"
            f"     вариантов моделей: {len(rows)}"
        )
    )
    return {"dataset": "benchmarks_summary.csv", "status": "НАЙДЕНО",
            "source": f"{quant_json.relative_to(PROJECT_ROOT) if quant_json.exists() else '—'} "
                      f"+ {thesis_json.relative_to(PROJECT_ROOT) if thesis_json.exists() else '—'}",
            "action": "—"}


def collect_corpus_stats(dry_run: bool) -> dict[str, str]:
    """
    (f) Count samples per class from latest dset_meta_only_*.csv →
    corpus_by_class.csv.

    Returns:
        Status record for the README table.
    """
    out = VKR_DATA / "corpus_by_class.csv"
    section = "f) Состав корпуса по классам (табл. состава)"

    meta_csv = _latest_dset_meta()
    if meta_csv is None:
        log.info(_miss(f"{section} — dset_meta_only_*.csv не найден в корне проекта"))
        return {"dataset": "corpus_by_class.csv", "status": "НЕ НАЙДЕНО",
                "source": "—",
                "action": "Запустить scripts/clf_dset/build_dset_meta.py"}

    with open(meta_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_meta = list(reader)

    # Detect column name (may be 'class' or 'label')
    col = None
    for candidate in ("class", "label", "word", "category"):
        if candidate in (reader.fieldnames or []):
            col = candidate
            break

    # fieldnames may already be consumed; reload if needed
    if col is None:
        with open(meta_csv, encoding="utf-8") as f:
            r2 = csv.DictReader(f)
            for cand in ("class", "label", "word", "category"):
                if cand in (r2.fieldnames or []):
                    col = cand
                    break

    if col is None:
        log.info(_warn(f"{section} — не найдена колонка класса в {meta_csv.name}"))
        return {"dataset": "corpus_by_class.csv", "status": "ПРЕДУПРЕЖДЕНИЕ",
                "source": str(meta_csv.relative_to(PROJECT_ROOT)),
                "action": "Проверить структуру dset_meta_only CSV"}

    counts: dict[str, int] = {}
    for r in rows_meta:
        lbl = r.get(col, "").strip()
        counts[lbl] = counts.get(lbl, 0) + 1

    export_rows = [
        {"class": lbl, "n_samples": cnt, "pct": f"{cnt/len(rows_meta)*100:.1f}"}
        for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1])
    ]

    _write_csv(out, export_rows, dry_run=dry_run)
    log.info(
        _ok(
            f"{section}\n"
            f"     src: {meta_csv.relative_to(PROJECT_ROOT)}\n"
            f"     классов: {len(export_rows)}, всего сэмплов: {len(rows_meta)}"
        )
    )
    return {"dataset": "corpus_by_class.csv", "status": "НАЙДЕНО",
            "source": str(meta_csv.relative_to(PROJECT_ROOT)), "action": "—"}


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------

def write_readme(records: list[dict[str, str]], dry_run: bool) -> None:
    """Write artifacts/vkr_data/_README.md with a status table."""
    readme_path = VKR_DATA / "_README.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# VKR Data — статус сбора данных",
        "",
        f"Сгенерировано: `collect_vkr_data.py` — {ts}",
        "",
        "| Набор данных | Статус | Источник | Действие |",
        "|---|---|---|---|",
    ]
    for rec in records:
        dataset = rec.get("dataset", "")
        status  = rec.get("status", "")
        source  = rec.get("source", "—")
        action  = rec.get("action", "—")
        emoji   = "✅" if "НАЙДЕНО" in status and "НЕ" not in status else "❌"
        lines.append(f"| `{dataset}` | {emoji} {status} | `{source}` | {action} |")

    lines += [
        "",
        "## Описание наборов данных",
        "",
        "| Файл | Использование |",
        "|---|---|",
        "| `training_curves.csv` | Рис. 4.2 — кривые обучения LoRA (loss/F1 по эпохам) |",
        "| `confusion_matrix.csv` | Рис. 4.1 — матрица ошибок |",
        "| `per_class_metrics.csv` | Табл. — precision/recall/F1/support по классам (промпт 10) |",
        "| `mahalanobis_distances.csv` | Рис. 4.3 — распределение расстояний OOD-детектора, порог τ |",
        "| `memory_24h.csv` | Рис. 4.4 — RSS памяти за 24 ч (промпт 07) |",
        "| `benchmarks_summary.csv` | Табл. 4.2 — задержка/F1/размер PyTorch FP32 / ONNX FP32 / INT8 (промпт 06) |",
        "| `corpus_by_class.csv` | Табл. состава корпуса — n_samples и % по классу (промпт 10) |",
        "",
        "## Для недостающих данных",
        "",
        "```",
        "# Классификационный отчёт (если нет eval_results.json)",
        "python scripts/train/eval_onnx_model.py",
        "",
        "# Бенчмарки ONNX INT8 vs FP32 (если нет benchmark_results.json)",
        "python scripts/train/benchmark_quantization.py",
        "",
        "# 24-часовой нагрузочный тест (memory_24h.csv)",
        "# Нужен отдельный stress-runner с psutil-логированием:",
        "# python tests/stress_test.py --duration 86400 --output logs/memory_24h.csv",
        "```",
    ]

    content = "\n".join(lines) + "\n"

    if dry_run:
        log.info("[dry-run] would write %s", readme_path.relative_to(PROJECT_ROOT))
        return

    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(content, encoding="utf-8")
    log.info("\n  → отчёт: %s", readme_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="VKR Data Auditor — collect and export VKR artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without writing any files.",
    )
    args = parser.parse_args()

    dry_run: bool = args.dry_run

    log.info("=" * 70)
    log.info("VKR Data Auditor  |  PROJECT_ROOT = %s", PROJECT_ROOT)
    if dry_run:
        log.info("  *** РЕЖИМ DRY-RUN: файлы не записываются ***")
    log.info("=" * 70)

    records: list[dict[str, str]] = []

    log.info("\n─── (a) Кривые обучения ────────────────────────────────────────────")
    records.append(collect_training_curves(dry_run))

    log.info("\n─── (b) Матрица ошибок + per-class метрики ─────────────────────────")
    records.append(collect_confusion_and_per_class(dry_run))

    log.info("\n─── (c) OOD-детектор — расстояния ──────────────────────────────────")
    records.append(collect_mahalanobis(dry_run))

    log.info("\n─── (d) 24-часовой нагрузочный тест — RSS памяти ───────────────────")
    records.append(collect_memory_log(dry_run))

    log.info("\n─── (e) Бенчмарки моделей ──────────────────────────────────────────")
    records.append(collect_benchmarks(dry_run))

    log.info("\n─── (f) Состав корпуса по классам ──────────────────────────────────")
    records.append(collect_corpus_stats(dry_run))

    log.info("\n─── Финальный отчёт ────────────────────────────────────────────────")
    write_readme(records, dry_run)

    # Summary table to stdout
    found  = sum(1 for r in records if "НАЙДЕНО" in r["status"] and "НЕ" not in r["status"])
    total  = len(records)
    log.info("")
    log.info("┌─────────────────────────────────────────────┐")
    log.info("│  Итог:  %d / %d наборов собрано               │", found, total)
    log.info("└─────────────────────────────────────────────┘")
    log.info("")
    for rec in records:
        ok = "✅" if "НАЙДЕНО" in rec["status"] and "НЕ" not in rec["status"] else "❌"
        log.info("  %s  %-38s  %s", ok, rec["dataset"], rec["status"])
    log.info("")


if __name__ == "__main__":
    main()
