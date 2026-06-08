"""
benchmark_compare_all.py — Сводная таблица сравнения всех методов.

Собирает результаты из JSON-файлов, сгенерированных:
  - benchmark_mfcc_svm.py        → mfcc_svm_results.json / mfcc_svm_noisy_results.json
  - benchmark_whisper.py         → whisper_tiny_results.json / whisper_tiny_noisy_results.json
  - eval_onnx_model.py           → eval_onnx_int8_results.json  (уже в lora_tune/…)

Если файл отсутствует — помечает ячейку как [NOT RUN].

Usage
-----
    # Сначала запустите все бенчмарки:
    python scripts/train/benchmark_mfcc_svm.py --data_csv dset_meta_only_2026-05-09_10-27-42.csv --noisy_test
    python scripts/train/benchmark_whisper.py  --data_csv dset_meta_only_2026-05-09_10-27-42.csv --noisy_test
    # (eval_onnx_int8_results.json уже существует)

    # Затем сводная таблица:
    python scripts/train/benchmark_compare_all.py

Output
------
artifacts/benchmarks/sota_comparison.json   — машиночитаемые результаты
artifacts/benchmarks/sota_comparison.txt    — таблица для ВКР (plaintext)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCH_DIR = _PROJECT_ROOT / "artifacts" / "benchmarks"
LORA_RUN_DIR = _PROJECT_ROOT / "lora_tune" / "models" / "run_2026-04-30_23-34-27"

# ── Source file registry ──────────────────────────────────────────────────────
# (method_name, clean_json, noisy_json)
SOURCES = [
    (
        "MFCC + SVM",
        BENCH_DIR / "mfcc_svm_results.json",
        BENCH_DIR / "mfcc_svm_noisy_results.json",
    ),
    (
        "Whisper-tiny (zero-shot)",
        BENCH_DIR / "whisper_tiny_results.json",
        BENCH_DIR / "whisper_tiny_noisy_results.json",
    ),
    (
        "LoRA-Wav2Vec2 + ONNX INT8",
        LORA_RUN_DIR / "eval_onnx_int8_results.json",
        None,   # noisy version not yet generated separately
    ),
]


def _load(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "[NOT RUN]"
    return f"{value:.{decimals}f}"


def build_row(name: str, clean: Optional[Dict], noisy: Optional[Dict]) -> Dict:
    return {
        "method":           name,
        "f1_clean":         clean.get("macro_f1")        if clean else None,
        "f1_noisy_12dB":    noisy.get("macro_f1")        if noisy else None,
        "latency_ms":       clean.get("mean_latency_ms") if clean else None,
        "n_samples":        clean.get("n_samples")       if clean else None,
        "accuracy_clean":   clean.get("accuracy")        if clean else None,
    }


def print_table(rows: list) -> str:
    header = (
        f"{'Метод':<35} {'F1 (clean)':>12} {'F1 (SNR 12 dБ)':>15} {'Latency мс':>12} {'N':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        lines.append(
            f"{r['method']:<35} "
            f"{fmt(r['f1_clean'], 4):>12} "
            f"{fmt(r['f1_noisy_12dB'], 4):>15} "
            f"{fmt(r['latency_ms'], 1):>12} "
            f"{str(r['n_samples'] or '[NOT RUN]'):>6}"
        )
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, clean_path, noisy_path in SOURCES:
        clean = _load(clean_path)
        noisy = _load(noisy_path)
        rows.append(build_row(name, clean, noisy))

    table_str = print_table(rows)
    print("\n" + table_str + "\n")

    # ── Save machine-readable ─────────────────────────────────────────────────
    out_json = BENCH_DIR / "sota_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON → {out_json}")

    # ── Save plaintext table ──────────────────────────────────────────────────
    out_txt = BENCH_DIR / "sota_comparison.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"Saved TXT  → {out_txt}")

    # ── Print instructions for missing results ────────────────────────────────
    missing = [r["method"] for r in rows if r["f1_clean"] is None]
    if missing:
        print("\n[!] Missing results for:")
        for m in missing:
            print(f"    - {m}")
        print(
            "\nRun the missing benchmarks first:\n"
            "  python scripts/train/benchmark_mfcc_svm.py "
            "--data_csv dset_meta_only_2026-05-09_10-27-42.csv --noisy_test\n"
            "  python scripts/train/benchmark_whisper.py  "
            "--data_csv dset_meta_only_2026-05-09_10-27-42.csv --noisy_test"
        )


if __name__ == "__main__":
    main()
