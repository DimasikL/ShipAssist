"""
scripts/vkr/plot_f1_vs_snr.py — Построение рис. 4.5: F1 vs SNR.

Читает artifacts/benchmarks/f1_vs_snr.csv (результат experiment_f1_vs_snr.py)
и строит кривую macro-F1 в зависимости от ОСШ для всех методов.
Сохраняет PNG 300 dpi в artifacts/plots/vkr_figures/fig_4_5_f1_vs_snr.png.

Usage:
    cd <PROJECT_ROOT>
    python scripts/vkr/plot_f1_vs_snr.py [--csv artifacts/benchmarks/f1_vs_snr.csv]

Notes:
    - Если CSV не найден, выводит подсказку запустить experiment_f1_vs_snr.py.
    - Подписи на русском, шрифт DejaVu (по умолчанию matplotlib).
    - Горизонтальная пунктирная линия F1 = 0.90 — граница приемлемой точности.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CSV  = PROJECT_ROOT / "artifacts" / "benchmarks" / "f1_vs_snr.csv"
DEFAULT_PLOT = PROJECT_ROOT / "artifacts" / "plots" / "vkr_figures" / "fig_4_5_f1_vs_snr.png"


def plot(csv_path: Path, save_path: Path) -> None:
    """Прочитать CSV и построить кривую F1 vs SNR.

    Args:
        csv_path:  Путь к f1_vs_snr.csv.
        save_path: Путь для сохранения PNG.

    Raises:
        FileNotFoundError: если CSV не найден.
        ValueError: если DataFrame пуст или не содержит нужных столбцов.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    # -- Загрузка данных -------------------------------------------------------
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV не найден: {csv_path}\n"
            "Сначала запустите: python scripts/vkr/experiment_f1_vs_snr.py"
        )

    df = pd.read_csv(csv_path)
    required_cols = {"snr_db", "method", "macro_f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {missing}")
    if df.empty:
        raise ValueError("CSV пуст.")

    logger.info("Загружено %d строк из %s", len(df), csv_path)

    # -- Настройка стиля -------------------------------------------------------
    style_map = {
        "LoRA-Wav2Vec2 ONNX INT8": dict(
            color="#1f77b4", lw=2.5, marker="o", ms=7,
            label="LoRA-Wav2Vec2 + ONNX INT8 (предложенный)", zorder=5
        ),
        "MFCC + SVM": dict(
            color="#ff7f0e", lw=1.8, marker="s", ms=6, ls="--",
            label="MFCC + SVM (базовая линия 1)"
        ),
        "Whisper-tiny": dict(
            color="#2ca02c", lw=1.8, marker="^", ms=6, ls="-.",
            label="Whisper-tiny (базовая линия 2)"
        ),
    }

    # -- Построение ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    methods = df["method"].unique()
    for method in methods:
        sub = df[df["method"] == method].copy()
        sub = sub.sort_values("snr_db", ascending=False)

        # inf (clean) → 22 для размещения на оси
        x_vals = sub["snr_db"].replace(float("inf"), 22.0).tolist()
        y_vals = sub["macro_f1"].tolist()

        kw = style_map.get(method, dict(lw=1.8, marker="D", ms=6, label=method))
        ax.plot(x_vals, y_vals, **kw)

        # Аннотация последней точки (−2 дБ)
        if x_vals:
            last_x, last_y = x_vals[-1], y_vals[-1]
            if not (isinstance(last_y, float) and np.isnan(last_y)):
                ax.annotate(
                    f"{last_y:.2f}",
                    xy=(last_x, last_y),
                    xytext=(last_x + 0.4, last_y + 0.02),
                    fontsize=9,
                    color=kw.get("color", "black"),
                )

    # -- Порог F1 = 0.90 -------------------------------------------------------
    ax.axhline(0.90, color="gray", lw=1.2, ls=":", zorder=1)
    ax.text(21.5, 0.905, "F1 = 0,90", fontsize=9, color="gray", va="bottom")

    # -- Ось X -----------------------------------------------------------------
    xticks  = [22, 20, 15, 12, 10, 8, 5, 2, 0, -2]
    xlabels = ["чистый", "20", "15", "12", "10", "8", "5", "2", "0", "−2"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.invert_xaxis()  # чистый слева → −2 дБ справа (ухудшение SNR)

    # -- Оси и заголовок -------------------------------------------------------
    ax.set_xlabel("ОСШ, дБ", fontsize=13, labelpad=8)
    ax.set_ylabel("Macro-F1", fontsize=13, labelpad=8)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=11)

    ax.set_title(
        "Рисунок 4.5 — Зависимость macro-F1 от уровня ОСШ",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5, -0.12,
        "Тестовая выборка: 123 записи, 5 дикторов (speaker-disjoint). "
        "Шум: гауссовский (seed=42, SNR-controlled).",
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )

    ax.grid(True, alpha=0.35, linestyle=":")
    ax.legend(fontsize=11, loc="lower left", framealpha=0.9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    # -- Сохранение ------------------------------------------------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("График сохранён → %s", save_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Построить рис. 4.5: F1 vs SNR.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Путь к f1_vs_snr.csv (по умолчанию: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_PLOT,
        help=f"Путь для сохранения PNG (по умолчанию: {DEFAULT_PLOT})",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа CLI."""
    args = _parse_args()
    plot(args.csv, args.out)


if __name__ == "__main__":
    main()
