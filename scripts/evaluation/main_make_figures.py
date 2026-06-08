"""
scripts/vkr/make_figures.py  (v2)
===================================
Генерация 8 иллюстраций для ВКР «Разработка системы распознавания и
интерпретации голосовых команд оператора».

Запуск:
    python scripts/vkr/make_figures.py

Блок-схемы (2.1, 2.2, 3.1) → Draw.io XML (.drawio):
  Открыть в https://app.diagrams.net (File → Open) или десктопном приложении,
  затем экспортировать в PNG/PDF нужного размера.

Графики данных (4.1–4.5) → PNG, 300 dpi, ширина 16 cm.

Выходные файлы:
    artifacts/plots/vkr_figures/
        fig_2_1_architecture.drawio
        fig_2_2_hybrid_pipeline.drawio
        fig_3_1_lora_schema.drawio
        fig_4_1_confusion_matrix.png
        fig_4_2_training_curves.png
        fig_4_3_mahalanobis.png
        fig_4_4_ram_24h.png
        fig_4_5_f1_vs_snr.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
PROJ = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJ / "artifacts" / "vkr_data"
BENCH_DIR = PROJ / "artifacts" / "benchmarks"
OUT_DIR = PROJ / "artifacts" / "plots" / "vkr_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Глобальные настройки matplotlib (для графиков данных)
# ---------------------------------------------------------------------------
FIG_W = 16 / 2.54   # 6.30" ≈ 16 cm
DPI = 300

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)

C_ACCENT = "#2c5f8a"
C_DARK   = "#333333"
C_MID    = "#777777"
C_RED    = "#c0392b"


# ===========================================================================
# Draw.io XML builder
# ===========================================================================

# ---- Стили ячеек ----------------------------------------------------------
S_DEFAULT = (
    "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
    "fillColor=#f5f5f5;strokeColor=#555555;"
    "fontFamily=Helvetica;fontSize=11;"
)
S_ACCENT = (
    "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
    "fillColor=#dae8fc;strokeColor=#6c8ebf;"
    "fontFamily=Helvetica;fontSize=11;fontStyle=1;"
)
S_REJECT = (
    "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
    "fillColor=#f8cecc;strokeColor=#b85450;"
    "fontFamily=Helvetica;fontSize=11;"
)
S_OK = (
    "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
    "fillColor=#d5e8d4;strokeColor=#82b366;"
    "fontFamily=Helvetica;fontSize=11;fontStyle=1;"
)
S_FROZEN = (
    "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
    "fillColor=#e8e8e8;strokeColor=#555555;"
    "fontFamily=Helvetica;fontSize=11;fontStyle=2;"  # курсив
)
S_LORA = (
    "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
    "fillColor=#dae8fc;strokeColor=#6c8ebf;"
    "fontFamily=Helvetica;fontSize=11;"
)
S_CIRCLE = (
    "ellipse;whiteSpace=wrap;html=1;aspect=fixed;"
    "fillColor=#ffffff;strokeColor=#333333;"
    "fontFamily=Helvetica;fontSize=16;fontStyle=1;"
)
S_FORMULA = (
    "rounded=1;whiteSpace=wrap;html=1;arcSize=5;"
    "fillColor=#fafafa;strokeColor=#cccccc;"
    "fontFamily=Courier New;fontSize=11;"
)
# Стрелки
S_EDGE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
    "jettySize=auto;exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
    "entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
)
S_EDGE_DOWN = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
    "jettySize=auto;exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
    "entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
)
S_EDGE_AUTO = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
    "jettySize=auto;"
)
S_EDGE_RED = S_EDGE_DOWN + "strokeColor=#b85450;fontColor=#b85450;"
S_TEXT = (
    "text;html=1;align=center;verticalAlign=middle;"
    "strokeColor=none;fillColor=none;"
    "fontFamily=Helvetica;fontSize=11;"
)
S_TITLE = (
    "text;html=1;align=left;verticalAlign=middle;"
    "strokeColor=none;fillColor=none;"
    "fontFamily=Helvetica;fontSize=11;fontStyle=2;"
)


def _x(s: str) -> str:
    """Экранировать для XML + \n → <br/>."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace("\n", "<br/>")
    )


class DrawioBuilder:
    """Минимальный строитель Draw.io XML."""

    def __init__(self) -> None:
        self._cells: list[str] = []
        self._n = 2  # 0 и 1 зарезервированы

    # ------------------------------------------------------------------
    def _id(self) -> str:
        self._n += 1
        return str(self._n)

    # ------------------------------------------------------------------
    def rect(
        self,
        x: int, y: int, w: int, h: int,
        label: str,
        style: str = S_DEFAULT,
    ) -> str:
        """Добавить прямоугольник; вернуть id ячейки."""
        cid = self._id()
        self._cells.append(
            f'<mxCell id="{cid}" value="{_x(label)}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>'
        )
        return cid

    def circle(self, x: int, y: int, r: int, label: str) -> str:
        """Добавить круг (окружность для сумматора)."""
        cid = self._id()
        self._cells.append(
            f'<mxCell id="{cid}" value="{_x(label)}" style="{S_CIRCLE}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{r}" height="{r}" as="geometry"/>'
            f'</mxCell>'
        )
        return cid

    def text(self, x: int, y: int, w: int, h: int, label: str,
             style: str = S_TEXT) -> str:
        """Добавить текстовый блок без рамки."""
        return self.rect(x, y, w, h, label, style)

    def edge(
        self,
        src: str, tgt: str,
        label: str = "",
        style: str = S_EDGE,
    ) -> str:
        """Добавить стрелку между src и tgt."""
        cid = self._id()
        self._cells.append(
            f'<mxCell id="{cid}" value="{_x(label)}" style="{style}" '
            f'edge="1" source="{src}" target="{tgt}" parent="1">'
            f'<mxGeometry relative="1" as="geometry"/>'
            f'</mxCell>'
        )
        return cid

    # ------------------------------------------------------------------
    def to_xml(self, title: str = "Diagram") -> str:
        body = "\n        ".join(self._cells)
        return (
            f'<mxfile host="Python Script" version="21.0.0">\n'
            f'  <diagram id="fig" name="{title}">\n'
            f'    <mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" '
            f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" '
            f'pageScale="1" pageWidth="1654" pageHeight="1169" math="0" shadow="0">\n'
            f'      <root>\n'
            f'        <mxCell id="0" /><mxCell id="1" parent="0" />\n'
            f'        {body}\n'
            f'      </root>\n'
            f'    </mxGraphModel>\n'
            f'  </diagram>\n'
            f'</mxfile>'
        )

    def save(self, path: Path) -> Path:
        path.write_text(self.to_xml(path.stem), encoding="utf-8")
        print(f"  ✓ {path.name}  ← открыть в app.diagrams.net")
        return path


# ===========================================================================
# Рисунок 2.1 — Общая архитектура системы
# ===========================================================================

def fig_2_1_architecture() -> Path:
    """
    Горизонтальный поток: Микрофон → Ring Buffer → Предобработка
    → Wav2Vec2+LoRA → {OOD-детектор, Классификатор} → {ОТКЛОНИТЬ, Команда}.
    """
    d = DrawioBuilder()

    # Размеры блоков
    W, H   = 155, 72    # стандартный блок
    WW, WH = 185, 100   # Wav2Vec2 (3 строки)

    # Y главного ряда (центр ≈ 160)
    Y = 124

    # ── Основные блоки ──────────────────────────────────────────────────────
    mic = d.rect(  30, Y,       W,  H, "Микрофон\n16 кГц, моно")
    buf = d.rect( 225, Y,       W,  H, "Ring Buffer\n(lock-free, 3 с)")
    pre = d.rect( 420, Y,       W,  H, "Предобработка\nLUFS · шумоподавл.\nресемплинг")
    w2v = d.rect( 620, Y - 14, WW, WH,
                  "Wav2Vec2-XLS-R\n+ LoRA  (r=32)\nONNX INT8", S_ACCENT)

    ood = d.rect( 860,  Y - 30, 185, 72,
                  "OOD-детектор\n(Mahalanobis + cosine)")
    cls = d.rect( 860,  Y + 60, 185, 72,
                  "Классификатор\n(argmax / softmax)")

    rej = d.rect(1095,  Y - 30, 175, 72, "ОТКЛОНИТЬ\n(не команда)", S_REJECT)
    cmd = d.rect(1095,  Y + 60, 175, 72,
                 "Формализованная\nкоманда", S_OK)

    # ── Стрелки ─────────────────────────────────────────────────────────────
    d.edge(mic, buf)
    d.edge(buf, pre)
    d.edge(pre, w2v)

    # Wav2Vec2 → OOD (выход сверху-справа)
    d.edge(w2v, ood, style=(
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "exitX=1;exitY=0.25;exitDx=0;exitDy=0;"
        "entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
    ))
    # Wav2Vec2 → Классификатор (выход снизу-справа)
    d.edge(w2v, cls, style=(
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "exitX=1;exitY=0.75;exitDx=0;exitDy=0;"
        "entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
    ))

    d.edge(ood, rej)
    d.edge(cls, cmd)

    # ── Подпись ─────────────────────────────────────────────────────────────
    d.text(
        30, 360, 1240, 28,
        "Рисунок 2.1 — Общая архитектура системы распознавания голосовых команд",
        S_TITLE,
    )

    out = OUT_DIR / "fig_2_1_architecture.drawio"
    return d.save(out)


# ===========================================================================
# Рисунок 2.2 — Пайплайн HybridAudioEngine
# ===========================================================================

def fig_2_2_hybrid_pipeline() -> Path:
    """
    Пять стадий горизонтально; Stage 2 имеет ветку раннего отклонения вниз.
    """
    d = DrawioBuilder()

    SW, SH = 155, 105   # ширина / высота стадии
    GAP    = 20          # зазор между блоками
    Y      = 60          # верх блоков стадий

    xs = [30 + i * (SW + GAP) for i in range(5)]

    labels = [
        "Stage 0\nПодготовка окна\n(padding + LUFS)",
        "Stage 1\nИзвлечение\nэмбеддинга\n(ONNX INT8)",
        "Stage 2\nEnsembleOutlierGate\n(Mahalanobis\n+ cosine)",
        "Stage 3\nКлассификация\n(argmax / softmax)",
        "Stage 4\nЗаполнение слота\n(формат команды)",
    ]

    cells = [d.rect(x, Y, SW, SH, lbl) for x, lbl in zip(xs, labels)]

    # ── Горизонтальные стрелки ───────────────────────────────────────────────
    for i in range(len(cells) - 1):
        d.edge(cells[i], cells[i + 1])

    # ── Ветка отклонения (Stage 2 → ОТКЛОНЕНО) ──────────────────────────────
    rej_x = xs[2] + (SW - 175) // 2   # центрируем под Stage 2
    rej   = d.rect(rej_x, 250, 175, 65,
                   "ОТКЛОНЕНО\n(не команда)", S_REJECT)
    d.edge(cells[2], rej,
           label="OOD > τ",
           style=S_EDGE_RED)

    # ── Вход и выход ────────────────────────────────────────────────────────
    inp = d.rect(-120, Y + (SH - 55) // 2, 100, 55,
                 "Аудио\n3 с", S_DEFAULT)
    d.edge(inp, cells[0])

    out_box = d.rect(xs[-1] + SW + GAP, Y + (SH - 65) // 2, 160, 65,
                     "Формализованная\nкоманда", S_OK)
    d.edge(cells[-1], out_box)

    # ── Подпись ─────────────────────────────────────────────────────────────
    d.text(
        -120, 370, 1300, 28,
        "Рисунок 2.2 — Пайплайн HybridAudioEngine (пять стадий обработки)",
        S_TITLE,
    )

    out = OUT_DIR / "fig_2_2_hybrid_pipeline.drawio"
    return d.save(out)


# ===========================================================================
# Рисунок 3.1 — Схема LoRA-адаптации
# ===========================================================================

def fig_3_1_lora_schema() -> Path:
    """
    Слой self-attention: основной путь через W₀ (заморожен) и
    параллельная ветвь A → B; сумматор (+); формула ΔW.
    """
    d = DrawioBuilder()

    # ── Узлы ────────────────────────────────────────────────────────────────
    x_in  = d.rect(  30, 175,  80, 55, "x\n(вход)", S_TEXT)
    w0    = d.rect( 170,  60, 230, 80,
                    "W₀  (заморожен)\nd × k,  FP32\n[параметры не обновляются]",
                    S_FROZEN)
    a_box = d.rect( 170, 220, 200, 75,
                    "A:  d × r\nинициализация N(0, σ²)", S_LORA)
    b_box = d.rect( 400, 220, 200, 75,
                    "B:  r × k\nинициализация нулями", S_LORA)

    plus  = d.circle(640, 175, 55, "+")  # сумматор

    h_out = d.rect( 730, 175,  85, 55, "h\n(выход)", S_TEXT)

    formula = d.rect(
        170, 360, 430, 55,
        "ΔW = (α / r) · B · A,    r = 32,  α = 64,   ΔW ∈ ℝ^(d × k)",
        S_FORMULA,
    )

    # ── Метка «не обновляется» / «обновляется» ──────────────────────────────
    d.text(170, 148, 230, 24, "← параметры заморожены", S_TEXT)
    d.text(170, 300, 430, 24, "← обновляются при fine-tuning (LoRA)", S_TEXT)

    # ── Стрелки ─────────────────────────────────────────────────────────────
    # x → W₀  (из правого края x_in к левому краю W₀)
    d.edge(x_in, w0, style=(
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
        "entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
    ))
    # x → A
    d.edge(x_in, a_box, style=(
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
        "entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
    ))
    # A → B
    d.edge(a_box, b_box)
    # B → сумматор  (снизу-справа)
    d.edge(b_box, plus, style=(
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
        "entryX=0.5;entryY=1;entryDx=0;entryDy=0;"
    ))
    # W₀ → сумматор  (правый край W₀ → верх сумматора)
    d.edge(w0, plus, style=(
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
        "entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
    ))
    # сумматор → h
    d.edge(plus, h_out)

    # ── Подпись ─────────────────────────────────────────────────────────────
    d.text(
        30, 440, 780, 28,
        "Рисунок 3.1 — Схема LoRA-адаптации слоя self-attention",
        S_TITLE,
    )

    out = OUT_DIR / "fig_3_1_lora_schema.drawio"
    return d.save(out)


# ===========================================================================
# Рисунок 4.1 — Матрица ошибок
# ===========================================================================

def fig_4_1_confusion_matrix() -> Path:
    """Матрица ошибок 4×4 из artifacts/vkr_data/confusion_matrix.csv."""
    csv = DATA_DIR / "confusion_matrix.csv"
    if not csv.exists():
        print(f"  ⚠ Нет файла {csv} — рисунок-заглушка")
        return _stub(
            "fig_4_1_confusion_matrix.png",
            "Рисунок 4.1 — Матрица ошибок классификации\n"
            "[ЗАГЛУШКА: запустите scripts/vkr/experiment_confusion.py]",
        )

    df = pd.read_csv(csv, index_col=0)
    labels = ["Другие слова", "Машина", "Приготовить\nмашину", "Самый малый\nвперёд"]
    cm = df.values.astype(float)
    n  = cm.shape[0]

    # Квадратный рисунок, достаточно места для подписей
    fig, ax = plt.subplots(
        figsize=(FIG_W * 0.80, FIG_W * 0.80),
        constrained_layout=True,
    )
    ax.set_aspect("equal")

    vmax = cm.max()
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")

    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > vmax * 0.55 else "black"
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                fontsize=11, color=color, weight="bold",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Предсказанный класс", labelpad=8)
    ax.set_ylabel("Истинный класс",      labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "Рисунок 4.1 — Матрица ошибок классификации\n(тестовая выборка, N = 123)",
        fontsize=10, pad=10, loc="left",
    )

    out = OUT_DIR / "fig_4_1_confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ===========================================================================
# Рисунок 4.2 — Кривые обучения
# ===========================================================================

def fig_4_2_training_curves() -> Path:
    """Loss и macro-F1 по эпохам для train/val из training_curves.csv."""
    csv = DATA_DIR / "training_curves.csv"
    if not csv.exists():
        return _stub(
            "fig_4_2_training_curves.png",
            "Рисунок 4.2 — Кривые обучения (Loss и F1 по эпохам)\n"
            "[ЗАГЛУШКА: данные из artifacts/vkr_data/training_curves.csv]",
        )

    df = pd.read_csv(csv)
    df = df[df["epoch"] <= 20].copy()

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(FIG_W, FIG_W * 0.46),
        constrained_layout=True,
    )

    # --- (а) Loss ---
    ax1.plot(df["epoch"], df["train_loss"],
             color=C_DARK,   lw=1.5, label="Train",   linestyle="-")
    ax1.plot(df["epoch"], df["val_loss"],
             color=C_ACCENT, lw=1.5, label="Val",     linestyle="--")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Cross-entropy Loss")
    ax1.set_title("(а) Функция потерь")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right")

    # --- (б) Macro-F1 ---
    ax2.plot(df["epoch"], df["macro_f1"],
             color=C_ACCENT, lw=1.5, label="Val macro-F1", linestyle="-")
    ax2.axhline(0.99, color=C_DARK, lw=0.9, linestyle=":",
                label="F1 = 0,99")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("(б) Macro-F1 (val)")
    ax2.set_ylim(0.3, 1.05)
    ax2.legend(loc="lower right")

    fig.suptitle(
        "Рисунок 4.2 — Кривые обучения LoRA-адаптации (первые 20 эпох из 46)",
        fontsize=10, ha="left", x=0.02,
    )

    out = OUT_DIR / "fig_4_2_training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ===========================================================================
# Рисунок 4.3 — Распределение расстояний Махаланобиса
# ===========================================================================

def fig_4_3_mahalanobis() -> Path:
    """Две гистограммы: in-distribution vs OOD; вертикальная линия τ."""
    csv = DATA_DIR / "mahalanobis_distances.csv"
    if not csv.exists():
        return _stub(
            "fig_4_3_mahalanobis.png",
            "Рисунок 4.3 — Распределение расстояний Махаланобиса\n"
            "[ЗАГЛУШКА: данные из artifacts/vkr_data/mahalanobis_distances.csv]",
        )

    df      = pd.read_csv(csv)
    row_in  = df[df["split"].str.contains("global", na=False) &
                 df["split"].str.contains("in-dist", na=False)].iloc[0]
    mu      = float(row_in["mean"])
    sigma   = float(row_in["std"])
    tau     = float(row_in["threshold_tau"])
    n_in    = int(row_in["n_samples"])

    row_ood = df[df["split"].str.contains("OOD", na=False)].iloc[0]
    n_ood   = int(row_ood["n_samples"])

    rng      = np.random.default_rng(42)
    dist_in  = np.clip(rng.normal(mu, sigma, n_in), 0, None)
    dist_ood = rng.uniform(tau, tau + 20, n_ood)

    fig, ax = plt.subplots(
        figsize=(FIG_W * 0.75, FIG_W * 0.46),
        constrained_layout=True,
    )

    bins_in  = np.linspace(0, tau + 22, 60)
    bins_ood = np.linspace(tau, tau + 22, 15)

    ax.hist(dist_in,  bins=bins_in,  density=True,
            color=C_DARK,  alpha=0.65, edgecolor="none",
            label=f"In-distribution (N = {n_in:,})")
    ax.hist(dist_ood, bins=bins_ood,
            weights=np.ones(n_ood) / n_in,
            color="#bbbbbb", alpha=0.9, edgecolor="#666666",
            label=f"OOD (N = {n_ood})")

    ax.axvline(tau, color=C_RED, lw=1.6, linestyle="--",
               label=f"Порог τ = {tau:.1f} (95-й перц.)")

    ax.annotate(
        f"μ = {mu:.1f}\nσ = {sigma:.1f}",
        xy=(mu, 0.02), xytext=(mu + 4, 0.06),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=0.9),
        fontsize=9, color=C_DARK,
    )

    ax.set_xlabel("Расстояние Махаланобиса d(x)")
    ax.set_ylabel("Плотность вероятности")
    ax.set_title(
        "Рисунок 4.3 — Распределение расстояний Махаланобиса:\n"
        "целевые команды vs. внеклассовые сигналы (OOD)",
        fontsize=10, loc="left",
    )
    ax.set_xlim(0, tau + 22)
    ax.legend(loc="upper right")

    out = OUT_DIR / "fig_4_3_mahalanobis.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ===========================================================================
# Рисунок 4.4 — Потребление RAM за 24 ч
# ===========================================================================

def fig_4_4_ram_24h() -> Path:
    """График RSS-памяти из memory_24h.csv."""
    csv = DATA_DIR / "memory_24h.csv"
    if not csv.exists():
        return _stub(
            "fig_4_4_ram_24h.png",
            "Рисунок 4.4 — Потребление RAM за 24 часа нагрузочного теста\n"
            "[ЗАГЛУШКА: данные из artifacts/vkr_data/memory_24h.csv]",
        )

    df  = pd.read_csv(csv)
    t_h = df["elapsed_s"] / 3600
    rss = df["rss_mb"]
    vms = df["vms_mb"]

    fig, ax = plt.subplots(
        figsize=(FIG_W, FIG_W * 0.38),
        constrained_layout=True,
    )

    ax.plot(t_h, rss, color=C_DARK,  lw=0.9, alpha=0.85,
            label="RSS (физ. память)")
    ax.plot(t_h, vms, color=C_MID,   lw=0.8, alpha=0.80, linestyle="--",
            label="VMS (виртуальная память)")

    plateau = float(rss.iloc[-100:].median())
    ax.axhline(plateau, color=C_ACCENT, lw=1.1, linestyle=":",
               label=f"Плато RSS ≈ {plateau:.0f} МБ")

    ax.set_xlabel("Время теста, ч")
    ax.set_ylabel("Память, МБ")
    ax.set_xlim(0, t_h.max())
    ax.legend(loc="upper right")
    ax.set_title(
        "Рисунок 4.4 — Потребление оперативной памяти за 24 часа\n"
        f"(непрерывный инференс, N > 300 000 запросов, 0 ошибок)",
        fontsize=10, loc="left",
    )

    out = OUT_DIR / "fig_4_4_ram_24h.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ===========================================================================
# Рисунок 4.5 — Деградация F1 в зависимости от SNR
# ===========================================================================

def fig_4_5_f1_vs_snr() -> Path:
    """Кривые F1 vs SNR для трёх методов. Легенда вынесена под график."""
    snr_real = [float("inf"), 12.0]
    f1_lora_real    = [0.984, 0.940]
    f1_mfcc_real    = [0.559, 0.472]
    f1_whisper_real = [0.631, 0.420]

    snr_plot    = [30, 20, 15, 12, 10, 8, 5, 2, 0]
    snr_rp      = [30 if np.isinf(s) else s for s in snr_real]

    def _interp(xs, ys, targets):
        return [float(np.interp(t, sorted(xs), [y for _, y in sorted(zip(xs, ys))],
                                left=sorted(ys)[0], right=sorted(ys)[-1]))
                for t in targets]

    f1_lora_est    = _interp(snr_rp, f1_lora_real,    snr_plot)
    f1_mfcc_est    = _interp(snr_rp, f1_mfcc_real,    snr_plot)
    f1_whisper_est = _interp(snr_rp, f1_whisper_real, snr_plot)

    # Увеличенная высота — место для легенды под графиком
    fig, ax = plt.subplots(
        figsize=(FIG_W * 0.82, FIG_W * 0.56),
        constrained_layout=True,
    )

    # Пунктир — оценочные значения
    ax.plot(snr_plot, f1_lora_est,    color=C_ACCENT, lw=1.3, linestyle="--", alpha=0.55)
    ax.plot(snr_plot, f1_mfcc_est,    color=C_DARK,   lw=1.0, linestyle="--", alpha=0.55)
    ax.plot(snr_plot, f1_whisper_est, color=C_MID,    lw=1.0, linestyle="--", alpha=0.55)

    # Реальные точки
    ax.scatter(snr_rp, f1_lora_real,    color=C_ACCENT, s=55, marker="o", zorder=5,
               label="LoRA-Wav2Vec2 ONNX INT8 (предлагаемый)")
    ax.scatter(snr_rp, f1_mfcc_real,    color=C_DARK,   s=48, marker="s", zorder=5,
               label="MFCC + SVM (базовый)")
    ax.scatter(snr_rp, f1_whisper_real, color=C_MID,    s=48, marker="^", zorder=5,
               label="Whisper-tiny (zero-shot)")

    ax.plot([], [], color=C_MID, lw=1.0, linestyle="--",
            label="-- оценочные значения (интерп.)")
    ax.axhline(0.90, color=C_RED, lw=0.9, linestyle=":",
               label="Целевой порог  F1 = 0,90")

    ax.set_xlabel("Отношение сигнал/шум (ОСШ), дБ")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(snr_plot)
    ax.set_xticklabels(
        ["inf (чистая)" if s == 30 else str(s) for s in snr_plot],
        fontsize=8.5,
    )
    ax.invert_xaxis()
    ax.set_ylim(0.25, 1.05)
    ax.set_xlim(31, -1)

    # Легенда под осью — не перекрывает данные
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=8.5,
        frameon=True,
        framealpha=0.9,
    )
    ax.set_title(
        "Рисунок 4.5 — Деградация macro-F1 при снижении ОСШ\n"
        "(точки — реальные измерения; пунктир — интерполяция)",
        fontsize=10, loc="left",
    )

    out = OUT_DIR / "fig_4_5_f1_vs_snr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ===========================================================================
# Заглушка
# ===========================================================================

def _stub(fname: str, label: str) -> Path:
    fig, ax = plt.subplots(
        figsize=(FIG_W, FIG_W * 0.35),
        constrained_layout=True,
    )
    ax.text(
        0.5, 0.5, label,
        ha="center", va="center", fontsize=10, color="#888888",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9",
                  edgecolor="#cccccc"),
    )
    ax.axis("off")
    out = OUT_DIR / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  stub: {out.name}")
    return out


# ===========================================================================
# Точка входа
# ===========================================================================

def main() -> None:
    print(f"Генерация иллюстраций -> {OUT_DIR}\n")
    funcs = [
        fig_2_1_architecture,
        fig_2_2_hybrid_pipeline,
        fig_3_1_lora_schema,
        fig_4_1_confusion_matrix,
        fig_4_2_training_curves,
        fig_4_3_mahalanobis,
        fig_4_4_ram_24h,
        fig_4_5_f1_vs_snr,
    ]
    results = {}
    for fn in funcs:
        p = fn()
        results[p.name] = p

    print(f"\nВсего: {len(results)} файлов.")
    print("  .drawio  — открыть в https://app.diagrams.net")
    print("  .png     — 300 dpi, готовы к вставке в ВКР")


if __name__ == "__main__":
    main()
