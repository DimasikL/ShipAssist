"""
scripts/vkr/plot_memory_24h.py — Figure 4.4: RSS memory over 24-hour load test.

Reads artifacts/vkr_data/memory_24h.csv (produced by collect_vkr_data.py
after the stress test) and saves a publication-ready PNG/PDF to
artifacts/plots/fig_4_4_memory_24h.png.

Usage
-----
    # After 24-h test completes, first collect the data:
    python scripts/vkr/collect_vkr_data.py

    # Then plot:
    python scripts/vkr/plot_memory_24h.py

    # Plot directly from the raw log (skip collect step):
    python scripts/vkr/plot_memory_24h.py --input logs/memory_24h.csv

    # Preview without saving:
    python scripts/vkr/plot_memory_24h.py --show

Google-style docstrings, no hardcoded absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution (works from any cwd)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Matplotlib — use non-interactive backend unless --show is requested
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")   # will switch to TkAgg/Qt5Agg if --show is passed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load memory_24h.csv and return (elapsed_h, rss_mb, lat_mean_ms).

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of three 1-D float arrays aligned by row index.
    """
    elapsed_s: list[float] = []
    rss_mb: list[float] = []
    lat_ms: list[float] = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                elapsed_s.append(float(row["elapsed_s"]))
                rss_mb.append(float(row["rss_mb"]))
                # lat_mean_ms may be 0 on the last row — keep it
                lat_ms.append(float(row.get("lat_mean_ms") or row.get("lat_ms") or 0))
            except (ValueError, KeyError):
                continue

    elapsed_h = np.array(elapsed_s) / 3600.0
    return elapsed_h, np.array(rss_mb), np.array(lat_ms)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(
    elapsed_h: np.ndarray,
    rss_mb: np.ndarray,
    lat_ms: np.ndarray,
    out_path: Path,
    show: bool = False,
) -> None:
    """Render and save Figure 4.4.

    Two-panel layout:
    * Top:    RSS memory (MB) over time — main thesis metric.
    * Bottom: Mean inference latency (ms) over time — sanity check.

    Args:
        elapsed_h: Time axis in hours.
        rss_mb:    RSS memory readings in MB.
        lat_ms:    Mean per-interval inference latency in ms.
        out_path:  Destination file (.png or .pdf).
        show:      If True, open an interactive matplotlib window.
    """
    # ── style ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
        "figure.dpi":       150,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.35,
        "grid.linestyle":   "--",
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    # ── Top panel: RSS memory ───────────────────────────────────────────────
    color_rss = "#1f77b4"

    ax1.plot(elapsed_h, rss_mb, color=color_rss, linewidth=1.4,
             label="RSS memory (MB)")

    # Reference line: initial stable value (median of first 10 % of points)
    n_ref = max(1, len(rss_mb) // 10)
    rss_baseline = float(np.median(rss_mb[:n_ref]))
    ax1.axhline(rss_baseline, color=color_rss, linestyle=":", linewidth=1.0,
                alpha=0.7, label=f"Baseline ≈ {rss_baseline:.0f} MB")

    # Mark min / max
    i_max = int(np.argmax(rss_mb))
    i_min = int(np.argmin(rss_mb))
    ax1.scatter([elapsed_h[i_max]], [rss_mb[i_max]], color="red",
                zorder=5, s=40, label=f"Peak {rss_mb[i_max]:.0f} MB")
    ax1.scatter([elapsed_h[i_min]], [rss_mb[i_min]], color="green",
                zorder=5, s=40, label=f"Min {rss_mb[i_min]:.0f} MB")

    delta = rss_mb[-1] - rss_mb[0]
    sign  = "+" if delta >= 0 else "−"
    ax1.text(
        0.98, 0.96,
        f"Δ RSS = {sign}{abs(delta):.1f} MB\n(start → end)",
        transform=ax1.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    ax1.set_ylabel("RSS Memory (MB)")
    ax1.set_title("Рис. 4.4 — Потребление памяти и задержка за 24 ч нагрузочного теста")
    ax1.legend(loc="upper right", framealpha=0.85)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # ── Bottom panel: latency ───────────────────────────────────────────────
    # Filter out zero-latency tail rows
    lat_valid = np.where(lat_ms > 0, lat_ms, np.nan)

    ax2.plot(elapsed_h, lat_valid, color="#ff7f0e", linewidth=1.0,
             label="Mean latency (ms)")
    lat_mean_overall = float(np.nanmean(lat_valid))
    ax2.axhline(lat_mean_overall, color="#ff7f0e", linestyle=":",
                linewidth=1.0, alpha=0.7,
                label=f"Mean ≈ {lat_mean_overall:.1f} ms")

    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Latency (ms)")
    ax2.legend(loc="upper right", framealpha=0.85)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # ── x-axis ticks every 2 h ─────────────────────────────────────────────
    x_max = max(elapsed_h[-1], 1.0) if len(elapsed_h) else 1.0
    ax2.set_xlim(0, x_max)
    step = 2.0 if x_max > 4 else 0.5
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax2.xaxis.set_minor_locator(mticker.MultipleLocator(step / 2))

    # ── annotations ────────────────────────────────────────────────────────
    total_inf = 0
    # Try to read total inferences from CSV — reuse elapsed_h length
    # (approximate: lat is per-interval, inferences are cumulative)
    # We encode this just as a text note
    n_hours = x_max
    ax2.text(
        0.02, 0.05,
        f"Duration: {n_hours:.1f} h  |  Latency: {lat_mean_overall:.1f} ms avg",
        transform=ax2.transAxes,
        fontsize=8, color="gray",
    )

    # ── save ───────────────────────────────────────────────────────────────
    if show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"  → saved: {out_path.relative_to(PROJECT_ROOT)}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 4.4: RSS memory over 24-h load test.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to memory CSV (default: artifacts/vkr_data/memory_24h.csv). "
            "Falls back to logs/memory_24h.csv if vkr_data version is absent."
        ),
    )
    parser.add_argument(
        "--output",
        default="artifacts/plots/fig_4_4_memory_24h.png",
        help="Output path relative to PROJECT_ROOT (default: artifacts/plots/fig_4_4_memory_24h.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive matplotlib window instead of saving.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    # Resolve input
    if args.input:
        csv_path = Path(args.input)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
    else:
        # Prefer the collected version; fall back to raw log
        csv_path = PROJECT_ROOT / "artifacts" / "vkr_data" / "memory_24h.csv"
        if not csv_path.exists():
            csv_path = PROJECT_ROOT / "logs" / "memory_24h.csv"

    if not csv_path.exists():
        print(
            f"ERROR: CSV not found at {csv_path}\n"
            "Run scripts/vkr/stress_runner.py first, then scripts/vkr/collect_vkr_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print(f"Reading: {csv_path.relative_to(PROJECT_ROOT)}")
    elapsed_h, rss_mb, lat_ms = _load_csv(csv_path)
    print(
        f"  Points : {len(elapsed_h)}\n"
        f"  Duration: {elapsed_h[-1]:.1f} h\n"
        f"  RSS range: {rss_mb.min():.1f} – {rss_mb.max():.1f} MB\n"
        f"  Latency avg: {lat_ms[lat_ms > 0].mean():.1f} ms"
        if len(elapsed_h) else "  (no data)"
    )

    make_figure(elapsed_h, rss_mb, lat_ms, out_path, show=args.show)


if __name__ == "__main__":
    main()
