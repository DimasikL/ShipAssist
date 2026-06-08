"""
scripts/debug_pt_vs_onnx.py — PT vs ONNX comparative diagnostic.

Usage
-----
  python scripts/debug_pt_vs_onnx.py --audio path/to/sample.wav
  python scripts/debug_pt_vs_onnx.py --audio sample.wav --pt artifacts/models/best_model
  python scripts/debug_pt_vs_onnx.py --audio sample.wav --suggest-temperature

Outputs
-------
* Markdown-style table with top-3 logits, confidence, argmax, and Δ
  between PyTorch and ONNX.
* Optional suggested temperature scaling factor for ``cfg.onnx.temperature``.
* Full session log appended to ``artifacts/logs/pt_onnx_diff.log``.

Both engines load the SAME audio through ``core.audio_utils.prepare_window``,
so any remaining numerical drift is attributable to the backend / quantisation
itself, not the preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Resolve project root so absolute imports work when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window
from core.config import settings
from core.engine import OnnxAudioEngine, TorchAudioEngine


# ── File logger ───────────────────────────────────────────────────────────────

def _make_diff_logger() -> logging.Logger:
    """Return a logger that appends to ``artifacts/logs/pt_onnx_diff.log``."""
    log_dir = settings.paths.artifacts_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pt_onnx_diff.log"

    diff_logger = logging.getLogger("pt_onnx_diff")
    diff_logger.setLevel(logging.INFO)
    diff_logger.propagate = False                       # don't double-log to root

    if not diff_logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        diff_logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        diff_logger.addHandler(ch)

    return diff_logger


# ── Top-K helper ──────────────────────────────────────────────────────────────

def _top_k(
    logits: np.ndarray, labels: List[str], k: int = 3
) -> List[Tuple[str, float]]:
    """Return ``[(label, logit)]`` for the top-*k* indices."""
    k = min(k, len(logits))
    idx = np.argsort(logits)[::-1][:k]
    return [(labels[i], float(logits[i])) for i in idx]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


# ── Suggested temperature ─────────────────────────────────────────────────────

def _suggest_temperature(pt_logits: np.ndarray, onnx_logits: np.ndarray) -> float:
    """Estimate a temperature that aligns ONNX confidence with PyTorch.

    Uses ratio of standard deviations: T ≈ std(ONNX) / std(PT). When ONNX
    logits are flatter (smaller std), T < 1 sharpens them; when ONNX is
    spikier, T > 1 softens them. Returns 1.0 when std is degenerate.
    """
    pt_std = float(np.std(pt_logits))
    onnx_std = float(np.std(onnx_logits))
    if pt_std == 0.0 or onnx_std == 0.0:
        return 1.0
    return round(onnx_std / pt_std, 4)


# ── Pretty printer ────────────────────────────────────────────────────────────

def _render_table(
    audio_path: str,
    pt_top: List[Tuple[str, float]],
    onnx_top: List[Tuple[str, float]],
    pt_logits: np.ndarray,
    onnx_logits: np.ndarray,
    pt_label: str,
    onnx_label: str,
    pt_conf: float,
    onnx_conf: float,
) -> str:
    """Render a Markdown-ish table summarising the comparison."""
    delta_logits = float(np.abs(pt_logits - onnx_logits).max())
    delta_top1 = float(abs(pt_logits.max() - onnx_logits.max()))

    lines: List[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  PT vs ONNX diagnostic — audio: {audio_path}")
    lines.append(f"  Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"{'#':<3}  {'Metric':<22}  {'PyTorch':<28}  {'ONNX':<28}")
    lines.append(f"{'-':<3}  {'-' * 22}  {'-' * 28}  {'-' * 28}")
    for i in range(3):
        pt_cell = f"{pt_top[i][0]!r}  {pt_top[i][1]:+.4f}" if i < len(pt_top) else "—"
        on_cell = f"{onnx_top[i][0]!r}  {onnx_top[i][1]:+.4f}" if i < len(onnx_top) else "—"
        lines.append(f"{i + 1:<3}  {'top-' + str(i + 1) + ' (logit)':<22}  {pt_cell:<28}  {on_cell:<28}")
    lines.append("")
    lines.append(f"{'argmax':<25}  PT={pt_label!r:<22}  ONNX={onnx_label!r}")
    lines.append(f"{'confidence (softmax)':<25}  PT={pt_conf:.4f}{'':<14}  ONNX={onnx_conf:.4f}")
    lines.append("")
    lines.append(f"  Δ logit max abs        : {delta_logits:.4e}")
    lines.append(f"  Δ logit (top-1 only)   : {delta_top1:.4e}")
    lines.append(f"  Δ confidence           : {abs(pt_conf - onnx_conf):.4f}")

    if pt_label != onnx_label:
        lines.append("")
        lines.append("  ⚠ ARGMAX MISMATCH — PT и ONNX выбирают разные классы.")
    if abs(pt_conf - onnx_conf) > 0.15:
        lines.append("  ⚠ Confidence drift > 0.15 — INT8 квантование «сплющивает» логиты.")

    lines.append("")
    return "\n".join(lines)


# ── Recalibration hint ────────────────────────────────────────────────────────

def _print_recalibration_hint(diff_logger: logging.Logger) -> None:
    """Log instructions for regenerating the INT8 calibration set."""
    diff_logger.info(
        "\n  Если деградация подтверждается, выполните:\n"
        "    1. Установите в configs/model.yaml: onnx.precision = \"fp32\"\n"
        "       — это уберёт INT8 как источник проблемы.\n"
        "    2. Если нужен INT8, увеличьте onnx.temperature (например 1.5–2.0)\n"
        "       либо включите onnx.auto_temperature: true.\n"
        "    3. Для полной перекалибровки: onnx.recalibrate: true и перезапустите\n"
        "       scripts/train/main_export_to_onnx.py с --quantize, передав\n"
        "       calib_data из 4 целевых фраз (см. RUNBOOK.md).\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PT vs ONNX logit/confidence diagnostic for ShipAssistant."
    )
    parser.add_argument(
        "--audio", required=True,
        help="Path to a target-phrase .wav file.",
    )
    parser.add_argument(
        "--pt", default=None,
        help=(
            "Override path to the PyTorch checkpoint directory "
            "(default: cfg.paths.best_model)."
        ),
    )
    parser.add_argument(
        "--onnx", default=None,
        help=(
            "Override path to the ONNX bundle directory "
            "(default: cfg.paths.onnx_model)."
        ),
    )
    parser.add_argument(
        "--precision", default=None, choices=["int8", "fp32", "fp16"],
        help="Override cfg.onnx.precision for this run.",
    )
    parser.add_argument(
        "--suggest-temperature", action="store_true",
        help="Print a suggested cfg.onnx.temperature aligning ONNX with PT.",
    )
    args = parser.parse_args()

    diff_logger = _make_diff_logger()

    pt_path = args.pt or str(settings.paths.best_model)
    onnx_path = args.onnx or str(settings.paths.onnx_model)
    precision = args.precision or settings.onnx.precision

    diff_logger.info(
        "PT vs ONNX diagnostic start — pt=%r onnx=%r precision=%s audio=%r",
        pt_path, onnx_path, precision, args.audio,
    )

    # 1. Load and prepare audio (same pipeline both engines see) ----------
    sr = settings.audio.sample_rate
    waveform, _ = load_wav(args.audio, target_sr=sr)
    target_samples = int(settings.audio.window_seconds * sr)
    prepared = prepare_window(waveform, target_samples=target_samples, do_normalize=True)
    diff_logger.info(
        "Audio prepared: original_samples=%d → target_samples=%d (sr=%d)",
        waveform.shape[0], prepared.shape[0], sr,
    )

    # 2. PyTorch reference ------------------------------------------------
    try:
        pt_engine = TorchAudioEngine(model_path=pt_path)
    except Exception as exc:
        diff_logger.error("PyTorch checkpoint load failed: %s", exc)
        diff_logger.error(
            "  Подсказка: укажите --pt <path> или положите чекпоинт по адресу %s.",
            settings.paths.best_model,
        )
        return 2
    pt_result = pt_engine.predict(prepared)

    # 3. ONNX engine (NO temperature here — we want the raw degradation) --
    try:
        onnx_engine = OnnxAudioEngine(
            onnx_dir=onnx_path,
            precision=precision,
            providers=list(settings.onnx.providers),
            temperature=1.0,                            # diagnostic mode
            adaptive_threshold=False,
            default_confidence=settings.recognition.default_confidence,
        )
    except Exception as exc:
        diff_logger.error("ONNX bundle load failed: %s", exc)
        return 3
    onnx_result = onnx_engine.predict(prepared)

    # 4. Render report ----------------------------------------------------
    pt_logits = pt_result["logits"]
    onnx_logits = onnx_result["logits"]

    # Both engines expose the same label ordering (id2label / config.labels);
    # we use ONNX labels because that is what is shipped in the bundle.
    labels = onnx_engine.labels
    if pt_engine.labels != labels:
        diff_logger.warning(
            "Label ordering differs PT vs ONNX! pt=%s onnx=%s — diagnostic "
            "will still run but cross-label deltas may be misleading.",
            pt_engine.labels, labels,
        )

    report = _render_table(
        audio_path=args.audio,
        pt_top=_top_k(pt_logits, pt_engine.labels, k=3),
        onnx_top=_top_k(onnx_logits, labels, k=3),
        pt_logits=pt_logits,
        onnx_logits=onnx_logits,
        pt_label=pt_result["label"],
        onnx_label=onnx_result["label"],
        pt_conf=pt_result["confidence"],
        onnx_conf=onnx_result["confidence"],
    )
    diff_logger.info(report)

    # 5. Suggested temperature (optional) ---------------------------------
    if args.suggest_temperature:
        suggested = _suggest_temperature(pt_logits, onnx_logits)
        diff_logger.info(
            "  Suggested cfg.onnx.temperature ≈ %.4f  "
            "(rule-of-thumb: T = std(ONNX) / std(PT))\n", suggested,
        )

    if pt_result["label"] != onnx_result["label"]:
        _print_recalibration_hint(diff_logger)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
