"""
calibrate_temperature.py — Find the optimal temperature T for INT8 ONNX logits.

Temperature scaling is a post-hoc calibration technique: the logits produced
by the quantised model are divided by T before softmax.  T > 1 softens the
distribution (useful when INT8 is overconfident); T < 1 sharpens it (useful
when INT8 is underconfident relative to PT).

The optimal T is found by minimising Negative Log-Likelihood (NLL) over the
test split using scipy.optimize.minimize_scalar.  The script also reports ECE
(Expected Calibration Error) before and after scaling so you can see whether
calibration actually changes anything meaningfully.

Usage
-----
    python scripts/train/calibrate_temperature.py \\
        --onnx_path  onnx_model/quant_benchmark/model_int8.onnx \\
        --run_dir    lora_tune/models/run_2026-04-30_23-34-27 \\
        --data_csv   dset_meta_only_2026-04-30_15-46-30.csv

    # Write the result into configs/model.yaml automatically:
    python scripts/train/calibrate_temperature.py ... --update_config

Output
------
    Prints the optimal T and ECE before / after.
    Saves calibration_results.json to --run_dir.
    Optionally patches the ``onnx.temperature`` key in configs/model.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Ensure project root is on sys.path for `core.*` imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test-split definition  (must stay in sync with eval_lora_model.py)
# ---------------------------------------------------------------------------
TEST_GROUPS = [
    "train_user_2",
    "drug slova2",
    "train_user_2_new",
    "drug slova2-new",
    "train user 4",
]

SR: int = 16_000


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax. Input shape: (N, C)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood at a given temperature.

    Args:
        logits:      (N, C) float32 raw logits.
        labels:      (N,)  int true class indices.
        temperature: Positive scalar dividing logits before softmax.

    Returns:
        Mean NLL (scalar).
    """
    probs = _softmax(logits / max(temperature, 1e-6))
    # Clip to avoid log(0)
    correct_probs = probs[np.arange(len(labels)), labels].clip(1e-9, 1.0)
    return float(-np.log(correct_probs).mean())


def ece(logits: np.ndarray, labels: np.ndarray, temperature: float, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) with equal-width confidence bins.

    ECE measures how well the model's confidence aligns with its accuracy.
    An ECE of 0 = perfectly calibrated.

    Args:
        logits:      (N, C) float32 raw logits.
        labels:      (N,)  int true class indices.
        temperature: Scalar temperature applied before softmax.
        n_bins:      Number of equal-width bins in [0, 1].

    Returns:
        ECE scalar in [0, 1].
    """
    probs = _softmax(logits / max(temperature, 1e-6))
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece_val += mask.mean() * abs(bin_conf - bin_acc)

    return float(ece_val)


# ---------------------------------------------------------------------------
# Logit collection
# ---------------------------------------------------------------------------

def collect_logits(
    onnx_path: str,
    df: pd.DataFrame,
    label2id: Dict[str, int],
    max_samples: int,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on the dataset and collect raw logits + true labels.

    Args:
        onnx_path:   Path to the ONNX model file.
        df:          DataFrame with ``audio_path`` and ``class`` columns.
        label2id:    Class-name → int mapping.
        max_samples: Canonical window length in samples.
        batch_size:  ORT batch size.

    Returns:
        ``(logits, labels)`` — (N, C) float32 and (N,) int arrays.
    """
    import onnxruntime as ort
    from core.audio_utils import load_wav, prepare_window

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = os.cpu_count() or 1
    opts.intra_op_num_threads = os.cpu_count() or 1
    session = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []
    rows = df.reset_index(drop=True)
    n = len(rows)

    for start in tqdm(range(0, n, batch_size), desc="Collecting logits"):
        batch_rows = rows.iloc[start : start + batch_size]
        batch_audio = []
        for _, row in batch_rows.iterrows():
            try:
                wav, _ = load_wav(row["audio_path"], target_sr=SR)
            except Exception as exc:
                logger.warning("Cannot load %s: %s — using silence.", row["audio_path"], exc)
                wav = np.zeros(max_samples, dtype=np.float32)
            batch_audio.append(
                prepare_window(wav, target_samples=max_samples, do_normalize=True)
            )
        batch_np = np.stack(batch_audio)  # (B, max_samples)
        outputs = session.run(None, {input_name: batch_np})
        logits: np.ndarray = outputs[0]  # (B, C)
        all_logits.append(logits.astype(np.float32))
        all_labels.extend([label2id[r["class"]] for _, r in batch_rows.iterrows()])

    return np.concatenate(all_logits, axis=0), np.array(all_labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Temperature optimisation
# ---------------------------------------------------------------------------

def find_optimal_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.1,
    t_max: float = 5.0,
) -> float:
    """Grid-search then refine with scipy.minimize_scalar.

    Falls back to grid search alone if scipy is not available.

    Args:
        logits: (N, C) float32.
        labels: (N,) int.
        t_min:  Lower bound for T search.
        t_max:  Upper bound for T search.

    Returns:
        Optimal temperature T (float).
    """
    # Coarse grid to seed the optimizer
    grid = np.linspace(t_min, t_max, 200)
    nll_grid = [nll(logits, labels, t) for t in grid]
    t_coarse = float(grid[np.argmin(nll_grid)])

    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            lambda t: nll(logits, labels, t),
            bounds=(t_min, t_max),
            method="bounded",
        )
        t_opt = float(result.x)
        logger.info("scipy.minimize_scalar converged: T=%.4f  NLL=%.6f", t_opt, result.fun)
    except ImportError:
        logger.warning("scipy not available — using coarse grid result only.")
        t_opt = t_coarse

    return t_opt


# ---------------------------------------------------------------------------
# YAML patcher
# ---------------------------------------------------------------------------

def _patch_yaml_temperature(yaml_path: str, temperature: float) -> None:
    """Update the ``onnx.temperature`` key in a YAML file in-place.

    Uses a line-by-line approach to avoid re-serialising the whole file
    (which would strip comments and formatting).

    Args:
        yaml_path:   Path to the YAML config file.
        temperature: New temperature value to write.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_onnx_block = False
    patched = False
    new_lines = []

    for line in lines:
        stripped = line.lstrip()
        # Detect entry into the `onnx:` block
        if stripped.startswith("onnx:") and ":" in line:
            in_onnx_block = True
        # Exit onnx block on a new top-level key (no leading spaces)
        elif in_onnx_block and line and not line[0].isspace() and not line.startswith("#"):
            in_onnx_block = False

        if in_onnx_block and stripped.startswith("temperature:"):
            indent = len(line) - len(stripped)
            new_lines.append(" " * indent + f"temperature: {temperature:.4f}\n")
            patched = True
        else:
            new_lines.append(line)

    if patched:
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        logger.info("Patched onnx.temperature → %.4f in %s", temperature, yaml_path)
    else:
        logger.warning(
            "Could not find 'temperature:' inside 'onnx:' block in %s. "
            "Add it manually:\n  onnx:\n    temperature: %.4f",
            yaml_path, temperature,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate ONNX INT8 temperature scaling on the test split."
    )
    parser.add_argument(
        "--onnx_path",
        required=True,
        help="Path to the ONNX model file (e.g. onnx_model/quant_benchmark/model_int8.onnx).",
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the LoRA run directory (contains best_model/config.json with id2label).",
    )
    parser.add_argument(
        "--data_csv",
        required=True,
        help="Path to the dataset metadata CSV.",
    )
    parser.add_argument(
        "--update_config",
        action="store_true",
        help="Patch configs/model.yaml with the optimal temperature value.",
    )
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--max_seconds", type=float, default=3.0)
    args = parser.parse_args()

    onnx_path  = Path(args.onnx_path)
    run_dir    = Path(args.run_dir)
    best_model = run_dir / "best_model"
    results_path = run_dir / "calibration_results.json"

    if not onnx_path.exists():
        logger.error("ONNX file not found: %s", onnx_path)
        sys.exit(1)

    # ── Load label map ──
    with open(best_model / "config.json", encoding="utf-8") as f:
        model_cfg = json.load(f)
    id2label: Dict[str, str] = model_cfg["id2label"]
    label2id: Dict[str, int] = {v: int(k) for k, v in id2label.items()}
    max_samples = int(args.max_seconds * SR)

    # ── Build test split ──
    df = pd.read_csv(args.data_csv)
    test_df = df[df["audio_group"].isin(TEST_GROUPS)].reset_index(drop=True)
    logger.info("Calibration split: %d samples", len(test_df))

    # ── Collect logits ──
    logger.info("Running ONNX inference to collect raw logits...")
    logits, labels = collect_logits(
        str(onnx_path), test_df, label2id, max_samples, args.batch_size
    )
    logger.info("Collected logits: shape=%s", logits.shape)

    # ── Metrics at T=1.0 (no scaling) ──
    nll_before  = nll(logits, labels, temperature=1.0)
    ece_before  = ece(logits, labels, temperature=1.0)
    logger.info("Before calibration — NLL: %.6f  ECE: %.4f", nll_before, ece_before)

    # ── Find optimal T ──
    logger.info("Optimising temperature (minimising NLL)...")
    t_opt = find_optimal_temperature(logits, labels)

    # ── Metrics at T_opt ──
    nll_after = nll(logits, labels, temperature=t_opt)
    ece_after = ece(logits, labels, temperature=t_opt)

    # ── Print report ──
    print()
    print("=" * 55)
    print("  Temperature Calibration Results")
    print("=" * 55)
    print(f"  Optimal T:         {t_opt:.4f}")
    print(f"  NLL  before (T=1): {nll_before:.6f}")
    print(f"  NLL  after  (T={t_opt:.2f}): {nll_after:.6f}  Δ={nll_after - nll_before:+.6f}")
    print(f"  ECE  before (T=1): {ece_before:.4f}")
    print(f"  ECE  after  (T={t_opt:.2f}): {ece_after:.4f}  Δ={ece_after - ece_before:+.4f}")
    print("=" * 55)

    # Interpret the result for the user.
    # We require BOTH a meaningful T shift AND an ECE improvement before
    # recommending a config change. With small test sets (<500 samples) the
    # NLL landscape is flat and the optimizer can land anywhere in ±0.3 of
    # T=1.0 without a real signal. ECE is the ground-truth calibration metric.
    nll_improvement = nll_before - nll_after          # positive = better
    ece_improvement = ece_before - ece_after          # positive = better
    meaningful = (
        abs(t_opt - 1.0) >= 0.10          # T must differ by ≥ 10%
        and nll_improvement >= 0.001       # NLL must improve by ≥ 0.001
        and ece_improvement >= 0.0         # ECE must not worsen
    )

    if not meaningful:
        print("\n  ✓ T ≈ 1.0 — INT8 logits are already well-calibrated.")
        print(f"    (NLL Δ={-nll_improvement:+.6f}, ECE Δ={-ece_improvement:+.4f} — noise level)")
        print("    No change needed in configs/model.yaml.")
    elif t_opt > 1.0:
        print(f"\n  ⚠ T > 1.0 ({t_opt:.4f}) — INT8 is overconfident.")
        print(f"    Set onnx.temperature: {t_opt:.4f} in configs/model.yaml.")
    else:
        print(f"\n  ⚠ T < 1.0 ({t_opt:.4f}) — INT8 is underconfident.")
        print(f"    Set onnx.temperature: {t_opt:.4f} in configs/model.yaml.")
    print()

    # ── Optionally update config (only when calibration is meaningful) ──
    if args.update_config:
        if meaningful:
            config_yaml = _PROJECT_ROOT / "configs" / "model.yaml"
            if not config_yaml.exists():
                config_yaml = _PROJECT_ROOT / "configs" / "default.yaml"
            _patch_yaml_temperature(str(config_yaml), t_opt)
        else:
            logger.info(
                "--update_config skipped: calibration improvement is within noise "
                "(ΔNLL=%.6f, ΔECE=%.4f). Keeping temperature: 1.0.",
                nll_after - nll_before, ece_after - ece_before,
            )

    # ── Save results ──
    output = {
        "onnx_path":    str(onnx_path),
        "n_samples":    int(len(labels)),
        "optimal_T":    t_opt,
        "nll_T1":       nll_before,
        "nll_Topt":     nll_after,
        "ece_T1":       ece_before,
        "ece_Topt":     ece_after,
        "nll_delta":    nll_after - nll_before,
        "ece_delta":    ece_after - ece_before,
        "well_calibrated": abs(t_opt - 1.0) < 0.03,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Calibration results saved → %s", results_path)


if __name__ == "__main__":
    main()
