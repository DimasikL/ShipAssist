"""
scripts/test_outlier_gate.py — Phase 4 gate: validate OutlierDetector rejection on noise.

Tests two scenarios:
  1. Synthetic noise (Gaussian white noise + pink noise) → expected ≥ 99 % rejected.
  2. Real command samples from clf_dset/test/ → expected ≤ 5 % rejected (low false-negative rate).

Usage:
    python scripts/test_outlier_gate.py \\
        --detector_path artifacts/models/outlier_detector.pkl \\
        --model_dir lora_tune/models/run_2026-04-30_23-34-27/best_model \\
        --num_samples 50 \\
        [--noise_dir artifacts/data/maritime_noise]   # optional real noise WAVs
        [--test_audio_dir clf_dset/test]              # real command WAVs

Gate pass condition:
    noise_rejection_rate  >= 0.99   (99 % noise clips rejected)
    command_pass_rate     >= 0.95   (≤ 5 % real commands rejected as noise)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_white_noise(duration_s: float, sr: int = 16_000) -> np.ndarray:
    """Gaussian white noise, unit variance."""
    n = int(duration_s * sr)
    return np.random.randn(n).astype(np.float32)


def _make_pink_noise(duration_s: float, sr: int = 16_000) -> np.ndarray:
    """Pink noise via 1/f shaping in frequency domain."""
    n = int(duration_s * sr)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    freqs[0] = 1.0  # avoid /0
    spectrum = (1.0 / np.sqrt(freqs)).astype(np.complex128)
    phases = np.exp(2j * np.pi * np.random.rand(len(spectrum)))
    spectrum *= phases
    signal = np.fft.irfft(spectrum, n=n).astype(np.float32)
    rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
    return signal / rms * 0.1  # normalise to ~0.1 RMS


def _make_dc_plus_click(duration_s: float, sr: int = 16_000) -> np.ndarray:
    """DC offset with random transient clicks — definitely not speech."""
    n = int(duration_s * sr)
    sig = np.full(n, 0.02, dtype=np.float32)
    click_positions = np.random.randint(0, n, size=5)
    sig[click_positions] = np.random.uniform(0.5, 1.0, size=5)
    return sig


def _load_wav(path: Path, target_sr: int = 16_000, max_seconds: float = 3.0) -> np.ndarray:
    """Load a WAV file and return a mono float32 waveform."""
    try:
        import soundfile as sf
    except ImportError:
        import scipy.io.wavfile as wav  # type: ignore
        rate, data = wav.read(str(path))
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data[: int(max_seconds * rate)]

    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if rate != target_sr:
        try:
            import torchaudio  # type: ignore
            import torch
            t = torch.from_numpy(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, rate, target_sr)
            data = t.squeeze(0).numpy()
        except Exception:
            pass  # keep original rate — EmbeddingExtractor will handle it
    return data[: int(max_seconds * target_sr)]


def _find_wav_files(
    root: Path,
    max_files: int = 50,
    exclude_subdirs: tuple = ("scr", "src", "source"),
) -> List[Path]:
    """Collect WAVs, skipping source-recording directories (long concatenations).

    Directories named 'scr', 'src', or 'source' contain multi-command source
    recordings that are not representative of single-command inference inputs.
    """
    wavs = [
        p for p in root.rglob("*.wav")
        if not any(part.lower() in exclude_subdirs for part in p.parts)
    ]
    random.shuffle(wavs)
    return wavs[:max_files]


# ── outlier logic (operates directly on the detector dict) ────────────────────

class _DictGate:
    """Thin wrapper around the raw dict produced by OutlierDetector.save()."""

    def __init__(self, state: dict) -> None:
        self.method: str = state["config"]["method"]
        self.threshold: float = state["threshold"]
        self.per_class_thresholds: dict = state.get("per_class_thresholds", {})
        self.class_centroids: dict = state["class_centroids"]  # {int: np.ndarray}
        self.covariance_inv: Optional[np.ndarray] = state.get("covariance_inv")
        self.id2label: dict = state.get("id2label", {})

    def is_outlier(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """Return (is_outlier, distance_to_nearest_centroid)."""
        emb = np.asarray(embedding, dtype=np.float32)
        best_dist = float("inf")

        for cls_id, centroid in self.class_centroids.items():
            c = np.asarray(centroid, dtype=np.float32)
            if self.method == "mahalanobis" and self.covariance_inv is not None:
                diff = emb - c
                cov_inv = np.asarray(self.covariance_inv, dtype=np.float32)
                dist = float(np.sqrt(np.dot(np.dot(diff, cov_inv), diff)))
            elif self.method == "cosine":
                n_emb = emb / (np.linalg.norm(emb) + 1e-12)
                n_c = c / (np.linalg.norm(c) + 1e-12)
                dist = float(1.0 - np.dot(n_emb, n_c))
            else:  # l2
                dist = float(np.linalg.norm(emb - c))

            if dist < best_dist:
                best_dist = dist

        return best_dist > self.threshold, best_dist


def _load_detector(path: Path) -> _DictGate:
    import pickle
    with open(path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"Expected dict in {path}, got {type(state)}")
    logger.info(
        "Outlier detector loaded: method=%s threshold=%.2f classes=%s",
        state["config"]["method"], state["threshold"],
        list(state.get("id2label", {}).values()),
    )
    return _DictGate(state)


def _build_embedding_extractor(model_dir: Path):
    """Load EmbeddingExtractor.

    Uses the same bypass as realtime_recognizer._load_wav2vec2_model to avoid
    the HuggingFace adapter-detection bug when adapter_model.safetensors is
    present alongside model.safetensors.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    from scripts.utils.outlier_detection import EmbeddingExtractor, OutlierConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model (bypass HF adapter detection) ──────────────────────────
    config = Wav2Vec2Config.from_pretrained(str(model_dir))
    model = Wav2Vec2ForSequenceClassification(config)

    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No weight file in {model_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s …", len(missing), missing[:3])
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir))

    cfg = OutlierConfig(embedding_layer="projector", max_audio_seconds=3.0)
    extractor = EmbeddingExtractor(
        model=model,
        feature_extractor=feature_extractor,
        device=device,
        config=cfg,
    )
    logger.info("EmbeddingExtractor ready (device=%s, model_dir=%s)", device, model_dir)
    return extractor


def _extract_embedding(extractor, waveform: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """Return pooled embedding vector for a single waveform.

    Writes a temp WAV file so EmbeddingExtractor.extract_single() can use its
    standard audio-loading path (same resampling and truncation logic).
    """
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, waveform, sr)
        emb = extractor.extract_single(tmp.name)

    import os
    os.unlink(tmp.name)
    return emb


# ── main logic ────────────────────────────────────────────────────────────────

def run_test(
    detector_path: Path,
    model_dir: Path,
    num_samples: int,
    noise_dir: Optional[Path],
    test_audio_dir: Optional[Path],
    sr: int = 16_000,
) -> dict:
    rng = np.random.default_rng(42)
    random.seed(42)

    gate = _load_detector(detector_path)
    extractor = _build_embedding_extractor(model_dir)

    results: dict = {}

    # ── 1. Noise test ─────────────────────────────────────────────────────────
    noise_clips: List[np.ndarray] = []

    if noise_dir and noise_dir.exists():
        wavs = _find_wav_files(noise_dir, max_files=num_samples)
        for w in wavs:
            noise_clips.append(_load_wav(w, target_sr=sr))
        logger.info("Loaded %d real noise WAVs from %s", len(noise_clips), noise_dir)

    # Pad with synthetic noise to reach num_samples
    needed = max(0, num_samples - len(noise_clips))
    generators = [_make_white_noise, _make_pink_noise, _make_dc_plus_click]
    for i in range(needed):
        gen = generators[i % len(generators)]
        dur = rng.uniform(0.5, 3.0)
        noise_clips.append(gen(float(dur), sr))

    logger.info("Testing %d noise samples (real=%d synthetic=%d)...",
                len(noise_clips), len(noise_clips) - needed, needed)

    noise_rejected = 0
    noise_distances: List[float] = []
    t0 = time.perf_counter()
    for clip in noise_clips:
        emb = _extract_embedding(extractor, clip, sr)
        outlier, dist = gate.is_outlier(emb)
        noise_distances.append(dist)
        if outlier:
            noise_rejected += 1
    noise_elapsed = time.perf_counter() - t0

    noise_rejection_rate = noise_rejected / len(noise_clips)
    results["noise"] = {
        "total": len(noise_clips),
        "rejected": noise_rejected,
        "rejection_rate": noise_rejection_rate,
        "avg_distance": float(np.mean(noise_distances)),
        "p95_distance": float(np.percentile(noise_distances, 95)),
        "elapsed_s": round(noise_elapsed, 2),
    }

    gate_pass_noise = noise_rejection_rate >= 0.99
    status_noise = "PASS" if gate_pass_noise else "FAIL"
    logger.info(
        "[%s] Noise rejection: %d/%d = %.1f %%  (threshold: 1189.7, detector_th: %.1f)",
        status_noise, noise_rejected, len(noise_clips), noise_rejection_rate * 100,
        gate.threshold,
    )

    # ── 2. Command pass-through test ──────────────────────────────────────────
    command_clips: List[Tuple[str, np.ndarray]] = []

    if test_audio_dir and test_audio_dir.exists():
        wavs = _find_wav_files(test_audio_dir, max_files=60)
        for w in wavs:
            command_clips.append((w.stem, _load_wav(w, target_sr=sr)))
        logger.info("Loaded %d real command WAVs from %s", len(command_clips), test_audio_dir)

    if command_clips:
        cmd_rejected = 0
        cmd_distances: List[float] = []
        for name, clip in command_clips:
            emb = _extract_embedding(extractor, clip, sr)
            outlier, dist = gate.is_outlier(emb)
            cmd_distances.append(dist)
            if outlier:
                cmd_rejected += 1
                logger.warning("  FALSE NEGATIVE: '%s'  dist=%.2f  (threshold=%.2f)", name, dist, gate.threshold)

        cmd_pass_rate = 1.0 - cmd_rejected / len(command_clips)
        results["commands"] = {
            "total": len(command_clips),
            "rejected_false_neg": cmd_rejected,
            "pass_rate": cmd_pass_rate,
            "avg_distance": float(np.mean(cmd_distances)),
        }

        gate_pass_cmd = cmd_pass_rate >= 0.95
        status_cmd = "PASS" if gate_pass_cmd else "FAIL"
        logger.info(
            "[%s] Command pass-through: %d/%d accepted = %.1f %%",
            status_cmd, len(command_clips) - cmd_rejected, len(command_clips), cmd_pass_rate * 100,
        )

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 — OutlierGate rejection test")
    p.add_argument(
        "--detector_path", type=Path,
        default=Path("artifacts/models/outlier_detector.pkl"),
        help="Path to outlier_detector.pkl",
    )
    p.add_argument(
        "--model_dir", type=Path,
        default=Path("lora_tune/models/run_2026-04-30_23-34-27/best_model"),
        help="LoRA checkpoint directory (used for embedding extraction)",
    )
    p.add_argument(
        "--noise_dir", type=Path, default=None,
        help="Directory with maritime noise WAVs (optional; falls back to synthetic)",
    )
    p.add_argument(
        "--test_audio_dir", type=Path,
        default=Path("clf_dset/test"),
        help="Directory with real command WAVs for false-negative rate check",
    )
    p.add_argument(
        "--num_samples", type=int, default=50,
        help="Number of noise clips to test (synthetic fills the gap if noise_dir is small)",
    )
    p.add_argument(
        "--output_json", type=Path,
        default=Path("artifacts/benchmarks/outlier_gate_test.json"),
        help="Where to write JSON results",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.detector_path.exists():
        logger.error("Detector not found: %s", args.detector_path)
        return 1
    if not args.model_dir.exists():
        logger.error("Model dir not found: %s", args.model_dir)
        return 1

    results = run_test(
        detector_path=args.detector_path,
        model_dir=args.model_dir,
        num_samples=args.num_samples,
        noise_dir=args.noise_dir,
        test_audio_dir=args.test_audio_dir,
    )

    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", args.output_json)

    # Final gate verdict
    noise_ok = results.get("noise", {}).get("rejection_rate", 0.0) >= 0.99
    cmd_ok = results.get("commands", {}).get("pass_rate", 1.0) >= 0.95

    print("\n" + "=" * 60)
    print("  Phase 4 — OutlierGate test summary")
    print("=" * 60)
    n = results.get("noise", {})
    print(f"  Noise rejection : {n.get('rejected',0)}/{n.get('total',0)} = "
          f"{n.get('rejection_rate',0)*100:.1f} %  "
          f"{'✓ PASS' if noise_ok else '✗ FAIL  (need ≥ 99 %)'}")
    if "commands" in results:
        c = results["commands"]
        print(f"  Command pass    : {c.get('total',0)-c.get('rejected_false_neg',0)}/{c.get('total',0)} = "
              f"{c.get('pass_rate',1.0)*100:.1f} %  "
              f"{'✓ PASS' if cmd_ok else '✗ FAIL  (need ≥ 95 %)'}")
    print(f"  Gate threshold  : {_load_detector(args.detector_path).threshold:.2f}")
    print("=" * 60)

    gate_passed = noise_ok and cmd_ok
    print(f"\n  Overall: {'✓ GATE PASSED' if gate_passed else '✗ GATE FAILED'}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
