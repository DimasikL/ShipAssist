"""scripts/vkr/benchmark_snr_profile.py — Extended SNR-profile benchmark.

Measures model quality metrics (macro-F1, per-class F1, WER, OOD degradation)
across a grid of SNR values, noise types, and model baselines. Designed for
thesis reproducibility: fixed seeds, statistical repetition, unified CSV + JSON.

Supported model backends:
    • ONNX Runtime   (--model_path: directory with onnx_config.json OR .onnx file)
    • HuggingFace    (--model_path: directory with config.json)
    • MFCC + SVM     (--baselines mfcc_svm  + --mfcc_svm_train_csv or --mfcc_svm_pkl)
    • Whisper Tiny   (--baselines whisper_tiny)
    • Wav2Vec2 base  (--baselines wav2vec2_base + --wav2vec2_base_path)

Noise types (synthetic, no external data needed):
    white, pink, babble, street, cafe, music, office, traffic, wind

Real noise files (optional, takes priority over synthetic):
    Place WAV clips at --noise_dir/{noise_type}/*.wav
    Prepare them with scripts/vkr/noise_augment_dataset.py

Output schema  (f1_vs_snr_full.csv):
    snr_db, noise_type, model, command_class, f1_mean, f1_std,
    wer, n_samples, is_ood, precision, recall, timestamp

Summary JSON (f1_vs_snr_summary.json):
    Per-model aggregated metrics + per-class breakdown + OOD delta table.

Example:
    python scripts/vkr/benchmark_snr_profile.py \\
        --model_path onnx_model/models/run_2026-05-22_09-50-17/ \\
        --test_csv artifacts/data/test_commands_snr.csv \\
        --baselines mfcc_svm,whisper_tiny,wav2vec2_base \\
        --mfcc_svm_train_csv artifacts/data/train_commands.csv \\
        --wav2vec2_base_path artifacts/models/wav2vec2_base/ \\
        --snr_values 2 5 8 10 12 15 20 \\
        --noise_types white babble street cafe music office traffic wind \\
        --per_class_breakdown \\
        --ood_flag_column is_ood \\
        --output_dir artifacts/benchmarks/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Bootstrap: resolve PROJECT_ROOT regardless of cwd
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]   # scripts/vkr/ → scripts/ → root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR: int = 16_000           # target sample rate, Hz
WIN_SAMPLES: int = 48_000  # 3 s window @ 16 kHz — must match training config
SNR_TOLERANCE_DB: float = 0.5

VALID_NOISE_TYPES: Tuple[str, ...] = (
    "white", "pink", "babble", "street", "cafe",
    "music", "office", "traffic", "wind",
)
BASELINE_IDS: Tuple[str, ...] = ("mfcc_svm", "whisper_tiny", "wav2vec2_base")

_MACRO_CLASS_LABEL = "macro"   # sentinel used in command_class column

# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def _rms(signal: np.ndarray) -> float:
    """Return RMS energy of a 1-D float32 signal.

    Args:
        signal: 1-D NumPy array.

    Returns:
        RMS value (scalar float).
    """
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2) + 1e-12))


def _measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure actual SNR in dB between clean and noisy signals.

    Args:
        clean: Reference clean signal, shape (N,).
        noisy: Noisy signal of the same shape.

    Returns:
        SNR in dB.
    """
    noise_component = noisy.astype(np.float64) - clean.astype(np.float64)
    sig_power = float(np.mean(clean.astype(np.float64) ** 2)) + 1e-12
    noi_power = float(np.mean(noise_component ** 2)) + 1e-12
    return 10.0 * np.log10(sig_power / noi_power)


def _generate_synthetic_noise(
    noise_type: str,
    n_samples: int,
    rng: np.random.Generator,
    babble_pool: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Generate a synthetic noise signal of the requested type.

    All types are self-contained: no external files are required.
    Spectral profiles are calibrated to match representative real-world spectra.

    Noise type profiles
    -------------------
    white   — flat-spectrum Gaussian noise (AWGN, -3 dB per octave)
    pink    — 1/f pink noise via cumulative-sum spectral shaping
    babble  — multi-speaker babble (randomly mixed utterances from *babble_pool*)
    street  — pink noise band-passed to 200–2000 Hz (traffic rumble)
    cafe    — pink noise low-passed to ≤500 Hz (ambient murmur)
    music   — pink noise band-passed 100–8000 Hz + 2 Hz amplitude modulation
    office  — 50 Hz HVAC hum + sparse burst noise (keyboard clicks, fan)
    traffic — brown noise (1/f²) with low-frequency emphasis + occasional horns
    wind    — low-pass shaped pink noise with slow 0.3 Hz amplitude modulation

    Args:
        noise_type:  One of VALID_NOISE_TYPES.
        n_samples:   Length of the output noise vector in samples.
        rng:         NumPy random generator (caller controls seed).
        babble_pool: List of clean speech waveforms required for 'babble'.

    Returns:
        Noise signal, float32, shape (n_samples,), normalised to unit RMS.

    Raises:
        ValueError: If noise_type is unsupported or babble_pool is missing.
    """
    from scipy.signal import butter, sosfilt  # type: ignore[import]

    t = np.arange(n_samples, dtype=np.float32) / SR

    if noise_type == "white":
        noise = rng.standard_normal(n_samples).astype(np.float32)

    elif noise_type == "pink":
        white = rng.standard_normal(n_samples).astype(np.float64)
        # Spectral shaping: 1/sqrt(f) rolloff ≈ pink noise
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-6
        pink_spectrum = spectrum / np.sqrt(np.abs(freqs))
        noise = np.fft.irfft(pink_spectrum, n=n_samples).astype(np.float32)

    elif noise_type == "babble":
        if not babble_pool:
            raise ValueError(
                "'babble' noise requires a non-empty babble_pool. "
                "Pass babble_pool to _generate_synthetic_noise."
            )
        mixed = np.zeros(n_samples, dtype=np.float32)
        n_spk = min(5, len(babble_pool))
        for idx in rng.integers(0, len(babble_pool), size=n_spk):
            src = babble_pool[int(idx)]
            if len(src) >= n_samples:
                off = rng.integers(0, len(src) - n_samples + 1)
                seg = src[off: off + n_samples]
            else:
                seg = np.tile(src, int(np.ceil(n_samples / len(src))))[:n_samples]
            mixed += seg.astype(np.float32)
        noise = mixed

    elif noise_type == "street":
        white = rng.standard_normal(n_samples).astype(np.float64)
        sos = butter(4, [200 / (SR / 2), 2000 / (SR / 2)], btype="band", output="sos")
        noise = sosfilt(sos, np.cumsum(white)).astype(np.float32)

    elif noise_type == "cafe":
        white = rng.standard_normal(n_samples).astype(np.float64)
        sos = butter(4, 500 / (SR / 2), btype="low", output="sos")
        noise = sosfilt(sos, np.cumsum(white)).astype(np.float32)

    elif noise_type == "music":
        # Broadband pink-like + 2 Hz AM (simulates rhythmic music energy)
        white = rng.standard_normal(n_samples).astype(np.float64)
        sos = butter(4, [100 / (SR / 2), 8000 / (SR / 2)], btype="band", output="sos")
        base = sosfilt(sos, np.cumsum(white)).astype(np.float32)
        am = (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t)).astype(np.float32)
        noise = base * am

    elif noise_type == "office":
        # 50 Hz HVAC hum + Poisson burst noise (keyboard) + 3 kHz fan whine
        hum = (0.3 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
        fan_sos = butter(4, [2800 / (SR / 2), 3200 / (SR / 2)], btype="band", output="sos")
        fan = 0.15 * sosfilt(
            fan_sos, rng.standard_normal(n_samples).astype(np.float64)
        ).astype(np.float32)
        # Sparse keyboard clicks: ~5 per second, 2 ms burst
        burst = np.zeros(n_samples, dtype=np.float32)
        n_clicks = int(5 * n_samples / SR)
        click_positions = rng.integers(0, n_samples - 32, size=n_clicks)
        for pos in click_positions:
            burst[pos: pos + 32] += rng.standard_normal(32).astype(np.float32) * 0.4
        noise = hum + fan + burst

    elif noise_type == "traffic":
        # Brown noise (1/f²) + low-frequency rumble + occasional horn pulse
        white = rng.standard_normal(n_samples).astype(np.float64)
        brown = np.cumsum(np.cumsum(white))  # double integration → -6 dB/oct
        brown_sos = butter(4, 600 / (SR / 2), btype="low", output="sos")
        brown = sosfilt(brown_sos, brown).astype(np.float32)
        # Horn pulses: ~0.5 per second, 150 ms, centred at 800 Hz
        horn = np.zeros(n_samples, dtype=np.float32)
        n_horns = max(1, int(0.5 * n_samples / SR))
        horn_len = int(0.15 * SR)
        horn_sos = butter(4, [600 / (SR / 2), 1000 / (SR / 2)], btype="band", output="sos")
        horn_tone = sosfilt(horn_sos, rng.standard_normal(horn_len).astype(np.float64)).astype(np.float32)
        for pos in rng.integers(0, n_samples - horn_len, size=n_horns):
            horn[pos: pos + horn_len] += horn_tone * 0.6
        noise = brown + horn

    elif noise_type == "wind":
        # Pink noise + strong low-pass + slow 0.3 Hz AM (gusts)
        white = rng.standard_normal(n_samples).astype(np.float64)
        sos = butter(6, 500 / (SR / 2), btype="low", output="sos")
        base = sosfilt(sos, np.cumsum(white)).astype(np.float32)
        am = (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)).astype(np.float32)
        noise = base * am

    else:
        raise ValueError(
            f"Unsupported noise_type '{noise_type}'. Choose from: {VALID_NOISE_TYPES}"
        )

    rms_val = _rms(noise)
    if rms_val > 1e-8:
        noise = (noise / rms_val).astype(np.float32)
    return noise


def _load_noise_from_file(
    noise_dir: Path,
    noise_type: str,
    n_samples: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Load a random noise clip from *noise_dir/{noise_type}/*.wav.

    Args:
        noise_dir:   Root directory containing per-type subdirectories.
        noise_type:  Noise type name (must match a subdirectory name).
        n_samples:   Required output length in samples.
        rng:         Random generator for file selection and offset.

    Returns:
        Float32 noise array of shape (n_samples,) normalised to unit RMS,
        or None if no matching files are found.
    """
    import librosa  # type: ignore[import]

    type_dir = noise_dir / noise_type
    if not type_dir.is_dir():
        return None

    wav_files = sorted(type_dir.glob("*.wav")) + sorted(type_dir.glob("*.WAV"))
    if not wav_files:
        return None

    chosen = wav_files[int(rng.integers(0, len(wav_files)))]
    try:
        wav, _ = librosa.load(str(chosen), sr=SR, mono=True, dtype=np.float32)
    except Exception as exc:
        logger.warning("Failed to load noise file %s: %s", chosen.name, exc)
        return None

    # Tile/trim to n_samples
    if len(wav) < n_samples:
        reps = int(np.ceil(n_samples / len(wav)))
        wav = np.tile(wav, reps)
    offset = int(rng.integers(0, max(1, len(wav) - n_samples + 1)))
    wav = wav[offset: offset + n_samples].astype(np.float32)

    rms_val = _rms(wav)
    if rms_val < 1e-8:
        return None
    return (wav / rms_val).astype(np.float32)


def _get_noise_signal(
    noise_type: str,
    n_samples: int,
    rng: np.random.Generator,
    babble_pool: Optional[List[np.ndarray]] = None,
    noise_dir: Optional[Path] = None,
) -> np.ndarray:
    """Return a noise signal, preferring real files over synthetic generation.

    Args:
        noise_type:  One of VALID_NOISE_TYPES.
        n_samples:   Output length in samples.
        rng:         Random generator.
        babble_pool: Required for 'babble' fallback synthesis.
        noise_dir:   Directory of real noise files (optional).

    Returns:
        Noise signal, float32, shape (n_samples,), unit RMS.
    """
    if noise_dir is not None and noise_type != "babble":
        real = _load_noise_from_file(noise_dir, noise_type, n_samples, rng)
        if real is not None:
            logger.debug("Using real noise file for type '%s'.", noise_type)
            return real

    return _generate_synthetic_noise(noise_type, n_samples, rng, babble_pool)


def add_noise_snr(
    clean: np.ndarray,
    snr_db: float,
    noise_type: str,
    rng: np.random.Generator,
    babble_pool: Optional[List[np.ndarray]] = None,
    noise_dir: Optional[Path] = None,
    validate: bool = True,
) -> np.ndarray:
    """Mix a clean signal with structured noise at a target SNR.

    Formula: noisy = clean + noise_unit * rms_clean * 10^(-snr_db / 20)

    Args:
        clean:       Input signal, float32, shape (N,).
        snr_db:      Target Signal-to-Noise Ratio in dB.
        noise_type:  Noise category (see VALID_NOISE_TYPES).
        rng:         Random generator for reproducibility.
        babble_pool: Required for 'babble' noise.
        noise_dir:   Directory with real noise WAV files (optional).
        validate:    If True, assert SNR within ±SNR_TOLERANCE_DB.

    Returns:
        Noisy signal, float32, shape (N,).

    Raises:
        AssertionError: If validate=True and actual SNR deviates excessively.
    """
    clean_f = clean.astype(np.float32)
    rms_clean = _rms(clean_f)
    noise_unit = _get_noise_signal(
        noise_type, len(clean_f), rng, babble_pool, noise_dir
    )
    noise_amplitude = rms_clean * (10.0 ** (-snr_db / 20.0))
    noisy = clean_f + noise_unit * noise_amplitude

    if validate:
        actual_snr = _measure_snr_db(clean_f, noisy)
        assert abs(actual_snr - snr_db) <= SNR_TOLERANCE_DB, (
            f"SNR validation failed: target={snr_db:.1f} dB, "
            f"actual={actual_snr:.2f} dB, "
            f"delta={abs(actual_snr - snr_db):.2f} > {SNR_TOLERANCE_DB}"
        )
    return noisy


def load_audio(path: Path, target_samples: int = WIN_SAMPLES) -> np.ndarray:
    """Load a mono WAV file resampled to SR, padded/cropped to target_samples.

    Args:
        path:           Path to the audio file.
        target_samples: Desired output length in samples.

    Returns:
        Float32 array of shape (target_samples,). Returns silence on failure.
    """
    import librosa  # type: ignore[import]

    try:
        wav, _ = librosa.load(str(path), sr=SR, mono=True, dtype=np.float32)
    except Exception as exc:
        logger.warning("Failed to load %s: %s — substituting silence.", path.name, exc)
        return np.zeros(target_samples, dtype=np.float32)

    if len(wav) < target_samples:
        wav = np.pad(wav, (0, target_samples - len(wav)))
    else:
        wav = wav[:target_samples]
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------


def _cache_key(
    audio_path: Path, snr_db: float, noise_type: str, seed: int
) -> str:
    """Generate a deterministic hex cache key from noise parameters.

    Args:
        audio_path: Source audio file path.
        snr_db:     SNR in dB.
        noise_type: Noise type string.
        seed:       Random seed.

    Returns:
        MD5 hex digest suitable for use as a filename.
    """
    tag = f"{audio_path}|{snr_db:.1f}|{noise_type}|{seed}"
    return hashlib.md5(tag.encode()).hexdigest()


def _load_or_create_noisy(
    audio_path: Path,
    snr_db: float,
    noise_type: str,
    seed: int,
    cache_dir: Optional[Path],
    babble_pool: Optional[List[np.ndarray]] = None,
    noise_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load a cached noisy waveform or create and cache it.

    Args:
        audio_path:  Source audio file.
        snr_db:      Target SNR in dB.
        noise_type:  Noise type string.
        seed:        Random seed for this repetition.
        cache_dir:   Directory for .npy cache files; None disables caching.
        babble_pool: Required for babble noise.
        noise_dir:   Directory with real noise WAV files (optional).

    Returns:
        Noisy waveform, float32, shape (WIN_SAMPLES,).
    """
    if cache_dir is not None:
        key = _cache_key(audio_path, snr_db, noise_type, seed)
        cache_file = cache_dir / f"{key}.npy"
        if cache_file.exists():
            return np.load(str(cache_file))

    clean = load_audio(audio_path)
    rng = np.random.default_rng(seed)
    noisy = add_noise_snr(clean, snr_db, noise_type, rng, babble_pool, noise_dir)

    if cache_dir is not None:
        np.save(str(cache_dir / f"{key}.npy"), noisy)

    return noisy


# ---------------------------------------------------------------------------
# WER utility
# ---------------------------------------------------------------------------


def _wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using the Wagner–Fischer algorithm.

    Args:
        reference:  Ground-truth label string.
        hypothesis: Model prediction string.

    Returns:
        WER ∈ [0, ∞).
    """
    ref = reference.lower().strip().split()
    hyp = hypothesis.lower().strip().split()
    n, m = len(ref), len(hyp)
    if n == 0:
        return float(m)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / n


# ---------------------------------------------------------------------------
# Model backend abstractions
# ---------------------------------------------------------------------------


class _BasePredictor:
    """Abstract base class for all inference backends.

    Subclasses must override ``predict_batch`` and set ``model_name``.
    """

    model_name: str = "unknown"

    def predict_batch(self, waveforms: List[np.ndarray]) -> List[str]:
        """Run inference on a batch of waveforms.

        Args:
            waveforms: List of float32 arrays, each shape (WIN_SAMPLES,).

        Returns:
            List of predicted label strings.
        """
        raise NotImplementedError


class OnnxPredictor(_BasePredictor):
    """ONNX Runtime inference backend.

    Reads onnx_config.json to locate the INT8 model file and label names.

    Args:
        model_path: Directory with onnx_config.json, or direct .onnx file.
        device:     'cuda' or 'cpu'.
    """

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        import onnxruntime as ort  # type: ignore[import]

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        if model_path.is_dir():
            config_file = model_path / "onnx_config.json"
            if not config_file.exists():
                raise FileNotFoundError(
                    f"onnx_config.json not found in {model_path}."
                )
            config = json.loads(config_file.read_text(encoding="utf-8"))
            onnx_file = model_path / config["model_int8"]
            self.label_names: List[str] = (
                config.get("label_names") or config.get("labels", [])
            )
            self.model_name = config.get("model_name", "ONNX_INT8")
        else:
            onnx_file = model_path
            self.label_names = []
            self.model_name = model_path.stem

        self._session = ort.InferenceSession(str(onnx_file), providers=providers)
        self._input_name: str = self._session.get_inputs()[0].name
        logger.info("OnnxPredictor: loaded %s", onnx_file.name)

    def predict_batch(self, waveforms: List[np.ndarray]) -> List[str]:
        """Run ONNX inference on a batch of waveforms."""
        predictions: List[str] = []
        for wav in waveforms:
            inp = wav[np.newaxis, :].astype(np.float32)
            logits = self._session.run(None, {self._input_name: inp})[0]
            idx = int(np.argmax(logits, axis=-1).ravel()[0])
            label = (
                self.label_names[idx]
                if self.label_names and idx < len(self.label_names)
                else str(idx)
            )
            predictions.append(label)
        return predictions


class HuggingFacePredictor(_BasePredictor):
    """HuggingFace AudioClassification inference backend.

    Works for any fine-tuned or base Wav2Vec2 checkpoint directory.

    Args:
        model_path:  Local directory with config.json + model weights.
        device:      'cuda' or 'cpu'.
        name_override: Override the model_name attribute (e.g., 'Wav2Vec2_base').
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        name_override: str = "",
    ) -> None:
        import torch  # type: ignore[import]
        from transformers import (  # type: ignore[import]
            AutoFeatureExtractor,
            AutoModelForAudioClassification,
        )

        self.model_name = name_override or model_path.name
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._extractor = AutoFeatureExtractor.from_pretrained(str(model_path))
        self._model = AutoModelForAudioClassification.from_pretrained(
            str(model_path)
        ).to(self._device)
        self._model.eval()
        logger.info(
            "HuggingFacePredictor: loaded %s (device=%s)",
            model_path.name,
            self._device,
        )

    def predict_batch(self, waveforms: List[np.ndarray]) -> List[str]:
        """Run HuggingFace model inference on a batch of waveforms."""
        import torch  # type: ignore[import]

        predictions: List[str] = []
        for wav in waveforms:
            inputs = self._extractor(
                wav, sampling_rate=SR, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            idx = int(logits.argmax(dim=-1).item())
            label = self._model.config.id2label.get(idx, str(idx))
            predictions.append(label)
        return predictions


class MFCCSVMPredictor(_BasePredictor):
    """MFCC + SVM baseline predictor.

    Feature vector (78-dim, matching benchmark_mfcc_svm.py):
        MFCC mean+std (26) + Δ mean+std (26) + ΔΔ mean+std (26)

    The SVM is trained once from ``train_csv`` at construction time, then
    optionally persisted as a pickle for fast re-use.

    Args:
        train_csv:     CSV with audio_path + label columns (training data).
                       Required if ``pretrained_pkl`` is None or missing.
        pretrained_pkl: Path to a pre-fitted sklearn Pipeline pickle.
                       Loaded instead of training if it exists.
        save_pkl:      If not None, save the fitted model here after training.
        svm_c:         SVM regularisation parameter C.
        max_per_class: Maximum train samples per class (0 = all).
    """

    model_name: str = "MFCC_SVM"
    N_MFCC: int = 13
    HOP_LENGTH: int = 512
    N_FFT: int = 2048

    def __init__(
        self,
        train_csv: Optional[Path] = None,
        pretrained_pkl: Optional[Path] = None,
        save_pkl: Optional[Path] = None,
        svm_c: float = 10.0,
        max_per_class: int = 3000,
    ) -> None:
        import pickle
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        if pretrained_pkl is not None and pretrained_pkl.exists():
            logger.info("MFCCSVMPredictor: loading pre-fitted model from %s", pretrained_pkl)
            with pretrained_pkl.open("rb") as f:
                data = pickle.load(f)
            self._pipeline: Pipeline = data["pipeline"]
            self._label_list: List[str] = data["label_list"]
            logger.info("Labels: %s", self._label_list)
            return

        if train_csv is None or not train_csv.exists():
            raise FileNotFoundError(
                "MFCCSVMPredictor requires either an existing pretrained_pkl "
                f"or a valid train_csv. Got train_csv={train_csv}, "
                f"pretrained_pkl={pretrained_pkl}."
            )

        logger.info("MFCCSVMPredictor: training SVM from %s", train_csv)
        df = pd.read_csv(train_csv, dtype=str)
        if {"audio_path", "label"} - set(df.columns):
            raise ValueError(
                "train_csv must have 'audio_path' and 'label' columns."
            )
        df["audio_path"] = df["audio_path"].apply(
            lambda p: PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)
        )
        df = df[df["audio_path"].apply(Path.exists)].reset_index(drop=True)
        self._label_list = sorted(df["label"].unique().tolist())
        label2id = {lbl: i for i, lbl in enumerate(self._label_list)}

        if max_per_class > 0:
            df = (
                df.groupby("label", group_keys=False)
                .apply(
                    lambda g: g.sample(min(len(g), max_per_class), random_state=42)
                )
                .reset_index(drop=True)
            )

        logger.info(
            "Extracting MFCC features from %d training samples…", len(df)
        )
        X, y = self._build_features(df, label2id)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=svm_c,
                    gamma="scale",
                    class_weight="balanced",
                    probability=False,
                    random_state=42,
                ),
            ),
        ])
        t0 = time.perf_counter()
        pipeline.fit(X, y)
        logger.info(
            "SVM trained in %.1f s on %d samples.", time.perf_counter() - t0, len(y)
        )
        self._pipeline = pipeline

        if save_pkl is not None:
            save_pkl.parent.mkdir(parents=True, exist_ok=True)
            import pickle
            with save_pkl.open("wb") as f:
                pickle.dump({"pipeline": pipeline, "label_list": self._label_list}, f)
            logger.info("Fitted SVM saved → %s", save_pkl)

    def _extract_mfcc(self, wav: np.ndarray) -> np.ndarray:
        """Extract 78-dim MFCC + Δ + ΔΔ feature vector.

        Args:
            wav: Float32 waveform of shape (WIN_SAMPLES,).

        Returns:
            Feature vector, float32, shape (78,).
        """
        import librosa  # type: ignore[import]

        mfcc = librosa.feature.mfcc(
            y=wav,
            sr=SR,
            n_mfcc=self.N_MFCC,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
        )
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        return np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            d1.mean(axis=1),   d1.std(axis=1),
            d2.mean(axis=1),   d2.std(axis=1),
        ]).astype(np.float32)

    def _build_features(
        self, df: pd.DataFrame, label2id: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for all rows in df.

        Args:
            df:       DataFrame with audio_path (Path) and label columns.
            label2id: Label → integer ID mapping.

        Returns:
            Tuple (X, y) of numpy arrays.
        """
        X, y = [], []
        errors = 0
        for _, row in df.iterrows():
            try:
                wav = load_audio(Path(row["audio_path"]))
                X.append(self._extract_mfcc(wav))
                y.append(label2id[row["label"]])
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    logger.warning("MFCC extraction failed for %s: %s", row["audio_path"], exc)
        if errors:
            logger.warning("Total MFCC extraction errors: %d / %d", errors, len(df))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def predict_batch(self, waveforms: List[np.ndarray]) -> List[str]:
        """Classify waveforms using the fitted MFCC + SVM pipeline.

        Args:
            waveforms: List of float32 arrays, each shape (WIN_SAMPLES,).

        Returns:
            List of predicted label strings.
        """
        feats = np.stack([self._extract_mfcc(w) for w in waveforms])
        ids = self._pipeline.predict(feats)
        return [self._label_list[int(i)] for i in ids]


class WhisperPredictor(_BasePredictor):
    """Whisper Tiny zero-shot baseline predictor.

    Transcribes audio with openai-whisper and maps the transcription to the
    nearest command label using keyword overlap (longest match wins).

    Args:
        label_keywords: Optional dict {label: [keyword, ...]} for matching.
                        If None, each label's words are used as its keywords.
        language:       ISO-639-1 language code passed to Whisper (e.g. 'ru').
        model_size:     Whisper model size: 'tiny', 'base', 'small', etc.
    """

    model_name: str = "Whisper_Tiny"

    def __init__(
        self,
        label_keywords: Optional[Dict[str, List[str]]] = None,
        language: str = "ru",
        model_size: str = "tiny",
    ) -> None:
        try:
            import whisper  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required for the Whisper baseline. "
                "Install with: pip install openai-whisper"
            ) from exc

        self._model = whisper.load_model(model_size)
        self._language = language
        self._label_keywords: Dict[str, List[str]] = label_keywords or {}
        self.model_name = f"Whisper_{model_size}"
        logger.info("WhisperPredictor: loaded %s, language=%s", model_size, language)

    def set_labels(self, labels: Sequence[str]) -> None:
        """Derive keyword maps from label strings if none were provided.

        Args:
            labels: Sequence of known label strings from the test set.
        """
        if not self._label_keywords:
            self._label_keywords = {lbl: lbl.lower().split() for lbl in labels}

    def _match_label(self, text: str) -> str:
        """Map a transcript to the best matching label via keyword overlap.

        Strategy: for each label, count how many of its keywords appear as
        substrings of the transcription. The label with the most matches
        (tie-broken by total keyword length) wins.

        Args:
            text: Lowercased transcription string.

        Returns:
            Best matching label string, or the first label as fallback.
        """
        if not self._label_keywords:
            return "unknown"

        best_label = next(iter(self._label_keywords))
        best_score = (-1, 0)

        for label, keywords in self._label_keywords.items():
            if not keywords:
                continue
            count = sum(1 for kw in keywords if kw in text)
            total_len = sum(len(kw) for kw in keywords if kw in text)
            score = (count, total_len)
            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict_batch(self, waveforms: List[np.ndarray]) -> List[str]:
        """Transcribe waveforms with Whisper and map to command labels.

        Args:
            waveforms: List of float32 arrays, each shape (WIN_SAMPLES,).

        Returns:
            List of predicted label strings.
        """
        predictions: List[str] = []
        for wav in waveforms:
            try:
                result = self._model.transcribe(
                    wav.astype(np.float32),
                    language=self._language,
                    fp16=False,
                    verbose=False,
                )
                transcript = result.get("text", "").lower().strip()
                predictions.append(self._match_label(transcript))
            except Exception as exc:
                logger.warning("Whisper transcription failed: %s", exc)
                predictions.append(self._match_label(""))
        return predictions


# ---------------------------------------------------------------------------
# Predictor factory
# ---------------------------------------------------------------------------


def build_predictor(model_path: Path, device: str) -> _BasePredictor:
    """Auto-detect model backend and instantiate the correct predictor.

    Detection priority:
        1. ONNX directory (contains onnx_config.json)
        2. ONNX file      (.onnx suffix)
        3. HuggingFace    (contains config.json)

    Args:
        model_path: Path to model file or directory.
        device:     Compute device, 'cuda' or 'cpu'.

    Returns:
        Instantiated predictor.

    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError:        If the backend cannot be determined.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if model_path.is_dir() and (model_path / "onnx_config.json").exists():
        return OnnxPredictor(model_path, device)
    if model_path.is_file() and model_path.suffix == ".onnx":
        return OnnxPredictor(model_path, device)
    if model_path.is_dir() and (model_path / "config.json").exists():
        return HuggingFacePredictor(model_path, device)

    raise ValueError(
        f"Cannot determine model backend for: {model_path}\n"
        "Expected: directory with onnx_config.json, .onnx file, or HF directory."
    )


def build_baseline_predictors(
    baseline_ids: List[str],
    mfcc_svm_train_csv: Optional[Path],
    mfcc_svm_pkl: Optional[Path],
    wav2vec2_base_path: Optional[Path],
    whisper_keywords_json: Optional[Path],
    device: str,
) -> List[_BasePredictor]:
    """Instantiate all requested baseline predictors.

    Args:
        baseline_ids:         List of baseline identifiers (e.g. ['mfcc_svm', 'whisper_tiny']).
        mfcc_svm_train_csv:   Training CSV for MFCC+SVM.
        mfcc_svm_pkl:         Pre-fitted pickle path for MFCC+SVM.
        wav2vec2_base_path:   Directory for Wav2Vec2 base HF model.
        whisper_keywords_json: JSON file mapping label → [keyword, ...].
        device:               Compute device.

    Returns:
        List of instantiated _BasePredictor objects (skips failed ones with a warning).
    """
    predictors: List[_BasePredictor] = []
    for bid in baseline_ids:
        try:
            if bid == "mfcc_svm":
                save_pkl = (
                    PROJECT_ROOT / "artifacts" / "benchmarks" / "mfcc_svm_model.pkl"
                    if mfcc_svm_pkl is None
                    else mfcc_svm_pkl
                )
                p = MFCCSVMPredictor(
                    train_csv=mfcc_svm_train_csv,
                    pretrained_pkl=mfcc_svm_pkl,
                    save_pkl=save_pkl,
                )
                predictors.append(p)

            elif bid == "whisper_tiny":
                kw: Optional[Dict[str, List[str]]] = None
                if whisper_keywords_json is not None and whisper_keywords_json.exists():
                    kw = json.loads(
                        whisper_keywords_json.read_text(encoding="utf-8")
                    )
                predictors.append(WhisperPredictor(label_keywords=kw))

            elif bid == "wav2vec2_base":
                if wav2vec2_base_path is None or not wav2vec2_base_path.exists():
                    raise FileNotFoundError(
                        f"--wav2vec2_base_path not set or missing: {wav2vec2_base_path}"
                    )
                p2 = HuggingFacePredictor(
                    wav2vec2_base_path, device, name_override="Wav2Vec2_base"
                )
                predictors.append(p2)

            else:
                logger.warning("Unknown baseline id '%s' — skipping.", bid)

        except Exception as exc:
            logger.warning("Failed to build baseline '%s': %s — skipping.", bid, exc)

    return predictors


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_test_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate the test CSV.

    Required columns:  audio_path, label
    Optional columns:  is_ood (bool — default False if missing)

    Args:
        csv_path: Path to the test CSV.

    Returns:
        Validated DataFrame with resolved Path objects in audio_path.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError:        If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    missing_cols = {"audio_path", "label"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Test CSV missing columns: {missing_cols}. Found: {list(df.columns)}"
        )

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    df["audio_path"] = df["audio_path"].apply(_resolve)
    df["label"] = df["label"].str.strip()

    if "is_ood" not in df.columns:
        df["is_ood"] = False
    else:
        df["is_ood"] = df["is_ood"].fillna("false").str.lower().isin(
            ("1", "true", "yes")
        )

    missing_files = df["audio_path"][~df["audio_path"].apply(Path.exists)]
    if not missing_files.empty:
        logger.warning(
            "%d audio file(s) not found — skipping:\n  %s",
            len(missing_files),
            "\n  ".join(str(p) for p in missing_files[:5]),
        )
        df = df[df["audio_path"].apply(Path.exists)].reset_index(drop=True)

    n_ood = int(df["is_ood"].sum())
    logger.info(
        "Test set: %d samples, %d OOD, labels=%s",
        len(df),
        n_ood,
        sorted(df["label"].unique().tolist()),
    )
    return df


def _build_babble_pool(df: pd.DataFrame, max_speakers: int = 50) -> List[np.ndarray]:
    """Load a pool of clean waveforms for babble noise synthesis.

    Args:
        df:            Test DataFrame with audio_path column.
        max_speakers:  Maximum number of waveforms to load.

    Returns:
        List of float32 waveforms.
    """
    rng_temp = np.random.default_rng(0)
    paths = df["audio_path"].tolist()
    rng_temp.shuffle(paths)  # type: ignore[arg-type]
    pool: List[np.ndarray] = []
    for p in paths[:max_speakers]:
        wav = load_audio(p)
        if _rms(wav) > 1e-4:
            pool.append(wav)
    logger.info("Babble pool: %d waveforms loaded.", len(pool))
    return pool


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def _evaluate_one(
    predictor: _BasePredictor,
    df_full: pd.DataFrame,
    snr_db: float,
    noise_type: str,
    seed: int,
    cache_dir: Optional[Path],
    babble_pool: Optional[List[np.ndarray]],
    noise_dir: Optional[Path],
    per_class_breakdown: bool,
) -> Dict[str, Any]:
    """Evaluate one (SNR, noise_type, seed) cell for a given predictor.

    Computes metrics separately for in-distribution and OOD subsets.

    Args:
        predictor:           Inference backend.
        df_full:             Full test DataFrame.
        snr_db:              Target SNR in dB (np.inf = clean signal).
        noise_type:          Noise type string.
        seed:                Random seed.
        cache_dir:           Directory for noisy-audio cache; None disables.
        babble_pool:         List of waveforms for babble noise.
        noise_dir:           Directory with real noise WAV files (optional).
        per_class_breakdown: If True, compute per-class F1 scores.

    Returns:
        Dict with keys: snr_db, noise_type, seed,
        in_dist: {f1_macro, per_class_f1, precision, recall, wer_mean, n_samples},
        ood:     {f1_macro, per_class_f1, precision, recall, wer_mean, n_samples}.
    """
    y_true_in: List[str] = []
    y_pred_in: List[str] = []
    wer_in: List[float] = []
    y_true_ood: List[str] = []
    y_pred_ood: List[str] = []
    wer_ood: List[float] = []

    for _, row in df_full.iterrows():
        audio_path: Path = row["audio_path"]
        true_label: str = row["label"]
        is_ood: bool = bool(row["is_ood"])

        if np.isinf(snr_db):
            wav = load_audio(audio_path)
        else:
            try:
                wav = _load_or_create_noisy(
                    audio_path, snr_db, noise_type, seed,
                    cache_dir, babble_pool, noise_dir,
                )
            except AssertionError as exc:
                logger.warning(
                    "SNR validation %s: %s — using clean audio.",
                    audio_path.name, exc,
                )
                wav = load_audio(audio_path)

        pred_label = predictor.predict_batch([wav])[0]

        if is_ood:
            y_true_ood.append(true_label)
            y_pred_ood.append(pred_label)
            wer_ood.append(_wer(true_label, pred_label))
        else:
            y_true_in.append(true_label)
            y_pred_in.append(pred_label)
            wer_in.append(_wer(true_label, pred_label))

    def _compute_metrics(
        y_true: List[str], y_pred: List[str], wers: List[float]
    ) -> Dict[str, Any]:
        if not y_true:
            return {
                "f1_macro": float("nan"), "per_class_f1": {},
                "precision": float("nan"), "recall": float("nan"),
                "wer_mean": float("nan"), "n_samples": 0,
            }
        labels = sorted(set(y_true))
        f1_mac = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        prec = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        per_class: Dict[str, float] = {}
        if per_class_breakdown:
            scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            per_class = {lbl: float(sc) for lbl, sc in zip(labels, scores)}
        return {
            "f1_macro": f1_mac, "per_class_f1": per_class,
            "precision": prec, "recall": rec,
            "wer_mean": float(np.mean(wers)) if wers else float("nan"),
            "n_samples": len(y_true),
        }

    return {
        "snr_db": float(snr_db),
        "noise_type": noise_type,
        "seed": seed,
        "in_dist": _compute_metrics(y_true_in, y_pred_in, wer_in),
        "ood": _compute_metrics(y_true_ood, y_pred_ood, wer_ood),
    }


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    raw_rows: List[Dict[str, Any]],
    model_name: str,
    test_csv_name: str,
    timestamp: str,
    per_class_breakdown: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate per-seed raw rows into the unified CSV schema and summary JSON.

    Output CSV schema:
        snr_db, noise_type, model, command_class, f1_mean, f1_std,
        wer, n_samples, is_ood, precision, recall, timestamp

    Args:
        raw_rows:            List of result dicts from _evaluate_one.
        model_name:          Human-readable model identifier.
        test_csv_name:       Basename of the test CSV.
        timestamp:           ISO-8601 timestamp string.
        per_class_breakdown: Whether per-class rows should be generated.

    Returns:
        Tuple (csv_df, summary_dict).
    """
    flat: List[Dict[str, Any]] = []
    for row in raw_rows:
        base = {"snr_db": row["snr_db"], "noise_type": row["noise_type"], "seed": row["seed"]}
        for split_name, is_ood_flag in [("in_dist", False), ("ood", True)]:
            m = row[split_name]
            if m["n_samples"] == 0:
                continue
            flat.append({
                **base,
                "is_ood": is_ood_flag,
                "f1_macro": m["f1_macro"],
                "precision": m["precision"],
                "recall": m["recall"],
                "wer_mean": m["wer_mean"],
                "n_samples": m["n_samples"],
                "per_class_f1": m["per_class_f1"],
            })

    if not flat:
        logger.warning("No valid evaluation results for model '%s'.", model_name)
        return pd.DataFrame(), {}

    csv_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple, List[Dict]] = {}
    for r in flat:
        grouped.setdefault((r["snr_db"], r["noise_type"], r["is_ood"]), []).append(r)

    all_classes: List[str] = sorted({
        cls
        for rows in grouped.values()
        for r in rows
        for cls in r["per_class_f1"]
    })

    for (snr_db, noise_type, is_ood), rows in grouped.items():
        f1_v = [r["f1_macro"] for r in rows if not np.isnan(r["f1_macro"])]
        wer_v = [r["wer_mean"] for r in rows if not np.isnan(r["wer_mean"])]
        prec_v = [r["precision"] for r in rows if not np.isnan(r["precision"])]
        rec_v = [r["recall"] for r in rows if not np.isnan(r["recall"])]

        csv_rows.append({
            "snr_db": snr_db, "noise_type": noise_type, "model": model_name,
            "command_class": _MACRO_CLASS_LABEL,
            "f1_mean": float(np.mean(f1_v)) if f1_v else float("nan"),
            "f1_std": float(np.std(f1_v)) if len(f1_v) > 1 else 0.0,
            "wer": float(np.mean(wer_v)) if wer_v else float("nan"),
            "n_samples": rows[0]["n_samples"],
            "is_ood": is_ood,
            "precision": float(np.mean(prec_v)) if prec_v else float("nan"),
            "recall": float(np.mean(rec_v)) if rec_v else float("nan"),
            "timestamp": timestamp,
        })

        if per_class_breakdown:
            for cls in all_classes:
                cls_v = [r["per_class_f1"][cls] for r in rows if cls in r["per_class_f1"]]
                if not cls_v:
                    continue
                csv_rows.append({
                    "snr_db": snr_db, "noise_type": noise_type, "model": model_name,
                    "command_class": cls,
                    "f1_mean": float(np.mean(cls_v)),
                    "f1_std": float(np.std(cls_v)) if len(cls_v) > 1 else 0.0,
                    "wer": float("nan"),
                    "n_samples": rows[0]["n_samples"],
                    "is_ood": is_ood,
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "timestamp": timestamp,
                })

    csv_df = pd.DataFrame(csv_rows)
    n_seeds = len({r["seed"] for r in raw_rows})

    def _frame_to_dict(frame: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for _, r in frame.iterrows():
            snr_key = "clean" if np.isinf(float(r["snr_db"])) else f"snr_{int(r['snr_db'])}"
            key = f"{snr_key}_{r['noise_type']}"
            f1 = float(r["f1_mean"])
            std = float(r["f1_std"])
            se = std / np.sqrt(max(n_seeds, 1))
            out[key] = {
                "f1": round(f1, 4),
                "f1_std": round(std, 4),
                "ci95": [round(f1 - 1.96 * se, 4), round(f1 + 1.96 * se, 4)],
                "precision": round(float(r["precision"]), 4),
                "recall": round(float(r["recall"]), 4),
                "wer": round(float(r["wer"]), 4),
                "n": int(r["n_samples"]),
            }
        return out

    # Per-class degradation: identify which commands degrade first
    per_class_summary: Dict[str, Any] = {}
    if per_class_breakdown and not csv_df.empty:
        class_rows = csv_df[
            (csv_df["command_class"] != _MACRO_CLASS_LABEL) & (~csv_df["is_ood"])
        ]
        for cls, grp in class_rows.groupby("command_class"):
            valid = grp.dropna(subset=["f1_mean"])
            if valid.empty:
                continue
            worst_idx = valid["f1_mean"].idxmin()
            per_class_summary[str(cls)] = {
                "min_f1": round(float(valid["f1_mean"].min()), 4),
                "worst_condition": (
                    f"snr_{int(valid.loc[worst_idx, 'snr_db'])}"
                    f"_{valid.loc[worst_idx, 'noise_type']}"
                ),
                "f1_by_condition": {
                    f"snr_{int(r['snr_db'])}_{r['noise_type']}": round(float(r["f1_mean"]), 4)
                    for _, r in grp.sort_values("snr_db").iterrows()
                },
            }

    macro_in = csv_df[(csv_df["command_class"] == _MACRO_CLASS_LABEL) & (~csv_df["is_ood"])]
    macro_ood = csv_df[(csv_df["command_class"] == _MACRO_CLASS_LABEL) & csv_df["is_ood"]]

    summary: Dict[str, Any] = {
        "model": model_name,
        "test_set": test_csv_name,
        "timestamp": timestamp,
        "n_repeats": n_seeds,
        "in_distribution": _frame_to_dict(macro_in),
        "ood": _frame_to_dict(macro_ood),
        "per_class_breakdown": per_class_summary,
    }
    return csv_df, summary


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------


def run_benchmark(
    model_path: Path,
    test_csv: Path,
    snr_values: List[float],
    noise_types: List[str],
    n_repeats: int,
    output_dir: Path,
    device: str,
    random_seed: int,
    include_clean: bool,
    disable_cache: bool,
    per_class_breakdown: bool,
    noise_dir: Optional[Path],
    baselines: List[str],
    mfcc_svm_train_csv: Optional[Path],
    mfcc_svm_pkl: Optional[Path],
    wav2vec2_base_path: Optional[Path],
    whisper_keywords_json: Optional[Path],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the full SNR-profile benchmark for all models and write outputs.

    Args:
        model_path:           Primary model path (ONNX or HF directory).
        test_csv:             Test CSV path.
        snr_values:           List of SNR dB values.
        noise_types:          List of noise type strings.
        n_repeats:            Number of random seed repetitions.
        output_dir:           Directory for output files.
        device:               Compute device.
        random_seed:          Base random seed.
        include_clean:        Also evaluate on clean signal.
        disable_cache:        Disable noisy-audio caching.
        per_class_breakdown:  Compute per-class F1 rows.
        noise_dir:            Directory with real noise WAV files (optional).
        baselines:            List of baseline IDs.
        mfcc_svm_train_csv:   Training CSV for MFCC+SVM.
        mfcc_svm_pkl:         Pre-fitted pickle for MFCC+SVM.
        wav2vec2_base_path:   HF directory for Wav2Vec2 base.
        whisper_keywords_json: JSON with label keyword mapping.

    Returns:
        Tuple (combined_df, combined_summary_dict).
    """
    invalid = set(noise_types) - set(VALID_NOISE_TYPES)
    if invalid:
        raise ValueError(f"Invalid noise_types: {invalid}. Valid: {VALID_NOISE_TYPES}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if disable_cache else (output_dir / "_noise_cache")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = load_test_csv(test_csv)

    main_pred = build_predictor(model_path, device)
    baseline_preds = build_baseline_predictors(
        baselines, mfcc_svm_train_csv, mfcc_svm_pkl,
        wav2vec2_base_path, whisper_keywords_json, device,
    )
    all_predictors: List[_BasePredictor] = [main_pred] + baseline_preds

    known_labels = sorted(df["label"].unique().tolist())
    for pred in all_predictors:
        if isinstance(pred, WhisperPredictor):
            pred.set_labels(known_labels)

    babble_pool: Optional[List[np.ndarray]] = None
    if "babble" in noise_types:
        babble_pool = _build_babble_pool(df)

    snr_grid: List[float] = ([float("inf")] if include_clean else []) + list(snr_values)
    seeds = [random_seed + i for i in range(n_repeats)]

    logger.info(
        "Benchmark: %d models x %d SNR x %d noise x %d repeats",
        len(all_predictors), len(snr_grid), len(noise_types), n_repeats,
    )

    all_csv_dfs: List[pd.DataFrame] = []
    all_summaries: Dict[str, Any] = {
        "test_set": test_csv.name, "timestamp": timestamp, "models": {}
    }

    for predictor in all_predictors:
        logger.info("--- Evaluating: %s ---", predictor.model_name)
        raw_rows: List[Dict[str, Any]] = []

        for snr_db in snr_grid:
            nt_list = ["clean"] if np.isinf(snr_db) else noise_types
            for noise_type in nt_list:
                for seed in seeds:
                    result = _evaluate_one(
                        predictor=predictor, df_full=df,
                        snr_db=snr_db, noise_type=noise_type, seed=seed,
                        cache_dir=cache_dir, babble_pool=babble_pool,
                        noise_dir=noise_dir, per_class_breakdown=per_class_breakdown,
                    )
                    raw_rows.append(result)
                    ood_n = result["ood"]["n_samples"]
                    logger.info(
                        "  SNR=%-5.1f  noise=%-10s  seed=%d  F1_in=%.4f  F1_ood=%s",
                        snr_db, noise_type, seed,
                        result["in_dist"]["f1_macro"],
                        f"{result['ood']['f1_macro']:.4f}" if ood_n > 0 else "n/a",
                    )

        csv_df, summary = aggregate_results(
            raw_rows,
            model_name=predictor.model_name,
            test_csv_name=test_csv.name,
            timestamp=timestamp,
            per_class_breakdown=per_class_breakdown,
        )
        if not csv_df.empty:
            all_csv_dfs.append(csv_df)
        all_summaries["models"][predictor.model_name] = summary

    combined_df = pd.concat(all_csv_dfs, ignore_index=True) if all_csv_dfs else pd.DataFrame()

    csv_path = output_dir / "f1_vs_snr_full.csv"
    if not combined_df.empty:
        combined_df.to_csv(csv_path, index=False, float_format="%.6f")
        logger.info("Full results -> %s  (%d rows)", csv_path, len(combined_df))
    else:
        logger.warning("No results to write.")

    json_path = output_dir / "f1_vs_snr_summary.json"
    json_path.write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Summary saved -> %s", json_path)
    _print_summary_table(combined_df)
    return combined_df, all_summaries


def _print_summary_table(df: pd.DataFrame) -> None:
    """Print a compact multi-model summary table to stdout.

    Args:
        df: Combined output DataFrame (all models, all conditions).
    """
    if df.empty:
        return
    macro_in = df[(df["command_class"] == _MACRO_CLASS_LABEL) & (~df["is_ood"])]
    if macro_in.empty:
        return
    print(f"\n{'='*80}")
    print("  SNR Benchmark -- macro F1 (in-distribution)")
    print(f"{'='*80}")
    print(f"  {'Model':<22}{'SNR(dB)':<10}{'Noise':<12}{'F1 mean':<10}{'F1 std':<10}{'WER':<8}")
    print(f"  {'-'*72}")
    for _, row in macro_in.sort_values(
        ["model", "noise_type", "snr_db"], ascending=[True, True, False]
    ).iterrows():
        snr_str = "clean" if np.isinf(float(row["snr_db"])) else str(int(row["snr_db"]))
        print(
            f"  {row['model']:<22}{snr_str:<10}{row['noise_type']:<12}"
            f"{row['f1_mean']:<10.4f}{row['f1_std']:<10.4f}{row['wer']:<8.4f}"
        )
    ood_df = df[(df["command_class"] == _MACRO_CLASS_LABEL) & df["is_ood"]]
    if not ood_df.empty:
        print(f"\n  OOD F1 (is_ood=True samples):")
        print(f"  {'-'*72}")
        for _, row in ood_df.sort_values(["model", "noise_type", "snr_db"]).iterrows():
            snr_str = "clean" if np.isinf(float(row["snr_db"])) else str(int(row["snr_db"]))
            print(f"  {row['model']:<22}{snr_str:<10}{row['noise_type']:<12}{row['f1_mean']:<10.4f}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with all parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "benchmark_snr_profile.py -- Evaluate speech command models "
            "across SNR levels, noise types, and baselines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=Path, required=True,
        help="Primary model path (ONNX directory/file or HuggingFace directory).")
    parser.add_argument("--test_csv", type=Path, required=True,
        help="CSV with test samples. Columns: audio_path, label. Optional: is_ood.")
    parser.add_argument("--ood_flag_column", type=str, default="is_ood",
        help="Column name in test_csv marking OOD samples.")
    parser.add_argument("--snr_values", type=float, nargs="+", metavar="DB",
        default=[2, 5, 8, 10, 12, 15, 20], help="SNR values in dB.")
    parser.add_argument("--noise_types", type=str, nargs="+", metavar="TYPE",
        default=["white", "babble"], choices=list(VALID_NOISE_TYPES),
        help=f"Noise types. Choices: {VALID_NOISE_TYPES}")
    parser.add_argument("--include_clean", action="store_true", default=False,
        help="Also evaluate on clean signal (SNR = +inf).")
    parser.add_argument("--baselines", type=str, default="",
        help="Comma-separated baselines: mfcc_svm,whisper_tiny,wav2vec2_base")
    parser.add_argument("--mfcc_svm_train_csv", type=Path, default=None,
        help="Training CSV (audio_path + label) for MFCC+SVM baseline.")
    parser.add_argument("--mfcc_svm_pkl", type=Path, default=None,
        help="Pre-fitted MFCC+SVM pickle path (saves training time on re-runs).")
    parser.add_argument("--wav2vec2_base_path", type=Path, default=None,
        help="HuggingFace directory for the Wav2Vec2 base (no LoRA) model.")
    parser.add_argument("--whisper_keywords_json", type=Path, default=None,
        help="JSON dict {label: [keyword, ...]} for Whisper label matching.")
    parser.add_argument("--noise_dir", type=Path, default=None,
        help="Directory with real noise WAV files: noise_dir/{noise_type}/*.wav")
    parser.add_argument("--n_repeats", type=int, default=5,
        help="Stochastic repetitions per (SNR, noise_type) cell.")
    parser.add_argument("--random_seed", type=int, default=42,
        help="Base random seed; actual seeds = random_seed + 0..n_repeats-1.")
    parser.add_argument("--per_class_breakdown", action="store_true", default=False,
        help="Compute and store per-class F1 rows in output CSV.")
    parser.add_argument("--output_dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmarks",
        help="Directory for output CSV and JSON files.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Compute device.")
    parser.add_argument("--no_cache", action="store_true", default=False,
        help="Disable noisy-audio caching.")
    parser.add_argument("--dry_run", action="store_true", default=False,
        help="Validate paths and model loading only -- skip inference.")
    return parser.parse_args()


def main() -> None:
    """Entry point: parse arguments and execute the benchmark."""
    args = _parse_args()
    logger.info("=== benchmark_snr_profile.py  |  %s ===",
                datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("  model_path  : %s", args.model_path)
    logger.info("  test_csv    : %s", args.test_csv)
    logger.info("  snr_values  : %s", args.snr_values)
    logger.info("  noise_types : %s", args.noise_types)
    logger.info("  baselines   : %s", args.baselines)
    logger.info("  per_class   : %s", args.per_class_breakdown)

    if args.dry_run:
        logger.info("DRY RUN -- validating paths only.")
        df = load_test_csv(args.test_csv)
        predictor = build_predictor(args.model_path, args.device)
        logger.info("Dry-run OK: %d samples, backend=%s", len(df), type(predictor).__name__)
        return

    baseline_list = (
        [b.strip() for b in args.baselines.split(",") if b.strip()]
        if args.baselines else []
    )
    invalid_b = set(baseline_list) - set(BASELINE_IDS)
    if invalid_b:
        logger.warning("Unknown baseline ids ignored: %s. Valid: %s", invalid_b, BASELINE_IDS)
        baseline_list = [b for b in baseline_list if b in BASELINE_IDS]

    t0 = time.perf_counter()
    run_benchmark(
        model_path=args.model_path,
        test_csv=args.test_csv,
        snr_values=args.snr_values,
        noise_types=args.noise_types,
        n_repeats=args.n_repeats,
        output_dir=args.output_dir,
        device=args.device,
        random_seed=args.random_seed,
        include_clean=args.include_clean,
        disable_cache=args.no_cache,
        per_class_breakdown=args.per_class_breakdown,
        noise_dir=args.noise_dir,
        baselines=baseline_list,
        mfcc_svm_train_csv=args.mfcc_svm_train_csv,
        mfcc_svm_pkl=args.mfcc_svm_pkl,
        wav2vec2_base_path=args.wav2vec2_base_path,
        whisper_keywords_json=args.whisper_keywords_json,
    )
    logger.info("Benchmark complete in %.1f s.", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
