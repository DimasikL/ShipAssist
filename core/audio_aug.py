from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, RoomSimulator

logger = logging.getLogger(__name__)

AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),

    Gain(min_gain_db=-12, max_gain_db=6, p=0.6),

    TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),

    RoomSimulator(
        min_size_x=3.0, max_size_x=7.0,
        min_size_y=3.0, max_size_y=7.0,
        min_size_z=2.4, max_size_z=3.2,
        min_absorption_value=0.2, max_absorption_value=0.6,
        p=0.4,
    ),
], p=1.0)


def augment_audio(input_path: Path, output_path: Path, aug_n, sample_rate=16000):
    audio_src, sr = librosa.load(input_path, sr=sample_rate)

    for aug_i in range(aug_n):
        audio = audio_src.copy()
        audio = AUGMENT(samples=audio, sample_rate=sample_rate)

        out_stem = output_path.parent / output_path.stem
        sf.write(
            str(out_stem) + f"_{aug_i}" + output_path.suffix,
            audio,
            sample_rate,
        )

    return audio


# ── Maritime noise augmentation ───────────────────────────────────────────────

def _generate_pink_noise(n_samples: int, sr: int) -> np.ndarray:
    """Generate pink (1/f) noise via spectral shaping.

    White noise is shaped in the frequency domain so that power falls off as
    1/f — the closest stationary surrogate for broadband maritime ambient noise
    (engine rumble, hull vibration, sea state).

    Args:
        n_samples: Number of samples to generate.
        sr: Sample rate in Hz (used to compute the frequency axis).

    Returns:
        Normalised float32 array of length *n_samples*, peak amplitude <= 1.0.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    freqs[0] = 1.0  # avoid division-by-zero for the DC component
    white = np.random.randn(n_samples).astype(np.float32)
    spectrum = np.fft.rfft(white)
    pink_spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(pink_spectrum, n=n_samples).astype(np.float32)
    peak = np.max(np.abs(pink))
    return pink / (peak + 1e-9)


def _load_noise_file(noise_path: Path, n_samples: int, sr: int) -> np.ndarray:
    """Load a noise WAV and tile/trim it to exactly *n_samples* frames.

    Args:
        noise_path: Absolute path to the noise WAV file.
        n_samples: Target length in samples.
        sr: Target sample rate (resampling applied if necessary).

    Returns:
        Float32 array of length *n_samples*, not yet amplitude-scaled.
    """
    noise, _ = librosa.load(noise_path, sr=sr, mono=True)
    if len(noise) < n_samples:
        repeats = (n_samples // len(noise)) + 1
        noise = np.tile(noise, repeats)
    noise = noise[:n_samples].astype(np.float32)
    return noise


def add_maritime_noise(
    audio: np.ndarray,
    sr: int,
    target_snr_db: float = 15.0,
    training: bool = False,
    noise_file: Optional[Path] = None,
) -> np.ndarray:
    """Mix maritime ambient noise into *audio* at a controlled SNR.

    This is a surrogate augmentation for the real maritime acoustic environment
    when dedicated recording equipment is unavailable.  It targets the primary
    defense metric (FPR-on-noise) by teaching the model to suppress broadband
    engine / sea-state interference.

    Behaviour
    ---------
    1. Enabled check  -- reads ``settings.augmentation.maritime_noise.enabled``
       and returns *audio* unchanged if False.
    2. Training gate  -- when ``training=False`` the function is a no-op so
       validation and test metrics remain uncontaminated.
    3. Probability gate -- applies with probability
       ``settings.augmentation.maritime_noise.probability`` (default 0.3).
    4. Noise source   -- tries to load ``artifacts/noise/maritime_sample.wav``
       (override via *noise_file*).  Falls back to pink (1/f) noise if absent.
    5. SNR mixing     -- scales noise to achieve *target_snr_db* relative to
       RMS power of *audio*.

    Args:
        audio: Input waveform as a float32 NumPy array (mono, shape (N,)).
        sr: Sample rate of *audio* in Hz.
        target_snr_db: Desired signal-to-noise ratio in dB.  Lower values
            produce more aggressive contamination.  Default 15 dB matches
            moderate vessel engine noise at 3-5 m distance.
        training: Set to True inside the training loop to enable stochastic
            application.  Must be False (default) during eval / inference.
        noise_file: Optional override for the noise WAV path.  When None
            the path is derived from ``settings.paths.artifacts_dir``.

    Returns:
        Noise-contaminated float32 waveform of the same shape as *audio*, or
        the original *audio* if the augmentation is skipped.

    Example:
        >>> import numpy as np
        >>> wav = np.random.randn(16000).astype(np.float32)
        >>> noisy = add_maritime_noise(wav, sr=16000, target_snr_db=15.0, training=True)
    """
    # -- 1. Enabled check via config ------------------------------------------
    try:
        from core.config import settings as _cfg
        if not _cfg.augmentation.maritime_noise.enabled:
            return audio
        probability = _cfg.augmentation.maritime_noise.probability
        if target_snr_db == 15.0:
            target_snr_db = _cfg.augmentation.maritime_noise.target_snr_db
        _artifacts_dir: Path = _cfg.paths.artifacts_dir
    except Exception:
        # Graceful degradation: config unavailable, use caller-supplied values.
        probability = 0.3
        _artifacts_dir = Path("artifacts")

    # -- 2. Training-only gate ------------------------------------------------
    if not training:
        return audio

    # -- 3. Stochastic gate ---------------------------------------------------
    if random.random() > probability:
        return audio

    # -- 4. Load or generate noise --------------------------------------------
    n_samples = len(audio)

    if noise_file is None:
        noise_file = _artifacts_dir / "noise" / "maritime_sample.wav"

    if noise_file.exists():
        try:
            noise = _load_noise_file(noise_file, n_samples, sr)
            logger.debug("Maritime noise loaded from %s", noise_file)
        except Exception as exc:
            logger.warning(
                "Failed to load maritime noise file %s (%s). "
                "Falling back to pink noise.",
                noise_file,
                exc,
            )
            noise = _generate_pink_noise(n_samples, sr)
    else:
        logger.info(
            "Maritime noise file not found at %s. "
            "Using pink noise surrogate (1/f spectrum).",
            noise_file,
        )
        noise = _generate_pink_noise(n_samples, sr)

    # -- 5. SNR-controlled mixing ---------------------------------------------
    signal_power = float(np.mean(audio.astype(np.float32) ** 2))
    noise_power = float(np.mean(noise ** 2))

    if noise_power < 1e-12 or signal_power < 1e-12:
        # Degenerate input -- return unchanged to avoid NaN propagation.
        return audio

    target_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    mixed = audio.astype(np.float32) + scale * noise

    logger.debug(
        "Maritime noise applied: SNR=%.1f dB, noise_scale=%.4f, training=%s",
        target_snr_db,
        scale,
        training,
    )

    return mixed
