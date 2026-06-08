"""
core/audio_utils.py — Unified audio preprocessing helpers for ShipAssistant.

Why this module exists
----------------------
PyTorch training (``src/data_utils.py`` → ``Wav2Vec2FeatureExtractor``) and ONNX
inference (``core/onnx_engine.py``) historically used **different** normalisation
recipes — that mismatch is the most common source of "ONNX confidence drop"
after INT8 quantisation. This module is the single source of truth for:

  * loading a .wav file at the model's target sample rate (mono, float32);
  * normalising amplitude (zero-mean / unit-variance, matching Wav2Vec2 FE);
  * padding / truncating to the canonical inference window.

All inference paths (PyTorch reference, ONNX engine, debug scripts) MUST go
through these helpers so the only remaining numerical difference is the
backend itself, never the input tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

# Wav2Vec2FeatureExtractor adds 1e-7 to *variance* before taking sqrt.
# We replicate that exactly to avoid a hidden numeric drift between
# train-time normalisation and inference-time normalisation.
_W2V2_VAR_EPS: float = 1e-7


# ── Loading ───────────────────────────────────────────────────────────────────

def load_wav(path: str | Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load a .wav file as mono float32 and resample to *target_sr*.

    Args:
        path:      Filesystem path to the audio file.
        target_sr: Required sample rate (Hz) for the model.

    Returns:
        Tuple of ``(waveform, sample_rate)`` where ``waveform`` is a 1-D
        float32 numpy array in range [-1.0, 1.0] and ``sample_rate ==
        target_sr``.

    Raises:
        FileNotFoundError: if *path* does not exist.
        RuntimeError:      on backend (torchaudio / soundfile) failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    # Prefer torchaudio for parity with training code; fall back to librosa.
    try:
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(p))
        if waveform.shape[0] > 1:                       # stereo → mono mix-down
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0).contiguous().numpy().astype(np.float32), target_sr

    except ImportError:                                 # pragma: no cover
        import librosa

        wav, _ = librosa.load(str(p), sr=target_sr, mono=True)
        return wav.astype(np.float32), target_sr


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_w2v2(waveform: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance normalisation matching ``Wav2Vec2FeatureExtractor``.

    The HuggingFace extractor (with ``do_normalize=True``, the default for
    Wav2Vec2) applies::

        x' = (x - mean(x)) / sqrt(var(x) + 1e-7)

    over the *unmasked* portion of the input. Since our inference pipeline
    feeds a single fixed-length window without padding tokens, the whole
    array is "unmasked" and the formula reduces to the line below.

    Args:
        waveform: 1-D float32 array.

    Returns:
        Normalised 1-D float32 array (same length as input).
    """
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    mean = float(waveform.mean())
    var = float(waveform.var())
    return ((waveform - mean) / np.sqrt(var + _W2V2_VAR_EPS)).astype(np.float32)


def pad_or_truncate(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros or right-truncate *waveform* to *target_samples*.

    Args:
        waveform:       1-D float32 array.
        target_samples: Desired length in samples (e.g. ``win_seconds * sr``).

    Returns:
        1-D float32 array of length ``target_samples``.
    """
    if waveform.shape[0] == target_samples:
        return waveform.astype(np.float32, copy=False)
    if waveform.shape[0] > target_samples:
        return waveform[:target_samples].astype(np.float32, copy=False)
    pad = np.zeros(target_samples - waveform.shape[0], dtype=np.float32)
    return np.concatenate([waveform.astype(np.float32, copy=False), pad])


# ── End-to-end ────────────────────────────────────────────────────────────────

def prepare_window(
    waveform: np.ndarray,
    target_samples: int,
    do_normalize: bool = True,
) -> np.ndarray:
    """Single canonical preprocessing pipeline used by every inference engine.

    Steps (in order): pad/truncate → normalise (optional).

    The order matters: we pad **before** normalising so the zero-padded tail
    contributes to mean/variance the same way ``Wav2Vec2FeatureExtractor``
    handles it when ``attention_mask`` is absent.

    Args:
        waveform:       1-D float32 array, already at target sample rate.
        target_samples: Canonical window length in samples.
        do_normalize:   Apply Wav2Vec2-style normalisation.

    Returns:
        1-D float32 array of length ``target_samples``.
    """
    arr = pad_or_truncate(waveform.astype(np.float32, copy=False), target_samples)
    if do_normalize:
        arr = normalize_w2v2(arr)
    return arr
