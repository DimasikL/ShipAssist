"""scripts/vkr/noise_augment_dataset.py — Prepare real noise clips for the SNR benchmark.

Downloads ESC-50 (Environmental Sound Classification, CC-BY) and organises
selected clips into per-type subdirectories that benchmark_snr_profile.py
reads via --noise_dir.

Noise type → ESC-50 category mapping:
    wind    → ESC-50 category 'wind'
    traffic → ESC-50 categories 'car_horn', 'engine', 'train'
    office  → ESC-50 categories 'keyboard_typing', 'clock_tick', 'vacuum_cleaner'
    music   → NOT in ESC-50: generated synthetically (patterned pink noise)

AudioSet note:
    For music noise, pass --audioset_music_dir pointing to a local directory
    of AudioSet WAV clips tagged with the "Music" ontology class.
    If absent, a synthetic music-like noise is generated instead.

Output layout:
    <noise_dir>/
      wind/      *.wav
      traffic/   *.wav
      office/    *.wav
      music/     *.wav   (synthetic or from AudioSet)

Usage:
    # Download ESC-50 and prepare all types:
    python scripts/vkr/noise_augment_dataset.py \\
        --output_dir artifacts/noise/ \\
        --esc50_dir  artifacts/noise/ESC-50/

    # Skip download if ESC-50 archive is already present:
    python scripts/vkr/noise_augment_dataset.py \\
        --output_dir artifacts/noise/ \\
        --esc50_dir  artifacts/noise/ESC-50/ \\
        --skip_download

    # Supply AudioSet music clips instead of synthetic:
    python scripts/vkr/noise_augment_dataset.py \\
        --output_dir artifacts/noise/ \\
        --esc50_dir  artifacts/noise/ESC-50/ \\
        --audioset_music_dir /path/to/audioset_music/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
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
SR: int = 16_000   # target sample rate for all output clips
CLIP_DURATION_S: float = 5.0   # ESC-50 clips are exactly 5 s
CLIP_SAMPLES: int = int(SR * CLIP_DURATION_S)

ESC50_URL = (
    "https://github.com/karoldvl/ESC-50/archive/master.zip"
)

# ESC-50 class name → our noise type
# Full ESC-50 category list: https://github.com/karoldvl/ESC-50
ESC50_CLASS_MAP: Dict[str, str] = {
    # wind
    "wind":           "wind",
    # traffic
    "car_horn":       "traffic",
    "engine":         "traffic",
    "train":          "traffic",
    # office
    "keyboard_typing": "office",
    "clock_tick":      "office",
    "vacuum_cleaner":  "office",
}

# How many clips to select per noise type (subset for fast re-runs)
MAX_CLIPS_PER_TYPE: int = 40

# ---------------------------------------------------------------------------
# ESC-50 utilities
# ---------------------------------------------------------------------------


def _download_esc50(esc50_dir: Path) -> None:
    """Download and extract the ESC-50 dataset if not already present.

    ESC-50 master.zip is ~600 MB (uncompressed ~2.4 GB after extracting audio).

    Args:
        esc50_dir: Target directory to extract the dataset into.
    """
    zip_path = esc50_dir.parent / "ESC-50-master.zip"

    if zip_path.exists():
        logger.info("ESC-50 zip already present: %s", zip_path)
    else:
        logger.info("Downloading ESC-50 from %s …", ESC50_URL)
        esc50_dir.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ESC50_URL, str(zip_path))
        logger.info("Downloaded → %s  (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    if esc50_dir.exists() and any(esc50_dir.iterdir()):
        logger.info("ESC-50 already extracted at %s", esc50_dir)
        return

    logger.info("Extracting %s …", zip_path)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(esc50_dir.parent))
    # The archive extracts to 'ESC-50-master/'
    extracted = esc50_dir.parent / "ESC-50-master"
    if extracted.exists() and extracted != esc50_dir:
        extracted.rename(esc50_dir)
    logger.info("Extracted → %s", esc50_dir)


def _load_esc50_meta(esc50_dir: Path) -> "pd.DataFrame":
    """Read the ESC-50 metadata CSV.

    Args:
        esc50_dir: Root of the ESC-50 dataset (contains meta/esc50.csv).

    Returns:
        DataFrame with columns: filename, category, esc10 (bool), …

    Raises:
        FileNotFoundError: If the metadata CSV is not found.
    """
    import pandas as pd

    meta_path = esc50_dir / "meta" / "esc50.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"ESC-50 metadata not found at {meta_path}. "
            "Run with --download to fetch the dataset first."
        )
    df = pd.read_csv(meta_path)
    # Normalise column name differences across ESC-50 versions
    df.columns = [c.lower().strip() for c in df.columns]
    if "category" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "category"})
    return df


def _resample_and_save(
    src: Path,
    dst: Path,
    target_sr: int = SR,
) -> bool:
    """Load *src*, resample to *target_sr*, and save as mono WAV at *dst*.

    Args:
        src:       Source audio file.
        dst:       Destination WAV file.
        target_sr: Target sample rate in Hz.

    Returns:
        True on success, False on error.
    """
    try:
        import librosa  # type: ignore[import]
        import soundfile as sf  # type: ignore[import]

        wav, _ = librosa.load(str(src), sr=target_sr, mono=True, dtype=np.float32)
        sf.write(str(dst), wav, target_sr, subtype="PCM_16")
        return True
    except Exception as exc:
        logger.warning("Failed to process %s: %s", src.name, exc)
        return False


def collect_esc50_clips(
    esc50_dir: Path,
    output_dir: Path,
    max_clips: int = MAX_CLIPS_PER_TYPE,
) -> Dict[str, int]:
    """Copy selected ESC-50 clips into per-type output subdirectories.

    Args:
        esc50_dir:  Root directory of the ESC-50 dataset.
        output_dir: Target noise directory (benchmark --noise_dir).
        max_clips:  Maximum number of clips to copy per noise type.

    Returns:
        Dict mapping noise_type → number of clips saved.
    """
    import pandas as pd

    audio_dir = esc50_dir / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"ESC-50 audio directory not found: {audio_dir}"
        )

    meta = _load_esc50_meta(esc50_dir)
    counts: Dict[str, int] = {}

    # Normalise category names (ESC-50 uses spaces in some versions)
    meta["category_norm"] = (
        meta["category"]
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    for esc_class, noise_type in ESC50_CLASS_MAP.items():
        subset = meta[meta["category_norm"] == esc_class]
        if subset.empty:
            logger.warning(
                "ESC-50 class '%s' not found in metadata. Skipping.", esc_class
            )
            continue

        type_dir = output_dir / noise_type
        type_dir.mkdir(parents=True, exist_ok=True)

        clip_count = 0
        for _, row in subset.iterrows():
            if clip_count >= max_clips:
                break
            src_wav = audio_dir / row["filename"]
            if not src_wav.exists():
                continue
            dst_wav = type_dir / f"{esc_class}_{row['filename']}"
            if dst_wav.exists():
                clip_count += 1
                continue
            if _resample_and_save(src_wav, dst_wav):
                clip_count += 1

        counts[noise_type] = counts.get(noise_type, 0) + clip_count
        logger.info(
            "  ESC-50 '%s' → noise_type '%s': %d clips saved.",
            esc_class, noise_type, clip_count,
        )

    return counts


# ---------------------------------------------------------------------------
# AudioSet music collector
# ---------------------------------------------------------------------------


def collect_audioset_music(
    audioset_dir: Path,
    output_dir: Path,
    max_clips: int = MAX_CLIPS_PER_TYPE,
) -> int:
    """Copy AudioSet music WAV clips to output_dir/music/.

    Assumes *audioset_dir* contains flat *.wav files labelled as music.
    No metadata required — all files are used.

    Args:
        audioset_dir: Directory of AudioSet music WAV files.
        output_dir:   Target noise directory.
        max_clips:    Maximum number of clips to copy.

    Returns:
        Number of clips saved.
    """
    music_dir = output_dir / "music"
    music_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(audioset_dir.glob("*.wav"))[:max_clips]
    count = 0
    for src in wav_files:
        dst = music_dir / src.name
        if dst.exists():
            count += 1
            continue
        if _resample_and_save(src, dst):
            count += 1

    logger.info("AudioSet music: %d clips saved → %s", count, music_dir)
    return count


# ---------------------------------------------------------------------------
# Synthetic music noise generator (fallback when no AudioSet clips available)
# ---------------------------------------------------------------------------


def generate_synthetic_music_clips(
    output_dir: Path,
    n_clips: int = 20,
    duration_s: float = CLIP_DURATION_S,
    seed: int = 0,
) -> int:
    """Generate synthetic music-like noise clips as WAV files.

    Method: pink noise band-passed to 100–8000 Hz + 2 Hz amplitude modulation
    simulates the spectral and rhythmic character of background music at a
    distance. Each clip uses a different random seed for diversity.

    Args:
        output_dir: Target noise directory (clips saved to output_dir/music/).
        n_clips:    Number of clips to generate.
        duration_s: Duration of each clip in seconds.
        seed:       Base random seed.

    Returns:
        Number of clips saved.
    """
    try:
        import soundfile as sf  # type: ignore[import]
        from scipy.signal import butter, sosfilt  # type: ignore[import]
    except ImportError as exc:
        logger.error(
            "soundfile + scipy required for synthetic generation. "
            "pip install soundfile scipy: %s",
            exc,
        )
        return 0

    music_dir = output_dir / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_s * SR)
    t = np.arange(n_samples) / SR

    sos = butter(4, [100 / (SR / 2), 8000 / (SR / 2)], btype="band", output="sos")
    count = 0

    for i in range(n_clips):
        dst = music_dir / f"synthetic_music_{i:04d}.wav"
        if dst.exists():
            count += 1
            continue

        rng = np.random.default_rng(seed + i)
        white = rng.standard_normal(n_samples).astype(np.float64)
        pink_approx = np.cumsum(white)
        filtered = sosfilt(sos, pink_approx).astype(np.float32)

        # Amplitude modulation: blend 2 Hz and 4 Hz rhythmic components
        am = (0.6 + 0.25 * np.sin(2 * np.pi * 2.0 * t)
              + 0.15 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, np.pi))).astype(np.float32)
        wave = filtered * am

        # Normalise to -18 dBFS
        peak = float(np.max(np.abs(wave)))
        if peak < 1e-8:
            continue
        wave = (wave / peak * 0.125).astype(np.float32)

        sf.write(str(dst), wave, SR, subtype="PCM_16")
        count += 1

    logger.info(
        "Synthetic music: %d clips generated → %s", count, music_dir
    )
    return count


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_noise_dir(output_dir: Path) -> None:
    """Log a summary of prepared noise clips for each type.

    Args:
        output_dir: The noise root directory (--noise_dir for benchmark).
    """
    logger.info("Noise library summary (%s):", output_dir)
    for noise_type in ("wind", "traffic", "office", "music"):
        type_dir = output_dir / noise_type
        if type_dir.is_dir():
            clips = list(type_dir.glob("*.wav"))
            logger.info("  %-12s  %d clip(s)", noise_type, len(clips))
        else:
            logger.warning("  %-12s  NOT FOUND (directory missing)", noise_type)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with all CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "noise_augment_dataset.py — Prepare real noise clips from ESC-50 "
            "(and optionally AudioSet) for the SNR robustness benchmark."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "noise",
        help="Root directory for prepared noise clips (pass to benchmark as --noise_dir).",
    )
    parser.add_argument(
        "--esc50_dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "noise" / "ESC-50",
        help="ESC-50 dataset root directory (extracted).",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        default=False,
        help="Skip ESC-50 download; use an already-extracted local copy.",
    )
    parser.add_argument(
        "--audioset_music_dir",
        type=Path,
        default=None,
        help=(
            "Directory of AudioSet WAV files tagged as music. "
            "If not provided, synthetic music clips are generated instead."
        ),
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=MAX_CLIPS_PER_TYPE,
        help="Maximum number of clips to prepare per noise type.",
    )
    parser.add_argument(
        "--n_synthetic_music",
        type=int,
        default=20,
        help="Number of synthetic music clips to generate (if no AudioSet music dir).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic generation.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        default=False,
        help="Only print a summary of existing clips without downloading anything.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: prepare noise clips for the benchmark."""
    args = _parse_args()

    if args.validate_only:
        validate_noise_dir(args.output_dir)
        return

    # -- ESC-50: wind, traffic, office --
    if not args.skip_download:
        _download_esc50(args.esc50_dir)
    else:
        logger.info("--skip_download set. Using existing ESC-50 at %s.", args.esc50_dir)

    if args.esc50_dir.exists():
        try:
            counts = collect_esc50_clips(
                esc50_dir=args.esc50_dir,
                output_dir=args.output_dir,
                max_clips=args.max_clips,
            )
            logger.info("ESC-50 collection complete: %s", counts)
        except Exception as exc:
            logger.error("ESC-50 collection failed: %s", exc)
    else:
        logger.warning(
            "ESC-50 directory not found: %s. "
            "Wind, traffic, and office types will fall back to synthetic noise.",
            args.esc50_dir,
        )

    # -- Music: AudioSet or synthetic --
    music_dir = args.output_dir / "music"
    if args.audioset_music_dir is not None and args.audioset_music_dir.is_dir():
        n_music = collect_audioset_music(
            audioset_dir=args.audioset_music_dir,
            output_dir=args.output_dir,
            max_clips=args.max_clips,
        )
        logger.info("Music clips from AudioSet: %d", n_music)
    else:
        logger.info(
            "No AudioSet music directory provided — generating %d synthetic clips.",
            args.n_synthetic_music,
        )
        n_music = generate_synthetic_music_clips(
            output_dir=args.output_dir,
            n_clips=args.n_synthetic_music,
            seed=args.seed,
        )
        logger.info("Synthetic music clips: %d", n_music)

    # -- Final validation --
    validate_noise_dir(args.output_dir)
    logger.info(
        "Done. Pass --noise_dir %s to benchmark_snr_profile.py.", args.output_dir
    )


if __name__ == "__main__":
    main()
