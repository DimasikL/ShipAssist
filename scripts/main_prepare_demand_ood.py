"""Prepare OOD noise dataset from DEMAND for speech command recognition testing.

Downloads DEMAND dataset (18 environments, 48kHz stereo), converts to 16kHz mono,
slices into 1-second fragments, and creates a mixed dataset from the project test
split (clf_dset/test/) at specified SNR levels.

Audio loading and preprocessing use core.audio_utils to guarantee parity with
the training pipeline.

Usage (from project root):
    python scripts/prepare_demand_ood.py
    python scripts/prepare_demand_ood.py --skip-download --skip-preprocess
    python scripts/prepare_demand_ood.py \\
        --demand-dir artifacts/demand_raw \\
        --noise-dir  artifacts/demand_noise \\
        --mixed-dir  artifacts/demand_mixed \\
        --output-csv artifacts/demand_ood_test.csv \\
        --snr-levels -2 0 3 5 10 15 20
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import random
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so core.* imports work regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DEMAND: 18 environments @ 48 kHz stereo 16-bit
# Zenodo record: https://zenodo.org/record/1227121
# ---------------------------------------------------------------------------
DEMAND_ENVS: List[str] = [
    "DKITCHEN", "DLIVING", "DOFFICE", "DWASHING",
    "NFIELD",   "NPARK",   "NRIVER",
    "OHALLWAY", "OMEETING","OOFFICE",
    "PCAFETER", "PRESTO",  "PSTATION",
    "SPSQUARE", "STRAFFIC",
    "TBUS",     "TCAR",    "TMETRO",
]

ZENODO_BASE = "https://zenodo.org/record/1227121/files"

# Some environments use a non-standard filename on Zenodo (checked 2026-06).
# Map env_name → actual zip filename stem; anything not listed uses "{env}_48k".
_ENV_FILENAME_OVERRIDE: Dict[str, str] = {
    "DOFFICE": "DOFFICE_48k",   # 404 on Zenodo record 1227121 — kept for retry
}  # extend here if other envs have different names
TARGET_SR = 16_000
SEGMENT_SAMPLES = TARGET_SR  # 1 second — matches model win_samples

# ---------------------------------------------------------------------------
# Label normalisation (mirrors eval_confusion_matrix.py)
# ---------------------------------------------------------------------------
_LABEL_NORM: Dict[str, str] = {
    "другие слова":       "другие слова",
    "negatives":          "другие слова",
    "машина":             "машина",
    "приготовить машину": "приготовить машину",
    "приготовить_машину": "приготовить машину",
    "самый малый вперед": "самый малый вперед",
    "самый_малый_вперед": "самый малый вперед",
    "самый_малый_вперёд": "самый малый вперед",
    "самый малый вперёд": "самый малый вперед",
}


def _normalise_label(raw: str) -> Optional[str]:
    """Map a directory-derived raw string to a canonical model label.

    Args:
        raw: Label string extracted from the directory path.

    Returns:
        Canonical label string, or None if unrecognised.
    """
    raw = raw.strip()
    if raw in _LABEL_NORM:
        return _LABEL_NORM[raw]
    stripped = re.sub(r"[\s_]x\d+$", "", raw).strip()
    if stripped in _LABEL_NORM:
        return _LABEL_NORM[stripped]
    spaced = stripped.replace("_", " ")
    return _LABEL_NORM.get(spaced)


def collect_test_files(test_dir: Path) -> List[Tuple[Path, str]]:
    """Recursively collect (wav_path, canonical_label) pairs from *test_dir*.

    Mirrors the logic in scripts/eval_confusion_matrix.py.  Skips files
    inside ``scr/`` and ``src/`` sub-folders (source originals).

    Args:
        test_dir: Root of the clf_dset/test directory.

    Returns:
        List of (absolute Path, canonical label) tuples.
    """
    samples: List[Tuple[Path, str]] = []
    skipped_unknown = 0

    for wav in sorted(test_dir.rglob("*.wav")):
        parts = wav.parts
        if "scr" in parts or "src" in parts:
            continue

        # Extract label from the immediate parent directory name
        label_raw = wav.parent.name
        label = _normalise_label(label_raw)
        if label is None:
            skipped_unknown += 1
            continue
        samples.append((wav, label))

    logger.info(
        "Test files discovered: %d  (skipped_unknown=%d)",
        len(samples), skipped_unknown,
    )
    return samples


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def build_download_urls() -> Dict[str, str]:
    """Return mapping env_name → Zenodo download URL for all 18 DEMAND environments.

    Uses _ENV_FILENAME_OVERRIDE for environments whose zip name differs from
    the default ``{env}_48k`` pattern.
    """
    return {
        env: f"{ZENODO_BASE}/{_ENV_FILENAME_OVERRIDE.get(env, env + '_48k')}.zip?download=1"
        for env in DEMAND_ENVS
    }


def download_with_progress(url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download *url* to *dest_path* with progress logging and zip validation.

    If the download is interrupted or the server returns a truncated file,
    the incomplete zip is removed so subsequent runs trigger a fresh download.

    Args:
        url: Direct download URL.
        dest_path: Destination file path (parent must exist).
        chunk_size: Bytes per read chunk (default 1 MiB).

    Raises:
        requests.HTTPError: If the server returns a non-2xx status.
        RuntimeError: If the downloaded file fails zip integrity check.
    """
    logger.info("Downloading %s → %s", url, dest_path.name)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with dest_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * chunk_size) < chunk_size:
                    logger.info("  %.0f%%  (%.0f / %.0f MiB)",
                                100 * downloaded / total,
                                downloaded / 1e6, total / 1e6)

    # Validate zip integrity — removes corrupt file so next run re-downloads it
    if not zipfile.is_zipfile(dest_path):
        dest_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is not a valid zip (truncated?): {dest_path.name}"
        )
    logger.info("Saved %s (%.1f MiB)", dest_path.name, dest_path.stat().st_size / 1e6)


def download_demand_dataset(demand_dir: Path, force: bool = False) -> None:
    """Download all 18 DEMAND environment zip archives.

    Existing zips are validated before skipping; corrupt/partial files from
    a previous interrupted download are automatically removed and re-downloaded.

    Args:
        demand_dir: Root directory for raw downloads.
        force: Re-download even if a valid zip already exists.
    """
    demand_dir.mkdir(parents=True, exist_ok=True)
    for env, url in build_download_urls().items():
        stem = _ENV_FILENAME_OVERRIDE.get(env, f"{env}_48k")
        zip_path = demand_dir / f"{stem}.zip"

        if zip_path.exists() and not force:
            if zipfile.is_zipfile(zip_path):
                logger.info("Skip %s (valid zip present)", env)
                continue
            else:
                logger.warning("Removing corrupt zip for %s, re-downloading.", env)
                zip_path.unlink()

        try:
            download_with_progress(url, zip_path)
        except (requests.RequestException, RuntimeError) as exc:
            logger.error("Download failed for %s: %s", env, exc)


# ---------------------------------------------------------------------------
# Noise preprocessing (stereo 48kHz → mono 16kHz → 1-second segments)
# ---------------------------------------------------------------------------

def _to_mono_16k_np(raw_bytes: bytes) -> np.ndarray:
    """Decode a WAV byte blob to a mono 16 kHz float32 numpy array.

    Uses torchaudio for resampling (parity with training pipeline).

    Args:
        raw_bytes: Raw bytes of a WAV file.

    Returns:
        1-D float32 numpy array at 16 kHz.
    """
    import torchaudio

    buf = io.BytesIO(raw_bytes)
    waveform, sr = torchaudio.load(buf)                    # (C, T)

    if waveform.shape[0] > 1:                              # stereo → mono
        waveform = waveform.mean(dim=0, keepdim=True)      # (1, T)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    return waveform.squeeze(0).contiguous().numpy().astype(np.float32)


def _slice_segments(waveform: np.ndarray, seg_len: int) -> List[np.ndarray]:
    """Split *waveform* into non-overlapping segments of *seg_len* samples.

    Trailing samples that don't fill a complete segment are discarded.

    Args:
        waveform: 1-D float32 numpy array.
        seg_len: Number of samples per segment.

    Returns:
        List of 1-D float32 arrays each of length *seg_len*.
    """
    n = len(waveform) // seg_len
    return [waveform[i * seg_len : (i + 1) * seg_len] for i in range(n)]


def preprocess_env(zip_path: Path, env_out_dir: Path, env_name: str) -> List[Path]:
    """Extract, convert, and slice one DEMAND zip into 1-second noise fragments.

    Args:
        zip_path: Path to the downloaded DEMAND zip archive.
        env_out_dir: Directory to save processed fragment WAVs.
        env_name: DEMAND environment identifier (e.g. "DKITCHEN").

    Returns:
        List of paths to saved WAV fragment files.
    """
    import soundfile as sf

    saved: List[Path] = []
    if not zip_path.exists():
        logger.warning("Zip not found, skipping: %s", zip_path)
        return saved

    env_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preprocessing %s …", env_name)

    if not zipfile.is_zipfile(zip_path):
        logger.warning(
            "Corrupt or incomplete zip for %s — skipping. "
            "Run without --skip-download to re-download.",
            env_name,
        )
        return saved

    with zipfile.ZipFile(zip_path, "r") as zf:
        wav_names = [n for n in zf.namelist() if n.endswith(".wav")]
        for wav_name in wav_names:
            try:
                raw = zf.read(wav_name)
                mono16k = _to_mono_16k_np(raw)
                segments = _slice_segments(mono16k, SEGMENT_SAMPLES)
                stem = Path(wav_name).stem
                for idx, seg in enumerate(segments):
                    out = env_out_dir / f"{stem}_seg{idx:05d}.wav"
                    sf.write(str(out), seg, TARGET_SR, subtype="PCM_16")
                    saved.append(out)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error processing %s in %s: %s", wav_name, env_name, exc)

    logger.info("  %s → %d fragments", env_name, len(saved))
    return saved


def preprocess_all_envs(
    demand_dir: Path, noise_dir: Path
) -> Dict[str, List[Path]]:
    """Extract and preprocess every available DEMAND environment.

    Args:
        demand_dir: Directory containing downloaded zip files.
        noise_dir: Root output directory for processed fragments.

    Returns:
        Mapping env_name → list of 1-second fragment paths.
    """
    result: Dict[str, List[Path]] = {}
    for env in DEMAND_ENVS:
        stem = _ENV_FILENAME_OVERRIDE.get(env, f"{env}_48k")
        frags = preprocess_env(demand_dir / f"{stem}.zip", noise_dir / env, env)
        if frags:
            result[env] = frags
    return result


def load_existing_fragments(noise_dir: Path) -> Dict[str, List[Path]]:
    """Discover already-preprocessed noise fragments from *noise_dir*.

    Args:
        noise_dir: Root directory containing one sub-directory per DEMAND env.

    Returns:
        Mapping env_name → sorted list of WAV paths.
    """
    result: Dict[str, List[Path]] = {}
    for env in DEMAND_ENVS:
        env_dir = noise_dir / env
        if env_dir.exists():
            frags = sorted(env_dir.glob("*.wav"))
            if frags:
                result[env] = frags
                logger.info("  %s: %d fragments", env, len(frags))
    return result


# ---------------------------------------------------------------------------
# SNR mixing (numpy, parity with core pipeline)
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray, eps: float = 1e-9) -> float:
    """Root-mean-square of a 1-D float32 array."""
    return float(np.sqrt(np.mean(x ** 2))) + eps


def mix_at_snr(
    speech_wav: np.ndarray,
    noise_wav: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix *speech_wav* with *noise_wav* at the desired SNR using RMS scaling.

    Noise length is matched to speech via tiling (too short) or random crop
    (too long).  Mixed output is clamped to [-1, 1].

    SNR definition:
        SNR_dB = 20 * log10(RMS_speech / RMS_noise_scaled)

    Args:
        speech_wav: Clean speech waveform, 1-D float32 at TARGET_SR.
        noise_wav:  Noise fragment, 1-D float32 at TARGET_SR.
        snr_db:     Desired SNR in decibels.

    Returns:
        Mixed 1-D float32 array of the same length as *speech_wav*.
    """
    n = len(speech_wav)

    # Match noise length
    if len(noise_wav) < n:
        reps = (n // len(noise_wav)) + 1
        noise_wav = np.tile(noise_wav, reps)
    if len(noise_wav) > n:
        start = random.randint(0, len(noise_wav) - n)
        noise_wav = noise_wav[start : start + n]

    rms_s = _rms(speech_wav)
    rms_n = _rms(noise_wav)
    rms_n_target = rms_s / (10.0 ** (snr_db / 20.0))
    noise_scaled = noise_wav * (rms_n_target / rms_n)

    return np.clip(speech_wav + noise_scaled, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_mixed_dataset(
    test_files: List[Tuple[Path, str]],
    env_fragments: Dict[str, List[Path]],
    mixed_dir: Path,
    snr_levels: List[float],
    seed: int = 42,
) -> List[Dict]:
    """Create mixed speech+noise files for every (speech, env, snr) combination.

    Args:
        test_files:    List of (wav_path, canonical_label) from collect_test_files.
        env_fragments: Mapping env_name → list of noise fragment paths.
        mixed_dir:     Root directory for mixed output WAVs.
        snr_levels:    List of SNR values in dB.
        seed:          Random seed for reproducibility.

    Returns:
        List of row dicts: filepath_mixed, label, snr_db, noise_env.
    """
    import soundfile as sf

    random.seed(seed)
    np.random.seed(seed)

    mixed_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    envs = list(env_fragments.keys())
    n_total = len(test_files)

    for rec_idx, (speech_path, label) in enumerate(test_files):
        try:
            speech_wav, _ = load_wav(speech_path, target_sr=TARGET_SR)
        except Exception as exc:  # noqa: BLE001
            logger.error("Cannot load %s: %s", speech_path, exc)
            continue

        # Canonical 1-second window (pad/truncate — no normalisation here;
        # normalisation happens inside the ONNX engine via prepare_window)
        if len(speech_wav) < SEGMENT_SAMPLES:
            speech_wav = np.pad(speech_wav, (0, SEGMENT_SAMPLES - len(speech_wav)))
        else:
            speech_wav = speech_wav[:SEGMENT_SAMPLES]

        for snr_db in snr_levels:
            for env_name in envs:
                noise_path = random.choice(env_fragments[env_name])
                try:
                    noise_wav, _ = load_wav(noise_path, target_sr=TARGET_SR)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Cannot load noise %s: %s", noise_path, exc)
                    continue

                mixed = mix_at_snr(speech_wav, noise_wav, snr_db)

                # snr tag: +10→p10, -2→n2
                snr_tag = f"snr{snr_db:+.0f}".replace("+", "p").replace("-", "n")
                out_dir = mixed_dir / env_name / snr_tag
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{speech_path.stem}_{env_name}_{snr_tag}.wav"

                sf.write(str(out_path), mixed, TARGET_SR, subtype="PCM_16")
                rows.append({
                    "filepath_mixed": str(out_path),
                    "label":          label,
                    "snr_db":         snr_db,
                    "noise_env":      env_name,
                })

        if (rec_idx + 1) % 50 == 0:
            logger.info("Mixed %d / %d speech files", rec_idx + 1, n_total)

    logger.info("Total mixed examples: %d", len(rows))
    return rows


def save_csv(rows: List[Dict], output_csv: Path) -> None:
    """Write mixed dataset rows to *output_csv*.

    Args:
        rows: List of dicts with keys filepath_mixed, label, snr_db, noise_env.
        output_csv: Destination path.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filepath_mixed", "label", "snr_db", "noise_env"]
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved → %s (%d rows)", output_csv, len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Prepare DEMAND OOD noise dataset for ShipAssistant evaluation."
    )
    p.add_argument("--test-dir", type=Path,
                   default=_PROJECT_ROOT / "clf_dset" / "test",
                   help="clf_dset/test directory (default: %(default)s)")
    p.add_argument("--demand-dir", type=Path,
                   default=_PROJECT_ROOT / "artifacts" / "demand_raw",
                   help="Directory for raw DEMAND zip downloads.")
    p.add_argument("--noise-dir", type=Path,
                   default=_PROJECT_ROOT / "artifacts" / "demand_noise",
                   help="Directory for preprocessed 16kHz noise fragments.")
    p.add_argument("--mixed-dir", type=Path,
                   default=_PROJECT_ROOT / "artifacts" / "demand_mixed",
                   help="Directory for mixed speech+noise WAV files.")
    p.add_argument("--output-csv", type=Path,
                   default=_PROJECT_ROOT / "artifacts" / "demand_ood_test.csv",
                   help="Output CSV path.")
    p.add_argument("--snr-levels", type=float, nargs="+",
                   default=[-2.0, 0.0, 3.0, 5.0, 10.0, 15.0, 20.0],
                   help="SNR levels in dB (default: -2 0 3 5 10 15 20).")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip downloading; use existing zips.")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip noise preprocessing; use existing fragments.")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download zips even if they already exist.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    return p.parse_args()


def main() -> None:
    """Run the full pipeline: download → preprocess → mix → save CSV."""
    args = parse_args()

    # 1. Download
    if not args.skip_download:
        logger.info("=== Stage 1/4: Downloading DEMAND ===")
        download_demand_dataset(args.demand_dir, force=args.force_download)
    else:
        logger.info("Stage 1/4: Skipped (--skip-download)")

    # 2. Preprocess noise
    if not args.skip_preprocess:
        logger.info("=== Stage 2/4: Preprocessing noise fragments ===")
        env_fragments = preprocess_all_envs(args.demand_dir, args.noise_dir)
    else:
        logger.info("=== Stage 2/4: Loading existing fragments ===")
        env_fragments = load_existing_fragments(args.noise_dir)

    if not env_fragments:
        logger.error("No noise fragments available. Aborting.")
        sys.exit(1)

    # 3. Collect test files
    logger.info("=== Stage 3/4: Collecting test files from %s ===", args.test_dir)
    if not args.test_dir.exists():
        logger.error("Test directory not found: %s", args.test_dir)
        sys.exit(1)
    test_files = collect_test_files(args.test_dir)
    if not test_files:
        logger.error("No test files found in %s", args.test_dir)
        sys.exit(1)

    # 4. Mix & save
    logger.info(
        "=== Stage 4/4: Mixing %d files × %d envs × %d SNRs ===",
        len(test_files), len(env_fragments), len(args.snr_levels),
    )
    rows = build_mixed_dataset(
        test_files, env_fragments, args.mixed_dir, args.snr_levels, seed=args.seed,
    )
    save_csv(rows, args.output_csv)
    logger.info("Done.")


if __name__ == "__main__":
    main()
