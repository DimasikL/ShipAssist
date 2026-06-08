"""
scripts/hybrid/demo_hybrid.py — Standalone demo for the Hybrid C+ engine.

This script is intentionally separate from ``scripts/demo_defense.py`` and does
NOT import from it. It demonstrates the full hybrid pipeline on a single audio
file or via live microphone input.

Modes
-----
  --wav FILE   : Run inference on a single .wav file and print the result.
  --mic        : Continuous microphone demo (Ctrl+C to stop).
  --bench N    : Benchmark N random synthetic inputs and report latency stats.

Usage
-----
    # Single file:
    python scripts/hybrid/demo_hybrid.py --wav path/to/audio.wav

    # Microphone:
    python scripts/hybrid/demo_hybrid.py --mic

    # Benchmark (no audio files needed):
    python scripts/hybrid/demo_hybrid.py --bench 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.hybrid.config import HybridConfig
from core.hybrid.factory import create_hybrid_engine
from core.audio_utils import load_wav, prepare_window
from core.logger import get_logger

logger = get_logger(__name__)

_SR = 16_000
_WIN_SAMPLES = 16_000

# ── ANSI colours ──────────────────────────────────────────────────────────────
_GRN = "\033[92m"
_YEL = "\033[93m"
_RED = "\033[91m"
_RST = "\033[0m"
_BLD = "\033[1m"


def _print_result(result: dict, wav_name: Optional[str] = None) -> None:
    """Pretty-print a predict() result dict.

    Args:
        result:   Output from ``HybridAudioEngine.predict()``.
        wav_name: Optional file name for display.
    """
    if "error" in result:
        print(f"{_RED}[ERROR]{_RST} {result['error']}")
        return

    label = result.get("full_label") or result.get("label") or "(none)"
    conf = result.get("confidence", 0.0)
    lat = result.get("latency_ms", 0.0)
    rejected = result.get("outlier_rejected", False)
    score = result.get("outlier_score", float("inf"))
    slot_val = result.get("slot_value")
    method = result.get("search_method", "?")

    header = f"{'FILE: ' + wav_name if wav_name else 'LIVE'}"
    colour = _RED if rejected else (_GRN if conf >= 0.75 else _YEL)

    print(f"\n{'─'*55}")
    print(f"{_BLD}{header}{_RST}")
    print(f"  Label:           {colour}{label!r}{_RST}")
    print(f"  Confidence:      {conf:.4f}")
    if slot_val is not None:
        print(f"  Slot value:      {slot_val:.1f}")
    print(f"  Outlier score:   {score:.4f}  {'[REJECTED]' if rejected else ''}")
    print(f"  Latency:         {lat:.1f} ms")
    print(f"  Search method:   {method}")


# ── Mode: single WAV file ─────────────────────────────────────────────────────

def run_wav(engine, wav_path: str) -> None:
    """Run inference on a single .wav file.

    Args:
        engine:   Loaded ``HybridAudioEngine``.
        wav_path: Path to the audio file.
    """
    p = Path(wav_path)
    if not p.exists():
        print(f"{_RED}File not found: {p}{_RST}")
        sys.exit(1)

    wav, _ = load_wav(str(p), target_sr=_SR)
    audio = prepare_window(wav, target_samples=_WIN_SAMPLES, do_normalize=True)
    result = engine.predict(audio)
    _print_result(result, wav_name=p.name)


# ── Mode: microphone ──────────────────────────────────────────────────────────

def run_mic(engine) -> None:
    """Continuous microphone demo with sliding window inference.

    Args:
        engine: Loaded ``HybridAudioEngine``.
    """
    try:
        import sounddevice as sd
    except ImportError:
        print(f"{_RED}sounddevice not installed. Run: pip install sounddevice{_RST}")
        sys.exit(1)

    from core.recognizer import RealTimeRecognizer

    recognizer = RealTimeRecognizer(
        sample_rate=_SR,
        window_s=1.0,
        stride_s=0.5,
    )

    def on_audio(chunk: np.ndarray) -> None:
        result = engine.predict(chunk)
        if result.get("label"):
            _print_result(result, wav_name="MIC")

    print(f"\n{_BLD}Hybrid C+ engine — microphone mode{_RST}")
    print("Listening... Press Ctrl+C to stop.\n")
    recognizer.start_stream(on_audio)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        recognizer.stop()
        print("\nStopped.")


# ── Mode: benchmark ───────────────────────────────────────────────────────────

def run_bench(engine, n: int) -> None:
    """Benchmark N random synthetic inference calls.

    Args:
        engine: Loaded ``HybridAudioEngine``.
        n:      Number of synthetic calls to run.
    """
    print(f"\n{_BLD}Benchmarking {n} calls with random noise input…{_RST}")
    latencies = []

    for i in range(n):
        audio = np.random.randn(_WIN_SAMPLES).astype(np.float32) * 0.01
        t0 = time.perf_counter()
        engine.predict(audio)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    print(f"\n{'─'*40}")
    print(f"  Calls:    {n}")
    print(f"  Mean:     {latencies.mean():.1f} ms")
    print(f"  P50:      {np.percentile(latencies, 50):.1f} ms")
    print(f"  P95:      {np.percentile(latencies, 95):.1f} ms")
    print(f"  Max:      {latencies.max():.1f} ms")
    print(f"{'─'*40}")
    print(
        f"  {'OK' if latencies.mean() < 500 else 'WARNING: exceeds 500ms target'}"
    )


# ── Entry-point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Load engine and dispatch to the selected mode.

    Args:
        args: Parsed CLI arguments.
    """
    # ── Load config ────────────────────────────────────────────────────
    model_yaml = Path(args.model_yaml)
    thresh_yaml = Path(args.thresholds_yaml)

    if model_yaml.exists() and thresh_yaml.exists():
        cfg = HybridConfig.from_yaml(model_yaml, thresh_yaml)
    else:
        logger.warning(
            "YAML configs not found (%s / %s). Using default config.",
            model_yaml, thresh_yaml,
        )
        cfg = HybridConfig()

    print(f"\n{_BLD}Loading HybridAudioEngine…{_RST}")
    engine = create_hybrid_engine(cfg)

    if not engine._loaded:
        print(
            f"{_YEL}Warning: engine loaded in degraded mode. "
            "Artefacts are missing — run the training scripts first.{_RST}"
        )
    else:
        print(f"{_GRN}Engine ready. Labels: {engine.labels}{_RST}")

    # ── Dispatch ───────────────────────────────────────────────────────
    if args.wav:
        run_wav(engine, args.wav)
    elif args.mic:
        run_mic(engine)
    elif args.bench is not None:
        run_bench(engine, args.bench)
    else:
        print("Specify --wav FILE, --mic, or --bench N.")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Hybrid C+ engine demo (does not touch demo_defense.py)."
    )
    parser.add_argument("--wav", default=None, help="Path to a .wav file.")
    parser.add_argument("--mic", action="store_true", help="Live microphone mode.")
    parser.add_argument("--bench", type=int, default=None, metavar="N",
                        help="Benchmark N random calls.")
    parser.add_argument(
        "--model_yaml",
        default="configs/hybrid/model.yaml",
        help="Path to the hybrid model config YAML.",
    )
    parser.add_argument(
        "--thresholds_yaml",
        default="configs/hybrid/thresholds.yaml",
        help="Path to the hybrid thresholds YAML.",
    )

    main(parser.parse_args())
