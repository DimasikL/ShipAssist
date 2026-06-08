"""
scripts/router_demo.py — SmartRouter interactive demo and benchmark tool.

Modes
-----
  --mode wav         Run the router on a single .wav file. Prints routing decision,
                     engine used, confidence, and latency.

  --mode interactive Live microphone demo showing real-time routing decisions with
                     colour-coded engine attribution (green=ONNX, blue=hybrid,
                     yellow=tie-break, red=rejected).

  --mode bench       Benchmark comparing three configurations side-by-side:
                       A) ONNX engine alone
                       B) Hybrid engine alone
                       C) SmartRouter (both engines)
                     Reports: P50/P95/P99 latency, routing distribution, fast-path
                     hit rate, and projected latency under each strategy.

Usage
-----
    # Single file:
    python scripts/router_demo.py --mode wav --wav path/to/audio.wav

    # Live mic:
    python scripts/router_demo.py --mode interactive

    # Benchmark with 200 random calls:
    python scripts/router_demo.py --mode bench --n 200

    # Benchmark on real WAV files from a directory:
    python scripts/router_demo.py --mode bench --wav_dir artifacts/data/samples/ --n 100

    # Use specific config:
    python scripts/router_demo.py --mode bench \\
        --routing_yaml configs/routing.yaml \\
        --model_yaml   configs/hybrid/model.yaml \\
        --thresh_yaml  configs/hybrid/thresholds.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.audio_utils import load_wav, prepare_window
from core.router import SmartRouter, RoutingConfig, create_router, TAG_ONNX, TAG_HYBRID, TAG_OUTLIER
from core.logger import get_logger

logger = get_logger(__name__)

_SR = 16_000
_WIN = 16_000

# ── ANSI helpers ──────────────────────────────────────────────────────────────
_RST = "\033[0m"
_BLD = "\033[1m"
_GRN = "\033[92m"   # ONNX result
_BLU = "\033[94m"   # Hybrid result
_YEL = "\033[93m"   # Tie-break / low-confidence
_RED = "\033[91m"   # Outlier rejected
_DIM = "\033[2m"


def _colour(engine_used: str) -> str:
    return {TAG_ONNX: _GRN, TAG_HYBRID: _BLU, TAG_OUTLIER: _RED}.get(engine_used, _YEL)


def _bar(value: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    n = max(0, min(width, int(value * width)))
    return filled * n + empty * (width - n)


# ── Engine loader ─────────────────────────────────────────────────────────────

def _load_engines(args: argparse.Namespace):
    """Load ONNX and Hybrid engines with graceful per-engine failure.

    Returns:
        Tuple of ``(onnx_engine, hybrid_engine)`` — either may be ``None``.
    """
    onnx_engine = None
    hybrid_engine = None

    # ONNX
    try:
        from core.engine import OnnxAudioEngine
        import yaml
        from pathlib import Path as _P

        # Try to find onnx_model path from configs
        onnx_dir = args.onnx_dir
        if not onnx_dir:
            base_yaml = _PROJECT_ROOT / "configs" / "base.yaml"
            if base_yaml.exists():
                with open(base_yaml, "r", encoding="utf-8") as f:
                    base_data = yaml.safe_load(f) or {}
                onnx_dir = base_data.get("paths", {}).get(
                    "onnx_model",
                    "onnx_model/models/run_2026-02-25_19-07-15/best_model",
                )

        if onnx_dir and _P(onnx_dir).exists():
            onnx_engine = OnnxAudioEngine(onnx_dir=str(onnx_dir))
            print(f"  {_GRN}✓{_RST} ONNX engine loaded from {onnx_dir!r}")
        else:
            print(f"  {_YEL}⚠{_RST}  ONNX model not found at {onnx_dir!r} — ONNX disabled")
    except Exception as exc:
        print(f"  {_RED}✗{_RST} ONNX load failed: {exc}")

    # Hybrid
    try:
        from core.hybrid.config import HybridConfig
        from core.hybrid.factory import create_hybrid_engine

        model_yaml = Path(args.model_yaml)
        thresh_yaml = Path(args.thresh_yaml)

        if model_yaml.exists() and thresh_yaml.exists():
            hybrid_cfg = HybridConfig.from_yaml(model_yaml, thresh_yaml)
        else:
            print(
                f"  {_YEL}⚠{_RST}  Hybrid YAML not found "
                f"({model_yaml} / {thresh_yaml}) — using defaults"
            )
            hybrid_cfg = HybridConfig()

        hybrid_engine = create_hybrid_engine(hybrid_cfg)
        status = f"{_GRN}✓{_RST}" if hybrid_engine._loaded else f"{_YEL}⚠{_RST} (degraded)"
        print(f"  {status} Hybrid engine — loaded={hybrid_engine._loaded}")
    except Exception as exc:
        print(f"  {_RED}✗{_RST} Hybrid load failed: {exc}")

    return onnx_engine, hybrid_engine


# ── Shared audio helper ───────────────────────────────────────────────────────

def _load_audio(path: str) -> Optional[np.ndarray]:
    """Load and prepare a .wav file for inference.

    Args:
        path: Path to the audio file.

    Returns:
        1-D float32 array ready for ``engine.predict()``, or ``None`` on error.
    """
    try:
        wav, _ = load_wav(path, target_sr=_SR)
        return prepare_window(wav, target_samples=_WIN, do_normalize=True)
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


# ── Mode: WAV file ────────────────────────────────────────────────────────────

def run_wav(router: SmartRouter, wav_path: str) -> None:
    """Run the router on a single .wav file and print a detailed result.

    Args:
        router:   Loaded ``SmartRouter``.
        wav_path: Path to the .wav file.
    """
    p = Path(wav_path)
    if not p.exists():
        print(f"{_RED}File not found: {p}{_RST}")
        sys.exit(1)

    audio = _load_audio(str(p))
    if audio is None:
        sys.exit(1)

    result = router.predict(audio)
    engine_used = result.get("engine_used", "?")
    colour = _colour(engine_used)
    label = result.get("full_label") or result.get("label") or "(none)"
    conf = result.get("confidence", 0.0)
    conf_m = result.get("confidence_mapped", conf)
    lat = result.get("router_latency_ms", 0.0)
    outlier = result.get("outlier_score", float("inf"))

    print(f"\n{'─'*60}")
    print(f"  File:             {p.name}")
    print(f"  Label:            {colour}{_BLD}{label}{_RST}")
    print(f"  Engine used:      {colour}{engine_used}{_RST}")
    print(f"  Confidence:       {conf:.4f}  (mapped → {conf_m:.4f})")
    print(f"  {_bar(conf_m)} {int(conf_m*100):3d}%")
    print(f"  Outlier score:    {outlier:.4f}")
    print(f"  Router latency:   {lat:.1f} ms")
    if result.get("slot_value") is not None:
        print(f"  Slot value:       {result['slot_value']:.1f}")
    print(f"{'─'*60}")


# ── Mode: interactive mic ─────────────────────────────────────────────────────

def run_interactive(router: SmartRouter) -> None:
    """Live microphone demo with real-time routing attribution display.

    Args:
        router: Loaded ``SmartRouter``.
    """
    try:
        import sounddevice as sd  # noqa: F401
    except ImportError:
        print(f"{_RED}sounddevice not installed. Run: pip install sounddevice{_RST}")
        sys.exit(1)

    from core.recognizer import RealTimeRecognizer

    summary = router.routing_summary()
    print(f"\n{_BLD}SmartRouter — Interactive Mode{_RST}")
    print(f"  ONNX  [{_GRN}green{_RST}] owns: {summary['known_phrases']}")
    print(f"  Hybrid[{_BLU}blue{_RST} ] owns: {summary['number_slot_intents']}")
    print(f"  Fast-path: {summary['fast_path_enabled']}")
    print("\nListening… Ctrl+C to stop.\n")

    call_counts: Dict[str, int] = defaultdict(int)
    total_latency: List[float] = []

    def on_audio(chunk: np.ndarray) -> None:
        result = router.predict(chunk)
        engine_used = result.get("engine_used", "?")
        label = result.get("full_label") or result.get("label") or ""
        if not label:
            return  # suppress noise

        colour = _colour(engine_used)
        conf_m = result.get("confidence_mapped", result.get("confidence", 0.0))
        lat = result.get("router_latency_ms", 0.0)

        call_counts[engine_used] += 1
        total_latency.append(lat)

        tag_str = f"[{engine_used:14s}]"
        bar = _bar(conf_m, width=15)
        print(
            f"  {colour}{tag_str}{_RST}  "
            f"{_BLD}{label:<35s}{_RST}  "
            f"conf={conf_m:.3f} {bar}  {lat:.0f}ms"
        )

    recognizer = RealTimeRecognizer(sample_rate=_SR, window_s=1.0, stride_s=0.5)
    recognizer.start_stream(on_audio)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        recognizer.stop()
        if total_latency:
            lats = np.array(total_latency)
            print(f"\n{'─'*55}")
            print(f"  Session stats:")
            print(f"  Calls:    {sum(call_counts.values())}")
            for eng, cnt in sorted(call_counts.items()):
                pct = 100 * cnt / max(sum(call_counts.values()), 1)
                print(f"  {eng:20s}  {cnt:4d}  ({pct:.0f}%)")
            print(f"  Lat P50:  {np.percentile(lats, 50):.1f} ms")
            print(f"  Lat P95:  {np.percentile(lats, 95):.1f} ms")
            print(f"{'─'*55}")


# ── Mode: benchmark ───────────────────────────────────────────────────────────

def run_bench(
    onnx_engine,
    hybrid_engine,
    router: SmartRouter,
    n: int,
    wav_dir: Optional[str],
) -> None:
    """Compare ONNX-alone, Hybrid-alone, and Router latency side-by-side.

    Args:
        onnx_engine:   ONNX engine instance (may be None).
        hybrid_engine: Hybrid engine instance (may be None).
        router:        Configured SmartRouter.
        n:             Number of inference calls per engine.
        wav_dir:       Optional directory of .wav files to use instead of noise.
    """
    print(f"\n{_BLD}SmartRouter Benchmark{_RST}  (n={n} calls per engine)\n")

    # Build audio inputs
    audio_inputs: List[np.ndarray] = []
    if wav_dir:
        wav_files = sorted(Path(wav_dir).glob("*.wav"))[:n]
        for wf in wav_files:
            a = _load_audio(str(wf))
            if a is not None:
                audio_inputs.append(a)
        if audio_inputs:
            print(f"  Using {len(audio_inputs)} real WAV files from {wav_dir!r}")
        else:
            print(f"  {_YEL}No WAV files found in {wav_dir!r} — falling back to synthetic noise{_RST}")

    if not audio_inputs:
        rng = np.random.default_rng(42)
        audio_inputs = [
            (rng.standard_normal(_WIN) * 0.02).astype(np.float32) for _ in range(n)
        ]
        print(f"  Using {n} synthetic noise clips (no real audio)")

    # Pad or cycle if fewer inputs than n
    while len(audio_inputs) < n:
        audio_inputs.extend(audio_inputs)
    audio_inputs = audio_inputs[:n]

    def _bench_engine(name: str, engine) -> Tuple[np.ndarray, Dict[str, int]]:
        if engine is None:
            print(f"  {_YEL}{name:20s}{_RST}  UNAVAILABLE — skipping")
            return np.array([]), {}
        lats: List[float] = []
        routing_counts: Dict[str, int] = defaultdict(int)
        for audio in audio_inputs:
            t0 = time.perf_counter()
            result = engine.predict(audio)
            elapsed = (time.perf_counter() - t0) * 1000.0
            # Use router_latency_ms if available (router already includes overhead)
            lats.append(result.get("router_latency_ms", elapsed))
            eng_used = result.get("engine_used", name.lower().replace(" ", "_"))
            routing_counts[eng_used] += 1
        return np.array(lats, dtype=np.float32), dict(routing_counts)

    configs = [
        ("ONNX alone", onnx_engine),
        ("Hybrid alone", hybrid_engine),
        ("SmartRouter", router),
    ]

    all_stats: List[Tuple[str, np.ndarray, Dict]] = []
    for label, eng in configs:
        lats, rcounts = _bench_engine(label, eng)
        all_stats.append((label, lats, rcounts))

    # ── Results table ──────────────────────────────────────────────────
    COL = 22
    print(f"\n{'─'*72}")
    print(f"  {'Engine':{COL}}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'Mean':>7}  {'Max':>7}")
    print(f"  {'─'*COL}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

    for name, lats, _ in all_stats:
        if lats.size == 0:
            print(f"  {name:{COL}}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}")
            continue
        ok = "✓" if np.percentile(lats, 95) < 500 else "✗"
        print(
            f"  {name:{COL}}"
            f"  {np.percentile(lats, 50):>6.1f}ms"
            f"  {np.percentile(lats, 95):>6.1f}ms"
            f"  {np.percentile(lats, 99):>6.1f}ms"
            f"  {lats.mean():>6.1f}ms"
            f"  {lats.max():>6.1f}ms"
            f"  {ok}"
        )
    print(f"{'─'*72}")

    # ── Routing distribution for the router ───────────────────────────
    _name, _lats, router_rcounts = all_stats[2]  # SmartRouter
    if router_rcounts:
        total_calls = sum(router_rcounts.values())
        print(f"\n  {_BLD}Router distribution ({total_calls} calls):{_RST}")
        for tag, cnt in sorted(router_rcounts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / max(total_calls, 1)
            colour = _colour(tag)
            bar = _bar(pct / 100, width=20)
            print(f"    {colour}{tag:20s}{_RST}  {cnt:4d}  {bar}  {pct:.0f}%")

    # ── Fast-path hit rate ─────────────────────────────────────────────
    onnx_routed = router_rcounts.get(TAG_ONNX, 0)
    hybrid_routed = router_rcounts.get(TAG_HYBRID, 0)
    outlier_routed = router_rcounts.get(TAG_OUTLIER, 0)
    total = sum(router_rcounts.values()) or 1
    fast_path_pct = 100 * onnx_routed / total

    print(f"\n  {_BLD}Fast-path analysis:{_RST}")
    print(f"    ONNX fast-path hits: {onnx_routed:4d} / {total}  ({fast_path_pct:.0f}%)")
    print(f"    Hybrid called:       {hybrid_routed + outlier_routed:4d} / {total}")

    # ── Latency savings estimate ───────────────────────────────────────
    router_lats, onnx_lats, hybrid_lats = all_stats[2][1], all_stats[0][1], all_stats[1][1]
    if router_lats.size > 0 and onnx_lats.size > 0 and hybrid_lats.size > 0:
        always_both_p50 = (np.percentile(onnx_lats, 50) + np.percentile(hybrid_lats, 50))
        router_p50 = np.percentile(router_lats, 50)
        saving_pct = 100 * (1 - router_p50 / max(always_both_p50, 1))
        print(f"\n  {_BLD}Latency vs. always-run-both:{_RST}")
        print(
            f"    Always-both P50:   {always_both_p50:.1f} ms  "
            f"(ONNX {np.percentile(onnx_lats,50):.1f} + Hybrid {np.percentile(hybrid_lats,50):.1f})"
        )
        print(f"    SmartRouter P50:   {router_p50:.1f} ms")
        print(f"    Latency saving:    {saving_pct:.0f}% on fast-path calls")

    print(f"\n  {_DIM}Target: P95 < 500 ms for maritime real-time use.{_RST}")
    print()


# ── Entry-point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Load engines, build router, dispatch to selected mode.

    Args:
        args: Parsed CLI arguments.
    """
    print(f"\n{_BLD}ShipAssistant SmartRouter Demo{_RST}")
    print("─" * 40)
    print("Loading engines…")

    onnx_engine, hybrid_engine = _load_engines(args)
    routing_cfg = RoutingConfig.from_yaml_or_default(args.routing_yaml)
    router = SmartRouter(routing_cfg, onnx_engine=onnx_engine, hybrid_engine=hybrid_engine)

    summary = router.routing_summary()
    print(f"\n  {_BLD}Router summary:{_RST}")
    for k, v in summary.items():
        print(f"    {k:<25s} {v}")
    print()

    if args.mode == "wav":
        if not args.wav:
            print(f"{_RED}--wav FILE is required for --mode wav{_RST}")
            sys.exit(1)
        run_wav(router, args.wav)

    elif args.mode == "interactive":
        run_interactive(router)

    elif args.mode == "bench":
        run_bench(onnx_engine, hybrid_engine, router, args.n, args.wav_dir)

    else:
        print(f"{_RED}Unknown mode: {args.mode!r}. Use wav, interactive, or bench.{_RST}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="SmartRouter demo: wav / interactive mic / latency benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/router_demo.py --mode wav --wav samples/test.wav
  python scripts/router_demo.py --mode interactive
  python scripts/router_demo.py --mode bench --n 200
  python scripts/router_demo.py --mode bench --wav_dir artifacts/data/samples/ --n 100
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["wav", "interactive", "bench"],
        required=True,
        help="Demo mode.",
    )
    parser.add_argument("--wav", default=None, help="Path to .wav file (--mode wav).")
    parser.add_argument(
        "--wav_dir", default=None,
        help="Directory of .wav files for bench mode (uses synthetic noise if absent).",
    )
    parser.add_argument("--n", type=int, default=100, help="Calls per engine in bench mode.")
    parser.add_argument(
        "--routing_yaml",
        default="configs/routing.yaml",
        help="Path to routing config YAML.",
    )
    parser.add_argument(
        "--model_yaml",
        default="configs/hybrid/model.yaml",
        help="Hybrid model config YAML.",
    )
    parser.add_argument(
        "--thresh_yaml",
        default="configs/hybrid/thresholds.yaml",
        help="Hybrid thresholds YAML.",
    )
    parser.add_argument(
        "--onnx_dir",
        default=None,
        help="ONNX bundle dir (overrides configs/base.yaml path).",
    )

    main(parser.parse_args())
