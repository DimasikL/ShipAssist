import argparse
import time

import numpy as np

from core.config import settings
from core.engine import AudioEngine, create_engine
from core.logger import get_logger
from core.recognizer import RealTimeRecognizer

logger = get_logger("Inference")


def main() -> None:
    """Real-time microphone recognition loop.

    Supports engine selection via --mode and automatic mic reconnect.
    """
    parser = argparse.ArgumentParser(description="ShipAssistant Local Voice Interface")
    parser.add_argument(
        "--mode",
        choices=["onnx", "torch"],
        default="onnx",
        help="Inference backend: 'onnx' (production) or 'torch' (debug)",
    )
    args = parser.parse_args()

    # ── Load engine ───────────────────────────────────────────────────────────
    logger.info("Starting local interface — mode=%s", args.mode)
    try:
        engine: AudioEngine = create_engine(settings, mode=args.mode)
        logger.info("Engine loaded — labels=%s", engine.labels)
    except Exception as exc:
        logger.critical("Failed to load engine: %s", exc, exc_info=True)
        return

    # ── Build recognizer ──────────────────────────────────────────────────────
    recognizer = RealTimeRecognizer(
        sample_rate=settings.audio.sample_rate,
        window_s=settings.audio.window_seconds,
        stride_s=settings.audio.stride_seconds,
    )

    def on_command_detected(audio_chunk: np.ndarray) -> None:
        """Callback invoked for each audio window ready for inference.

        Runs in the sounddevice background thread — any uncaught exception
        here would kill the thread silently, so all errors are caught and
        logged instead of propagated.
        """
        try:
            audio_data = audio_chunk.flatten().astype(np.float32)
            result = engine.predict(audio_data)
            label: str = result["label"]
            conf: float = result["confidence"]

            threshold = settings.recognition.per_label_thresholds.get(
                label, settings.recognition.default_confidence
            )

            if conf >= threshold:
                logger.info("DETECTED: %s (conf=%.2f)", label.upper(), conf)
            elif conf > settings.recognition.noise_log_threshold:
                logger.debug("Low-confidence match: %s (conf=%.2f)", label, conf)

        except Exception as exc:
            # Log and continue — do NOT re-raise; re-raising would terminate
            # the sounddevice callback thread and stop all future detections.
            logger.error("Inference error in callback: %s", exc, exc_info=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        recognizer.start_stream(callback=on_command_detected)
        logger.info("Listening... (Ctrl+C to stop)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        recognizer.stop()


if __name__ == "__main__":
    main()
