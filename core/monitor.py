import threading
import time
from typing import Dict, Optional

import psutil
import torch

from core.logger import get_logger

logger = get_logger(__name__)


class ResourceMonitor:
    """Background thread that periodically logs RAM, VRAM, and CPU usage.

    Thresholds and polling interval are sourced from
    ``settings.monitor`` (configs/base.yaml → monitor section) so they
    can be tuned without touching source code.
    """

    def __init__(self, interval_seconds: Optional[int] = None, ram_alert_mb: Optional[float] = None) -> None:
        from core.config import settings

        self.interval: int = interval_seconds if interval_seconds is not None else settings.monitor.interval_seconds
        self.ram_alert_mb: float = ram_alert_mb if ram_alert_mb is not None else settings.monitor.ram_alert_mb

        self.is_running: bool = False
        self._thread: Optional[threading.Thread] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_usage(self) -> Dict[str, float]:
        """Collect current process resource statistics."""
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024 * 1024)

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_reserved() / (1024 * 1024)

        return {
            "ram_usage_mb": round(ram_mb, 2),
            "vram_usage_mb": round(vram_mb, 2),
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
        }

    def _monitor_loop(self) -> None:
        while self.is_running:
            usage = self._get_usage()
            logger.info("System health check", extra={"extra_data": usage})

            if usage["ram_usage_mb"] > self.ram_alert_mb:
                logger.error(
                    "High RAM usage: %.1f MB (alert threshold: %.0f MB)",
                    usage["ram_usage_mb"],
                    self.ram_alert_mb,
                )

            time.sleep(self.interval)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background monitoring thread (idempotent)."""
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(
            "ResourceMonitor started — interval=%ds ram_alert=%.0f MB",
            self.interval, self.ram_alert_mb,
        )

    def stop(self) -> None:
        """Stop monitoring and join the background thread."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
        logger.info("ResourceMonitor stopped.")
