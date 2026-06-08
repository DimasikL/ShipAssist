import logging
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict


class _WinSafeRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that silently skips rotation on Windows when the
    log file is locked by another process (PermissionError / WinError 32).

    On Windows, ``os.rename`` fails if the destination file is open by any
    process. The base class propagates this as a logging error that clutters
    stdout. This subclass catches the error during ``doRollover`` and falls
    back to simple appending, which is always safe.
    """

    def doRollover(self) -> None:  # type: ignore[override]
        """Attempt rotation; silently skip on Windows lock errors."""
        try:
            super().doRollover()
        except PermissionError:
            # File is locked by another process — skip rotation this cycle.
            # The file will be rotated on the next attempt once the lock is released.
            pass

class JsonFormatter(logging.Formatter):
    """Форматирует записи лога в JSON."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_record["data"] = record.extra_data

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)

def setup_app_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Настраивает логгер с ротацией файлов.

    Args:
        name:         Logger name (e.g. module ``__name__``).
        log_level:    Root log level string (DEBUG / INFO / WARNING / …).
        log_dir:      Directory where ``app.log`` is written.
        max_bytes:    Maximum size of a single log file before rotation
                      (sourced from ``settings.logging.max_bytes``).
        backup_count: Number of rotated files to retain
                      (sourced from ``settings.logging.backup_count``).

    Returns:
        Configured :class:`logging.Logger` instance with a rotating file
        handler and a console handler, both using :class:`JsonFormatter`.

    Note:
        Callers should prefer :func:`get_logger` which automatically pulls
        ``max_bytes`` / ``backup_count`` from the centralised config.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    if logger.hasHandlers():
        logger.handlers.clear()

    # Форматтер
    formatter = JsonFormatter()

    # Файл для всех логов — размер и количество ротаций берутся из конфига.
    # _WinSafeRotatingFileHandler: на Windows RotatingFileHandler.doRollover()
    # падает с PermissionError (WinError 32) если app.log открыт другим процессом.
    # Подкласс перехватывает ошибку и пропускает ротацию без вывода в stderr.
    app_handler = _WinSafeRotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
        delay=True,
    )
    app_handler.setFormatter(formatter)

    # Консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(app_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Облегченная функция получения логгера для модулей.

    Pulls log level, directory, max file size, and backup count from
    ``core.config.settings`` (configs/base.yaml → logging section).
    Falls back to safe built-in defaults if the config is not yet loaded
    (e.g., during early bootstrap or in isolated test runs).

    Args:
        name: Logger name — typically the calling module's ``__name__``.

    Returns:
        Configured :class:`logging.Logger` with file + console handlers.
    """
    try:
        from core.config import settings
        level = settings.logging.level
        directory = settings.paths.logs_dir
        # Rotation limits sourced from settings (configs/base.yaml → logging)
        max_bytes = settings.logging.max_bytes
        backup_count = settings.logging.backup_count
    except Exception:
        # Fallback если конфиг ещё не загружен (ранняя стадия инициализации)
        level = "INFO"
        directory = "logs"
        max_bytes = 10 * 1024 * 1024  # 10 MB
        backup_count = 5

    return setup_app_logger(
        name,
        log_level=level,
        log_dir=str(directory),
        max_bytes=max_bytes,
        backup_count=backup_count,
    )

# CHANGED: Инициализация корневого логгера проекта
logger = get_logger("ShipAssistant")