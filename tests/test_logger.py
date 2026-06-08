import os
import json
from pathlib import Path
from core.logger import get_logger, JsonFormatter
from core.logger import setup_app_logger, get_logger

def test_logger_json_format(tmp_path):
    from core.logger import setup_app_logger
    import json

    # CHANGED: Создаем изолированный логгер для теста
    test_log_dir = tmp_path / "test_logs"
    logger = setup_app_logger("test_json", log_level="INFO", log_dir=str(test_log_dir))

    logger.info("Test message", extra={"extra_data": {"key": "val"}})

    # CHANGED: Принудительно закрываем хендлеры, чтобы Windows отпустила файл
    for handler in logger.handlers:
        handler.close()

    log_file = test_log_dir / "app.log"
    assert log_file.exists()

    with open(log_file, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())
        assert data["message"] == "Test message"