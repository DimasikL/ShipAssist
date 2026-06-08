import sys
import os

# Добавляем текущую директорию в путь, чтобы импорты src. и core. работали
sys.path.append(os.getcwd())

def test_imports():
    print("--- Проверка системы ShipAssistant ---")

    try:
        from core.config import settings
        print(f"✅ Конфигурация загружена. Проект: {settings.paths.base_dir}")
    except Exception as e:
        print(f"❌ Ошибка в core/config.py: {e}")
        return

    try:
        from core.logger import logger
        logger.info("Проверка системы логирования")
        print("✅ Логгер работает (проверьте папку logs/)")
    except Exception as e:
        print(f"❌ Ошибка в core/logger.py: {e}")

    try:
        from src.data_utils import parse_metadata
        print("✅ Модуль данных (src/data_utils.py) найден")
    except Exception as e:
        print(f"❌ Ошибка в src/data_utils.py: {e}")

    try:
        from src.train import Trainer
        print("✅ Модуль обучения (src/train.py) найден")
    except Exception as e:
        print(f"❌ Ошибка в src/train.py: {e}")

    try:
        from src.api import app
        print("✅ REST API (src/api.py) готов к запуску")
    except Exception as e:
        print(f"❌ Ошибка в src/api.py: {e}")

if __name__ == "__main__":
    test_imports()