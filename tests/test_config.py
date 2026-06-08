import os
import pytest
import yaml
from core.config import Settings

def test_config_load_success(test_config_path):
    """Простая загрузка."""
    cfg = Settings.load(test_config_path)
    assert cfg.audio.sample_rate == 16000

def test_config_env_override(test_config_path):
    """CHANGED: Проверка переопределения через ENV с удалением ключа из словаря."""
    with open(test_config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Удаляем порт из словаря, чтобы он пришел ТОЛЬКО из ENV
    if 'api' in data:
        data['api'].pop('port', None)

    # Устанавливаем переменную окружения
    os.environ["SHIP_API__PORT"] = "9999"

    try:
        # Инициализируем — Pydantic найдет порт в ENV
        cfg = Settings(**data)
        assert cfg.api.port == 9999
    finally:
        # Обязательная чистка
        if "SHIP_API__PORT" in os.environ:
            del os.environ["SHIP_API__PORT"]