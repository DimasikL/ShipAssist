class ShipAssistantError(Exception):
    """Базовое исключение проекта."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class AudioDeviceError(ShipAssistantError):
    """Ошибка доступа к микрофону или звуковой карте."""
    pass

class AudioFormatError(ShipAssistantError):
    """Файл поврежден или имеет неверный формат (не 16кГц/моно)."""
    pass

class RecognitionError(ShipAssistantError):
    """Ошибка в процессе инференса модели."""
    pass

class ConfigError(ShipAssistantError):
    """Ошибка в файле конфигурации или параметрах окружения."""
    pass

class ModelLoadError(ShipAssistantError):
    """Ошибка при загрузке весов модели или инициализации ONNX."""
    pass