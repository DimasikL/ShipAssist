import pytest
from core.exceptions import ShipAssistantError, AudioDeviceError

def test_exceptions_inheritance():
    """Проверка правильности наследования исключений."""
    with pytest.raises(ShipAssistantError):
        raise AudioDeviceError("Mic failed")

def test_exception_message():
    """Проверка сохранения сообщения об ошибке."""
    msg = "Custom error"
    ex = AudioDeviceError(msg)
    assert str(ex) == msg
    assert ex.message == msg