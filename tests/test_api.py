import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "model_loaded" in body
    assert "uptime_seconds" in body


def test_api_recognize_success(mocker, raw_audio):
    """Recognition endpoint with engine and soundfile fully mocked."""
    mock_engine = mocker.Mock()
    mock_engine.labels = ["other", "command"]
    mock_engine.predict.return_value = {
        "label": "command",
        "confidence": 0.9,
        "probs": np.array([0.1, 0.9]),
        "logits": np.array([-2.2, 2.2]),
        "latency_ms": 12.5,
    }
    # Engine now lives in app.state, not as a module-level global
    app.state.engine = mock_engine

    mocker.patch("soundfile.read", return_value=(raw_audio, 16000))

    buf = io.BytesIO()
    sf.write(buf, raw_audio, 16000, format="WAV")
    buf.seek(0)

    response = client.post(
        "/recognize",
        files={"file": ("test.wav", buf, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["command"] == "command"
    assert response.json()["confidence"] == pytest.approx(0.9)


def test_api_recognize_no_engine():
    """Returns 503 when engine is not loaded."""
    app.state.engine = None
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    buf.seek(0)

    response = client.post(
        "/recognize",
        files={"file": ("test.wav", buf, "audio/wav")},
    )
    assert response.status_code == 503


def test_api_recognize_unsupported_format(mocker):
    """Returns 400 for non-audio file extensions."""
    mock_engine = mocker.Mock()
    mock_engine.labels = ["other"]
    app.state.engine = mock_engine

    response = client.post(
        "/recognize",
        files={"file": ("audio.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400


def test_api_recognize_file_too_large(mocker):
    """Returns 413 when uploaded file exceeds max_file_size_mb."""
    mock_engine = mocker.Mock()
    app.state.engine = mock_engine

    # Patch the limit to 1 byte so any real file triggers it
    mocker.patch("src.api.settings.api.max_file_size_mb", 0)

    buf = io.BytesIO(b"\x00" * 10)
    response = client.post(
        "/recognize",
        files={"file": ("test.wav", buf, "audio/wav")},
    )
    assert response.status_code == 413
