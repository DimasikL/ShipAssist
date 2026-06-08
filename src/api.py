import os
import time
import tempfile
import numpy as np
import soundfile as sf
import soxr
from datetime import datetime
from typing import List, Optional
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import settings
from core.engine import AudioEngine, create_engine
from core.logger import get_logger

logger = get_logger("API")

_start_time = time.time()
_log_storage: deque = deque(maxlen=100)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the inference engine on startup; clean up on shutdown."""
    onnx_path = settings.paths.onnx_model
    logger.info("Loading model from: %s", onnx_path)

    if not onnx_path.exists():
        # Log and continue — /health will report model_loaded=false.
        logger.error("Model directory not found: %s. Run ONNX export first.", onnx_path)
        app.state.engine = None
    else:
        try:
            app.state.engine = create_engine(settings)
            logger.info("Engine ready — type=%s", settings.model.type)
        except Exception as exc:
            logger.error("Engine initialisation failed: %s", exc, exc_info=True)
            app.state.engine = None

    yield

    logger.info("API shutting down.")
    app.state.engine = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ShipAssistant API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class RecognitionResult(BaseModel):
    command: str
    confidence: float
    timestamp: str
    latency_ms: float


class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_engine(request: Request) -> AudioEngine:
    """Retrieve engine from app.state; raise 503 if not loaded."""
    engine: Optional[AudioEngine] = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Recognition engine not loaded")
    return engine


def _load_and_normalise(path: str) -> np.ndarray:
    """Read audio file, convert to mono float32, resample to 16 000 Hz if needed."""
    audio, sr = sf.read(path)

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Resample to model sample rate if the uploaded file differs
    target_sr: int = settings.audio.sample_rate
    if sr != target_sr:
        logger.debug("Resampling %d Hz → %d Hz", sr, target_sr)
        audio = soxr.resample(audio, sr, target_sr, quality="HQ")

    return audio


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthStatus)
async def health_check(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.engine is not None,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


@app.get("/commands", response_model=List[str])
async def get_commands(request: Request):
    return _get_engine(request).labels


@app.post("/recognize", response_model=RecognitionResult)
async def recognize_audio(request: Request, file: UploadFile = File(...)):
    engine = _get_engine(request)

    # ── Validate format ───────────────────────────────────────────────────────
    allowed = {".wav", ".mp3", ".m4a"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Allowed: {sorted(allowed)}",
        )

    # ── Enforce file size limit ───────────────────────────────────────────────
    content = await file.read()
    max_bytes = settings.api.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(content) / 1024 / 1024:.1f} MB "
                f"(limit {settings.api.max_file_size_mb} MB)"
            ),
        )

    t_start = time.perf_counter()
    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        audio = _load_and_normalise(tmp_path)
        prediction = engine.predict(audio)

        result = {
            "command": prediction["label"],
            "confidence": prediction["confidence"],
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 3),
        }
        _log_storage.append(result)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Recognition failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/logs")
async def get_logs(limit: int = Query(10, ge=1, le=100)):
    return list(_log_storage)[-limit:]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)
