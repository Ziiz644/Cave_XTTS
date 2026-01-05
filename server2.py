import os
import re
import time
import json
import hashlib
import tempfile
from typing import Optional, Dict, Any, Generator

import httpx
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    Body,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# Coqui TTS
from TTS.api import TTS

APP_NAME = "xtts-service"

# ---------------- Config (ENV) ----------------
MODEL_NAME = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
USE_GPU = os.getenv("USE_GPU", "true").lower() in ("1", "true", "yes", "y")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "ar").strip().lower()  # "ar" or "en"

# If you want a fallback local voice in the container
DEFAULT_SPEAKER_WAV = os.getenv("DEFAULT_SPEAKER_WAV", "").strip()

OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))

# Speaker download hardening
MAX_SPEAKER_WAV_BYTES = int(os.getenv("MAX_SPEAKER_WAV_BYTES", str(8 * 1024 * 1024)))  # 8MB
SPEAKER_HTTP_TIMEOUT = float(os.getenv("SPEAKER_HTTP_TIMEOUT", "20"))
ALLOW_SPEAKER_URL = os.getenv("ALLOW_SPEAKER_URL", "true").lower() in ("1", "true", "yes", "y")

# Streaming chunk size
STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", str(64 * 1024)))  # 64KB

# Concurrency control (VERY important on GPU)
MAX_CONCURRENT_SYNTH = int(os.getenv("MAX_CONCURRENT_SYNTH", "2"))

# CORS
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8100,http://127.0.0.1:8100,https://api.mda.sa,https://mda.sa,https://www.mda.sa",
).split(",")

# ------------------------------------------------

app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Request-Id"],
)

tts: Optional[TTS] = None

# Simple process-level semaphore to protect GPU/CPU
try:
    import threading
    _sem = threading.BoundedSemaphore(MAX_CONCURRENT_SYNTH)
except Exception:
    _sem = None


# ---------------- Models ----------------

class TtsJsonRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    language: str = Field(default=DEFAULT_LANG)
    # Provide ONE of these:
    speaker_wav_url: Optional[str] = None  # Wasabi URL / presigned URL
    speaker_wav_path: Optional[str] = None  # path inside container (optional)

    @field_validator("language")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        v = (v or DEFAULT_LANG).strip().lower()
        if v not in ("ar", "en"):
            raise ValueError('language must be "ar" or "en"')
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("text is required")
        if len(v) > MAX_TEXT_CHARS:
            raise ValueError(f"text too long (max {MAX_TEXT_CHARS} chars)")
        return v


# ---------------- Utilities ----------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_filename(name: str) -> str:
    name = (name or "audio").strip()
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    if not name.endswith(".wav"):
        name += ".wav"
    return name


def _tmp_file(prefix: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path


def _cleanup_files(*paths: str) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def _stream_file(path: str, chunk_size: int = STREAM_CHUNK_SIZE) -> Generator[bytes, None, None]:
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


async def _download_to_temp_wav(url: str) -> str:
    """
    Downloads speaker wav from URL to a temp file with size limit.
    Returns local path. Raises HTTPException on failure.
    """
    if not ALLOW_SPEAKER_URL:
        raise HTTPException(status_code=400, detail="speaker_wav_url is disabled on this server")

    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="speaker_wav_url is empty")

    url = url.strip()

    # Use a deterministic temp name (helps re-use in higher layers if you cache later)
    tmp_path = _tmp_file("xtts_ref_", ".wav")

    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    timeout = httpx.Timeout(SPEAKER_HTTP_TIMEOUT)

    total = 0
    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise HTTPException(status_code=400, detail=f"Failed to download speaker wav (HTTP {resp.status_code})")

                # Optional checks
                ctype = (resp.headers.get("content-type") or "").lower()
                # Some object stores return octet-stream; accept it.
                if ctype and ("audio" not in ctype and "octet-stream" not in ctype):
                    # Don’t hard fail; but it’s a useful guardrail in prod.
                    pass

                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_SPEAKER_WAV_BYTES:
                            raise HTTPException(status_code=400, detail="speaker wav too large")
                        f.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="Downloaded speaker wav is empty")

        return tmp_path

    except HTTPException:
        _cleanup_files(tmp_path)
        raise
    except Exception as e:
        _cleanup_files(tmp_path)
        raise HTTPException(status_code=400, detail=f"Error downloading speaker wav: {str(e)}")


def _write_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "ref.wav")[1] or ".wav"
    tmp_path = _tmp_file("xtts_ref_", suffix)

    # NOTE: UploadFile.file is a SpooledTemporaryFile; reading it is fine for typical WAV sizes.
    data = upload.file.read()
    if not data:
        _cleanup_files(tmp_path)
        raise HTTPException(status_code=400, detail="speaker_wav upload is empty")

    if len(data) > MAX_SPEAKER_WAV_BYTES:
        _cleanup_files(tmp_path)
        raise HTTPException(status_code=400, detail="speaker wav too large")

    with open(tmp_path, "wb") as f:
        f.write(data)

    return tmp_path


def _resolve_ref_path(
    speaker_upload: Optional[UploadFile],
    speaker_wav_url: Optional[str],
    speaker_wav_path: Optional[str],
) -> Dict[str, Optional[str]]:
    """
    Resolve speaker reference WAV source.
    Returns: { "ref_path": ..., "tmp_ref": ... }  (tmp_ref is for cleanup)
    """
    # Upload takes priority
    if speaker_upload is not None:
        tmp_ref = _write_upload_to_temp(speaker_upload)
        return {"ref_path": tmp_ref, "tmp_ref": tmp_ref}

    # URL will be downloaded by the caller (async), so here just pass through intent
    if speaker_wav_url:
        return {"ref_path": None, "tmp_ref": None}

    # Container path
    sp = (speaker_wav_path or DEFAULT_SPEAKER_WAV).strip()
    if sp:
        if not os.path.exists(sp):
            raise HTTPException(status_code=400, detail=f"speaker_wav_path not found: {sp}")
        return {"ref_path": sp, "tmp_ref": None}

    raise HTTPException(status_code=400, detail="Provide speaker_wav upload, speaker_wav_url, or speaker_wav_path/DEFAULT_SPEAKER_WAV")


def _acquire_sem_or_503():
    if _sem is None:
        return None
    ok = _sem.acquire(timeout=60)  # avoid hanging forever
    if not ok:
        raise HTTPException(status_code=503, detail="TTS server is busy, try again")
    return True


def _release_sem():
    if _sem is not None:
        try:
            _sem.release()
        except Exception:
            pass


# ---------------- Lifecycle ----------------

@app.on_event("startup")
def load_model():
    global tts
    # Load once
    tts = TTS(MODEL_NAME, gpu=USE_GPU)


@app.get("/health")
def health():
    return {
        "service": APP_NAME,
        "model": MODEL_NAME,
        "gpu": USE_GPU,
        "default_lang": DEFAULT_LANG,
        "max_text_chars": MAX_TEXT_CHARS,
        "max_concurrent_synth": MAX_CONCURRENT_SYNTH,
    }


# ---------------- Endpoints ----------------

@app.post("/tts")
async def tts_multipart(
    background_tasks: BackgroundTasks,
    request: Request,
    text: str = Form(...),
    language: str = Form(DEFAULT_LANG),
    speaker_wav: UploadFile = File(None),
    # Optional: allow URL in multipart form too
    speaker_wav_url: Optional[str] = Form(None),
    # Optional: allow container path in multipart
    speaker_wav_path: Optional[str] = Form(None),
):
    """
    Multipart endpoint (works with browser forms, but your Spring is currently sending JSON).
    Supports:
      - text (form)
      - language (form)
      - speaker_wav (file) OR speaker_wav_url (form) OR speaker_wav_path (form)
    Returns: audio/wav (StreamingResponse)
    """
    global tts
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"text too long (max {MAX_TEXT_CHARS} chars)")

    language = (language or DEFAULT_LANG).strip().lower()
    if language not in ("ar", "en"):
        raise HTTPException(status_code=400, detail='language must be "ar" or "en"')

    # Concurrency control
    _acquire_sem_or_503()

    tmp_ref = None
    out_path = None
    try:
        ref_info = _resolve_ref_path(speaker_wav, speaker_wav_url, speaker_wav_path)

        # If URL is provided, download it now (async)
        ref_path = ref_info["ref_path"]
        if ref_path is None and speaker_wav_url:
            tmp_ref = await _download_to_temp_wav(speaker_wav_url)
            ref_path = tmp_ref

        # Synthesize to temp wav
        out_path = _tmp_file("xtts_out_", ".wav")

        tts.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=language,
            file_path=out_path,
        )

        # cleanup after response is finished sending
        background_tasks.add_task(_cleanup_files, tmp_ref, out_path)

        filename = _safe_filename("tts")
        headers = {
            "Content-Disposition": f'inline; filename="{filename}"',
        }

        return StreamingResponse(
            _stream_file(out_path),
            media_type="audio/wav",
            headers=headers,
            background=background_tasks,
        )

    finally:
        _release_sem()


@app.post("/tts_json")
async def tts_json_stream(
    background_tasks: BackgroundTasks,
    payload: TtsJsonRequest = Body(...),
):
    """
    JSON endpoint (BEST for your Spring backend)
    Request:
    {
      "text": "...",
      "language": "ar",
      "speaker_wav_url": "https://.../ref.wav"   // recommended
      // OR "speaker_wav_path": "/app/voices/ref.wav"
    }

    Returns: audio/wav streamed as chunked response.
    """
    global tts
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = payload.text
    language = payload.language

    # Concurrency control
    _acquire_sem_or_503()

    tmp_ref = None
    out_path = None
    try:
        # Resolve reference voice
        speaker_wav_url = (payload.speaker_wav_url or "").strip() or None
        speaker_wav_path = (payload.speaker_wav_path or "").strip() or None

        if speaker_wav_url:
            tmp_ref = await _download_to_temp_wav(speaker_wav_url)
            ref_path = tmp_ref
        else:
            ref_path = speaker_wav_path or DEFAULT_SPEAKER_WAV
            if not ref_path:
                raise HTTPException(status_code=400, detail="Provide speaker_wav_url or speaker_wav_path/DEFAULT_SPEAKER_WAV")
            if not os.path.exists(ref_path):
                raise HTTPException(status_code=400, detail=f"speaker_wav_path not found: {ref_path}")

        # Synthesize
        out_path = _tmp_file("xtts_out_", ".wav")

        tts.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=language,
            file_path=out_path,
        )

        background_tasks.add_task(_cleanup_files, tmp_ref, out_path)

        headers = {"Content-Disposition": 'inline; filename="tts.wav"'}

        return StreamingResponse(
            _stream_file(out_path),
            media_type="audio/wav",
            headers=headers,
            background=background_tasks,
        )

    finally:
        _release_sem()


@app.get("/info")
def info():
    """Small introspection endpoint for ops checks."""
    return {
        "service": APP_NAME,
        "model": MODEL_NAME,
        "gpu": USE_GPU,
        "cors_origins": [o.strip() for o in CORS_ORIGINS if o.strip()],
        "allow_speaker_url": ALLOW_SPEAKER_URL,
        "max_speaker_wav_bytes": MAX_SPEAKER_WAV_BYTES,
        "max_text_chars": MAX_TEXT_CHARS,
        "max_concurrent_synth": MAX_CONCURRENT_SYNTH,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
    }
