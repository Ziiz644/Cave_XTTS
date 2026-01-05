import os
import re
import time
import tempfile
from typing import Optional, Dict, Any, Generator

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from TTS.api import TTS

APP_NAME = "xtts-service"

# ---------------- Config (ENV) ----------------
MODEL_NAME = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
USE_GPU = os.getenv("USE_GPU", "true").lower() in ("1", "true", "yes", "y")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "ar").strip().lower()  # "ar" or "en"

DEFAULT_SPEAKER_WAV = os.getenv("DEFAULT_SPEAKER_WAV", "").strip()

# XTTS v2 commonly outputs 24000. We'll treat this as the ONLY supported PCM sample rate unless you resample.
OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))

# Speaker download hardening
MAX_SPEAKER_WAV_BYTES = int(os.getenv("MAX_SPEAKER_WAV_BYTES", str(8 * 1024 * 1024)))  # 8MB
SPEAKER_HTTP_TIMEOUT = float(os.getenv("SPEAKER_HTTP_TIMEOUT", "20"))
ALLOW_SPEAKER_URL = os.getenv("ALLOW_SPEAKER_URL", "true").lower() in ("1", "true", "yes", "y")

# Chunking for "stream-like" behavior (generate sentence by sentence)
# Smaller = more responsive streaming.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "220"))

# Concurrency control (VERY important on GPU)
MAX_CONCURRENT_SYNTH = int(os.getenv("MAX_CONCURRENT_SYNTH", "1"))

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8100,http://127.0.0.1:8100,https://api.mda.sa,https://mda.sa,https://www.mda.sa",
).split(",")

# ------------------------------------------------

app = FastAPI(title=APP_NAME, version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Request-Id", "X-Audio-Format", "X-Sample-Rate"],
)

tts: Optional[TTS] = None

# GPU semaphore
try:
    import threading
    _sem = threading.BoundedSemaphore(MAX_CONCURRENT_SYNTH)
except Exception:
    _sem = None

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\u061F])\s+")


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    language: str = Field(default=DEFAULT_LANG)

    # Provide ONE:
    speaker_wav_url: Optional[str] = None  # presigned URL (Wasabi)
    speaker_wav_path: Optional[str] = None  # inside container

    # Output control
    format: str = Field(default="pcm")  # "pcm" or "wav"
    sample_rate: int = Field(default=OUTPUT_SAMPLE_RATE)  # for pcm path we only support OUTPUT_SAMPLE_RATE

    @field_validator("language")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        v = (v or DEFAULT_LANG).strip().lower()
        if v not in ("ar", "en"):
            raise ValueError('language must be "ar" or "en"')
        return v

    @field_validator("format")
    @classmethod
    def validate_fmt(cls, v: str) -> str:
        v = (v or "pcm").strip().lower()
        if v not in ("pcm", "wav"):
            raise ValueError('format must be "pcm" or "wav"')
        return v


def chunk_text(text: str, max_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _acquire_sem_or_503():
    if _sem is None:
        return
    ok = _sem.acquire(timeout=60)
    if not ok:
        raise HTTPException(status_code=503, detail="TTS server is busy, try again")


def _release_sem():
    if _sem is None:
        return
    try:
        _sem.release()
    except Exception:
        pass


async def _download_to_temp(url: str) -> str:
    if not ALLOW_SPEAKER_URL:
        raise HTTPException(status_code=400, detail="speaker_wav_url is disabled")
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="speaker_wav_url is empty")

    fd, tmp_path = tempfile.mkstemp(prefix="xtts_ref_", suffix=".wav")
    os.close(fd)

    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    timeout = httpx.Timeout(SPEAKER_HTTP_TIMEOUT)
    total = 0

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
            async with client.stream("GET", url.strip()) as resp:
                if resp.status_code >= 400:
                    raise HTTPException(status_code=400, detail=f"Failed to download speaker wav (HTTP {resp.status_code})")

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
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Error downloading speaker wav: {str(e)}")


def _float_to_s16le_pcm(audio: np.ndarray) -> bytes:
    """
    XTTS usually returns float32 numpy in [-1,1]. Convert to signed 16-bit little-endian PCM.
    """
    if audio is None:
        return b""
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes(order="C")


@app.on_event("startup")
def load_model():
    global tts
    tts = TTS(MODEL_NAME, gpu=USE_GPU)


@app.get("/health")
def health():
    return {
        "service": APP_NAME,
        "model": MODEL_NAME,
        "gpu": USE_GPU,
        "default_lang": DEFAULT_LANG,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "max_text_chars": MAX_TEXT_CHARS,
        "max_concurrent_synth": MAX_CONCURRENT_SYNTH,
        "max_chunk_chars": MAX_CHUNK_CHARS,
    }


@app.post("/tts")
async def tts_stream(payload: TtsRequest = Body(...)):
    """
    JSON endpoint designed for Spring WebFlux:
    - returns STREAMED bytes
    - for format="pcm": application/octet-stream (s16le mono)
    - for format="wav": audio/wav (NOT recommended for your current Java streamPcm)
    """
    global tts
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"text too long (max {MAX_TEXT_CHARS} chars)")

    language = payload.language
    fmt = payload.format
    sample_rate = int(payload.sample_rate or OUTPUT_SAMPLE_RATE)

    # IMPORTANT: if you need resampling, we can add it later, but keep it strict for stability.
    if fmt == "pcm" and sample_rate != OUTPUT_SAMPLE_RATE:
        raise HTTPException(
            status_code=400,
            detail=f"PCM sample_rate must be {OUTPUT_SAMPLE_RATE} for this server (got {sample_rate})",
        )

    _acquire_sem_or_503()

    tmp_ref = None
    try:
        # Resolve speaker reference
        speaker_wav_url = (payload.speaker_wav_url or "").strip() or None
        speaker_wav_path = (payload.speaker_wav_path or "").strip() or None

        if speaker_wav_url:
            tmp_ref = await _download_to_temp(speaker_wav_url)
            ref_path = tmp_ref
        else:
            ref_path = speaker_wav_path or DEFAULT_SPEAKER_WAV
            if not ref_path:
                raise HTTPException(status_code=400, detail="Provide speaker_wav_url or speaker_wav_path/DEFAULT_SPEAKER_WAV")
            if not os.path.exists(ref_path):
                raise HTTPException(status_code=400, detail=f"speaker_wav_path not found: {ref_path}")

        # Chunk the text so client can start playing earlier (pseudo-streaming)
        chunks = chunk_text(text, MAX_CHUNK_CHARS)

        def gen() -> Generator[bytes, None, None]:
            try:
                for ch in chunks:
                    # Generate per chunk
                    audio = tts.tts(text=ch, speaker_wav=ref_path, language=language)

                    if fmt == "pcm":
                        yield _float_to_s16le_pcm(audio)
                    else:
                        # WAV streaming not used by your Java streamPcm
                        # If you ever need it, we can add in-memory WAV encoding here.
                        raise RuntimeError("format=wav not implemented in generator")
            finally:
                pass

        headers = {
            "X-Audio-Format": fmt,
            "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
        }

        if fmt == "pcm":
            return StreamingResponse(gen(), media_type="application/octet-stream", headers=headers)
        else:
            # Not used in your current backend path
            raise HTTPException(status_code=400, detail="Use format=pcm for streaming integration")

    finally:
        # cleanup + release semaphore
        if tmp_ref and os.path.exists(tmp_ref):
            try:
                os.remove(tmp_ref)
            except Exception:
                pass
        _release_sem()
