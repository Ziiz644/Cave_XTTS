import os
import re
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import httpx
import torchaudio

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from TTS.api import TTS

APP_NAME = "xtts-service"

# ---------------- Config (ENV) ----------------
MODEL_NAME = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
USE_GPU = os.getenv("USE_GPU", "true").lower() in ("1", "true", "yes", "y")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "ar").strip().lower()  # "ar" or "en"

DEFAULT_SPEAKER_WAV = os.getenv("DEFAULT_SPEAKER_WAV", "").strip()

OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))

MAX_SPEAKER_WAV_BYTES = int(os.getenv("MAX_SPEAKER_WAV_BYTES", str(8 * 1024 * 1024)))  # 8MB
SPEAKER_HTTP_TIMEOUT = float(os.getenv("SPEAKER_HTTP_TIMEOUT", "20"))
ALLOW_SPEAKER_URL = os.getenv("ALLOW_SPEAKER_URL", "true").lower() in ("1", "true", "yes", "y")

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "220"))
MAX_CONCURRENT_SYNTH = int(os.getenv("MAX_CONCURRENT_SYNTH", "1"))

# Cache settings
REF_CACHE_DIR = Path(os.getenv("REF_CACHE_DIR", "/tmp/xtts_ref_cache"))
REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# If true: keep cached refs and never delete them
CACHE_SPEAKER_WAV = os.getenv("CACHE_SPEAKER_WAV", "true").lower() in ("1", "true", "yes", "y")

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8100,http://127.0.0.1:8100,https://api.mda.sa,https://mda.sa,https://www.mda.sa",
).split(",")

# ------------------------------------------------

app = FastAPI(title=APP_NAME, version="1.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Request-Id", "X-Audio-Format", "X-Sample-Rate"],
)

tts: Optional[TTS] = None

# Concurrency control (GPU-friendly)
try:
    import threading
    _sem = threading.BoundedSemaphore(MAX_CONCURRENT_SYNTH)
except Exception:
    _sem = None

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\u061F])\s+")


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    language: str = Field(default=DEFAULT_LANG)

    # Prefer this name:
    speaker_wav_url: Optional[str] = None
    speaker_wav_path: Optional[str] = None

    # Backward compat: accept "speaker_wav" from older clients
    speaker_wav: Optional[str] = None

    format: str = Field(default="pcm")  # "pcm" or "wav"
    sample_rate: int = Field(default=OUTPUT_SAMPLE_RATE)

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

    @model_validator(mode="after")
    def normalize_speaker_fields(self):
        # If caller used speaker_wav, treat it as speaker_wav_url
        if (not self.speaker_wav_url) and self.speaker_wav:
            self.speaker_wav_url = self.speaker_wav
        return self


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


def _float_to_s16le_pcm(audio: np.ndarray) -> bytes:
    if audio is None:
        return b""
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes(order="C")


def _url_cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


async def _download_and_cache_ref(url: str) -> str:
    """
    Downloads speaker wav from URL and saves atomically to /tmp/xtts_ref_cache/<sha1>.wav.
    Returns the local file path.
    """
    if not ALLOW_SPEAKER_URL:
        raise HTTPException(status_code=400, detail="speaker_wav_url is disabled")
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="speaker_wav_url is empty")

    url = url.strip()
    key = _url_cache_key(url)

    # If caching enabled, reuse same file always
    cached_path = REF_CACHE_DIR / f"{key}.wav"
    if CACHE_SPEAKER_WAV and cached_path.exists() and cached_path.stat().st_size > 1024:
        return str(cached_path)

    # Otherwise download to a unique tmp and (optionally) replace cached
    tmp_path = REF_CACHE_DIR / f"{key}.tmp.{uuid.uuid4().hex}.wav"

    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    timeout = httpx.Timeout(SPEAKER_HTTP_TIMEOUT)
    total = 0

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise HTTPException(status_code=400, detail=f"Failed to download speaker wav (HTTP {resp.status_code})")

                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_SPEAKER_WAV_BYTES:
                            raise HTTPException(status_code=400, detail="speaker wav too large (max 8MB)")
                        f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())

        if total < 1024:
            raise HTTPException(status_code=400, detail="Downloaded speaker wav is too small / invalid")

        # Validate that torchaudio can open it (catches corrupt partial downloads)
        torchaudio.load(str(tmp_path))

        if CACHE_SPEAKER_WAV:
            os.replace(tmp_path, cached_path)  # atomic
            return str(cached_path)
        else:
            return str(tmp_path)

    except HTTPException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

    except Exception as e:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Error downloading speaker wav: {str(e)}")


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
        "cache_speaker_wav": CACHE_SPEAKER_WAV,
        "ref_cache_dir": str(REF_CACHE_DIR),
    }


@app.post("/tts")
async def tts_stream(request: Request, payload: TtsRequest = Body(...)):
    global tts
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    req_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"text too long (max {MAX_TEXT_CHARS} chars)")

    language = payload.language
    fmt = payload.format
    sample_rate = int(payload.sample_rate or OUTPUT_SAMPLE_RATE)

    if fmt != "pcm":
        raise HTTPException(status_code=400, detail="Use format=pcm for streaming integration")

    if sample_rate != OUTPUT_SAMPLE_RATE:
        raise HTTPException(
            status_code=400,
            detail=f"PCM sample_rate must be {OUTPUT_SAMPLE_RATE} for this server (got {sample_rate})",
        )

    speaker_wav_url = (payload.speaker_wav_url or "").strip() or None
    speaker_wav_path = (payload.speaker_wav_path or "").strip() or None

    # Acquire semaphore ONCE for the whole streaming request
    _acquire_sem_or_503()

    # Resolve ref_path BEFORE creating generator (so errors return cleanly)
    tmp_ref_to_delete: Optional[str] = None
    if speaker_wav_url:
        ref_path = await _download_and_cache_ref(speaker_wav_url)
        # If caching disabled, we downloaded to a unique tmp that we must delete later
        if not CACHE_SPEAKER_WAV:
            tmp_ref_to_delete = ref_path
    else:
        ref_path = speaker_wav_path or DEFAULT_SPEAKER_WAV
        if not ref_path:
            _release_sem()
            raise HTTPException(status_code=400, detail="Provide speaker_wav_url or speaker_wav_path/DEFAULT_SPEAKER_WAV")
        if not os.path.exists(ref_path):
            _release_sem()
            raise HTTPException(status_code=400, detail=f"speaker_wav_path not found: {ref_path}")

    chunks = chunk_text(text, MAX_CHUNK_CHARS)
    if not chunks:
        _release_sem()
        raise HTTPException(status_code=400, detail="text produced no chunks")

    def gen() -> Generator[bytes, None, None]:
        """
        IMPORTANT:
        - Cleanup + semaphore release must happen here, not in endpoint finally.
        - Otherwise temp files get deleted while streaming is still running.
        """
        try:
            for ch in chunks:
                # Optional: skip chunks that are only emojis / punctuation (prevents weird edge cases)
                if not ch.strip():
                    continue

                audio = tts.tts(text=ch, speaker_wav=ref_path, language=language)
                yield _float_to_s16le_pcm(audio)

        finally:
            # Cleanup temp ref only when streaming is DONE
            if tmp_ref_to_delete and os.path.exists(tmp_ref_to_delete):
                try:
                    os.remove(tmp_ref_to_delete)
                except Exception:
                    pass
            _release_sem()

    headers = {
        "X-Request-Id": req_id,
        "X-Audio-Format": "pcm",
        "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
    }

    return StreamingResponse(gen(), media_type="application/octet-stream", headers=headers)
import io
import wave

def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)

def _pcm16_to_wav_bytes(pcm16: np.ndarray, sample_rate: int) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes(order="C"))
    return bio.getvalue()
from fastapi.responses import Response

@app.post("/tts/wav")
async def tts_wav(request: Request, payload: TtsRequest = Body(...)):
    """
    Non-streaming endpoint for lab testing:
    Returns one WAV file (audio/wav) that contains the full synthesized text.
    """
    global tts
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    req_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"text too long (max {MAX_TEXT_CHARS} chars)")

    language = payload.language
    sample_rate = int(payload.sample_rate or OUTPUT_SAMPLE_RATE)

    # For WAV endpoint, we allow returning WAV always.
    # If caller sends format=pcm, we still return WAV (lab needs WAV).
    # If you want strict behavior, you can enforce format == "wav".
    if sample_rate != OUTPUT_SAMPLE_RATE:
        raise HTTPException(
            status_code=400,
            detail=f"sample_rate must be {OUTPUT_SAMPLE_RATE} for this server (got {sample_rate})",
        )

    speaker_wav_url = (payload.speaker_wav_url or "").strip() or None
    speaker_wav_path = (payload.speaker_wav_path or "").strip() or None

    _acquire_sem_or_503()

    tmp_ref_to_delete: Optional[str] = None
    try:
        # Resolve reference audio
        if speaker_wav_url:
            ref_path = await _download_and_cache_ref(speaker_wav_url)
            if not CACHE_SPEAKER_WAV:
                tmp_ref_to_delete = ref_path
        else:
            ref_path = speaker_wav_path or DEFAULT_SPEAKER_WAV
            if not ref_path:
                raise HTTPException(status_code=400, detail="Provide speaker_wav_url or speaker_wav_path/DEFAULT_SPEAKER_WAV")
            if not os.path.exists(ref_path):
                raise HTTPException(status_code=400, detail=f"speaker_wav_path not found: {ref_path}")

        chunks = chunk_text(text, MAX_CHUNK_CHARS)
        if not chunks:
            raise HTTPException(status_code=400, detail="text produced no chunks")

        # Build final PCM int16 buffer
        parts: list[np.ndarray] = []

        # Optional: small silence between chunks to avoid hard joins
        gap_ms = int(os.getenv("WAV_JOIN_GAP_MS", "70"))  # tweak 40-120ms
        gap_samples = int((gap_ms / 1000.0) * OUTPUT_SAMPLE_RATE)
        gap = np.zeros((gap_samples,), dtype=np.int16) if gap_samples > 0 else None

        for i, ch in enumerate(chunks):
            if not ch.strip():
                continue
            audio_f32 = tts.tts(text=ch, speaker_wav=ref_path, language=language)
            pcm16 = _float_to_int16(audio_f32)
            parts.append(pcm16)
            if gap is not None and i < len(chunks) - 1:
                parts.append(gap)

        if not parts:
            raise HTTPException(status_code=400, detail="no audio produced")

        merged = np.concatenate(parts, axis=0)
        wav_bytes = _pcm16_to_wav_bytes(merged, OUTPUT_SAMPLE_RATE)

        headers = {
            "X-Request-Id": req_id,
            "X-Audio-Format": "wav",
            "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
            "Content-Disposition": f'attachment; filename="xtts_{req_id}.wav"',
        }

        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    finally:
        if tmp_ref_to_delete and os.path.exists(tmp_ref_to_delete):
            try:
                os.remove(tmp_ref_to_delete)
            except Exception:
                pass
        _release_sem()
