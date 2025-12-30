import io
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse

# Coqui TTS
from TTS.api import TTS

APP_NAME = "xtts-service"

# -------- Config (ENV) ----------
MODEL_NAME = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
USE_GPU = os.getenv("USE_GPU", "true").lower() in ("1", "true", "yes", "y")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "ar")  # "ar" or "en"
DEFAULT_SPEAKER_WAV = os.getenv("DEFAULT_SPEAKER_WAV", "")  # optional path inside container
OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))

# Optional: limit text length to protect server
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))

app = FastAPI(title=APP_NAME)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8100",
        "http://127.0.0.1:8100",
        "https://mda-ai-backend:8080",
        "https://api.mda.sa",
        "https://mda.sa",
        "https://www.mda.sa",
        # add your deployed frontend origin later, e.g.
        # "https://cave.mda.sa"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # helpful for FileResponse filename
)
tts: Optional[TTS] = None


@app.on_event("startup")
def load_model():
    global tts
    # Load model once on startup
    tts = TTS(MODEL_NAME, gpu=USE_GPU)


@app.get("/health")
def health():
    return {
        "service": APP_NAME,
        "model": MODEL_NAME,
        "gpu": USE_GPU,
        "default_lang": DEFAULT_LANG,
    }


def _write_upload_to_temp(upload: UploadFile) -> str:
    # Save uploaded WAV to a temp file for XTTS
    suffix = os.path.splitext(upload.filename or "ref.wav")[1] or ".wav"
    fd, path = tempfile.mkstemp(prefix="xtts_ref_", suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    language: str = Form(DEFAULT_LANG),  # "ar" or "en"
    speaker_wav: UploadFile = File(None),
):
    """
    Generate a WAV from text.
    - text: required
    - language: "ar" or "en"
    - speaker_wav: optional upload. If not provided, DEFAULT_SPEAKER_WAV must exist.
    Returns: audio/wav bytes
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

    # Resolve reference voice
    ref_path = None
    temp_ref = None

    try:
        if speaker_wav is not None:
            temp_ref = _write_upload_to_temp(speaker_wav)
            ref_path = temp_ref
        else:
            if not DEFAULT_SPEAKER_WAV:
                raise HTTPException(status_code=400, detail="speaker_wav not provided and DEFAULT_SPEAKER_WAV not set")
            if not os.path.exists(DEFAULT_SPEAKER_WAV):
                raise HTTPException(status_code=400, detail=f"DEFAULT_SPEAKER_WAV not found: {DEFAULT_SPEAKER_WAV}")
            ref_path = DEFAULT_SPEAKER_WAV

        # Generate to temp wav file, then return bytes
        out_fd, out_path = tempfile.mkstemp(prefix="xtts_out_", suffix=".wav")
        os.close(out_fd)

        # XTTS API writes to file
        tts.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=language,
            file_path=out_path,
        )

        with open(out_path, "rb") as f:
            wav_bytes = f.read()

        return Response(content=wav_bytes, media_type="audio/wav")

    finally:
        # Cleanup temp files
        if temp_ref and os.path.exists(temp_ref):
            try:
                os.remove(temp_ref)
            except Exception:
                pass
        # cleanup output
        try:
            if "out_path" in locals() and out_path and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass


@app.post("/tts_json")
async def tts_json(
    payload: dict,
):
    """
    JSON alternative:
    {
      "text": "...",
      "language": "ar",
      "speaker_wav_path": "/app/voices/ref.wav"
    }

    Note: This method only supports speaker WAV already on disk (no upload).
    """
    global tts
    if tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = (payload.get("text") or "").strip()
    language = (payload.get("language") or DEFAULT_LANG).strip().lower()
    speaker_wav_path = (payload.get("speaker_wav_path") or DEFAULT_SPEAKER_WAV).strip()

    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"text too long (max {MAX_TEXT_CHARS} chars)")
    if language not in ("ar", "en"):
        raise HTTPException(status_code=400, detail='language must be "ar" or "en"')
    if not speaker_wav_path or not os.path.exists(speaker_wav_path):
        raise HTTPException(status_code=400, detail="Valid speaker_wav_path is required")

    out_fd, out_path = tempfile.mkstemp(prefix="xtts_out_", suffix=".wav")
    os.close(out_fd)

    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=out_path,
        )
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        return Response(content=wav_bytes, media_type="audio/wav")
    finally:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
