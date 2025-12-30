FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy code
COPY requirements.txt /app/requirements.txt
COPY server.py /app/server.py

# Create venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install CUDA-enabled PyTorch (CUDA 12.1 wheels)
# Your driver is CUDA 12.8 which is backward compatible for runtime.
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps (Coqui TTS + FastAPI)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Optional: place a default reference voice inside container
# COPY voices/ref.wav /app/voices/ref.wav
# ENV DEFAULT_SPEAKER_WAV=/app/voices/ref.wav

EXPOSE 8000

# Performance defaults (you can override at runtime)
ENV USE_GPU=true
ENV XTTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
ENV DEFAULT_LANG=ar
ENV MAX_TEXT_CHARS=2000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
