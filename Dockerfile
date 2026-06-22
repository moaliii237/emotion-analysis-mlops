# EmotionScope Backend - Production Image
FROM python:3.11-slim

LABEL maintainer="group-1-nlp"
LABEL description="Emotion MLOps - Backend API with monitoring, YouTube, and frontend"
LABEL version="0.4.0"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    prometheus-fastapi-instrumentator \
    yt-dlp \
    imageio-ffmpeg \
    python-multipart

RUN pip install --no-cache-dir \
    "torch==2.4.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-deps

RUN pip install --no-cache-dir --force-reinstall \
    "transformers>=4.40,<4.45" \
    "tokenizers<0.20"

RUN pip install --no-cache-dir --force-reinstall "numpy<2"

COPY pyproject.toml .
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir spacy && \
    python -m spacy download en_core_web_sm

RUN python -c "import whisper; whisper.load_model('base')"

RUN TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python -c "from transformers import MarianMTModel, MarianTokenizer; MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-nl'); MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-nl')"

RUN mkdir -p /app/logs

EXPOSE 8000

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

CMD ["uvicorn", "emotion_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
