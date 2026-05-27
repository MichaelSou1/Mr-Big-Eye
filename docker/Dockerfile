# Mr-Big-Eye ingest image
# Base: official PyTorch 2.7 + CUDA 12.6 (driver >=525 still works; needs torch>=2.6
# because transformers 5.x enforces CVE-2025-32434 and refuses torch.load with older torch).
# We intentionally do NOT bake models or data into the image — they bind-mount at runtime,
# so the image stays small (~5GB) and you can swap weights without rebuilding.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    MBE_ROOT=/app

# System deps:
# - ffmpeg: scene detection + audio extraction
# - libsndfile1: soundfile (ASR audio loading)
# - libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender1: opencv runtime (scenedetect, rapidocr)
# - git: pip occasionally pulls VCS refs
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cacheable layer). torch is already in the base image,
# so the constraint torch>=2.1.0 from requirements.txt is satisfied without re-download.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy code last so source edits don't bust the deps cache.
# data/ models/ .env are bind-mounted at run time (see docker/run-ingest.sh).
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY eval/ /app/eval/

# Default entrypoint: ingest the audiovisual manifest.
# Override via `docker run ... bash` for interactive debugging.
ENTRYPOINT ["python", "-m", "scripts.ingest_videomme"]
