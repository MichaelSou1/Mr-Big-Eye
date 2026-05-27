#!/usr/bin/env bash
# Launch ingest inside the container.
# Expects the following on the host (paths can be overridden via env vars below):
#   ${MBE_HOST_ROOT}/data/uploads/{video_id}.mp4   — 60 pre-staged videos
#   ${MBE_HOST_ROOT}/eval/audiovisual/video_manifest.json
#   ${MBE_HOST_ROOT}/models/bge-m3/               — local BGE-M3 weights
#   ${MBE_HOST_ROOT}/models/siglip2-so400m-patch14-384/
#   ${MBE_HOST_ROOT}/.env                          — DASHSCOPE / HF_TOKEN / etc.
#
# Cache outputs land in:
#   ${MBE_HOST_ROOT}/data/cache/                   — pipeline-derived assets (rsync back)
#
# Single-GPU example:  bash docker/run-ingest.sh
# Pick a GPU:          GPU=1 bash docker/run-ingest.sh
# 4-way parallel:      see comments at bottom.
set -euo pipefail

IMAGE="${IMAGE:-mbe-ingest:latest}"
MBE_HOST_ROOT="${MBE_HOST_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-mbe-ingest-gpu${GPU}}"

# ModelScope SenseVoice + FSMN-VAD download to ~/.cache/modelscope on first run.
# Persist across container restarts via a named volume so we don't re-download (~900MB).
MODELSCOPE_CACHE_VOL="${MODELSCOPE_CACHE_VOL:-mbe-modelscope-cache}"
HF_CACHE_VOL="${HF_CACHE_VOL:-mbe-hf-cache}"

echo "Image:       ${IMAGE}"
echo "Host root:   ${MBE_HOST_ROOT}"
echo "GPU:         ${GPU}"
echo "Container:   ${CONTAINER_NAME}"
echo

# Sanity checks
for required in \
    "${MBE_HOST_ROOT}/.env" \
    "${MBE_HOST_ROOT}/eval/audiovisual/video_manifest.json" \
    "${MBE_HOST_ROOT}/data/uploads" \
    "${MBE_HOST_ROOT}/models/bge-m3"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: missing ${required}" >&2
        exit 1
    fi
done

docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --gpus "device=${GPU}" \
    --shm-size=8g \
    --env-file "${MBE_HOST_ROOT}/.env" \
    -e MBE_ROOT=/app \
    -e APP_CUDA_VISIBLE_DEVICES=0 \
    -e MODELS_DEVICE=cuda:0 \
    -v "${MBE_HOST_ROOT}/data:/app/data" \
    -v "${MBE_HOST_ROOT}/models:/app/models:ro" \
    -v "${MODELSCOPE_CACHE_VOL}:/root/.cache/modelscope" \
    -v "${HF_CACHE_VOL}:/root/.cache/huggingface" \
    "${IMAGE}"

# --- 4×GPU parallel pattern -------------------------------------------------
# The current ingest script processes the manifest sequentially. To use all four
# RTX 3080s at once, split the manifest by GPU index (e.g. video_id hash mod 4)
# and launch 4 containers in parallel — each binds a distinct GPU:
#
#   for g in 0 1 2 3; do
#       MBE_INGEST_SHARD="${g}/4" GPU=${g} \
#           CONTAINER_NAME=mbe-ingest-gpu${g} \
#           bash docker/run-ingest.sh &
#   done
#   wait
#
# The shard env var is honored by scripts/ingest_videomme.py (see MBE_INGEST_SHARD
# handling there). Sequential single-GPU is fine for a one-off 60-video batch
# (~4h); shard only if you want ~1h wall time.
