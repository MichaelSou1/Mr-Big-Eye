#!/usr/bin/env bash
# §6 cross-family: serve InternVL3-8B (different vision tower than Qwen3-VL) TP=2 on GPU2,3 :30003.
set -euo pipefail
ENV_NAME="${ENV_NAME:-vllm-qwen}"
MODEL_DIR="${MODEL_DIR:-/home/gpus/models/InternVL3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-InternVL3-8B}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-30003}"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_IMAGES="${MAX_IMAGES:-18}"
TP="${TP:-2}"
LOG_FILE="${LOG_FILE:-/home/gpus/logs/serve-vlm-internvl.log}"
PID_FILE="${PID_FILE:-/home/gpus/logs/serve-vlm-internvl.pid}"

[[ -d "$MODEL_DIR" ]] || { echo "model dir missing: $MODEL_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$LOG_FILE")"
source /home/gpus/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
  --trust-remote-code \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "InternVL3-8B launching pid $(cat "$PID_FILE") on GPU $GPUS :$PORT -> $LOG_FILE"
