#!/usr/bin/env bash
# Serve Qwen3-VL-8B-Instruct (the "8b" cell) TP=2 on GPU 0,1 at the default LOCAL_VLM port 30000.
# Run the gate/probe with LOCAL_VLM_MODEL_NAME=Qwen3-VL-8B-Instruct LOCAL_VLM_BASE_URL=...:30000/v1.
set -euo pipefail
ENV_NAME="${ENV_NAME:-vllm-qwen}"
MODEL_DIR="${MODEL_DIR:-/home/gpus/models/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-8B-Instruct}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-30000}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_IMAGES="${MAX_IMAGES:-18}"
TP="${TP:-2}"
LOG_FILE="${LOG_FILE:-/home/gpus/logs/serve-vlm-8b.log}"
PID_FILE="${PID_FILE:-/home/gpus/logs/serve-vlm-8b.pid}"

[[ -d "$MODEL_DIR" ]] || { echo "model dir missing: $MODEL_DIR" >&2; exit 1; }
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running pid $(cat "$PID_FILE")"; exit 0
fi
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
echo "VLM-8B launching pid $(cat "$PID_FILE") on GPU $GPUS :$PORT -> $LOG_FILE"
