#!/usr/bin/env bash
# Serve Qwen3-VL-32B-Instruct-AWQ (the strong-perception base + sighted critic) TP=2.
# Defaults to GPU 2,3 port 30001. gpu-mem-util 0.85 (0.90 -> vllm masked_scatter crash, see progress.md).
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm-qwen}"
MODEL_DIR="${MODEL_DIR:-/home/gpus/models/Qwen3-VL-32B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-32B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
GPUS="${CUDA_VISIBLE_DEVICES:-2,3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_IMAGES="${MAX_IMAGES:-18}"
LOG_FILE="${LOG_FILE:-/home/gpus/logs/serve-vlm-32b.log}"
PID_FILE="${PID_FILE:-/home/gpus/logs/serve-vlm-32b.pid}"

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
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
  --trust-remote-code \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "VLM-32B launching pid $(cat "$PID_FILE") on GPU $GPUS :$PORT -> $LOG_FILE"
