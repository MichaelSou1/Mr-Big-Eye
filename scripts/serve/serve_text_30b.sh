#!/usr/bin/env bash
# Serve Qwen3-30B-A3B-Instruct-2507 (AWQ-4bit) for orchestrator + rewriter,
# on GPUs 1,2,3 (TP=3), port 30001. OpenAI-compatible w/ qwen25 tool-calling.
# Matches ORCHESTRATOR_*/REWRITER_* in .env. MoE → moe_wna16 quant for AWQ.
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm-qwen}"
MODEL_DIR="${MODEL_DIR:-/home/gpus/DLT/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-30B-A3B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
GPUS="${CUDA_VISIBLE_DEVICES:-1,2,3}"
TP_SIZE="${TP_SIZE:-3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
LOG_FILE="${LOG_FILE:-/home/gpus/logs/serve-text-30b.log}"
PID_FILE="${PID_FILE:-/home/gpus/logs/serve-text-30b.pid}"

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
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --trust-remote-code \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "TEXT-30B launching pid $(cat "$PID_FILE") on GPU $GPUS TP$TP_SIZE :$PORT -> $LOG_FILE"
