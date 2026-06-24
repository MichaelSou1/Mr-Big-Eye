#!/usr/bin/env bash
# Wait for the 8B gate to finish (frees GPU 0,1), then serve the dense 32B-AWQ (TP=2) and run the
# n=400 gate as the "32b" cell. Backgroundable; logs to /home/gpus/logs/wu1.
set -uo pipefail
cd /home/gpus/Mr-Big-Eye-internalization
LOGD=/home/gpus/logs/wu1; mkdir -p "$LOGD"
AWQ=/home/gpus/models/Qwen3-VL-32B-Instruct-AWQ
DUMP=data/distill/chartqa/test_cases_400.jsonl
IMG=/home/gpus/mbe_data/chartqa_test_images
CR(){ conda run --no-capture-output -n mbe-up "$@"; }
say(){ echo "[$(date +%H:%M:%S)] $*"; }
n8(){ python3 -c "import json;print(sum(1 for l in open('data/distill/results/results.jsonl') if l.strip() and json.loads(l).get('model_id')=='8b' and json.loads(l).get('n')==400))"; }
n32(){ python3 -c "import json;print(sum(1 for l in open('data/distill/results/results.jsonl') if l.strip() and json.loads(l).get('model_id')=='32b' and json.loads(l).get('n')==400))"; }

say "waiting for 8B gate to finish (11 rows)..."
while [ "$(n8)" -lt 11 ]; do sleep 30; done
say "8B gate done. stopping 8B server."
if [ -f /home/gpus/logs/serve-vlm-8b.pid ]; then
  kill "$(cat /home/gpus/logs/serve-vlm-8b.pid)" 2>/dev/null
  pkill -P "$(cat /home/gpus/logs/serve-vlm-8b.pid)" 2>/dev/null || true
  rm -f /home/gpus/logs/serve-vlm-8b.pid
fi
sleep 10

if [ "$(n32)" -ge 11 ]; then say "32b already has 11 rows — skipping"; exit 0; fi

say "serving 32B-AWQ TP=2 on GPU0,1 :30000"
PORT=30000 CUDA_VISIBLE_DEVICES=0,1 MODEL_DIR="$AWQ" SERVED_MODEL_NAME=Qwen3-VL-32B-Instruct \
  LOG_FILE="$LOGD/serve-32b-awq.log" PID_FILE=/home/gpus/logs/serve-vlm-32b.pid \
  bash scripts/serve/serve_vlm_32b.sh
for i in $(seq 1 150); do
  curl -s --max-time 2 http://127.0.0.1:30000/v1/models 2>/dev/null | grep -q "Qwen3-VL-32B" && { say "32B-AWQ up after ~$((i*5))s"; break; }
  sleep 5
done
curl -s --max-time 30 http://127.0.0.1:30000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-VL-32B-Instruct","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":5,"temperature":0}' \
  >/dev/null 2>&1 && say "32B-AWQ generation OK" || { say "32B-AWQ NOT responding — abort"; exit 1; }

say "running 32B gate (n=400, 5 seeds, conc 8)"
LOCAL_VLM_BASE_URL=http://127.0.0.1:30000/v1 LOCAL_VLM_MODEL_NAME=Qwen3-VL-32B-Instruct \
  CR python -u scripts/run_chartqa_gate.py --model-id 32b --dump "$DUMP" --img-dir "$IMG" \
  --methods self_reflect orch_reflect_blind --seeds 5 --concurrency 8 > "$LOGD/gate_32b_n400.log" 2>&1
say "32B gate finished. stopping server."
[ -f /home/gpus/logs/serve-vlm-32b.pid ] && { kill "$(cat /home/gpus/logs/serve-vlm-32b.pid)" 2>/dev/null; pkill -P "$(cat /home/gpus/logs/serve-vlm-32b.pid)" 2>/dev/null || true; rm -f /home/gpus/logs/serve-vlm-32b.pid; }
say "32B GATE PIPELINE DONE"
touch "$LOGD/GATE32B_DONE"
