#!/usr/bin/env bash
# WU-1 remainder (v2, after DeepSeek 402): complete orch_reflect_blind with a LOCAL 8B orchestrator,
# run the 32B gate, then the in-process 1.3/1.4/1.5 pipeline. Resumable; logs under /home/gpus/logs/wu1.
# Assumes 8B server up on :30000 (GPU0,1) and 4B server up on :30002 (GPU2) at start.
set -uo pipefail
cd /home/gpus/Mr-Big-Eye-internalization
LOGD=/home/gpus/logs/wu1; mkdir -p "$LOGD"
DUMP=data/distill/chartqa/test_cases_400.jsonl
IMG=/home/gpus/mbe_data/chartqa_test_images
AWQ=/home/gpus/models/Qwen3-VL-32B-Instruct-AWQ
CR(){ conda run --no-capture-output -n mbe-up "$@"; }
say(){ echo "[$(date +%H:%M:%S)] $*"; }
ORCH_BASE=http://127.0.0.1:30000/v1; ORCH_MODEL=Qwen3-VL-8B-Instruct
n_orch(){ python3 -c "import json;print(sum(1 for l in open('$1') if l.strip() and json.loads(l).get('method')=='orch_reflect_blind' and json.loads(l).get('model_id')=='$2' and json.loads(l).get('n')==400))" 2>/dev/null || echo 0; }
n_meth(){ python3 -c "import json;print(sum(1 for l in open('$1') if l.strip() and json.loads(l).get('method')=='$3' and json.loads(l).get('model_id')=='$2' and json.loads(l).get('n')==400))" 2>/dev/null || echo 0; }

# ---------- Phase 1: 8B + 4B orch (local 8B orchestrator), concurrent ----------
P8=""; P4=""
if [ "$(n_orch data/distill/results/results.jsonl 8b)" -lt 5 ]; then
  say "8B orch (local orchestrator) ..."
  env LOCAL_VLM_BASE_URL=http://127.0.0.1:30000/v1 LOCAL_VLM_MODEL_NAME=Qwen3-VL-8B-Instruct \
      ORCHESTRATOR_API_BASE_URL=$ORCH_BASE ORCHESTRATOR_MODEL_NAME=$ORCH_MODEL ORCHESTRATOR_API_KEY=EMPTY \
      conda run --no-capture-output -n mbe-up python -u scripts/run_chartqa_gate.py --model-id 8b \
      --dump "$DUMP" --img-dir "$IMG" --methods orch_reflect_blind --seeds 5 --concurrency 8 --no-free-form \
      > "$LOGD/orch_8b.log" 2>&1 &
  P8=$!
else say "8B orch already complete"; fi
if [ "$(n_orch data/distill/results/results_4b_n400.jsonl 4b)" -lt 5 ]; then
  say "4B orch (local orchestrator) ..."
  env LOCAL_VLM_BASE_URL=http://127.0.0.1:30002/v1 LOCAL_VLM_MODEL_NAME=Qwen3-VL-4B-Instruct \
      ORCHESTRATOR_API_BASE_URL=$ORCH_BASE ORCHESTRATOR_MODEL_NAME=$ORCH_MODEL ORCHESTRATOR_API_KEY=EMPTY \
      MBE_RESULTS_PATH=data/distill/results/results_4b_n400.jsonl \
      conda run --no-capture-output -n mbe-up python -u scripts/run_chartqa_gate.py --model-id 4b \
      --dump "$DUMP" --img-dir "$IMG" --methods orch_reflect_blind --seeds 5 --concurrency 8 --no-free-form \
      > "$LOGD/orch_4b.log" 2>&1 &
  P4=$!
else say "4B orch already complete"; fi
[ -n "$P8" ] && wait "$P8" && say "8B orch done"
[ -n "$P4" ] && wait "$P4" && say "4B orch done"

# ---------- Phase 2: 32B gate (free + self_reflect + orch w/ local orchestrator) ----------
if [ "$(n_meth data/distill/results/results.jsonl 32b self_reflect)" -lt 5 ]; then
  say "stopping 4B server (free GPU2 for 32B TP=2)"
  [ -f /home/gpus/logs/serve-vlm-4b.pid ] && { kill "$(cat /home/gpus/logs/serve-vlm-4b.pid)" 2>/dev/null; rm -f /home/gpus/logs/serve-vlm-4b.pid; }
  sleep 8
  say "serving 32B-AWQ TP=2 on GPU2,3 :30001"
  PORT=30001 CUDA_VISIBLE_DEVICES=2,3 MODEL_DIR="$AWQ" SERVED_MODEL_NAME=Qwen3-VL-32B-Instruct \
    LOG_FILE="$LOGD/serve-32b-awq.log" PID_FILE=/home/gpus/logs/serve-vlm-32b.pid bash scripts/serve/serve_vlm_32b.sh
  for i in $(seq 1 150); do curl -s --max-time 2 http://127.0.0.1:30001/v1/models 2>/dev/null | grep -q Qwen3-VL-32B && { say "32B-AWQ up"; break; }; sleep 5; done
  curl -s --max-time 30 http://127.0.0.1:30001/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-VL-32B-Instruct","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":5,"temperature":0}' >/dev/null 2>&1 \
    && say "32B-AWQ gen OK" || { say "32B-AWQ NOT responding — abort"; exit 1; }
  say "running 32B gate (VLM=32B:30001, orch=8B:30000)"
  env LOCAL_VLM_BASE_URL=http://127.0.0.1:30001/v1 LOCAL_VLM_MODEL_NAME=Qwen3-VL-32B-Instruct \
      ORCHESTRATOR_API_BASE_URL=$ORCH_BASE ORCHESTRATOR_MODEL_NAME=$ORCH_MODEL ORCHESTRATOR_API_KEY=EMPTY \
      conda run --no-capture-output -n mbe-up python -u scripts/run_chartqa_gate.py --model-id 32b \
      --dump "$DUMP" --img-dir "$IMG" --methods self_reflect orch_reflect_blind --seeds 5 --concurrency 8 \
      > "$LOGD/gate_32b_n400.log" 2>&1
  say "32B gate done"
else say "32B gate already complete"; fi

# ---------- Phase 3: stop all servers, merge 4B, run in-process ----------
say "stopping all vLLM servers for in-process phase"
for pf in /home/gpus/logs/serve-vlm-8b.pid /home/gpus/logs/serve-vlm-4b.pid /home/gpus/logs/serve-vlm-32b.pid; do
  [ -f "$pf" ] && { kill "$(cat "$pf")" 2>/dev/null; rm -f "$pf"; }
done
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 12

if [ ! -f "$LOGD/MERGED_4B" ]; then
  say "merging 4B side-file -> main store"
  cat data/distill/results/results_4b_n400.jsonl >> data/distill/results/results.jsonl
  touch "$LOGD/MERGED_4B"; say "store now $(wc -l < data/distill/results/results.jsonl) rows"
fi

say "launching in-process pipeline (1.3/1.4/1.5)"
bash scripts/run_wu1_inproc.sh >> "$LOGD/inproc_pipeline.log" 2>&1
say "in-process pipeline returned"
ls -la "$LOGD/INPROC_DONE" 2>/dev/null || say "INPROC_DONE missing — check $LOGD/inproc_pipeline.log"
touch "$LOGD/WU1_REST_V2_DONE"; say "run_wu1_rest_v2 COMPLETE"
