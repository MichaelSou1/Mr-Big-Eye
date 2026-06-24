#!/usr/bin/env bash
# Re-run orch_reflect_blind for all 3 scales with the ORIGINAL DeepSeek orchestrator (deepseek-v4-flash
# from .env), replacing the temporary local-8B-orchestrator rows. Run AFTER the core in-process phase
# (1.3/1.4/1.5) finishes and all GPUs are free. Requires DeepSeek balance (will abort if 402).
# Logs under /home/gpus/logs/wu1.
set -uo pipefail
cd /home/gpus/Mr-Big-Eye-internalization
LOGD=/home/gpus/logs/wu1; mkdir -p "$LOGD"
DUMP=data/distill/chartqa/test_cases_400.jsonl
IMG=/home/gpus/mbe_data/chartqa_test_images
AWQ=/home/gpus/models/Qwen3-VL-32B-Instruct-AWQ
M8B=/home/gpus/models/Qwen3-VL-8B-Instruct
say(){ echo "[$(date +%H:%M:%S)] $*"; }
DS_KEY=$(grep '^ORCHESTRATOR_API_KEY=' .env | cut -d= -f2-)

# --- preflight: DeepSeek reachable + has balance? ---
say "DeepSeek preflight ..."
code=$(curl -s -o /tmp/ds_pf.json -w "%{http_code}" --max-time 40 -x http://127.0.0.1:7890 \
  https://api.deepseek.com/v1/chat/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${DS_KEY}" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":50}')
say "DeepSeek http=$code"
if [ "$code" = "402" ]; then say "ABORT: DeepSeek still 402 (top up first)"; exit 2; fi
if [ "$code" != "200" ]; then say "ABORT: DeepSeek unexpected http=$code"; head -c 300 /tmp/ds_pf.json; exit 3; fi

# --- remove the local-8B-orch n=400 rows (4b/8b/32b) so DeepSeek rows don't collide ---
say "removing local-8B orch rows from store + 4B side-file"
python3 - <<'PY'
import json
for p in ("data/distill/results/results.jsonl",):
    rows=[l for l in open(p) if l.strip()]
    keep=[l for l in rows if not (json.loads(l).get("method")=="orch_reflect_blind"
          and json.loads(l).get("n")==400 and json.loads(l).get("model_id") in ("4b","8b","32b"))]
    open(p,"w").write("".join(keep)); print(p, len(rows),"->",len(keep))
PY
cp data/distill/results/results.jsonl "$LOGD/results.pre_dsorch.jsonl"

# --- Phase 1: serve 8B (GPU0,1) + 4B (GPU2); run 8B+4B orch with DeepSeek, concurrent ---
say "serving 8B (GPU0,1 :30000) + 4B (GPU2 :30002)"
PORT=30000 CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve/serve_vlm_8b.sh
PORT=30002 CUDA_VISIBLE_DEVICES=2 LOG_FILE=/home/gpus/logs/serve-vlm-4b.log PID_FILE=/home/gpus/logs/serve-vlm-4b.pid bash scripts/serve/serve_vlm_4b.sh
for i in $(seq 1 120); do
  curl -s --max-time 2 http://127.0.0.1:30000/v1/models 2>/dev/null | grep -q Qwen3-VL-8B \
   && curl -s --max-time 2 http://127.0.0.1:30002/v1/models 2>/dev/null | grep -q Qwen3-VL-4B && { say "8B+4B up"; break; }
  sleep 5
done

say "8B orch (DeepSeek) ..."
env LOCAL_VLM_BASE_URL=http://127.0.0.1:30000/v1 LOCAL_VLM_MODEL_NAME=Qwen3-VL-8B-Instruct \
  conda run --no-capture-output -n mbe-up python -u scripts/run_chartqa_gate.py --model-id 8b \
  --dump "$DUMP" --img-dir "$IMG" --methods orch_reflect_blind --seeds 5 --concurrency 8 --no-free-form \
  > "$LOGD/dsorch_8b.log" 2>&1 &
P8=$!
say "4B orch (DeepSeek) ..."
env LOCAL_VLM_BASE_URL=http://127.0.0.1:30002/v1 LOCAL_VLM_MODEL_NAME=Qwen3-VL-4B-Instruct \
  MBE_RESULTS_PATH=data/distill/results/results_4b_dsorch.jsonl \
  conda run --no-capture-output -n mbe-up python -u scripts/run_chartqa_gate.py --model-id 4b \
  --dump "$DUMP" --img-dir "$IMG" --methods orch_reflect_blind --seeds 5 --concurrency 8 --no-free-form \
  > "$LOGD/dsorch_4b.log" 2>&1 &
P4=$!
wait "$P8"; say "8B orch done"
wait "$P4"; say "4B orch done"
# merge 4B side-file
cat data/distill/results/results_4b_dsorch.jsonl >> data/distill/results/results.jsonl
say "merged 4B DeepSeek-orch rows"

# --- Phase 2: stop 4B, serve 32B-AWQ (GPU2,3); run 32B orch with DeepSeek ---
say "stopping 4B; serving 32B-AWQ (GPU2,3 :30001)"
[ -f /home/gpus/logs/serve-vlm-4b.pid ] && { kill "$(cat /home/gpus/logs/serve-vlm-4b.pid)" 2>/dev/null; rm -f /home/gpus/logs/serve-vlm-4b.pid; }
sleep 8
PORT=30001 CUDA_VISIBLE_DEVICES=2,3 MODEL_DIR="$AWQ" SERVED_MODEL_NAME=Qwen3-VL-32B-Instruct \
  LOG_FILE="$LOGD/serve-32b-awq.log" PID_FILE=/home/gpus/logs/serve-vlm-32b.pid bash scripts/serve/serve_vlm_32b.sh
for i in $(seq 1 150); do curl -s --max-time 2 http://127.0.0.1:30001/v1/models 2>/dev/null | grep -q Qwen3-VL-32B && { say "32B up"; break; }; sleep 5; done
say "32B orch (DeepSeek) ..."
env LOCAL_VLM_BASE_URL=http://127.0.0.1:30001/v1 LOCAL_VLM_MODEL_NAME=Qwen3-VL-32B-Instruct \
  conda run --no-capture-output -n mbe-up python -u scripts/run_chartqa_gate.py --model-id 32b \
  --dump "$DUMP" --img-dir "$IMG" --methods orch_reflect_blind --seeds 5 --concurrency 8 --no-free-form \
  > "$LOGD/dsorch_32b.log" 2>&1
say "32B orch done"

# --- stop servers, refresh report ---
for pf in /home/gpus/logs/serve-vlm-8b.pid /home/gpus/logs/serve-vlm-32b.pid; do
  [ -f "$pf" ] && { kill "$(cat "$pf")" 2>/dev/null; rm -f "$pf"; }
done
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
say "DeepSeek orch re-run COMPLETE"
python3 scripts/wu1_report.py
touch "$LOGD/DSORCH_DONE"
