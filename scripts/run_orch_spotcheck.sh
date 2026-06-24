#!/usr/bin/env bash
# Cheap validation: re-run 8B orch_reflect_blind with the ORIGINAL DeepSeek orchestrator (deepseek-v4-flash
# from .env), n=400, 2 seeds, into a SEPARATE results file. Compare net to the stored local-8B-orchestrator
# 8B net (~-0.092). Consistent -> keep all 8B-orchestrator orch rows. Different -> escalate to full re-run.
# Run AFTER the core in-process phase finishes (all GPUs free). Aborts cleanly if DeepSeek 402.
set -uo pipefail
cd /home/gpus/Mr-Big-Eye-internalization
LOGD=/home/gpus/logs/wu1; mkdir -p "$LOGD"
DUMP=data/distill/chartqa/test_cases_400.jsonl
IMG=/home/gpus/mbe_data/chartqa_test_images
SPOT=/home/gpus/logs/wu1/spot_8b_dsorch.jsonl
say(){ echo "[$(date +%H:%M:%S)] $*"; }
DS_KEY=$(grep '^ORCHESTRATOR_API_KEY=' .env | cut -d= -f2-)

say "DeepSeek preflight ..."
code=$(curl -s -o /tmp/ds_pf.json -w "%{http_code}" --max-time 40 -x http://127.0.0.1:7890 \
  https://api.deepseek.com/v1/chat/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${DS_KEY}" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":50}')
say "DeepSeek http=$code"
[ "$code" = "402" ] && { say "ABORT: DeepSeek still 402 (top up first)"; exit 2; }
[ "$code" != "200" ] && { say "ABORT: DeepSeek http=$code"; head -c 300 /tmp/ds_pf.json; exit 3; }

say "serving 8B (GPU0,1 :30000)"
PORT=30000 CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve/serve_vlm_8b.sh
for i in $(seq 1 120); do curl -s --max-time 2 http://127.0.0.1:30000/v1/models 2>/dev/null | grep -q Qwen3-VL-8B && { say "8B up"; break; }; sleep 5; done

rm -f "$SPOT"
say "8B orch with DeepSeek (n=400, 2 seeds) -> $SPOT"
env LOCAL_VLM_BASE_URL=http://127.0.0.1:30000/v1 LOCAL_VLM_MODEL_NAME=Qwen3-VL-8B-Instruct \
  MBE_RESULTS_PATH="$SPOT" \
  conda run --no-capture-output -n mbe-up python -u scripts/run_chartqa_gate.py --model-id 8b \
  --dump "$DUMP" --img-dir "$IMG" --methods orch_reflect_blind --seeds 2 --concurrency 8 --no-free-form \
  > "$LOGD/spot_8b_dsorch.log" 2>&1

# stop 8B server
[ -f /home/gpus/logs/serve-vlm-8b.pid ] && { kill "$(cat /home/gpus/logs/serve-vlm-8b.pid)" 2>/dev/null; rm -f /home/gpus/logs/serve-vlm-8b.pid; }
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

say "=== COMPARISON: 8B orch net  (DeepSeek spot vs stored local-8B) ==="
python3 - "$SPOT" <<'PY'
import json, sys, statistics
spot=[json.loads(l) for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("method")=="orch_reflect_blind"]
ds_nets=[r["bootstrap"]["net"] for r in spot]
ds_mean=statistics.mean(ds_nets) if ds_nets else None
# stored local-8B orch (5 seeds) from main store
store=[json.loads(l) for l in open("data/distill/results/results.jsonl")
       if json.loads(l).get("method")=="orch_reflect_blind" and json.loads(l).get("model_id")=="8b"
       and json.loads(l).get("n")==400]
loc_nets=[r["bootstrap"]["net"] for r in store]
loc_mean=statistics.mean(loc_nets) if loc_nets else None
print(f"  DeepSeek-orch (k={len(ds_nets)}): per-seed {['%+.3f'%n for n in ds_nets]}  mean={ds_mean}")
print(f"  local-8B-orch (k={len(loc_nets)}): mean={loc_mean}")
if ds_mean is not None and loc_mean is not None:
    diff=abs(ds_mean-loc_mean)
    same_sign = (ds_mean<0)==(loc_mean<0)
    verdict = "CONSISTENT (keep local-8B orch)" if (same_sign and diff<=0.03) else "DIFFERENT -> consider full DeepSeek re-run"
    print(f"  |Δ|={diff:.3f}  same_sign={same_sign}  => {verdict}")
PY
touch "$LOGD/SPOT_DONE"
say "spot-check done"
