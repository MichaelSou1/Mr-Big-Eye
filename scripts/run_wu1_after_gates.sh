#!/usr/bin/env bash
# Chain the rest of WU-1 after the gates: wait for the 32B gate (GATE32B_DONE marker) and the 4B
# gate (11 rows), merge the 4B side-results into the canonical store, stop all vLLM servers, then
# run the in-process 1.3/1.4/1.5 pipeline. Fully autonomous; logs under /home/gpus/logs/wu1.
set -uo pipefail
cd /home/gpus/Mr-Big-Eye-internalization
LOGD=/home/gpus/logs/wu1; mkdir -p "$LOGD"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
n4(){ wc -l < data/distill/results/results_4b_n400.jsonl 2>/dev/null; }

say "waiting for GATE32B_DONE marker..."
while [ ! -f "$LOGD/GATE32B_DONE" ]; do sleep 30; done
say "32B gate done."

say "waiting for 4B gate (11 rows)..."
while [ "$(n4)" -lt 11 ]; do sleep 30; done
say "4B gate done ($(n4) rows)."

# stop the 4B server (frees GPU 2)
if [ -f /home/gpus/logs/serve-vlm-4b.pid ]; then
  kill "$(cat /home/gpus/logs/serve-vlm-4b.pid)" 2>/dev/null
  pkill -P "$(cat /home/gpus/logs/serve-vlm-4b.pid)" 2>/dev/null || true
  rm -f /home/gpus/logs/serve-vlm-4b.pid
fi
# belt-and-suspenders: kill any lingering vLLM servers so the in-process jobs get all GPUs
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 12

# merge 4B side-file into the canonical store (idempotent via marker)
if [ ! -f "$LOGD/MERGED_4B" ]; then
  say "merging results_4b_n400.jsonl -> results.jsonl"
  cat data/distill/results/results_4b_n400.jsonl >> data/distill/results/results.jsonl
  touch "$LOGD/MERGED_4B"
  say "merged. store now $(wc -l < data/distill/results/results.jsonl) rows"
else
  say "4B already merged (marker present)"
fi

say "GPU state before in-process jobs:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

say "launching in-process pipeline (1.3/1.4/1.5)"
bash scripts/run_wu1_inproc.sh >> "$LOGD/inproc_pipeline.log" 2>&1
say "in-process pipeline returned. marker:"
ls -la "$LOGD/INPROC_DONE" 2>/dev/null || say "INPROC_DONE not present — check inproc_pipeline.log"
touch "$LOGD/WU1_AFTER_GATES_DONE"
say "run_wu1_after_gates.sh COMPLETE"
