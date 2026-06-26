#!/usr/bin/env bash
# Resume only the remaining TabMWP Full-SFT causal batteries.
# This intentionally does not train or re-evaluate the Full-SFT checkpoint.
set -euo pipefail

cd /home/gpus/Mr-Big-Eye-internalization

PY=${PY:-/home/gpus/anaconda3/envs/mbe-up/bin/python}
MODEL=${MODEL:-data/distill/poc/full_8b_tabmwp}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
LOG=${LOG:-/home/gpus/logs/full_sft_8b_nonvideo}
CACHE=${CACHE:-data/distill/poc/paraphrase_cache_full8b_tabmwp_mimo.jsonl}
BASE_CACHE=${BASE_CACHE:-data/distill/poc/paraphrase_cache_full8b_tabmwp_mimo_base_cot.jsonl}
PRESENT_OUT=${PRESENT_OUT:-data/distill/poc/battery_full8b_tabmwp_present.json}
MASKED_OUT=${MASKED_OUT:-data/distill/poc/battery_full8b_tabmwp_masked.json}
REQUIRE_GPU_IDLE=${REQUIRE_GPU_IDLE:-1}
MAX_GPU_USED_MB=${MAX_GPU_USED_MB:-2048}
MIN_GPU_FREE_MB=${MIN_GPU_FREE_MB:-16000}
MIN_DISK_FREE_GB=${MIN_DISK_FREE_GB:-40}
REQUIRE_ORCHESTRATOR=${REQUIRE_ORCHESTRATOR:-1}
ORCHESTRATOR_HOST_HINT=${ORCHESTRATOR_HOST_HINT:-xiaomimimo.com}
EXPECTED_INTERVENTIONS=${EXPECTED_INTERVENTIONS:-corrupt,delete,filler,paraphrase,shuffle,truncate}

export CUDA_VISIBLE_DEVICES="$GPUS"
export NO_PROXY='*' no_proxy='*'
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$LOG" data/distill/poc/logs

LOCK=${LOCK:-/tmp/mbe_full_sft_8b_tabmwp_battery.lock}
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another TabMWP Full-SFT battery run is active (lock: $LOCK)."
  exit 3
fi

say() {
  echo "[$(date '+%F %T')] $*"
}

battery_output_ready() {
  local out=$1
  [[ -s "$out" ]] || return 1
  EXPECTED_INTERVENTIONS="$EXPECTED_INTERVENTIONS" "$PY" - "$out" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = {x for x in os.environ["EXPECTED_INTERVENTIONS"].split(",") if x}
try:
    obj = json.loads(path.read_text())
except Exception as exc:
    print(f"{path}: invalid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)
summary = obj.get("summary") or {}
got = set((summary.get("interventions") or {}).keys())
missing = sorted(expected - got)
details = obj.get("details") or []
answer_keys = set()
for row in details:
    answers = row.get("answers") if isinstance(row, dict) else None
    if isinstance(answers, dict):
        answer_keys.update(answers.keys())
missing_answers = sorted(expected - answer_keys)
has_answers = bool(details and answer_keys)
if missing or not has_answers or missing_answers:
    reasons = []
    if missing:
        reasons.append("missing interventions=" + ",".join(missing))
    if not has_answers:
        reasons.append("details[].answers missing")
    if missing_answers:
        reasons.append("missing answer variants=" + ",".join(missing_answers))
    print(f"{path}: incomplete ({'; '.join(reasons)})", file=sys.stderr)
    raise SystemExit(1)
print(f"{path}: ready")
PY
}

backup_incomplete_output() {
  local out=$1
  [[ -e "$out" ]] || return 0
  local stamp backup
  stamp=$(date '+%Y%m%d_%H%M%S')
  backup="${out}.incomplete.${stamp}"
  say "Existing output is incomplete; moving aside: $out -> $backup"
  mv "$out" "$backup"
}

preflight() {
  say "Preflight: gpus=$GPUS require_gpu_idle=$REQUIRE_GPU_IDLE max_gpu_used_mb=$MAX_GPU_USED_MB min_gpu_free_mb=$MIN_GPU_FREE_MB min_disk_free_gb=$MIN_DISK_FREE_GB require_orchestrator=$REQUIRE_ORCHESTRATOR"

  local active
  active=$(
    ps -eo pid,ppid,stat,etime,cmd \
      | awk '/scripts\/(battery_n400\.py|run_full_sft_8b_nonvideo\.sh|eval_full_sft_n400\.py|poc_sft_full_8b\.py)/ {print}' \
      || true
  )
  if [[ -n "$active" ]]; then
    echo "Refusing to resume while related experiment processes are active:"
    echo "$active"
    exit 5
  fi

  local free_gb
  free_gb=$(df -BG --output=avail /home/gpus | tail -1 | tr -dc '0-9')
  if [[ -z "$free_gb" || "$free_gb" -lt "$MIN_DISK_FREE_GB" ]]; then
    echo "Refusing to resume: /home/gpus has ${free_gb:-unknown}GB free, need >= ${MIN_DISK_FREE_GB}GB."
    exit 6
  fi

  if [[ ! -d "$MODEL" ]]; then
    echo "Missing Full-SFT checkpoint directory: $MODEL"
    exit 4
  fi
  if ! battery_output_ready "$PRESENT_OUT" >/dev/null 2>&1 || ! battery_output_ready "$MASKED_OUT" >/dev/null 2>&1; then
    if ! find "$MODEL" -maxdepth 1 -type f \
      \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) \
      -size +1M -print -quit | grep -q .; then
      echo "Missing large Full-SFT weight shard under $MODEL; cannot generate remaining battery outputs."
      exit 7
    fi
  fi

  if [[ "$REQUIRE_ORCHESTRATOR" == "1" ]]; then
    ORCHESTRATOR_HOST_HINT="$ORCHESTRATOR_HOST_HINT" "$PY" - <<'PY'
import os
from urllib.parse import urlparse

from app.config import settings

base = settings.orchestrator_api_base_url
key = settings.orchestrator_api_key
model = settings.orchestrator_model_name
hint = os.environ.get("ORCHESTRATOR_HOST_HINT", "").strip()
host = urlparse(base).netloc if base else ""
missing = []
if not base:
    missing.append("ORCHESTRATOR_API_BASE_URL")
if not key:
    missing.append("ORCHESTRATOR_API_KEY")
if not model:
    missing.append("ORCHESTRATOR_MODEL_NAME")
if missing:
    raise SystemExit("Missing orchestrator config: " + ", ".join(missing))
if hint and hint not in base:
    raise SystemExit(f"ORCHESTRATOR_API_BASE_URL host mismatch: expected substring {hint!r}, got host {host!r}")
print(f"orchestrator preflight ok: host={host} model={model} key_set=yes")
PY
  fi

  if [[ "$REQUIRE_GPU_IDLE" == "1" ]]; then
    local selected=",${GPUS},"
    local seen=0 bad=0 line idx used free util
    while IFS= read -r line; do
      IFS=',' read -r idx used free util <<<"$line"
      idx=$(echo "$idx" | tr -d ' ')
      used=$(echo "$used" | tr -d ' ')
      free=$(echo "$free" | tr -d ' ')
      util=$(echo "$util" | tr -d ' ')
      if [[ "$selected" == *",$idx,"* ]]; then
        seen=$((seen + 1))
        if [[ -z "$used" || -z "$free" || "$used" -gt "$MAX_GPU_USED_MB" || "$free" -lt "$MIN_GPU_FREE_MB" ]]; then
          echo "GPU $idx is not idle enough: used=${used:-unknown}MB free=${free:-unknown}MB util=${util:-unknown}%."
          bad=$((bad + 1))
        fi
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
    if [[ "$seen" -eq 0 ]]; then
      echo "Could not match CUDA_VISIBLE_DEVICES='$GPUS' against nvidia-smi GPU indices."
      exit 8
    fi
    if [[ "$bad" -ne 0 ]]; then
      echo "Refusing to resume: selected GPU(s) are busy. Override only when intentional with REQUIRE_GPU_IDLE=0."
      exit 9
    fi
  fi

  say "Preflight passed: /home/gpus free=${free_gb}GB."
}

run_battery() {
  local mode=$1 out=$2 log=$3
  local mask_arg=()
  if [[ "$mode" == "masked" ]]; then
    mask_arg=(--mask-image)
  fi
  if battery_output_ready "$out" >/dev/null 2>&1; then
    say "Skip $mode: $out already exists and passes completeness checks."
    return 0
  fi
  backup_incomplete_output "$out"
  say "Run TabMWP Full-SFT battery: $mode"
  "$PY" scripts/battery_n400.py \
    --base "$MODEL" --quant none --adapter none \
    --scale-tag full8b_tabmwp \
    --dump data/distill/tabmwp/test_cases_400.jsonl \
    --img-dir /home/gpus/mbe_data/tabmwp_test_images \
    --n 400 "${mask_arg[@]}" \
    --out "$out" \
    --paraphrase-cache "$CACHE" \
    --base-cache "$BASE_CACHE" \
    --paraphrase-workers "${PARAPHRASE_WORKERS:-4}" \
    --release-model-during-paraphrase \
    > "$log" 2>&1
  grep -E 'n_eval|base free-form|paraphrase|BATTERY|snap_rate|follow_rate' "$log" | tail -20 || true
  battery_output_ready "$out"
}

preflight

if [[ "${PRECHECK_ONLY:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  say "PRECHECK_ONLY/DRY_RUN requested; exiting before model load."
  exit 0
fi

say "GPU snapshot before resume:"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

run_battery present "$PRESENT_OUT" "$LOG/battery_tabmwp_present.log"
run_battery masked "$MASKED_OUT" "$LOG/battery_tabmwp_masked.log"

if battery_output_ready "$PRESENT_OUT" >/dev/null 2>&1 && battery_output_ready "$MASKED_OUT" >/dev/null 2>&1 && [[ "${KEEP_CHECKPOINTS:-0}" != "1" ]]; then
  say "Both battery outputs exist; removing large TabMWP weight shards."
  find "$MODEL" -maxdepth 1 -type f \
    \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'training_args.bin' \) \
    -print -delete || true
fi

if battery_output_ready "$PRESENT_OUT" >/dev/null 2>&1 && battery_output_ready "$MASKED_OUT" >/dev/null 2>&1; then
  say "Running CPU-only finalization audit/export."
  bash scripts/finalize_full_sft_8b_nonvideo.sh
fi

say "TabMWP Full-SFT battery resume complete."
