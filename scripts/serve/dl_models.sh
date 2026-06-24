#!/usr/bin/env bash
# WU-1 enabler: fetch the missing VLMs from ModelScope (4B + dense 32B AWQ + dense 32B bf16).
# Staged small->large so the gate (needs 4B, 32B-AWQ) can start before the 65GB bf16 lands.
set -uo pipefail
MS=/home/gpus/anaconda3/bin/modelscope
log(){ echo "[$(date +%H:%M:%S)] $*"; }

dl(){  # $1=modelscope id  $2=local dir
  local id="$1" dir="$2"
  if [[ -f "$dir/.ok" ]]; then log "SKIP $id (already complete)"; return 0; fi
  log "START $id -> $dir"
  "$MS" download --model "$id" --local_dir "$dir" 2>&1 | grep -iE "error|fail|Successfully|100%\|" | tail -2
  if [[ -f "$dir/config.json" || -n "$(ls "$dir"/*.safetensors 2>/dev/null)" ]]; then
    touch "$dir/.ok"; log "DONE  $id ($(du -sh "$dir" 2>/dev/null | cut -f1))"
  else
    log "FAIL  $id (no weights found in $dir)"; return 1
  fi
}

dl "Qwen/Qwen3-VL-4B-Instruct"            /home/gpus/models/Qwen3-VL-4B-Instruct
dl "QuantTrio/Qwen3-VL-32B-Instruct-AWQ"  /home/gpus/models/Qwen3-VL-32B-Instruct-AWQ
dl "Qwen/Qwen3-VL-32B-Instruct"           /home/gpus/models/Qwen3-VL-32B-Instruct
log "ALL DOWNLOADS FINISHED"
