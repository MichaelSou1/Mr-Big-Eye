#!/usr/bin/env bash
# Fan out teacher trajectory capture across GPUs: one harness process per
# pre-split case shard, pinned to a GPU (for the local retrieval encoders).
# Each shard keeps its own prediction/judge cache (shards are disjoint).
#
# Usage: scripts/run_teacher_4gpu.sh [ENV] [GPUS] [DELAY]
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
ENV="${1:-mbe-ingest}"
IFS=',' read -ra GPUS <<< "${2:-0,1,2,3}"
DELAY="${3:-0.0}"

# SHARD_DIR may be overridden (e.g. SHARD_DIR=data/distillation/v2/teacher_shards)
# so a new capture round doesn't clobber an earlier round's shards / caches.
SHARD_DIR="${SHARD_DIR:-$ROOT/data/distillation/teacher_shards}"
echo "env=$ENV gpus=${GPUS[*]} delay=$DELAY shard_dir=$SHARD_DIR"
pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  cases="$SHARD_DIR/cases_${i}.jsonl"
  [ -f "$cases" ] || { echo "missing $cases"; continue; }
  echo "  shard $i on GPU $gpu: $cases"
  CUDA_VISIBLE_DEVICES="$gpu" MBE_ROOT="$ROOT" PYTHONPATH="$ROOT" \
    nohup conda run --no-capture-output -n "$ENV" python scripts/eval_harness.py \
      --cases "$cases" \
      --output "$SHARD_DIR/report_${i}.json" \
      --save-full-trajectory \
      --trajectory-out "$SHARD_DIR/traj_${i}.jsonl" \
      --judge \
      --judge-cache "$SHARD_DIR/judge_cache_${i}.jsonl" \
      --prediction-cache "$SHARD_DIR/pred_cache_${i}.jsonl" \
      --per-case-delay-sec "$DELAY" \
    > "$SHARD_DIR/log_${i}.log" 2>&1 &
  pids+=("$!")
done
echo "shard PIDs: ${pids[*]}"
wait "${pids[@]}"
echo "ALL_TEACHER_SHARDS_DONE"
