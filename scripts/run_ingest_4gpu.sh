#!/usr/bin/env bash
# Fan out Video-MME ingest across GPUs via MBE_INGEST_SHARD sharding.
# One process per GPU, each pinned with CUDA_VISIBLE_DEVICES (so cuda:0 inside
# the process maps to that physical card). Already-ingested videos are skipped.
#
# Usage:
#   scripts/run_ingest_4gpu.sh [ENV] [GPUS] [NSHARDS]
#     ENV     conda env to run in           (default: mbe-ingest)
#     GPUS    comma list of physical GPUs    (default: 0,1,2,3)
#     NSHARDS number of shards               (default: number of GPUs)
#
# Logs: data/distillation/ingest_logs/shard_<gpu>.log
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

ENV="${1:-mbe-ingest}"
IFS=',' read -ra GPUS <<< "${2:-0,1,2,3}"
NSHARDS="${3:-${#GPUS[@]}}"

LOGDIR="$ROOT/data/distillation/ingest_logs"
mkdir -p "$LOGDIR"

echo "env=$ENV gpus=${GPUS[*]} nshards=$NSHARDS"
pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  log="$LOGDIR/shard_${gpu}.log"
  echo "  launching shard $i/$NSHARDS on GPU $gpu → $log"
  CUDA_VISIBLE_DEVICES="$gpu" MBE_INGEST_SHARD="$i/$NSHARDS" MBE_ROOT="$ROOT" \
    nohup conda run --no-capture-output -n "$ENV" python -m scripts.ingest_videomme \
    > "$log" 2>&1 &
  pids+=("$!")
done

echo "shard PIDs: ${pids[*]}"
echo "tail logs:  tail -f $LOGDIR/shard_*.log"
wait "${pids[@]}"
echo "ALL_SHARDS_DONE"
