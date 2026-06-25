#!/usr/bin/env bash
# N3 — natural-image (V*-style) pole probe on the GENERAL BASE (no chart-SFT adapter).
# Two chains in PARALLEL on disjoint GPUs; within each chain present runs before masked.
# Interventions: corrupt + shuffle + local controls (filler/truncate/delete); paraphrase
# SKIPPED (needs DeepSeek API, not load-bearing for the follow/other verdict).
# quant=nf4 (matches WU-2 ChartQA 8B; lets 8B fit on one 3080).
# Serving must be STOPPED (these load models in-process). Usage: bash scripts/run_n3.sh
set -u
PY=/home/gpus/anaconda3/envs/mbe-up/bin/python
cd /home/gpus/Mr-Big-Eye-internalization || exit 1
export NO_PROXY='*' no_proxy='*'
mkdir -p data/distill/poc/logs
DUMP=data/distill/natcount/test_cases_400.jsonl
IMG=/home/gpus/mbe_data/natcount_test_images
INTERV="corrupt shuffle truncate delete filler"

run_chain() {  # tag base gpus batch
  local tag=$1 base=$2 gpus=$3 batch=$4
  echo "[$tag] present start $(date +%H:%M:%S) gpus=$gpus"
  CUDA_VISIBLE_DEVICES=$gpus $PY scripts/battery_n400.py \
    --base "$base" --adapter none --quant nf4 --scale-tag "$tag" \
    --dump "$DUMP" --img-dir "$IMG" --n 400 --batch-size "$batch" \
    --interventions $INTERV \
    --out data/distill/poc/battery_${tag}_present.json \
    > data/distill/poc/logs/battery_${tag}_present.log 2>&1
  echo "[$tag] present done $(date +%H:%M:%S) rc=$?"
  echo "[$tag] masked start $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpus $PY scripts/battery_n400.py \
    --base "$base" --adapter none --quant nf4 --scale-tag "$tag" --mask-image \
    --dump "$DUMP" --img-dir "$IMG" --n 400 --batch-size "$batch" \
    --interventions $INTERV \
    --out data/distill/poc/battery_${tag}_masked.json \
    > data/distill/poc/logs/battery_${tag}_masked.log 2>&1
  echo "[$tag] masked done $(date +%H:%M:%S) rc=$?"
}

run_chain natcount8b  /home/gpus/models/Qwen3-VL-8B-Instruct  0     8 &
P8=$!
run_chain natcount32b /home/gpus/models/Qwen3-VL-32B-Instruct 1,2,3 8 &
P32=$!
wait $P8 $P32
echo "N3 GRID DONE $(date +%H:%M:%S)"
