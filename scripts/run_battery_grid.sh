#!/usr/bin/env bash
# WU-2 — run the full battery grid {8B,32B} x {present,masked} at n=400.
# Two chains run in PARALLEL on disjoint GPUs; within each chain present runs BEFORE masked
# (present populates the paraphrase cache; masked reuses it — base-CoT is image-present in both,
# cache keyed by CoT md5). Separate cache files per scale avoid concurrent-append races.
set -u
PY=/home/gpus/anaconda3/envs/mbe-up/bin/python
cd /home/gpus/Mr-Big-Eye-internalization || exit 1
mkdir -p data/distill/poc/logs

run_chain() {  # tag base adapter gpus batch
  local tag=$1 base=$2 adapter=$3 gpus=$4 batch=$5
  local cache=data/distill/poc/paraphrase_cache_${tag}.jsonl
  echo "[$tag] present start $(date +%H:%M:%S) gpus=$gpus"
  CUDA_VISIBLE_DEVICES=$gpus $PY scripts/battery_n400.py \
    --base "$base" --adapter "$adapter" --scale-tag "$tag" \
    --n 400 --batch-size "$batch" --paraphrase-cache "$cache" \
    --out data/distill/poc/battery_${tag}_present.json \
    > data/distill/poc/logs/battery_${tag}_present.log 2>&1
  echo "[$tag] present done $(date +%H:%M:%S) rc=$?"
  echo "[$tag] masked start $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpus $PY scripts/battery_n400.py \
    --base "$base" --adapter "$adapter" --scale-tag "$tag" --mask-image \
    --n 400 --batch-size "$batch" --paraphrase-cache "$cache" \
    --out data/distill/poc/battery_${tag}_masked.json \
    > data/distill/poc/logs/battery_${tag}_masked.log 2>&1
  echo "[$tag] masked done $(date +%H:%M:%S) rc=$?"
}

run_chain 8b  /home/gpus/models/Qwen3-VL-8B-Instruct  data/distill/poc/lora_8b_chartqa        0     8 &
P8=$!
run_chain 32b /home/gpus/models/Qwen3-VL-32B-Instruct data/distill/poc/lora_32b_chartqa/epoch_1 1,2,3 8 &
P32=$!
wait $P8 $P32
echo "GRID DONE $(date +%H:%M:%S)"
