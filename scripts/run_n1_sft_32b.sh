#!/usr/bin/env bash
# N1 (B2) 32B replication — QLoRA-SFT Qwen3-VL-32B, B2 curriculum vs vanilla, SAME recipe,
# parallel on disjoint GPU pairs. 3 epochs (parity with 8B); we take epoch_3 (no test-selection).
# Per-epoch eval points at a 40-case monitoring slice. Serving must be STOPPED.
set -u
PY=/home/gpus/anaconda3/envs/mbe-up/bin/python
cd /home/gpus/Mr-Big-Eye-internalization || exit 1
export NO_PROXY='*' no_proxy='*'
BASE=/home/gpus/models/Qwen3-VL-32B-Instruct
SLICE=data/distill/finqa/test_slice40.jsonl
IMG=/home/gpus/mbe_data/finqa_test_images
mkdir -p data/distill/poc/logs

train() { # tag data out gpus
  local tag=$1 data=$2 out=$3 gpus=$4
  echo "[$tag] 32B SFT start $(date +%H:%M:%S) gpus=$gpus -> $out"
  CUDA_VISIBLE_DEVICES=$gpus $PY scripts/poc_sft_32b_qlora.py \
    --base "$BASE" --data "$data" --out "$out" --epochs 3 --quant nf4 \
    --test-dump "$SLICE" --test-img-dir "$IMG" --batch-size 8 \
    > data/distill/poc/logs/n1_sft32b_${tag}.log 2>&1
  echo "[$tag] 32B SFT done $(date +%H:%M:%S) rc=$?"
}

train b2      data/distill/finqa/curriculum_dev_strict.jsonl data/distill/poc/lora_32b_finqa_b2      0,1 &
P1=$!
train vanilla data/distill/finqa/curriculum_dev_none.jsonl   data/distill/poc/lora_32b_finqa_vanilla 2,3 &
P2=$!
wait $P1 $P2
echo "N1 32B SFT BOTH DONE $(date +%H:%M:%S)"
