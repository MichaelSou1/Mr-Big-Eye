#!/usr/bin/env bash
# N1 (B2) — SFT two 8B LoRA adapters with the SAME recipe, in parallel on disjoint GPU pairs:
#   B2      = FinQA dev STRICT curriculum (③: >=2-op, multi-cell, flippable-operand, not single-cell-readable)
#   vanilla = FinQA dev NONE (matched-size random sample; includes 1-op / single-cell-readable shortcuts)
# Only difference is the curriculum filter -> isolates "does removing the read-shortcut at TRAIN time
# make the student's internalized chain load-bearing?". Serving must be STOPPED. Usage: bash scripts/run_n1_sft.sh
set -u
PY=/home/gpus/anaconda3/envs/mbe-up/bin/python
cd /home/gpus/Mr-Big-Eye-internalization || exit 1
export NO_PROXY='*' no_proxy='*'
BASE=/home/gpus/models/Qwen3-VL-8B-Instruct
EPOCHS=3
mkdir -p data/distill/poc/logs

train() { # tag data out gpus
  local tag=$1 data=$2 out=$3 gpus=$4
  echo "[$tag] SFT start $(date +%H:%M:%S) gpus=$gpus -> $out"
  CUDA_VISIBLE_DEVICES=$gpus $PY scripts/poc_sft.py \
    --base "$BASE" --data "$data" --out "$out" --epochs $EPOCHS \
    > data/distill/poc/logs/n1_sft_${tag}.log 2>&1
  echo "[$tag] SFT done $(date +%H:%M:%S) rc=$?"
}

train b2      data/distill/finqa/curriculum_dev_strict.jsonl data/distill/poc/lora_8b_finqa_b2      0,1 &
P1=$!
train vanilla data/distill/finqa/curriculum_dev_none.jsonl   data/distill/poc/lora_8b_finqa_vanilla 2,3 &
P2=$!
wait $P1 $P2
echo "N1 SFT BOTH DONE $(date +%H:%M:%S)"
