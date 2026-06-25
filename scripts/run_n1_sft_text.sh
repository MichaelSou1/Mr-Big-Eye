#!/usr/bin/env bash
# N1 text-teacher arm — SFT 8B LoRA, B2-text (strict, fluent answer-conditioned teacher) vs
# vanilla-text (none), SAME recipe, parallel on disjoint GPU pairs. Removes the gold-program
# CoT-format confound (teacher chains are fluent natural language). Serving must be STOPPED.
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
    > data/distill/poc/logs/n1_sft_text_${tag}.log 2>&1
  echo "[$tag] SFT done $(date +%H:%M:%S) rc=$?"
}

train b2text      data/distill/finqa/curriculum_dev_strict_text.jsonl data/distill/poc/lora_8b_finqa_b2_text      0,1 &
P1=$!
train vanillatext data/distill/finqa/curriculum_dev_none_text.jsonl   data/distill/poc/lora_8b_finqa_vanilla_text 2,3 &
P2=$!
wait $P1 $P2
echo "N1 TEXT SFT BOTH DONE $(date +%H:%M:%S)"
