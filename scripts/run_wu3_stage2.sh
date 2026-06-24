#!/usr/bin/env bash
# WU-3 stage 2 (serving must be STOPPED first — these load models in-process on the GPUs).
# Usage: bash scripts/run_wu3_stage2.sh <8b|32b>
# Runs: LoRA SFT on TabMWP teacher CoT -> n=400 paired eval (Δacc + CI + McNemar)
#       -> WU-2 battery present + masked (corrupt/shuffle/re-perception/...).
set -uo pipefail
CELL="${1:?usage: run_wu3_stage2.sh <8b|32b>}"
PY=/home/gpus/anaconda3/envs/mbe-up/bin/python
cd /home/gpus/Mr-Big-Eye-internalization
export NO_PROXY='*' no_proxy='*'
LOG=/home/gpus/logs/wu3
mkdir -p "$LOG"
COT=data/distill/poc/tabmwp_cot_train.jsonl
TEST=data/distill/tabmwp/test_cases_400.jsonl
TEST60=data/distill/tabmwp/test_cases_60.jsonl
IMG=/home/gpus/mbe_data/tabmwp_test_images

if [[ "$CELL" == "8b" ]]; then
  BASE=/home/gpus/models/Qwen3-VL-8B-Instruct ; QUANT=none ; ADIR=data/distill/poc/lora_8b_tabmwp
elif [[ "$CELL" == "32b" ]]; then
  BASE=/home/gpus/models/Qwen3-VL-32B-Instruct ; QUANT=nf4 ; ADIR=data/distill/poc/lora_32b_tabmwp
else echo "bad cell $CELL"; exit 2; fi

echo "===== WU-3 stage2 cell=$CELL base=$BASE quant=$QUANT  $(date +%H:%M:%S) ====="

# --- 1. SFT ---
echo "--- [1/4] SFT ($CELL) ---"
if [[ "$CELL" == "8b" ]]; then
  $PY scripts/poc_sft.py --base "$BASE" --data "$COT" --out "$ADIR" --epochs 2 \
     > "$LOG/sft_${CELL}.log" 2>&1 || { echo "SFT FAILED"; tail -20 "$LOG/sft_${CELL}.log"; exit 1; }
  ADAPTER="$ADIR"
else
  $PY scripts/poc_sft_32b_qlora.py --base "$BASE" --data "$COT" \
     --test-dump "$TEST60" --test-img-dir "$IMG" --out "$ADIR" --epochs 5 --quant nf4 \
     > "$LOG/sft_${CELL}.log" 2>&1 || { echo "SFT FAILED"; tail -20 "$LOG/sft_${CELL}.log"; exit 1; }
  BEST=$($PY -c "import json;print(json.load(open('$ADIR/train_summary.json'))['best_epoch'])")
  ADAPTER="$ADIR/epoch_$BEST"
  echo "best epoch = $BEST -> $ADAPTER"
fi
echo "SFT done -> adapter=$ADAPTER"

# --- 2. n=400 paired eval ---
echo "--- [2/4] eval n=400 ($CELL) ---"
$PY scripts/eval_sft_n400.py --base "$BASE" --quant "$QUANT" --adapters "$ADAPTER" \
   --test-dump "$TEST" --test-img-dir "$IMG" \
   --out "data/distill/poc/eval_sft_${CELL}_tabmwp_n400.json" --tag "tabmwp_${CELL}" \
   > "$LOG/eval_${CELL}.log" 2>&1 || { echo "EVAL FAILED"; tail -20 "$LOG/eval_${CELL}.log"; exit 1; }
grep -E 'SUMMARY|net=|McNemar|base_acc' "$LOG/eval_${CELL}.log" | tail -6

# --- 3. battery present ---
echo "--- [3/4] battery present ($CELL) ---"
$PY scripts/battery_n400.py --base "$BASE" --quant "$QUANT" --adapter "$ADAPTER" \
   --scale-tag "tabmwp${CELL}" --dump "$TEST" --img-dir "$IMG" --n 400 \
   --out "data/distill/poc/battery_tabmwp${CELL}_present.json" \
   --paraphrase-cache "data/distill/poc/paraphrase_cache_tabmwp${CELL}.jsonl" \
   > "$LOG/battery_${CELL}_present.log" 2>&1 || { echo "BATTERY present FAILED"; tail -20 "$LOG/battery_${CELL}_present.log"; exit 1; }
grep -E 'n_eval|snap_rate|BATTERY' "$LOG/battery_${CELL}_present.log" | tail -4

# --- 4. battery masked ---
echo "--- [4/4] battery masked ($CELL) ---"
$PY scripts/battery_n400.py --base "$BASE" --quant "$QUANT" --adapter "$ADAPTER" \
   --scale-tag "tabmwp${CELL}" --dump "$TEST" --img-dir "$IMG" --n 400 --mask-image \
   --out "data/distill/poc/battery_tabmwp${CELL}_masked.json" \
   --paraphrase-cache "data/distill/poc/paraphrase_cache_tabmwp${CELL}.jsonl" \
   > "$LOG/battery_${CELL}_masked.log" 2>&1 || { echo "BATTERY masked FAILED"; tail -20 "$LOG/battery_${CELL}_masked.log"; exit 1; }
grep -E 'n_eval|BATTERY' "$LOG/battery_${CELL}_masked.log" | tail -3

echo "===== WU-3 stage2 cell=$CELL DONE  $(date +%H:%M:%S) ====="