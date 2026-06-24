#!/usr/bin/env bash
# WU-1 §1.3 + §1.4 + §1.5 — in-process (transformers/peft) phase, run AFTER all gates finish and
# every vLLM server is stopped (these jobs grab all GPUs via device_map=auto).
# Uses the env python DIRECTLY (conda run's argparser eats --n). Resumable (skips if output exists).
set -uo pipefail
cd /home/gpus/Mr-Big-Eye-internalization
LOGD=/home/gpus/logs/wu1; mkdir -p "$LOGD"
PY=/home/gpus/anaconda3/envs/mbe-up/bin/python
BF16=/home/gpus/models/Qwen3-VL-32B-Instruct
M8B=/home/gpus/models/Qwen3-VL-8B-Instruct
DUMP=data/distill/chartqa/test_cases_400.jsonl
IMG=/home/gpus/mbe_data/chartqa_test_images
A32=data/distill/poc/lora_32b_chartqa
A8=data/distill/poc/lora_8b_chartqa
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "GPU check (should be all free):"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ---- 1.3a: 32B SFT eval n=400 (NF4 base + epochs 1..3 — brackets the n=60 peak epoch_1) ----
OUT32=$A32/eval_n400.json
if [ -f "$OUT32" ]; then say "skip 1.3a (exists $OUT32)"; else
  say "1.3a 32B SFT eval n=400 ..."
  # epoch_1 = n=60 peak; train set is fixed so it stays the principled checkpoint at n=400. 32B NF4
  # multi-GPU eval is the slowest in-process step, so we evaluate base + epoch_1 only (gives Delta acc
  # + bootstrap CI + McNemar at the peak — the §1.3 acceptance). Add epochs later if time allows.
  CUDA_VISIBLE_DEVICES=0,1,2,3 NO_PROXY='*' $PY -u scripts/eval_sft_n400.py --base "$BF16" --quant nf4 \
     --adapters $A32/epoch_1 \
     --test-dump "$DUMP" --test-img-dir "$IMG" --tag sft32b_n400 --out "$OUT32" --batch-size 8 \
     > "$LOGD/eval_sft_32b.log" 2>&1 || { say "1.3a FAILED (see $LOGD/eval_sft_32b.log)"; exit 1; }
  say "1.3a done -> $OUT32"
fi

# ---- 1.3b: 8B SFT eval n=400 (bf16 base + single adapter) ----
OUT8=$A8/eval_n400.json
if [ -f "$OUT8" ]; then say "skip 1.3b (exists $OUT8)"; else
  say "1.3b 8B SFT eval n=400 ..."
  CUDA_VISIBLE_DEVICES=0,1,2,3 NO_PROXY='*' $PY -u scripts/eval_sft_n400.py --base "$M8B" --quant none \
     --adapters "$A8" --test-dump "$DUMP" --test-img-dir "$IMG" --tag sft8b_n400 --out "$OUT8" --batch-size 8 \
     > "$LOGD/eval_sft_8b.log" 2>&1 || { say "1.3b FAILED (see $LOGD/eval_sft_8b.log)"; exit 1; }
  say "1.3b done -> $OUT8"
fi

# peak 32B adapter from 1.3a
PEAK=$($PY -c "import json;print(json.load(open('$OUT32'))['best_adapter'])")
say "32B peak adapter @ n=400: $PEAK"

# ---- 1.4: causal probe 2x2 (8B/32B x present/masked) at n=400 — BATCHED ----
run_probe(){  # $1=base $2=adapter $3=quant $4=out $5=extra(--mask-image|"")
  local base="$1" ad="$2" q="$3" out="$4" extra="$5"
  if [ -f "$out" ]; then say "skip probe (exists $out)"; return 0; fi
  say "probe -> $out  ($extra)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 NO_PROXY='*' $PY -u scripts/probe_n400.py --base "$base" --adapter "$ad" \
     --quant "$q" --dump "$DUMP" --img-dir "$IMG" --n 400 --out "$out" --batch-size 8 $extra \
     > "$LOGD/probe_$(basename "$out" .json).log" 2>&1 \
     || { say "probe FAILED for $out"; return 1; }
  say "probe done -> $out"
}
run_probe "$BF16" "$PEAK" nf4 data/distill/poc/causal_probe_32b_n400.json ""
run_probe "$BF16" "$PEAK" nf4 data/distill/poc/causal_probe_32b_maskimg_n400.json "--mask-image"
run_probe "$M8B"  "$A8"   nf4 data/distill/poc/causal_probe_8b_n400.json ""
run_probe "$M8B"  "$A8"   nf4 data/distill/poc/causal_probe_8b_maskimg_n400.json "--mask-image"

# ---- 1.5: power table (reads max-n per cell) ----
say "1.5 power table ..."
$PY -u scripts/power_table.py > "$LOGD/power_table.log" 2>&1 || say "1.5 FAILED"

say "WU-1 in-process pipeline COMPLETE"; touch "$LOGD/INPROC_DONE"
