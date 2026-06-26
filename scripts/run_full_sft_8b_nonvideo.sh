#!/usr/bin/env bash
# Run Qwen3-VL-8B Full-SFT replicas of the non-video LoRA SFT arms, then run the
# matching causal probes. GPU policy: one Full-SFT/eval/probe process at a time.
#
# Usage:
#   bash scripts/run_full_sft_8b_nonvideo.sh chartqa
#   bash scripts/run_full_sft_8b_nonvideo.sh tabmwp
#   bash scripts/run_full_sft_8b_nonvideo.sh finqa_b2
#   bash scripts/run_full_sft_8b_nonvideo.sh all
#
# Arms:
#   chartqa, tabmwp, finqa_b2, finqa_vanilla, finqa_b2_text, finqa_vanilla_text
#
# By default, large full-model weight shards are deleted after their eval/probe
# finishes, because the local disk cannot hold all six 8B checkpoints. Set
# KEEP_CHECKPOINTS=1 to keep them.
set -euo pipefail

cd /home/gpus/Mr-Big-Eye-internalization

PY=${PY:-/home/gpus/anaconda3/envs/mbe-up/bin/python}
BASE=${BASE:-/home/gpus/models/Qwen3-VL-8B-Instruct}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES="$GPUS"
export NO_PROXY='*' no_proxy='*'
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG=${LOG:-/home/gpus/logs/full_sft_8b_nonvideo}
mkdir -p "$LOG" data/distill/poc/logs

LOCK=${LOCK:-/tmp/mbe_full_sft_8b_nonvideo.lock}
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another Full-SFT non-video run is active (lock: $LOCK)."
  exit 3
fi

say() {
  echo "[$(date '+%F %T')] $*"
}

require_space() {
  local need_gb=${1:-25}
  local avail
  avail=$(df -BG /home/gpus | awk 'NR==2 {gsub("G","",$4); print $4}')
  if (( avail < need_gb )); then
    echo "Need at least ${need_gb}G free on /home/gpus, found ${avail}G. Clean space first."
    exit 4
  fi
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
}

cleanup_checkpoint() {
  local out=$1
  if [[ "${KEEP_CHECKPOINTS:-0}" == "1" ]]; then
    say "KEEP_CHECKPOINTS=1; keeping full checkpoint at $out"
    return 0
  fi
  say "Removing large weight shards from $out; keeping configs, tokenizer, summaries, and results."
  find "$out" -maxdepth 1 -type f \
    \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'training_args.bin' \) \
    -print -delete || true
}

train_full() {
  local arm=$1 data=$2 out=$3 epochs=$4
  require_space 25
  say "GPU snapshot before train $arm:"
  gpu_snapshot
  say "Full-SFT train arm=$arm epochs=$epochs data=$data out=$out gpus=$CUDA_VISIBLE_DEVICES"
  "$PY" scripts/poc_sft_full_8b.py \
    --base "$BASE" --data "$data" --out "$out" --epochs "$epochs" \
    --strategy device_map --workers 2 --lr "${LR:-2e-5}" \
    --freeze-embeddings --freeze-first-layers "${FREEZE_FIRST_LAYERS:-3}" \
    > "$LOG/train_${arm}.log" 2>&1
  say "Full-SFT train done arm=$arm"
}

eval_chartlike() {
  local arm=$1 out=$2 dump=$3 img=$4 eval_out=$5 tag=$6
  say "Full-SFT n=400 eval arm=$arm"
  "$PY" scripts/eval_full_sft_n400.py \
    --base "$BASE" --full-models "${tag}=${out}" --quant none \
    --test-dump "$dump" --test-img-dir "$img" \
    --out "$eval_out" --tag "$tag" \
    > "$LOG/eval_${arm}.log" 2>&1
  grep -E 'SUMMARY|test_acc=|McNemar|base_acc' "$LOG/eval_${arm}.log" | tail -8 || true
}

battery_chartlike() {
  local arm=$1 out=$2 dump=$3 img=$4 tag=$5 present_out=$6 masked_out=$7 cache=$8
  local base_cache="${cache%.jsonl}_base_cot.jsonl"
  local extra=()
  if [[ "${USE_BASE_COT_CACHE:-1}" == "1" ]]; then
    extra+=(--base-cache "$base_cache")
  fi
  if [[ "${RELEASE_MODEL_DURING_PARAPHRASE:-1}" == "1" ]]; then
    extra+=(--release-model-during-paraphrase)
  fi
  say "Causal battery present arm=$arm"
  "$PY" scripts/battery_n400.py \
    --base "$out" --quant none --adapter none \
    --scale-tag "$tag" --dump "$dump" --img-dir "$img" --n 400 \
    --out "$present_out" --paraphrase-cache "$cache" "${extra[@]}" \
    > "$LOG/battery_${arm}_present.log" 2>&1
  grep -E 'n_eval|snap_rate|BATTERY' "$LOG/battery_${arm}_present.log" | tail -6 || true

  say "Causal battery masked arm=$arm"
  "$PY" scripts/battery_n400.py \
    --base "$out" --quant none --adapter none \
    --scale-tag "$tag" --dump "$dump" --img-dir "$img" --n 400 --mask-image \
    --out "$masked_out" --paraphrase-cache "$cache" "${extra[@]}" \
    > "$LOG/battery_${arm}_masked.log" 2>&1
  grep -E 'n_eval|BATTERY' "$LOG/battery_${arm}_masked.log" | tail -4 || true
}

probe_finqa() {
  local arm=$1 out=$2 mode=$3
  say "FinQA targeted causal probe arm=$arm mode=$mode"
  "$PY" scripts/battery_n1_targeted.py \
    --base "$out" --base-mode-name "$mode" --quant none \
    --dump data/distill/finqa/curriculum_test_strict.jsonl \
    --out "data/distill/poc/battery_n1_${mode}.json" \
    > "$LOG/battery_${arm}.log" 2>&1
  grep -E 'TARGETED|base_acc|mode|snap|follow|n_targeted' "$LOG/battery_${arm}.log" | tail -18 || true
}

run_arm() {
  local arm=$1
  case "$arm" in
    chartqa)
      local out=data/distill/poc/full_8b_chartqa
      train_full "$arm" data/distill/poc/chartqa_cot_train.jsonl "$out" 2
      eval_chartlike "$arm" "$out" data/distill/chartqa/test_cases_400.jsonl \
        /home/gpus/mbe_data/chartqa_test_images "$out/eval_n400.json" full8b_chartqa
      battery_chartlike "$arm" "$out" data/distill/chartqa/test_cases_400.jsonl \
        /home/gpus/mbe_data/chartqa_test_images full8b_chartqa \
        data/distill/poc/battery_full8b_chartqa_present.json \
        data/distill/poc/battery_full8b_chartqa_masked.json \
        data/distill/poc/paraphrase_cache_full8b_chartqa.jsonl
      cleanup_checkpoint "$out"
      ;;
    tabmwp)
      local out=data/distill/poc/full_8b_tabmwp
      train_full "$arm" data/distill/poc/tabmwp_cot_train.jsonl "$out" 2
      eval_chartlike "$arm" "$out" data/distill/tabmwp/test_cases_400.jsonl \
        /home/gpus/mbe_data/tabmwp_test_images data/distill/poc/eval_full_sft_8b_tabmwp_n400.json full8b_tabmwp
      battery_chartlike "$arm" "$out" data/distill/tabmwp/test_cases_400.jsonl \
        /home/gpus/mbe_data/tabmwp_test_images full8b_tabmwp \
        data/distill/poc/battery_full8b_tabmwp_present.json \
        data/distill/poc/battery_full8b_tabmwp_masked.json \
        data/distill/poc/paraphrase_cache_full8b_tabmwp_mimo.jsonl
      cleanup_checkpoint "$out"
      ;;
    finqa_b2)
      local out=data/distill/poc/full_8b_finqa_b2
      train_full "$arm" data/distill/finqa/curriculum_dev_strict.jsonl "$out" 3
      probe_finqa "$arm" "$out" full8b_finqa_b2
      cleanup_checkpoint "$out"
      ;;
    finqa_vanilla)
      local out=data/distill/poc/full_8b_finqa_vanilla
      train_full "$arm" data/distill/finqa/curriculum_dev_none.jsonl "$out" 3
      probe_finqa "$arm" "$out" full8b_finqa_vanilla
      cleanup_checkpoint "$out"
      ;;
    finqa_b2_text)
      local out=data/distill/poc/full_8b_finqa_b2_text
      train_full "$arm" data/distill/finqa/curriculum_dev_strict_text.jsonl "$out" 3
      probe_finqa "$arm" "$out" full8b_finqa_b2_text
      cleanup_checkpoint "$out"
      ;;
    finqa_vanilla_text)
      local out=data/distill/poc/full_8b_finqa_vanilla_text
      train_full "$arm" data/distill/finqa/curriculum_dev_none_text.jsonl "$out" 3
      probe_finqa "$arm" "$out" full8b_finqa_vanilla_text
      cleanup_checkpoint "$out"
      ;;
    *)
      echo "Unknown arm: $arm"
      exit 2
      ;;
  esac
}

selection=${1:-all}
if [[ "$selection" == "all" ]]; then
  arms=(chartqa tabmwp finqa_b2 finqa_vanilla finqa_b2_text finqa_vanilla_text)
else
  IFS=',' read -r -a arms <<< "$selection"
fi

say "Starting Full-SFT non-video run arms=${arms[*]} gpus=$CUDA_VISIBLE_DEVICES keep=${KEEP_CHECKPOINTS:-0}"
for arm in "${arms[@]}"; do
  say "===== ARM $arm START ====="
  run_arm "$arm"
  say "===== ARM $arm DONE ====="
done
say "All requested Full-SFT non-video arms complete."
