#!/usr/bin/env bash
# Phase D: serve the distilled orchestrator (Qwen2.5-7B + LoRA) via vLLM with an
# OpenAI-compatible API + Hermes tool-call parser (spec §7.1).
#
# Then point the harness at it (spec §7.2) and run on the held-out videos:
#   ORCHESTRATOR_API_BASE_URL=http://<host>:8001/v1
#   ORCHESTRATOR_API_KEY=EMPTY
#   ORCHESTRATOR_MODEL_NAME=mrbigeye_orch      # == the --lora-modules name below
# Keep VLM_*/JUDGE_* unchanged so only the orchestrator differs.
#
# Usage: distill_serve_vllm.sh <base_model_dir> <adapter_dir> [gpu] [port]
set -euo pipefail

BASE="${1:?base model dir, e.g. /home/gpus/models/Qwen2.5-7B-Instruct}"
ADAPTER="${2:?LoRA adapter dir, e.g. /home/gpus/Mr-Big-Eye/data/distillation/ckpt_full}"
GPU="${3:-1}"
PORT="${4:-8001}"
VLLM_PY="${VLLM_PY:-$HOME/anaconda3/envs/vllm-qwen/bin/vllm}"

# 7B bf16 ~14GB fits one 20GB 3080 + KV cache; eval is serial so no TP needed
# (no NVLink → tensor-parallel would only slow PCIe transfers).
CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_PY" serve "$BASE" \
  --enable-lora \
  --lora-modules "mrbigeye_orch=${ADAPTER}" \
  --max-lora-rank 32 \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
