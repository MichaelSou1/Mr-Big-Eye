#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -n "${SGLANG_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES}"
fi

if [[ -z "${CUDA_HOME:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CUDA_TARGET_DIR="${CONDA_PREFIX}/targets/x86_64-linux"
  if [[ -d "${CUDA_TARGET_DIR}/include" ]]; then
    export CPATH="${CUDA_TARGET_DIR}/include:${CPATH:-}"
  fi
  if [[ -d "${CUDA_TARGET_DIR}/lib" ]]; then
    export LIBRARY_PATH="${CUDA_TARGET_DIR}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${CUDA_TARGET_DIR}/lib:${LD_LIBRARY_PATH:-}"
    if [[ -d "${CUDA_TARGET_DIR}/lib/stubs" && ! -e "${CONDA_PREFIX}/lib/stubs" ]]; then
      ln -s "${CUDA_TARGET_DIR}/lib/stubs" "${CONDA_PREFIX}/lib/stubs"
    fi
  fi
fi

MODEL_NAME="${VLM_MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
MODEL_LOCAL_DIR="${VLM_MODEL_LOCAL_DIR:-./models/Qwen3-VL-2B-Instruct}"
MODEL_PATH="${MODEL_NAME}"
if [[ -d "${MODEL_LOCAL_DIR}" ]] && [[ -n "$(find "${MODEL_LOCAL_DIR}" -maxdepth 1 -type f -print -quit)" ]]; then
  MODEL_PATH="${MODEL_LOCAL_DIR}"
fi

ARGS=(
  serve
  --model-path "${MODEL_PATH}"
  --served-model-name "${SGLANG_SERVED_MODEL_NAME:-${MODEL_NAME}}"
  --host "${SGLANG_HOST:-127.0.0.1}"
  --port "${SGLANG_PORT:-30000}"
  --tp-size "${SGLANG_TP_SIZE:-1}"
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.72}"
  --enable-multimodal
  --trust-remote-code
)

if [[ -n "${SGLANG_CHAT_TEMPLATE:-}" ]]; then
  ARGS+=(--chat-template "${SGLANG_CHAT_TEMPLATE}")
fi

if [[ -n "${SGLANG_ATTENTION_BACKEND:-}" ]]; then
  ARGS+=(--attention-backend "${SGLANG_ATTENTION_BACKEND}")
fi

if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-false}" == "true" ]]; then
  ARGS+=(--disable-cuda-graph)
fi

if [[ "${SGLANG_DISABLE_OVERLAP_SCHEDULE:-false}" == "true" ]]; then
  ARGS+=(--disable-overlap-schedule)
fi

sglang "${ARGS[@]}"
