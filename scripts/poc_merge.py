#!/usr/bin/env python
"""Merge the §11 LoRA adapter into the base 8B and save a standalone model for vLLM serving
(so §11.2 eval reuses run_chartqa_gate on the SAME 60 ChartQA cases as the pre-SFT 8B)."""
from __future__ import annotations
import argparse
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/gpus/Workout-Coach/hf_cache/modelscope/Qwen--Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default="data/distill/poc/lora_8b_chartqa")
    ap.add_argument("--out", default="/home/gpus/models/Qwen3-VL-8B-ChartQA-SFT")
    args = ap.parse_args()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
    print(f"merged -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
