#!/usr/bin/env python
"""QLoRA dry-run — empirically measure peak VRAM for k-bit-base + LoRA SFT of a Qwen3-VL VLM
on the ChartQA reasoning CoT records, WITHOUT a full training run.

Why: the §11 "pure-reasoning internalization" target is the 32B-charts cell, which needs a
4-bit base (bf16-32B is 64GB and won't fit 4x20GB nor the 46GB free disk). This validates the
QLoRA stack (bnb NF4 + peft LoRA + vision frozen + fwd/bwd) end-to-end and reports per-GPU peak
VRAM. Run on the local bf16 8B first (stack + footprint anchor); the 32B base scales the NF4
weight footprint ~linearly (8B NF4 ~5GB -> 32B NF4 ~17GB), activations are batch/len-bound and
near-constant, so an 8B measurement that leaves >ample headroom proves the 32B fits.

Manual fwd/bwd loop (not Trainer) for clean, deterministic peak-VRAM measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gpu_report(tag: str) -> None:
    n = torch.cuda.device_count()
    parts = []
    for i in range(n):
        peak = torch.cuda.max_memory_allocated(i) / 1024**3
        resv = torch.cuda.max_memory_reserved(i) / 1024**3
        parts.append(f"GPU{i} alloc_peak={peak:.2f}G reserved_peak={resv:.2f}G")
    print(f"[{tag}] " + " | ".join(parts), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/gpus/Workout-Coach/hf_cache/modelscope/Qwen--Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", default="data/distill/poc/chartqa_cot_train.jsonl")
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    ap.add_argument("--steps", type=int, default=4, help="fwd/bwd steps to measure peak")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3VLForConditionalGeneration)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    recs = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"records: {len(recs)} | base={args.base} | quant={args.quant}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tok = processor.tokenizer

    quant_cfg = None
    if args.quant == "nf4":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    gpu_report("after load")

    # freeze vision tower
    for n, p in model.named_parameters():
        if "visual" in n or "vision" in n:
            p.requires_grad = False

    if args.quant == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    def build(rec):
        img = Image.open(rec["image_path"]).convert("RGB")
        target = f"{rec['cot']}\nANSWER: {rec['answer']}"
        user_msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Solve step by step, end with 'ANSWER: <final>'.\n\n"
                                     f"Question: {rec['question']}"}]}]
        full_msg = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
        full = processor.apply_chat_template(full_msg, tokenize=True, return_dict=True,
                                             add_generation_prompt=False, return_tensors="pt")
        prompt = processor.apply_chat_template(user_msg, tokenize=True, return_dict=True,
                                               add_generation_prompt=True, return_tensors="pt")
        plen = prompt["input_ids"].shape[1]
        seq_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
        item = {k: (v[0] if k in seq_keys else v) for k, v in full.items()}
        labels = item["input_ids"].clone()
        labels[:plen] = -100
        item["labels"] = labels
        return item

    # build a few items, longest-first so the FIRST step hits a near-worst-case length
    items = []
    for r in recs:
        try:
            it = build(r)
            if it["input_ids"].shape[0] <= args.max_len:
                items.append(it)
        except Exception as e:
            print("build skip:", str(e)[:80])
        if len(items) >= max(args.steps * 3, 12):
            break
    items.sort(key=lambda d: d["input_ids"].shape[0], reverse=True)
    print(f"usable items: {len(items)}; token lens (top): "
          f"{[int(d['input_ids'].shape[0]) for d in items[:args.steps]]}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    dev0 = torch.device("cuda:0")

    for step in range(args.steps):
        it = items[step % len(items)]
        batch = {"input_ids": it["input_ids"].unsqueeze(0).to(dev0),
                 "attention_mask": it["attention_mask"].unsqueeze(0).to(dev0),
                 "labels": it["labels"].unsqueeze(0).to(dev0)}
        if "mm_token_type_ids" in it:
            batch["mm_token_type_ids"] = it["mm_token_type_ids"].unsqueeze(0).to(dev0)
        if "pixel_values" in it:
            batch["pixel_values"] = it["pixel_values"].to(dev0)
        if "image_grid_thw" in it:
            batch["image_grid_thw"] = it["image_grid_thw"].to(dev0)
        opt.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        opt.step()
        print(f"step {step} loss={loss.item():.4f} len={int(it['input_ids'].shape[0])}", flush=True)
        gpu_report(f"after step {step}")

    print("\n=== PEAK SUMMARY ===")
    gpu_report("final peak")
    total_peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3
    print(f"TOTAL alloc peak across GPUs: {total_peak:.2f} G")
    print("Stack OK: NF4 base + LoRA fwd/bwd ran." if args.quant == "nf4"
          else "Stack OK: full-precision fwd/bwd ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
