#!/usr/bin/env python
"""Spec §11.2 — LoRA-SFT the small VLM (Qwen3-VL-8B) on (image, question, CoT, answer) for the
ChartQA reasoning residual. Vision tower frozen; LoRA on the LLM projections. Supervises only
the assistant turn (CoT + 'ANSWER: ...'); prompt tokens masked to -100. bf16, model sharded
across the visible GPUs, gradient checkpointing. A PoC: small data, few epochs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/gpus/Workout-Coach/hf_cache/modelscope/Qwen--Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", default="data/distill/poc/chartqa_cot_train.jsonl")
    ap.add_argument("--out", default="data/distill/poc/lora_8b_chartqa")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    recs = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"SFT records: {len(recs)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tok = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    # freeze vision tower
    for n, p in model.named_parameters():
        if "visual" in n or "vision" in n:
            p.requires_grad = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

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
        # only sequence tensors carry a leading batch dim to strip; pixel_values/image_grid_thw
        # are flattened across images (no batch dim) — keep them whole.
        seq_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
        item = {k: (v[0] if k in seq_keys else v) for k, v in full.items()}
        labels = item["input_ids"].clone()
        labels[:plen] = -100
        item["labels"] = labels
        return item

    dataset = [build(r) for r in recs]
    dataset = [d for d in dataset if d["input_ids"].shape[0] <= args.max_len]
    print(f"usable (<= {args.max_len} tok): {len(dataset)}", flush=True)

    def collate(batch):
        # pad to longest in batch
        maxlen = max(b["input_ids"].shape[0] for b in batch)
        out = {}
        ids = torch.full((len(batch), maxlen), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)
        lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        has_mmtt = "mm_token_type_ids" in batch[0]
        mmtt = torch.zeros((len(batch), maxlen), dtype=torch.long) if has_mmtt else None
        pix, grid = [], []
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            ids[i, :L] = b["input_ids"]; att[i, :L] = 1; lab[i, :L] = b["labels"]
            if has_mmtt: mmtt[i, :L] = b["mm_token_type_ids"]
            if "pixel_values" in b: pix.append(b["pixel_values"])
            if "image_grid_thw" in b: grid.append(b["image_grid_thw"])
        out["input_ids"] = ids; out["attention_mask"] = att; out["labels"] = lab
        if has_mmtt: out["mm_token_type_ids"] = mmtt
        if pix: out["pixel_values"] = torch.cat(pix, dim=0)
        if grid: out["image_grid_thw"] = torch.cat(grid, dim=0)
        return out

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=args.lr, bf16=True, logging_steps=5,
        save_strategy="no", report_to=[], gradient_checkpointing=True, warmup_ratio=0.05,
        lr_scheduler_type="cosine", remove_unused_columns=False, dataloader_num_workers=2)
    trainer = Trainer(model=model, args=targs, train_dataset=dataset, data_collator=collate)
    trainer.train()
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"SAVED LoRA adapter -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
