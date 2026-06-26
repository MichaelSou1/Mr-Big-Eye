#!/usr/bin/env python
"""Full-SFT Qwen3-VL-8B on the existing (image, question, CoT, answer) records.

This mirrors ``poc_sft.py``'s prompt/loss construction, but trains the full text
model parameters instead of LoRA adapters. The vision tower stays frozen by
default, matching the original LoRA SFT setup. The default launch uses a single
Python process with ``device_map=auto`` so one job occupies the visible GPUs
without spawning one model copy per GPU:

  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/poc_sft_full_8b.py \
    --base /home/gpus/models/Qwen3-VL-8B-Instruct \
    --data data/distill/poc/chartqa_cot_train.jsonl \
    --out data/distill/poc/full_8b_chartqa

The script is deliberately single-run/single-checkpoint: disk is tight on the
local box, so orchestration should run one Full-SFT arm, evaluate/probe it, then
optionally remove the generated checkpoint before training the next arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/gpus/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-target-len", type=int, default=0,
                    help="Skip examples with assistant target length above this value when >0.")
    ap.add_argument("--loss-chunk-tokens", type=int, default=128,
                    help="Compute CE over assistant logits in token chunks to reduce peak VRAM; 0 disables.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-records", type=int, default=0,
                    help="Debug/smoke: keep only the first N records when >0.")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Debug/smoke: stop after this many optimizer steps when >0.")
    ap.add_argument("--optim", default="paged_adamw_8bit",
                    choices=["paged_adamw_8bit", "adamw_torch"])
    ap.add_argument("--strategy", choices=["device_map", "fsdp"], default="device_map",
                    help="device_map is the local 4x20GB default. fsdp is experimental.")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--cpu-offload", action="store_true",
                    help="Enable FSDP CPU offload. Safer on VRAM, much slower.")
    ap.add_argument("--gpu0-max-memory", default="",
                    help="device_map mode: cap cuda:0 memory, e.g. 17GiB, to leave room for activations/optimizer.")
    ap.add_argument("--gpu-max-memory", default="",
                    help="device_map mode: cap other CUDA devices, e.g. 19GiB.")
    ap.add_argument("--train-vision", action="store_true",
                    help="Also train the vision tower. Default freezes it to match LoRA SFT.")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="Freeze token embeddings. Useful when dense embedding grads exhaust cuda:0.")
    ap.add_argument("--freeze-first-layers", type=int, default=0,
                    help="Freeze the first N language decoder layers to reduce cuda:0 training memory.")
    ap.add_argument("--no-save", action="store_true",
                    help="Smoke mode: train but do not write the full checkpoint.")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tok = processor.tokenizer

    recs = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.max_records and args.max_records > 0:
        recs = recs[:args.max_records]
    if is_main_process():
        print(f"Full-SFT records: {len(recs)} from {args.data}", flush=True)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.strategy == "device_map":
        load_kwargs["device_map"] = "auto"
        if args.gpu0_max_memory or args.gpu_max_memory:
            max_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    value = args.gpu0_max_memory if i == 0 and args.gpu0_max_memory else args.gpu_max_memory
                    if value:
                        max_memory[i] = value
            if max_memory:
                load_kwargs["max_memory"] = max_memory
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.base, **load_kwargs)
    model.config.use_cache = False

    if not args.train_vision:
        for name, p in model.named_parameters():
            if "visual" in name or "vision" in name:
                p.requires_grad = False
    if args.freeze_embeddings:
        for name, p in model.named_parameters():
            if "embed_tokens" in name:
                p.requires_grad = False
    if args.freeze_first_layers and args.freeze_first_layers > 0:
        prefixes = [
            f"model.language_model.layers.{i}."
            for i in range(args.freeze_first_layers)
        ]
        for name, p in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                p.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        if args.freeze_embeddings:
            def make_inputs_require_grad(_module, _inputs, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        else:
            model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"trainable params: {trainable:,} / {total:,} ({trainable / total:.2%})", flush=True)
        print("GPU policy: one Full-SFT job at a time; no concurrent eval/probe.", flush=True)
        print(f"parallelism strategy: {args.strategy}", flush=True)
        if "max_memory" in load_kwargs:
            print(f"device_map max_memory: {load_kwargs['max_memory']}", flush=True)

    user_instr = "Solve step by step, end with 'ANSWER: <final>'.\n\nQuestion: "

    def build(rec):
        img = Image.open(rec["image_path"]).convert("RGB")
        target = f"{rec['cot']}\nANSWER: {rec['answer']}"
        user_msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": user_instr + rec["question"]},
        ]}]
        full_msg = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
        full = processor.apply_chat_template(
            full_msg, tokenize=True, return_dict=True,
            add_generation_prompt=False, return_tensors="pt")
        prompt = processor.apply_chat_template(
            user_msg, tokenize=True, return_dict=True,
            add_generation_prompt=True, return_tensors="pt")
        plen = prompt["input_ids"].shape[1]
        seq_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
        item = {k: (v[0] if k in seq_keys else v) for k, v in full.items()}
        labels = item["input_ids"].clone()
        labels[:plen] = -100
        item["labels"] = labels
        item["target_len"] = int((labels != -100).sum().item())
        return item

    dataset = []
    skipped = 0
    for rec in recs:
        try:
            item = build(rec)
            if args.max_target_len and item["target_len"] > args.max_target_len:
                skipped += 1
            elif item["input_ids"].shape[0] <= args.max_len:
                dataset.append(item)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            if is_main_process():
                print(f"build skip: {str(e)[:120]}", flush=True)
    if is_main_process():
        print(f"usable (<= {args.max_len} tok): {len(dataset)}; skipped={skipped}", flush=True)
    if not dataset:
        raise RuntimeError("No usable training records after preprocessing.")

    def collate(batch):
        maxlen = max(b["input_ids"].shape[0] for b in batch)
        ids = torch.full((len(batch), maxlen), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)
        lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        has_mmtt = "mm_token_type_ids" in batch[0]
        mmtt = torch.zeros((len(batch), maxlen), dtype=torch.long) if has_mmtt else None
        pix, grid = [], []
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            ids[i, :L] = b["input_ids"]
            att[i, :L] = 1
            lab[i, :L] = b["labels"]
            if has_mmtt:
                mmtt[i, :L] = b["mm_token_type_ids"]
            if "pixel_values" in b:
                pix.append(b["pixel_values"])
            if "image_grid_thw" in b:
                grid.append(b["image_grid_thw"])
        out = {"input_ids": ids, "attention_mask": att, "labels": lab}
        if has_mmtt:
            out["mm_token_type_ids"] = mmtt
        if pix:
            out["pixel_values"] = torch.cat(pix, dim=0)
        if grid:
            out["image_grid_thw"] = torch.cat(grid, dim=0)
        out["target_len"] = torch.tensor([int(b["target_len"]) for b in batch], dtype=torch.long)
        return out

    def tensor_to_device(batch, device):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def first_cuda_device():
        for p in model.parameters():
            if p.device.type == "cuda":
                return p.device
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def manual_device_map_train():
        """Small-data full-SFT loop for model-parallel ``device_map=auto``.

        ``Trainer`` wraps outputs and converts them to fp32; for Qwen3-VL this can
        allocate an extra full-sequence logits tensor. Here we request only the
        assistant tail logits and compute the CausalLM loss ourselves.
        """
        import random
        try:
            import bitsandbytes as bnb
        except Exception:
            bnb = None

        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        params = [p for p in model.parameters() if p.requires_grad]
        if args.optim == "paged_adamw_8bit":
            if bnb is None:
                raise RuntimeError("bitsandbytes is required for --optim paged_adamw_8bit")
            opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr)
        else:
            opt = torch.optim.AdamW(params, lr=args.lr)

        micro_per_epoch = len(dataset)
        updates_per_epoch = (micro_per_epoch + args.grad_accum - 1) // args.grad_accum
        total_updates = int(args.max_steps) if args.max_steps and args.max_steps > 0 \
            else max(1, int(args.epochs * updates_per_epoch))
        warmup = int(total_updates * args.warmup_ratio)

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_updates - warmup))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        input_device = first_cuda_device()
        opt.zero_grad(set_to_none=True)
        global_step = 0
        micro = 0
        running = 0.0
        epoch = 0
        rng = random.Random(0)

        if is_main_process():
            print(f"manual loop: total_updates={total_updates} grad_accum={args.grad_accum} "
                  f"input_device={input_device}", flush=True)

        while global_step < total_updates:
            epoch += 1
            order = list(range(len(dataset)))
            rng.shuffle(order)
            for idx in order:
                item = dataset[idx]
                batch = collate([item])
                target_len = int(batch.pop("target_len")[0].item())
                labels = batch.pop("labels")
                batch = tensor_to_device(batch, input_device)
                # Need one extra hidden position: position before the first assistant token
                # predicts that first assistant token.
                logits_to_keep = max(1, target_len + 1)
                out = model(**batch, logits_to_keep=logits_to_keep)
                logits = out.logits[:, :-1, :]
                target = labels[labels != -100].to(logits.device)
                if logits.shape[1] != target.numel():
                    logits = logits[:, -target.numel():, :]
                if args.loss_chunk_tokens and args.loss_chunk_tokens > 0:
                    total_loss = None
                    total_tokens = 0
                    for start in range(0, target.numel(), args.loss_chunk_tokens):
                        end = min(target.numel(), start + args.loss_chunk_tokens)
                        chunk_logits = logits[:, start:end, :].reshape(-1, logits.shape[-1])
                        chunk_target = target[start:end].reshape(-1)
                        chunk_loss = F.cross_entropy(
                            chunk_logits.float(), chunk_target, reduction="sum")
                        total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss
                        total_tokens += int(chunk_target.numel())
                    loss = total_loss / max(1, total_tokens)
                else:
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1))
                (loss / args.grad_accum).backward()
                running += float(loss.detach().cpu())
                micro += 1
                if micro % args.grad_accum == 0 or idx == order[-1]:
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                    if is_main_process() and (global_step % args.logging_steps == 0 or global_step == 1):
                        avg = running / max(1, micro)
                        lr = sched.get_last_lr()[0]
                        print(json.dumps({"step": global_step, "total_steps": total_updates,
                                          "epoch": epoch, "loss": round(avg, 6),
                                          "lr": lr}, ensure_ascii=False), flush=True)
                    if global_step >= total_updates:
                        break
        metrics = {"train_loss": running / max(1, micro),
                   "train_micro_steps": micro, "train_steps": global_step, "epoch": epoch}
        return metrics

    targs_kwargs = {
        "output_dir": args.out,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "save_total_limit": 1,
        "save_only_model": True,
        "report_to": [],
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "remove_unused_columns": False,
        "dataloader_num_workers": args.workers,
        "optim": args.optim,
    }
    fsdp_opts = None
    if args.strategy == "fsdp":
        fsdp_opts = "full_shard auto_wrap"
        if args.cpu_offload:
            fsdp_opts += " offload"
        targs_kwargs.update({
            "fsdp": fsdp_opts,
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": ["Qwen3VLTextDecoderLayer"],
                "use_orig_params": True,
                "activation_checkpointing": True,
            },
        })
    else:
        targs_kwargs.update({
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        })
    targs = TrainingArguments(**targs_kwargs)
    if args.strategy == "device_map":
        train_metrics = manual_device_map_train()
        trainer = None
    else:
        trainer = Trainer(model=model, args=targs, train_dataset=dataset, data_collator=collate)
        train_result = trainer.train()
        train_metrics = train_result.metrics

    if not args.no_save:
        if trainer is not None:
            trainer.save_model(args.out)
        else:
            model.save_pretrained(args.out)
        if is_main_process():
            processor.save_pretrained(args.out)
            summary = {
                "base": args.base,
                "data": args.data,
                "out": args.out,
                "epochs": args.epochs,
                "lr": args.lr,
                "max_len": args.max_len,
                "max_target_len": args.max_target_len,
                "loss_chunk_tokens": args.loss_chunk_tokens,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "optim": args.optim,
                "strategy": args.strategy,
                "fsdp": fsdp_opts,
                "train_vision": bool(args.train_vision),
                "freeze_embeddings": bool(args.freeze_embeddings),
                "freeze_first_layers": args.freeze_first_layers,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": trainable / total if total else 0.0,
                "n_records": len(recs),
                "n_usable": len(dataset),
                "train_metrics": train_metrics,
            }
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "full_sft_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"SAVED Full-SFT checkpoint -> {args.out}", flush=True)
    elif is_main_process():
        print("NO-SAVE smoke run completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
