#!/usr/bin/env python
"""Phase C: LoRA SFT of the orchestrator on captured tool-call trajectories.

Self-contained transformers + peft trainer that consumes the Phase B sharegpt/
OpenAI dataset ({"messages":[...], "tools":[...]}) and trains with the model's
own chat template (Qwen). Loss is computed ONLY on the final assistant message
(completion-only masking) — the prefix/system/tool turns are masked, realizing
spec §5.7/§6's "only the last assistant gets loss".

Why not LLaMA-Factory here: the orchestrator emits *parallel* tool_calls in one
message, which the bundled LLaMA-Factory-main glaive format (one function_call
per turn) cannot represent. apply_chat_template handles it natively.

Runs in an env with a CUDA-enabled torch + transformers + peft (e.g. vlm_dapo).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_tool_calls(messages: list[dict]) -> list[dict]:
    """Ensure assistant tool_call arguments are dicts (chat templates json-encode
    them); our Phase B output stores arguments as a JSON string."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        fn["arguments"] = {}
                calls.append({**tc, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def build_example(tokenizer, sample: dict, cutoff_len: int) -> dict | None:
    """Tokenize one sample, masking everything but the final assistant message.

    Renders to text first (apply_chat_template returns a BatchEncoding dict under
    tokenize=True in transformers 5.x, so we tokenize the rendered string instead)
    and masks the prompt prefix so loss falls only on the final assistant turn.
    """
    messages = _normalize_tool_calls(sample["messages"])
    tools = sample.get("tools") or None
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tools=tools, add_generation_prompt=True, tokenize=False
        )
        full_text = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=False, tokenize=False
        )
    except Exception:  # noqa: BLE001
        return None
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return None
    if len(full_ids) > cutoff_len:
        return None  # never truncate the target; drop overlong samples
    # The generation-prompt text is a prefix of the full text for chat templates
    # like Qwen; clamp to the common length so the mask is never out of range.
    n_prompt = len(prompt_ids)
    if full_ids[:n_prompt] != prompt_ids:
        common = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            common += 1
        n_prompt = common
    labels = [-100] * n_prompt + list(full_ids[n_prompt:])
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch

        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path (e.g. Qwen2.5-7B-Instruct).")
    parser.add_argument("--train", default="data/distillation/train.jsonl")
    parser.add_argument("--val", default="data/distillation/val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs for a smoke run.")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--cutoff-len", type=int, default=6144)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="QLoRA: load the base model in 4-bit nf4 (fits 7B LoRA on a 20GB card).",
    )
    parser.add_argument(
        "--use-liger",
        action="store_true",
        help="Use Liger fused linear cross-entropy (avoids materializing full-vocab "
        "fp32 logits — needed for long sequences on a 20GB card).",
    )
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable eval. Eval runs non-fused CE (Liger only fuses in train mode), "
        "which OOMs on long val samples; train loss + the held-out comparison suffice.",
    )
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Log loss/lr/grad_norm/config to Weights & Biases.")
    parser.add_argument("--wandb-project", default="mbe-distill")
    parser.add_argument("--run-name", default=None, help="W&B run name (also the TrainingArguments run_name).")
    args = parser.parse_args()

    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = _read_jsonl(Path(args.train))
    val_rows = _read_jsonl(Path(args.val)) if Path(args.val).exists() else []

    def encode(rows):
        out = []
        for r in rows:
            ex = build_example(tokenizer, r, args.cutoff_len)
            if ex is not None:
                out.append(ex)
        return out

    train_ds = encode(train_rows)
    val_ds = [] if args.no_eval else encode(val_rows)
    print(f"encoded train={len(train_ds)}/{len(train_rows)} val={len(val_ds)}/{len(val_rows)} "
          f"(dropped overlong/invalid)")
    if not train_ds:
        print("ERROR: no trainable samples after encoding.", file=sys.stderr)
        return 2
    lens = sorted(len(e["input_ids"]) for e in train_ds)
    print(f"train seq_len: max={lens[-1]} p50={lens[len(lens)//2]} p95={lens[int(0.95*len(lens))-1]}")

    quant_config = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _load(attn: str):
        kwargs: dict[str, Any] = {"attn_implementation": attn, "trust_remote_code": True}
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        else:
            kwargs["dtype"] = torch.bfloat16
        return AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

    try:
        model = _load("flash_attention_2")
    except Exception as exc:  # noqa: BLE001
        print(f"flash_attention_2 unavailable ({exc}); using sdpa")
        model = _load("sdpa")
    model.config.use_cache = False

    if args.load_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        trainer_gc = False  # already enabled by prepare_model_for_kbit_training
    else:
        model.enable_input_require_grads()
        trainer_gc = True

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,  # eval logits are materialized per batch; keep tiny
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=trainer_gc,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.save_steps if val_ds else None,
        # Eval must not gather full-vocab logits (OOM at long seq); we only track
        # eval loss. Liger falls back to non-fused CE in eval mode, so per-sample
        # logits are still transient-only with this on.
        prediction_loss_only=True,
        report_to=(["wandb"] if args.wandb else []),
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        use_liger_kernel=args.use_liger,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds or None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
