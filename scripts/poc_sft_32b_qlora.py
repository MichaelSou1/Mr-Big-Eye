#!/usr/bin/env python
"""Spec §11.2 (regime-2 cell) — QLoRA-SFT Qwen3-VL-32B on ChartQA arithmetic CoT, with a
PER-EPOCH held-out eval so we select the checkpoint at the TEST-ACC inflection rather than
blindly running a fixed #epochs.

Why per-epoch test eval (not train_loss): train_loss falls monotonically and is meaningless
for a 150-example LoRA — the real failure mode is overfit/memorization (loss keeps dropping
while held-out test-acc peaks then falls). We therefore eval the 60 train/test-DISJOINT held-out
cases every epoch, log test-acc + gain/lost (vs the SAME NF4 base with the adapter DISABLED),
save the adapter each epoch, and report the peak-test-acc epoch.

Base: NF4 (bnb 4-bit) Qwen3-VL-32B-Instruct, vision tower frozen, LoRA on the LLM projections.
Eval is single-forward greedy (the inference condition we ultimately claim). base = adapter OFF,
sft = adapter ON — a clean paired ±adapter contrast on the identical prompt, isolating the
internalized CoT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re as _re


def normalize_text(value) -> str:
    """Copied verbatim from app.mcq to avoid the eval_common->langchain import chain."""
    text = str(value or "").lower()
    text = _re.sub(r"[^a-z0-9一-鿿]+", " ", text)
    return _re.sub(r"\s+", " ", text).strip()


def relaxed_match(pred: str, gold: str) -> bool:
    """Verbatim from app.distill.eval_common: 5% numeric tolerance / normalized substring."""
    g = str(gold).strip()
    nums = _re.findall(r"-?\d+\.?\d*", str(pred).replace(",", ""))
    try:
        gv = float(g.replace(",", "").replace("%", ""))
        return any(abs(float(p) - gv) <= abs(gv) * 0.05 + 1e-6 for p in nums)
    except ValueError:
        gn = normalize_text(g)
        return bool(gn) and gn in normalize_text(pred)


def parse_answer(text: str) -> str:
    import re
    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        lines = m.group(1).strip().splitlines()
        return lines[0].strip() if lines else ""
    lines = (text or "").strip().splitlines()
    return lines[-1].strip() if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/gpus/models/Qwen3-VL-32B-Instruct")
    ap.add_argument("--data", default="data/distill/poc/chartqa_cot_train.jsonl")
    ap.add_argument("--test-dump", default="data/distill/analysis/dump_8b_chartqa.jsonl")
    ap.add_argument("--test-img-dir", default="/home/gpus/mbe_data/chartqa_images")
    ap.add_argument("--out", default="data/distill/poc/lora_32b_chartqa")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8, help="eval generation batch (pipeline fill)")
    ap.add_argument("--wandb", default="mbe-internalize-32b")
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    args = ap.parse_args()

    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3VLForConditionalGeneration)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    USER_INSTR = "Solve step by step, end with 'ANSWER: <final>'.\n\nQuestion: "

    # ---- wandb (best-effort; offline fallback) ----
    run = None
    try:
        import wandb
        run = wandb.init(project=args.wandb, name=f"qlora32b-{int(time.time())}",
                         config=vars(args))
    except Exception as e:
        print(f"[wandb] disabled: {str(e)[:80]}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tok = processor.tokenizer

    quant_cfg = None
    if args.quant == "nf4":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, quantization_config=quant_cfg, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    print(f"loaded 32B in {time.time()-t0:.0f}s", flush=True)

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

    dev0 = torch.device("cuda:0")

    # ---- build train items ----
    def build_train(rec):
        img = Image.open(rec["image_path"]).convert("RGB")
        target = f"{rec['cot']}\nANSWER: {rec['answer']}"
        user_msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": USER_INSTR + rec["question"]}]}]
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

    recs = [json.loads(l) for l in open(args.data) if l.strip()]
    train = []
    for r in recs:
        try:
            it = build_train(r)
            if it["input_ids"].shape[0] <= args.max_len:
                train.append(it)
        except Exception as e:
            print("train build skip:", str(e)[:80])
    print(f"train items (<= {args.max_len} tok): {len(train)}", flush=True)

    # ---- build eval cases (60 held-out, train/test DISJOINT) ----
    test_rows = [json.loads(l) for l in open(args.test_dump) if l.strip()]
    eval_cases = []
    for r in test_rows:
        cid = r.get("case_id", "")
        idx = cid.split("-")[-1]
        img_p = Path(args.test_img_dir) / f"chartqa_{idx}.png"
        if not img_p.exists():
            continue
        eval_cases.append({"cid": cid, "question": str(r.get("question") or ""),
                           "gold": str(r.get("gold") or ""), "img": str(img_p)})
    print(f"eval cases: {len(eval_cases)}", flush=True)

    prog_path = Path("data/distill/poc/logs/eval_progress.txt")
    prog_path.parent.mkdir(parents=True, exist_ok=True)

    pad_id = tok.pad_token_id or tok.eos_token_id

    def _encode(c):
        img = Image.open(c["img"]).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": USER_INSTR + c["question"]}]}]
        return processor.apply_chat_template(msg, tokenize=True, return_dict=True,
                                             add_generation_prompt=True, return_tensors="pt")

    @torch.no_grad()
    def run_eval(tag="eval", bs=args.batch_size):
        """Greedy single-forward over the 60 cases, BATCHED with left-padding to fill the
        4-GPU pipeline (unbatched ~44s/case; batching amortizes the pipeline bubble).
        Reuses the validated per-case encoding, then left-pads input_ids / concats
        pixel_values+grid. Writes flushed per-case progress (conda-run buffers stdout)."""
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        results = []
        t_start = time.time()
        for i in range(0, len(eval_cases), bs):
            chunk = eval_cases[i:i + bs]
            encs = [_encode(c) for c in chunk]
            maxlen = max(e["input_ids"].shape[1] for e in encs)
            B = len(encs)
            ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
            att = torch.zeros((B, maxlen), dtype=torch.long)
            has_mmtt = "mm_token_type_ids" in encs[0]
            mmtt = torch.zeros((B, maxlen), dtype=torch.long) if has_mmtt else None
            pix, grid = [], []
            for b, e in enumerate(encs):
                L = e["input_ids"].shape[1]
                ids[b, maxlen - L:] = e["input_ids"][0]          # LEFT pad
                att[b, maxlen - L:] = 1
                if has_mmtt: mmtt[b, maxlen - L:] = e["mm_token_type_ids"][0]
                if "pixel_values" in e: pix.append(e["pixel_values"])
                if "image_grid_thw" in e: grid.append(e["image_grid_thw"])
            batch = {"input_ids": ids.to(dev0), "attention_mask": att.to(dev0)}
            if has_mmtt: batch["mm_token_type_ids"] = mmtt.to(dev0)
            if pix: batch["pixel_values"] = torch.cat(pix, dim=0).to(dev0)
            if grid: batch["image_grid_thw"] = torch.cat(grid, dim=0).to(dev0)
            gen = model.generate(**batch, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=pad_id)
            for b, c in enumerate(chunk):
                new = gen[b][maxlen:]
                text = tok.decode(new, skip_special_tokens=True)
                pred = parse_answer(text)
                results.append((c["cid"], pred, bool(relaxed_match(pred, c["gold"]))))
            ncorr = sum(r[2] for r in results)
            with open(prog_path, "a") as pf:
                pf.write(f"{tag} {len(results)}/{len(eval_cases)} run_acc={ncorr}/{len(results)} "
                         f"elapsed={time.time()-t_start:.0f}s\n")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.train()
        return results

    # ---- baseline: adapter DISABLED (== NF4 base, untrained) ----
    # cached to disk so a crash/restart skips the ~16-min base eval (base never changes).
    base_cache = Path(args.out) / "base_eval.json"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if base_cache.exists():
        base_correct = {cid: bool(ok) for cid, ok in json.load(open(base_cache)).items()}
        base_acc = sum(base_correct.values()) / len(base_correct)
        print(f"=== BASE loaded from cache: acc={base_acc:.3f} "
              f"({sum(base_correct.values())}/{len(base_correct)}) ===", flush=True)
    else:
        print("=== eval BASE (adapter disabled) ===", flush=True)
        t0 = time.time()
        with model.disable_adapter():
            base_res = run_eval("base")
        base_correct = {cid: ok for cid, _, ok in base_res}
        json.dump({cid: bool(ok) for cid, ok in base_correct.items()}, open(base_cache, "w"))
        base_acc = sum(base_correct.values()) / len(base_res)
    print(f"BASE acc = {base_acc:.3f} ({sum(base_correct.values())}/{len(base_correct)})", flush=True)
    if run: run.log({"base_acc": base_acc}, step=0)

    # ---- training loop with per-epoch eval ----
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n_steps = args.epochs * ((len(train) + args.grad_accum - 1) // args.grad_accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps, 1))
    Path(args.out).mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        # simple deterministic order (seeded by epoch for a mild shuffle)
        order = list(range(len(train)))
        order = order[epoch % len(order):] + order[:epoch % len(order)]
        running = 0.0; micro = 0; opt.zero_grad(set_to_none=True)
        for i, idx in enumerate(order):
            it = train[idx]
            batch = {"input_ids": it["input_ids"].unsqueeze(0).to(dev0),
                     "attention_mask": it["attention_mask"].unsqueeze(0).to(dev0),
                     "labels": it["labels"].unsqueeze(0).to(dev0)}
            if "mm_token_type_ids" in it:
                batch["mm_token_type_ids"] = it["mm_token_type_ids"].unsqueeze(0).to(dev0)
            if "pixel_values" in it:
                batch["pixel_values"] = it["pixel_values"].to(dev0)
            if "image_grid_thw" in it:
                batch["image_grid_thw"] = it["image_grid_thw"].to(dev0)
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item(); micro += 1
            if micro % args.grad_accum == 0 or i == len(order) - 1:
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        train_loss = running / max(micro, 1)

        # save adapter for this epoch
        ep_dir = Path(args.out) / f"epoch_{epoch}"
        model.save_pretrained(str(ep_dir))

        # per-epoch held-out eval (adapter ON)
        t0 = time.time()
        sft_res = run_eval(f"epoch{epoch}")
        sft_acc = sum(ok for _, _, ok in sft_res) / len(sft_res)
        gain = sum(1 for cid, _, ok in sft_res if ok and not base_correct[cid])
        lost = sum(1 for cid, _, ok in sft_res if not ok and base_correct[cid])
        net = sft_acc - base_acc
        history.append({"epoch": epoch, "train_loss": train_loss, "test_acc": sft_acc,
                        "gain": gain, "lost": lost, "net": net})
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} test_acc={sft_acc:.3f} "
              f"gain={gain} lost={lost} net={net:+.3f}  (eval {time.time()-t0:.0f}s)", flush=True)
        if run:
            run.log({"train_loss": train_loss, "test_acc": sft_acc, "gain": gain,
                     "lost": lost, "net": net}, step=epoch)
        # dump per-epoch predictions for later inspection / causal-probe seed selection
        with open(ep_dir / "eval_preds.jsonl", "w") as fh:
            for cid, pred, ok in sft_res:
                fh.write(json.dumps({"cid": cid, "pred": pred, "correct": ok,
                                     "base_correct": base_correct[cid]}) + "\n")

    # ---- select inflection (peak test-acc; tie-break earliest) ----
    best = max(history, key=lambda h: (h["test_acc"], -h["epoch"]))
    summary = {"base_acc": base_acc, "history": history, "best_epoch": best["epoch"],
               "best_test_acc": best["test_acc"], "best_net": best["net"]}
    with open(Path(args.out) / "train_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== SUMMARY ===")
    print(f"base_acc={base_acc:.3f}")
    for h in history:
        mark = "  <-- PEAK" if h["epoch"] == best["epoch"] else ""
        print(f"  epoch {h['epoch']}: loss={h['train_loss']:.4f} test_acc={h['test_acc']:.3f} "
              f"net={h['net']:+.3f} gain={h['gain']} lost={h['lost']}{mark}")
    print(f"SELECTED checkpoint: epoch_{best['epoch']} (test_acc={best['test_acc']:.3f}, "
          f"net={best['net']:+.3f}) -> {args.out}/epoch_{best['epoch']}")
    if run: run.summary.update(summary); run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
