#!/usr/bin/env python
"""WU-1 §1.3 — re-evaluate the EXISTING ChartQA SFT adapters on the scaled n=400 held-out set.

The 150-CoT train set is fixed, so the adapters are unchanged — only eval-n grows (isolating the
power effect). This loads the NF4/bf16 base once, evaluates the base (adapter DISABLED) and each
supplied adapter checkpoint with the same batched greedy single-forward eval as
``poc_sft_32b_qlora.py``, and reports per-checkpoint Δacc with a PAIRED bootstrap CI (base vs
adapter on identical cases) AND McNemar p on the discordant pairs. Selects the peak-test-acc
checkpoint. Writes a JSON next to the adapters; does NOT touch the result store or the original
train_summary.json.

Works for both cells:
  32B:  --base /home/gpus/models/Qwen3-VL-32B-Instruct --quant nf4 \
        --adapters data/distill/poc/lora_32b_chartqa/epoch_{1..5}
  8B :  --base /home/gpus/models/Qwen3-VL-8B-Instruct  --quant none \
        --adapters data/distill/poc/lora_8b_chartqa
Run in env `mbe-up`.
"""
from __future__ import annotations

import argparse
import json
import os
import re as _re
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

from app.distill.eval_stats import mcnemar, paired_bootstrap_net  # numpy-only, no langchain chain


def normalize_text(value) -> str:
    text = str(value or "").lower()
    text = _re.sub(r"[^a-z0-9一-鿿]+", " ", text)
    return _re.sub(r"\s+", " ", text).strip()


def relaxed_match(pred: str, gold: str) -> bool:
    g = str(gold).strip()
    nums = _re.findall(r"-?\d+\.?\d*", str(pred).replace(",", ""))
    try:
        gv = float(g.replace(",", "").replace("%", ""))
        return any(abs(float(p) - gv) <= abs(gv) * 0.05 + 1e-6 for p in nums)
    except ValueError:
        gn = normalize_text(g)
        return bool(gn) and gn in normalize_text(pred)


def parse_answer(text: str) -> str:
    m = _re.search(r"ANSWER:\s*(.+)", text, _re.IGNORECASE | _re.DOTALL)
    if m:
        lines = m.group(1).strip().splitlines()
        return lines[0].strip() if lines else ""
    lines = (text or "").strip().splitlines()
    return lines[-1].strip() if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="adapter dirs (epoch_1..5 or one)")
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    ap.add_argument("--test-dump", default="data/distill/chartqa/test_cases_400.jsonl")
    ap.add_argument("--test-img-dir", default="/home/gpus/mbe_data/chartqa_test_images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="sft")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=320)
    args = ap.parse_args()

    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    USER_INSTR = "Solve step by step, end with 'ANSWER: <final>'.\n\nQuestion: "
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tok = processor.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    quant_cfg = None
    if args.quant == "nf4":
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, quantization_config=quant_cfg, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.config.use_cache = True
    # attach all adapters under distinct names
    names = []
    for i, ad in enumerate(args.adapters):
        name = f"ad{i}"
        if i == 0:
            model = PeftModel.from_pretrained(model, ad, adapter_name=name)
        else:
            model.load_adapter(ad, adapter_name=name)
        names.append((name, ad))
    model.eval()
    dev0 = torch.device("cuda:0")
    print(f"loaded base+{len(names)} adapters ({args.quant}) in {time.time()-t0:.0f}s", flush=True)

    # ---- eval cases ----
    rows = [json.loads(l) for l in open(args.test_dump) if l.strip()]
    eval_cases = []
    for r in rows:
        cid = r.get("case_id", ""); idx = cid.rsplit("-", 1)[-1]
        prefix = cid.rsplit("-", 1)[0] or "chartqa"
        img_p = Path(args.test_img_dir) / f"{prefix}_{idx}.png"
        if img_p.exists():
            eval_cases.append({"cid": cid, "question": str(r.get("question") or ""),
                               "gold": str(r.get("gold") or ""), "img": str(img_p)})
    print(f"eval cases: {len(eval_cases)}", flush=True)
    prog = Path("data/distill/poc/logs/eval_n400_progress.txt"); prog.parent.mkdir(parents=True, exist_ok=True)

    def _encode(c):
        img = Image.open(c["img"]).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": USER_INSTR + c["question"]}]}]
        return processor.apply_chat_template(msg, tokenize=True, return_dict=True,
                                             add_generation_prompt=True, return_tensors="pt")

    @torch.no_grad()
    def run_eval(tag, bs):
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
                ids[b, maxlen - L:] = e["input_ids"][0]
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
                text = tok.decode(gen[b][maxlen:], skip_special_tokens=True)
                results.append((c["cid"], parse_answer(text), bool(relaxed_match(parse_answer(text), c["gold"]))))
            with open(prog, "a") as pf:
                pf.write(f"{tag} {len(results)}/{len(eval_cases)} "
                         f"acc={sum(r[2] for r in results)}/{len(results)} "
                         f"elapsed={time.time()-t_start:.0f}s\n")
        return results

    # ---- base (adapter disabled) ----
    print("=== eval BASE (adapter disabled) ===", flush=True)
    with model.disable_adapter():
        base_res = run_eval("base", args.batch_size)
    base_correct = {cid: ok for cid, _, ok in base_res}
    base_acc = sum(base_correct.values()) / len(base_res)
    print(f"BASE acc = {base_acc:.3f} ({sum(base_correct.values())}/{len(base_res)})", flush=True)

    # ---- each adapter ----
    per = []
    for name, ad in names:
        model.set_adapter(name)
        t0 = time.time()
        res = run_eval(name, args.batch_size)
        sft_correct = {cid: ok for cid, _, ok in res}
        cids = [cid for cid, _, _ in res]
        b_ok = [int(base_correct[c]) for c in cids]
        s_ok = [int(sft_correct[c]) for c in cids]
        boot = paired_bootstrap_net(b_ok, s_ok)
        mc = mcnemar(boot["gain"], boot["lost"])
        sft_acc = sum(s_ok) / len(s_ok)
        rec = {"adapter": ad, "test_acc": sft_acc, "base_acc": base_acc,
               "net": sft_acc - base_acc, "gain": boot["gain"], "lost": boot["lost"],
               "boot_ci": [boot["ci_lo"], boot["ci_hi"]], "boot_net": boot["net"],
               "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_p": mc["p"], "n": len(s_ok)}
        per.append(rec)
        print(f"[{Path(ad).name}] test_acc={sft_acc:.3f} net={rec['net']:+.3f} "
              f"CI[{boot['ci_lo']:+.3f},{boot['ci_hi']:+.3f}] gain={boot['gain']} lost={boot['lost']} "
              f"McNemar b={mc['b']} c={mc['c']} p={mc['p']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(Path(args.out).with_suffix(f".{Path(ad).name}.preds.jsonl"), "w") as fh:
            for cid, pred, ok in res:
                fh.write(json.dumps({"cid": cid, "pred": pred, "correct": ok,
                                     "base_correct": base_correct[cid]}) + "\n")

    best = max(per, key=lambda h: (h["test_acc"], )) if per else None
    out = {"tag": args.tag, "base": args.base, "quant": args.quant, "n_eval": len(eval_cases),
           "base_acc": base_acc, "per_adapter": per,
           "best_adapter": best["adapter"] if best else None,
           "best_test_acc": best["test_acc"] if best else None,
           "best_net": best["net"] if best else None,
           "best_mcnemar_p": best["mcnemar_p"] if best else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n=== SFT n=400 SUMMARY ===")
    print(f"base_acc={base_acc:.3f}")
    for h in per:
        mark = "  <-- PEAK" if best and h["adapter"] == best["adapter"] else ""
        print(f"  {Path(h['adapter']).name}: test_acc={h['test_acc']:.3f} net={h['net']:+.3f} "
              f"gain={h['gain']} lost={h['lost']} McNemar p={h['mcnemar_p']:.4f}{mark}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
