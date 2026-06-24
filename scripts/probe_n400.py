#!/usr/bin/env python
"""WU-1 §1.4 — causal probe (2a counterfactual) at scale (n=400), BATCHED.

Same probe as ``poc_causal_probe_32b.py`` (corrupt one intermediate number vs shuffle the CoT
sentences, force the model to finish from the edited CoT, measure answer FLIP), but the base-CoT
generation and the force-continue generations are BATCHED (left-pad, like ``eval_sft_n400.py``)
so n=400 is tractable on the 3080s instead of the ~hours the per-case loop would take.

Identical decode (greedy), identical edit RNG order (Random(0), corrupt then shuffle per kept case
in input order) → results match the unbatched probe up to batching numerics. Supports --mask-image
(force-continue WITHOUT the chart, so the edited CoT is the only info source). Run in env `mbe-up`
via the env python directly (NOT `conda run`, whose arg parser eats --n).
"""
from __future__ import annotations

import argparse
import json
import os
import random
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


def extract_answer(text: str) -> str:
    m = _re.search(r"ANSWER:\s*(.+)", text, _re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip() if m.group(1).strip() else ""
    lines = (text or "").strip().splitlines()
    return lines[-1].strip() if lines else ""


def corrupt_number(cot, rng):
    nums = list(_re.finditer(r"-?\d+\.?\d*", cot))
    if not nums:
        return None
    m = rng.choice(nums)
    v = m.group(0)
    try:
        f = float(v); nv = str(int(f * 2 + 7)) if f == int(f) else f"{f*2+7:.1f}"
    except ValueError:
        return None
    return cot[:m.start()] + nv + cot[m.end():]


def shuffle_cot(cot, rng):
    sents = _re.split(r"(?<=[.\n])", cot)
    sents = [s for s in sents if s.strip()]
    rng.shuffle(sents)
    return "".join(sents)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    ap.add_argument("--dump", default="data/distill/chartqa/test_cases_400.jsonl")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/chartqa_test_images")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--cont-new", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mask-image", action="store_true")
    args = ap.parse_args()
    rng = random.Random(0)

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
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval(); model.config.use_cache = True
    dev0 = torch.device("cuda:0")
    print(f"loaded base+adapter ({args.quant}) in {time.time()-t0:.0f}s", flush=True)

    prog = Path("data/distill/poc/logs/probe_n400_progress.txt"); prog.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def gen_batch(items, max_new):
        """items: list of {content:[...]} chat messages. Returns list of decoded continuations."""
        encs = [processor.apply_chat_template([{"role": "user", "content": it}], tokenize=True,
                                              return_dict=True, add_generation_prompt=True,
                                              return_tensors="pt") for it in items]
        out_texts = []
        B = len(encs)
        maxlen = max(e["input_ids"].shape[1] for e in encs)
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
        g = model.generate(**batch, max_new_tokens=max_new, do_sample=False, pad_token_id=pad_id)
        for b in range(B):
            out_texts.append(tok.decode(g[b][maxlen:], skip_special_tokens=True))
        return out_texts

    def run_batched(items, max_new, tag):
        res = [None] * len(items)
        t_s = time.time()
        for i in range(0, len(items), args.batch_size):
            chunk = items[i:i + args.batch_size]
            outs = gen_batch([c["content"] for c in chunk], max_new)
            for j, o in enumerate(outs):
                res[i + j] = o
            with open(prog, "a") as pf:
                pf.write(f"{tag} {min(i+args.batch_size,len(items))}/{len(items)} "
                         f"elapsed={time.time()-t_s:.0f}s\n")
        return res

    # ---- load cases ----
    rows = [json.loads(l) for l in open(args.dump) if l.strip()][:args.n]
    cases = []
    for r in rows:
        cid = r["case_id"]; idx = cid.split("-")[-1]
        imgp = Path(args.img_dir) / f"chartqa_{idx}.png"
        if imgp.exists():
            cases.append({"cid": cid, "q": str(r["question"]), "gold": str(r["gold"]),
                          "img": Image.open(imgp).convert("RGB")})
    print(f"probe cases: {len(cases)} (mask_image={args.mask_image})", flush=True)

    # ---- Phase 1: batched base-CoT generation (image present) ----
    base_items = [{"content": [{"type": "image", "image": c["img"]},
                               {"type": "text", "text": USER_INSTR + c["q"]}]} for c in cases]
    base_out = run_batched(base_items, args.max_new, "base")

    # keep cases the model gets right WITH a CoT
    kept = []
    for c, out in zip(cases, base_out):
        base_cot = out.split("ANSWER:")[0].strip()
        base_ans = extract_answer(out)
        if base_cot and relaxed_match(base_ans, c["gold"]):
            kept.append({**c, "base_cot": base_cot, "base_ans": base_ans})
    print(f"n_eval (correct + CoT): {len(kept)}", flush=True)

    # ---- build edits (rng order = kept order; corrupt then shuffle per case) ----
    for k in kept:
        k["corrupt"] = corrupt_number(k["base_cot"], rng)
        k["shuffle"] = shuffle_cot(k["base_cot"], rng)

    def cont_item(c, edited):
        txt = {"type": "text", "text": f"Question: {c['q']}\n\nReasoning so far:\n{edited}\n\n"
               "Given ONLY the reasoning above, state the final answer now as 'ANSWER: <final>'."}
        content = [txt] if args.mask_image else [{"type": "image", "image": c["img"]}, txt]
        return {"content": content}

    # ---- Phase 2: batched force-continue for corrupt + shuffle ----
    corrupt_idx = [i for i, k in enumerate(kept) if k["corrupt"]]
    corrupt_items = [cont_item(kept[i], kept[i]["corrupt"]) for i in corrupt_idx]
    corrupt_out = run_batched(corrupt_items, args.cont_new, "corrupt") if corrupt_items else []
    shuffle_items = [cont_item(k, k["shuffle"]) for k in kept]
    shuffle_out = run_batched(shuffle_items, args.cont_new, "shuffle")

    flips_corrupt = flips_shuffle = 0
    details = []
    corrupt_map = {ci: o for ci, o in zip(corrupt_idx, corrupt_out)}
    for i, k in enumerate(kept):
        flip_c = False
        if i in corrupt_map:
            a_c = extract_answer(corrupt_map[i]); flip_c = not relaxed_match(a_c, k["base_ans"])
            flips_corrupt += int(flip_c)
        a_s = extract_answer(shuffle_out[i]); flip_s = not relaxed_match(a_s, k["base_ans"])
        flips_shuffle += int(flip_s)
        details.append({"case_id": k["cid"], "base_ans": k["base_ans"],
                        "flip_corrupt": bool(i in corrupt_map and flip_c), "flip_shuffle": bool(flip_s)})

    n_eval = len(kept)
    summary = {"n_eval": n_eval, "mask_image": bool(args.mask_image),
               "flip_rate_corrupt": flips_corrupt / n_eval if n_eval else 0.0,
               "flip_rate_shuffle": flips_shuffle / n_eval if n_eval else 0.0,
               "flips_corrupt": flips_corrupt, "flips_shuffle": flips_shuffle}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "details": details},
                                         ensure_ascii=False, indent=2))
    print("\n=== CAUSAL PROBE n=400 (batched) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
