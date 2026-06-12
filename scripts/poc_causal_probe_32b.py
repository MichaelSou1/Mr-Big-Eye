#!/usr/bin/env python
"""Spec §11.3 — causal probe (2a counterfactual) for the QLoRA-32B ChartQA SFT, IN-PROCESS
(NF4 base + LoRA adapter; the merged-bf16-served path is impossible here — 64GB disk + vLLM
can't host bnb+peft). On test cases the SFT answers correctly WITH a CoT: take the model's own
CoT, (a) corrupt ONE intermediate numeric value, (b) shuffle the CoT sentences (control), force
the model to finish from the edited CoT, and measure the answer FLIP rate.

Load-bearing reasoning => corrupting an intermediate flips the answer at a HIGHER rate than
shuffling; perception-transfer / post-hoc CoT (the 8B regime-1 result: 3% corrupt-flip) flips
little because the model just re-reads the chart. The regime-2 (32B) hypothesis is a markedly
higher corrupt-flip than the 8B's 3%.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
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
    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip() if m.group(1).strip() else ""
    lines = (text or "").strip().splitlines()
    return lines[-1].strip() if lines else ""


def corrupt_number(cot, rng):
    nums = list(re.finditer(r"-?\d+\.?\d*", cot))
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
    sents = re.split(r"(?<=[.\n])", cot)
    sents = [s for s in sents if s.strip()]
    rng.shuffle(sents)
    return "".join(sents)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/gpus/models/Qwen3-VL-32B-Instruct")
    ap.add_argument("--adapter", default="data/distill/poc/lora_32b_chartqa/epoch_1")
    ap.add_argument("--dump", default="data/distill/analysis/dump_8b_chartqa.jsonl")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/chartqa_images")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="data/distill/poc/causal_probe_32b.json")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--mask-image", action="store_true",
                    help="control: force_continue WITHOUT the chart, so the (edited) CoT is the "
                         "only info source — disambiguates 'CoT non-load-bearing' vs 're-read shortcut'")
    args = ap.parse_args()
    rng = random.Random(0)

    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3VLForConditionalGeneration)
    from peft import PeftModel

    USER_INSTR = "Solve step by step, end with 'ANSWER: <final>'.\n\nQuestion: "
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tok = processor.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, quantization_config=quant_cfg, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    dev0 = torch.device("cuda:0")
    print(f"loaded 32B+adapter in {time.time()-t0:.0f}s", flush=True)

    @torch.no_grad()
    def gen(content_msg, max_new):
        enc = processor.apply_chat_template([{"role": "user", "content": content_msg}],
                                            tokenize=True, return_dict=True,
                                            add_generation_prompt=True, return_tensors="pt")
        enc = {k: v.to(dev0) for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=pad_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def force_continue(img, q, edited_cot):
        txt = {"type": "text", "text": f"Question: {q}\n\nReasoning so far:\n{edited_cot}\n\n"
               "Given ONLY the reasoning above, state the final answer now as 'ANSWER: <final>'."}
        msg = [txt] if args.mask_image else [{"type": "image", "image": img}, txt]
        return extract_answer(gen(msg, 64))

    rows = [json.loads(l) for l in open(args.dump) if l.strip()][:args.n]
    prog = Path("data/distill/poc/logs/probe_progress.txt"); prog.parent.mkdir(parents=True, exist_ok=True)
    flips_corrupt, flips_shuffle, n_eval = 0, 0, 0
    details = []
    t_start = time.time()
    for k, r in enumerate(rows):
        cid = r["case_id"]; idx = cid.split("-")[-1]
        imgp = Path(args.img_dir) / f"chartqa_{idx}.png"
        if not imgp.exists():
            continue
        q, gold = str(r["question"]), str(r["gold"])
        img = Image.open(imgp).convert("RGB")
        base_out = gen([{"type": "image", "image": img},
                        {"type": "text", "text": USER_INSTR + q}], args.max_new)
        base_cot = base_out.split("ANSWER:")[0].strip()
        base_ans = extract_answer(base_out)
        if not relaxed_match(base_ans, gold) or not base_cot:
            with open(prog, "a") as pf:
                pf.write(f"{k+1}/{len(rows)} {cid} SKIP (wrong/no-cot) n_eval={n_eval} "
                         f"elapsed={time.time()-t_start:.0f}s\n")
            continue
        n_eval += 1
        cc = corrupt_number(base_cot, rng); flip_c = False
        if cc:
            a_c = force_continue(img, q, cc)
            flip_c = not relaxed_match(a_c, base_ans)
            flips_corrupt += int(flip_c)
        sc = shuffle_cot(base_cot, rng)
        a_s = force_continue(img, q, sc)
        flip_s = not relaxed_match(a_s, base_ans)
        flips_shuffle += int(flip_s)
        details.append({"case_id": cid, "base_ans": base_ans,
                        "flip_corrupt": bool(cc and flip_c), "flip_shuffle": bool(flip_s)})
        with open(prog, "a") as pf:
            pf.write(f"{k+1}/{len(rows)} {cid} corrupt_flip={cc and flip_c} shuffle_flip={flip_s} "
                     f"n_eval={n_eval} corrupt={flips_corrupt} shuffle={flips_shuffle} "
                     f"elapsed={time.time()-t_start:.0f}s\n")

    summary = {"n_eval": n_eval,
               "flip_rate_corrupt": flips_corrupt / n_eval if n_eval else 0.0,
               "flip_rate_shuffle": flips_shuffle / n_eval if n_eval else 0.0,
               "flips_corrupt": flips_corrupt, "flips_shuffle": flips_shuffle}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "details": details},
                                         ensure_ascii=False, indent=2))
    print("\n=== CAUSAL PROBE (2a) — QLoRA-32B epoch_1 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n8B reference (regime-1): corrupt-flip 3%, shuffle-flip 15% (perception-transfer).")
    print("Load-bearing reasoning => corrupt-flip markedly higher than the 8B's 3%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
