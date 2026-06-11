#!/usr/bin/env python
"""Spec §11.3 — the 2a counterfactual causal probe (the cheap, convincing one). On ChartQA
test cases the SFT model answers correctly WITH a CoT: take the model's own CoT, (a) corrupt
ONE intermediate numeric value, (b) shuffle the CoT sentences (control). Force the model to
continue from the edited CoT to its final answer; measure answer FLIP rate.

Real step-by-step reasoning: corrupting a load-bearing intermediate flips the answer at a HIGHER
rate than shuffling (which only scrambles order); a template/post-hoc CoT flips little either way.
Reports flip(corrupt) vs flip(shuffle). Runs against a served model (vLLM OpenAI endpoint).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from PIL import Image
from app.distill.eval_common import relaxed_match
from app.vqa import _pil_to_data_url


def _chat(base, model, content, max_tokens=256, stop=None):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "temperature": 0.0, "max_tokens": max_tokens}
    if stop:
        payload["stop"] = stop
    r = httpx.post(f"{base.rstrip('/')}/chat/completions", json=payload, timeout=180, trust_env=False)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def _content(img_url, q, cot_prefix=None):
    text = "Solve step by step, end with 'ANSWER: <final>'.\n\nQuestion: " + q
    c = [{"type": "image_url", "image_url": {"url": img_url}}, {"type": "text", "text": text}]
    return c


def extract_answer(text):
    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    return (m.group(1).strip() if m else text.strip().splitlines()[-1] if text.strip() else "")


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


def force_continue(base, model, img_url, q, edited_cot):
    """Feed the edited CoT back as the start of the assistant's reasoning; ask for the ANSWER line."""
    prompt = [{"type": "image_url", "image_url": {"url": img_url}},
              {"type": "text", "text": f"Question: {q}\n\nReasoning so far:\n{edited_cot}\n\n"
                                       "Given ONLY the reasoning above, state the final answer now as "
                                       "'ANSWER: <final>'."}]
    return extract_answer(_chat(base, model, prompt, max_tokens=64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="served SFT model base_url e.g. http://127.0.0.1:30004/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", default="data/distill/analysis/dump_8b_chartqa.jsonl")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/chartqa_images")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/distill/poc/causal_probe.json")
    args = ap.parse_args()
    rng = random.Random(0)

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    flips_corrupt, flips_shuffle, n_eval = 0, 0, 0
    details = []
    for r in rows[:args.n]:
        cid = r["case_id"]; idx = cid.split("-")[-1]
        img = Path(args.img_dir) / f"chartqa_{idx}.png"
        if not img.exists():
            continue
        q, gold = str(r["question"]), str(r["gold"])
        url = _pil_to_data_url(Image.open(img).convert("RGB"), quality=75, max_side=768)
        base_out = _chat(args.base, args.model, _content(url, q), max_tokens=300)
        base_cot = base_out.split("ANSWER:")[0].strip()
        base_ans = extract_answer(base_out)
        if not relaxed_match(base_ans, gold) or not base_cot:
            continue  # only probe cases it gets right with a CoT
        n_eval += 1
        cc = corrupt_number(base_cot, rng)
        if cc:
            a_c = force_continue(args.base, args.model, url, q, cc)
            flip_c = not relaxed_match(a_c, base_ans)
            flips_corrupt += int(flip_c)
        sc = shuffle_cot(base_cot, rng)
        a_s = force_continue(args.base, args.model, url, q, sc)
        flip_s = not relaxed_match(a_s, base_ans)
        flips_shuffle += int(flip_s)
        details.append({"case_id": cid, "base_ans": base_ans, "flip_corrupt": bool(cc and flip_c),
                        "flip_shuffle": bool(flip_s)})
        print(f"  {cid}: corrupt_flip={cc and flip_c} shuffle_flip={flip_s}", flush=True)

    summary = {"n_eval": n_eval,
               "flip_rate_corrupt": flips_corrupt / n_eval if n_eval else 0.0,
               "flip_rate_shuffle": flips_shuffle / n_eval if n_eval else 0.0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2))
    print("\n=== CAUSAL PROBE (2a) ===\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print("Real reasoning: flip_rate_corrupt notably > 0 (a load-bearing intermediate matters); "
          "interpret vs shuffle control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
