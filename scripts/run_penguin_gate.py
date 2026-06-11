#!/usr/bin/env python
"""§6 cross-family (NEW vision-encoder paradigm) — Penguin-VL-{2B,8B}. Penguin uses an
LLM-based vision encoder (init from Qwen3-0.6B, bidirectional attn + 2D-RoPE) instead of a
contrastive CLIP/SigLIP ViT — the cleanest test of whether the perception-bound regime is a
property of the encoder paradigm or of the task.

Runs in the `vllm-qwen` env (transformers 4.57.6 loads Penguin's custom `penguinvl_qwen3`;
the project's transformers-5.9 env breaks its processor). To avoid the langchain-pulling app
modules, this script is SELF-CONTAINED: inline read_json / frame-manifest / DeepSeek orch /
grading (via app.mcq) / bootstrap (app.distill.eval_stats) / result-store append — schema
matches `data/distill/results/results.jsonl` so regen_tables/build_map pick Penguin up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
import torch
from dotenv import load_dotenv
from PIL import Image

from app.mcq import normalize_text, parse_candidates, selected_candidate, text_contains_option
from app.distill.eval_stats import agg_seed_nets, paired_bootstrap_net

RESULTS = Path("data/distill/results/results.jsonl")
_CLASH = os.environ.get("CLASH_PROXY", "http://127.0.0.1:7890")

# ---------- DeepSeek orchestrator (inline) ----------
def orch(messages, *, temp=0.7, seed=None, max_tokens=2048) -> str:
    base = os.environ["ORCHESTRATOR_API_BASE_URL"]; key = os.environ.get("ORCHESTRATOR_API_KEY")
    payload = {"model": os.environ["ORCHESTRATOR_MODEL_NAME"], "messages": messages,
               "temperature": temp, "max_tokens": max_tokens}
    if seed is not None:
        payload["seed"] = int(seed)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    is_local = ("127.0.0.1" in base) or ("localhost" in base)
    ckw = {"timeout": 300, "trust_env": False}
    if not is_local:
        ckw["proxy"] = _CLASH
    last = None
    for attempt in range(4):
        try:
            with httpx.Client(**ckw) as c:
                r = c.post(f"{base.rstrip('/')}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"].get("content") or ""
        except Exception as e:
            last = e
            import time; time.sleep(1.5 * (attempt + 1))
    raise last


_CRITIC_SYS = (
    "You are a careful visual-reasoning critic. A vision model answered a multiple-choice "
    "question about video frames. You CANNOT see the frames; you only see its reading. If its "
    "answer may be wrong, list up to 3 specific visual sub-questions to re-check against the "
    "frames (counting, ordering, who-does-what, fine detail). Output ONLY a JSON array of "
    "strings (empty array if clearly reliable).")
_INTEGRATE_SYS = ("Integrate the vision model's readings and choose the single best option. "
                  "Reply with the option letter and the option text only.")


def parse_subqs(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x) for x in arr][:3] if isinstance(arr, list) else []
    except Exception:
        return []


# ---------- grading (inline, app.mcq only) ----------
def relaxed_match(pred, gold):
    g = str(gold).strip()
    nums = re.findall(r"-?\d+\.?\d*", str(pred).replace(",", ""))
    try:
        gv = float(g.replace(",", "").replace("%", ""))
        return any(abs(float(p) - gv) <= abs(gv) * 0.05 + 1e-6 for p in nums)
    except ValueError:
        gn = normalize_text(g)
        return bool(gn) and gn in normalize_text(pred)


def _gold_candidate(gold, cands):
    g = str(gold).strip()
    bare = re.fullmatch(r"\(?([A-Ea-e])\)?\.?", g)
    if bare:
        lab = bare.group(1).upper()
        for c in cands:
            if c["label"].upper() == lab:
                return c
    c = selected_candidate(gold, cands)
    if c is not None:
        return c
    gn = normalize_text(gold).rstrip(".")
    for c in cands:
        cn = normalize_text(c["text"]).rstrip(".")
        if gn and (gn in cn or cn in gn):
            return c
    return None


def grade(question, gold, answer) -> bool:
    cands = parse_candidates(question)
    if not cands:
        return relaxed_match(answer, gold)
    gc = _gold_candidate(gold, cands)
    if gc is None:
        return relaxed_match(answer, gold)
    sel = selected_candidate(answer, cands)
    letter_ok = bool(sel and sel["label"] == gc["label"])
    gold_text = text_contains_option(answer, gc["text"])
    other_text = any(c["label"] != gc["label"] and text_contains_option(answer, c["text"]) for c in cands)
    letter_luck = letter_ok and not gold_text and other_text
    return bool(letter_ok and not letter_luck)


# ---------- Penguin inference ----------
_MODEL = _PROC = None


def _load(model_dir):
    global _MODEL, _PROC
    from transformers import AutoModelForCausalLM, AutoProcessor
    _PROC = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    n = torch.cuda.device_count()
    # balance weights across visible GPUs (naive 'auto' fills card 0 → no headroom for the
    # vision-encoder + long-video activation spike → OOM). Cap per-card to leave headroom.
    max_memory = None
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, device_map="auto", max_memory=max_memory,
        torch_dtype=torch.bfloat16).eval()


@torch.no_grad()
def penguin_answer(question, frames, *, temp=0.0, max_new_tokens=512):
    mm = ({"type": "image", "image": frames[0]} if len(frames) == 1
          else {"type": "video", "video": list(frames), "num_frames": len(frames)})
    conv = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [mm, {"type": "text", "text": question}]}]
    inputs = _PROC(conversation=conv, return_tensors="pt", add_generation_prompt=True)
    dev = next(_MODEL.parameters()).device
    inputs = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    plen = inputs["input_ids"].shape[1]
    gkw = dict(max_new_tokens=max_new_tokens, do_sample=temp > 0)
    if temp > 0:
        gkw["temperature"] = temp; gkw["top_p"] = 1.0
    for attempt in range(2):  # per-case OOM resilience: empty cache + retry once, else blank
        try:
            out = _MODEL.generate(**inputs, **gkw)
            # Penguin's custom generate() returns ONLY new tokens; guard input-included case too.
            seq = out[0][plen:] if out.shape[1] > plen else out[0]
            return _PROC.tokenizer.decode(seq, skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == 1:
                print("    [OOM on case -> blank]", flush=True)
                return ""


def m_free(case, seed=0):
    a = penguin_answer(case["question"], case["_frames"], temp=0.0)
    return a, a


def m_orch(case, seed=0):
    q = case["question"]; frames = case["_frames"]
    free = penguin_answer(q, frames, temp=0.0)
    critic = orch([{"role": "system", "content": _CRITIC_SYS},
                   {"role": "user", "content": f"QUESTION:\n{q}\n\nVISION MODEL ANSWER:\n{free}"}], seed=seed)
    sub_qa = []
    for sq in parse_subqs(critic):
        a = penguin_answer(sq, frames, temp=0.0)
        sub_qa.append(f"Q: {sq}\nA: {a}")
    if sub_qa:
        final = orch([{"role": "system", "content": _INTEGRATE_SYS},
                      {"role": "user", "content": f"QUESTION:\n{q}\n\nINITIAL READING:\n{free}\n\n"
                       "RE-CHECK Q&A:\n" + "\n\n".join(sub_qa)}], seed=seed)
    else:
        final = free
    return free, final


METHODS = {"free_form": m_free, "orch_reflect_blind": m_orch}


def run_method(dataset, model_id, method, cases, seeds, decode, n_frames):
    fn = METHODS[method]
    split_hash = hashlib.sha1("\n".join(sorted(c["case_id"] for c in cases)).encode()).hexdigest()[:12]
    per_seed = []
    for seed in seeds:
        rows = []
        for i, c in enumerate(cases):
            free, meth = fn(c, seed)
            rows.append({"case_id": c["case_id"], "free_correct": grade(c["question"], c["gold"], free),
                         "method_correct": grade(c["question"], c["gold"], meth)})
            if (i + 1) % 10 == 0:
                print(f"  [{method} seed{seed}] {i+1}/{len(cases)}", flush=True)
        f = [int(r["free_correct"]) for r in rows]; m = [int(r["method_correct"]) for r in rows]
        boot = paired_bootstrap_net(f, m)
        fp = hashlib.sha1(f"{dataset}{model_id}{method}{seed}{split_hash}{decode}".encode()).hexdigest()[:12]
        row = {"dataset": dataset, "model_id": model_id, "method": method, "seed": seed,
               "n": len(cases), "free_acc": sum(f) / len(f), "method_acc": sum(m) / len(m),
               "bootstrap": boot, "fingerprint": {"fp": fp}, "rows": rows}
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        per_seed.append(boot["net"])
        print(f"[{model_id}/{dataset}/{method} seed{seed}] free={row['free_acc']:.3f} "
              f"method={row['method_acc']:.3f} net={boot['net']:+.3f} "
              f"CI[{boot['ci_lo']:+.3f},{boot['ci_hi']:+.3f}] gain={boot['gain']} lost={boot['lost']}", flush=True)
    agg = agg_seed_nets(per_seed)
    print(f"  {method:22s} net_mean={agg['net_mean']:+.3f} ± {agg['net_std']:.3f} (k={agg['k']})", flush=True)


def load_next(keep, max_frames=0):
    out = []
    for p in sorted(Path("data/distill/pilot/trajectories").glob("*.json")):
        t = json.load(open(p)); c = t.get("case") or {}
        cid = c.get("case_id")
        if keep is not None and cid not in keep:
            continue
        manifest = (t.get("state") or {}).get("sampler_frames") or []
        frames = []
        for it in manifest:
            if isinstance(it, dict) and it.get("path") and Path(it["path"]).exists():
                frames.append(Image.open(it["path"]).convert("RGB"))
        if max_frames and len(frames) > max_frames:  # even subsample (memory on the big model)
            step = len(frames) / max_frames
            frames = [frames[int(i * step)] for i in range(max_frames)]
        if frames:
            out.append({"case_id": cid, "question": str(c.get("question") or ""),
                        "gold": str(c.get("reference_answer") or ""), "_frames": frames})
    return out


def load_chartqa(dump, img_dir):
    out = []
    for r in (json.loads(l) for l in open(dump) if l.strip()):
        cid = r.get("case_id", ""); img = Path(img_dir) / f"chartqa_{cid.split('-')[-1]}.png"
        if img.exists():
            out.append({"case_id": cid, "question": str(r.get("question") or ""),
                        "gold": str(r.get("gold") or ""), "_frames": [Image.open(img).convert("RGB")]})
    return out


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--dataset", default="next", choices=["next", "chartqa"])
    ap.add_argument("--methods", nargs="+", default=["free_form"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    if args.dataset == "next":
        keep = set(json.load(open("data/distill/analysis/partition_next.json"))["evidence_in"])
        keep &= set(json.load(open("data/distill/analysis/label_audit_next.jsonl"))["clean_ids"])
        cases = load_next(keep, args.max_frames); nframes = args.max_frames or 16
    else:
        cases = load_chartqa("data/distill/analysis/dump_8b_chartqa.jsonl", "/home/gpus/mbe_data/chartqa_images")
        nframes = 1
    print(f"=== Penguin gate: {args.model_id} / {args.dataset} n={len(cases)} ===", flush=True)
    _load(args.model_dir)
    if "free_form" in args.methods:
        run_method(args.dataset, args.model_id, "free_form", cases, [0],
                   {"temperature": 0.0}, nframes)
    for m in [x for x in args.methods if x != "free_form"]:
        run_method(args.dataset, args.model_id, m, cases, list(range(args.seeds)),
                   {"temperature": 0.7}, nframes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
