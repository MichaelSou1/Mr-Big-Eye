#!/usr/bin/env python
"""Dump FULL per-case model outputs for weakness analysis: question, gold, the
VLM's free-form answer (+reasoning), the orchestrator's sub-questions, the VLM's
per-sub-question visual reads, and the integrated final answer. Lets you see
whether a failure is PERCEPTION (the read is wrong) or REASONING (reads right,
integration wrong). Works on whichever VLM env LOCAL_VLM points at.

Usage: python scripts/dump_case_outputs.py --dataset next|clevrer|chartqa --tag 32b --n 60
Output: data/distill/analysis/dump_<tag>_<dataset>.jsonl  (one rich record per case)
"""
from __future__ import annotations
import os as _os
for _k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): _os.environ.pop(_k,None)
_os.environ["NO_PROXY"]="*"; _os.environ["no_proxy"]="*"

import argparse, asyncio, io, json, re, sys
from pathlib import Path
ROOT = Path("/home/gpus/Mr-Big-Eye-internalization")
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from PIL import Image
from app.config import settings
from app.distill.common import read_json, sampler_frame_manifest
from app.distill.frames import uniform_frame_manifest
from app.distill.filter_consistency import _answer_matches
from app.vqa import LocalVLMBackbone
from scripts.diag_orch_reflection import _orch, _parse_subqs
from scripts.diag_chartqa_gap import relaxed_match


def _open(paths_ts):
    fr, ts = [], []
    for p, t in paths_ts:
        if Path(p).exists():
            fr.append(Image.open(p).convert("RGB")); ts.append(float(t))
    return fr, ts


async def run_verbose(bb, q, frames, ts):
    """free-form + orchestrated reflection, returning ALL intermediate text."""
    free = await bb.answer_question(q, frames, ts)
    critic = _orch([
        {"role": "system", "content":
         "You are a careful visual-reasoning critic. A vision model answered a "
         "multiple-choice question about video frames. You CANNOT see the frames; "
         "you only see its reading. If its answer may be wrong, list up to 3 "
         "specific visual sub-questions to re-check (counting, ordering, "
         "who-does-what, fine detail). Output ONLY a JSON array of strings."},
        {"role": "user", "content": f"QUESTION:\n{q}\n\nVISION MODEL ANSWER:\n{free}"}])
    subqs = _parse_subqs(critic)
    sub_qa = []
    for sq in subqs:
        a = await bb.answer_question(sq, frames, ts)
        sub_qa.append({"subq": sq, "vlm_read": a})
    if sub_qa:
        final = _orch([
            {"role": "system", "content":
             "Integrate the vision model's readings and choose the single best "
             "option. Reply with the option letter and the option text only."},
            {"role": "user", "content":
             f"QUESTION:\n{q}\n\nINITIAL READING:\n{free}\n\nRE-CHECK Q&A:\n" +
             "\n\n".join(f"Q: {x['subq']}\nA: {x['vlm_read']}" for x in sub_qa)}])
    else:
        final = free
    return free, critic, sub_qa, final


def load_dataset(name, n):
    """Yield (case_id, question, gold, frames, ts, matcher)."""
    if name == "next":
        for p in sorted((ROOT/"data/distill/pilot/trajectories").glob("*.json")):
            t = read_json(p); c = t["case"]
            fr, ts = _open([(x["path"], x["timestamp"]) for x in sampler_frame_manifest(t)])
            if fr:
                yield c["case_id"], c["question"], c.get("reference_answer"), c.get("question_type"), fr, ts, _answer_matches
    elif name == "clevrer":
        from app.cache import get_video_status
        for line in open(ROOT/"data/eval/datasets/clevrer_pilot/cases.jsonl"):
            if not line.strip(): continue
            c = json.loads(line)
            if not c.get("video_id") or get_video_status(c["video_id"]) != "done": continue
            fr, ts = _open([(x["path"], x["timestamp"]) for x in uniform_frame_manifest(c["video_id"])])
            if fr:
                yield c["case_id"], c["question"], c.get("reference_answer"), c.get("question_type"), fr, ts, _answer_matches
    elif name == "chartqa":
        import pandas as pd
        from huggingface_hub import hf_hub_download
        df = pd.read_parquet(hf_hub_download("HuggingFaceM4/ChartQA",
                "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet", repo_type="dataset",
                local_files_only=True))
        cnt = 0
        for i in range(min(len(df), n*3)):
            if cnt >= n: break
            r = df.iloc[i]; cell = r["image"]; raw = cell["bytes"] if isinstance(cell, dict) else cell
            try: img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception: continue
            gold = r["label"]
            try:
                gold = gold[0]
            except (TypeError, IndexError, KeyError):
                pass
            gold = str(gold)
            cnt += 1
            yield f"chartqa-{i}", str(r["query"]), gold, "chart", [img], [0.0], (lambda q,g,a: relaxed_match(a,g))


async def main_async():
    load_dotenv(str(ROOT/".env"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["next","clevrer","chartqa"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=70)
    args = ap.parse_args()
    bb = LocalVLMBackbone()
    out = ROOT/f"data/distill/analysis/dump_{args.tag}_{args.dataset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok=n_done=0
    with open(out, "w") as f:
        for cid, q, gold, qtype, fr, ts, match in load_dataset(args.dataset, args.n):
            try:
                free, critic, sub_qa, final = await run_verbose(bb, q, fr, ts)
            except Exception as e:
                print(f"{cid}: ERR {str(e)[:60]}", flush=True); continue
            fok = bool(match(q, gold, free)); ook = bool(match(q, gold, final))
            rec = {"case_id": cid, "qtype": qtype, "question": q, "gold": gold,
                   "n_frames": len(fr),
                   "free_answer": free, "free_correct": fok,
                   "critic_subqs_raw": critic, "sub_qa": sub_qa,
                   "final_answer": final, "orch_correct": ook}
            f.write(json.dumps(rec, ensure_ascii=False)+"\n"); f.flush()
            n_done+=1; n_ok+=fok
            print(f"{cid}: free={fok} orch={ook} nsub={len(sub_qa)}", flush=True)
    print(f"\nDUMP DONE {args.tag}/{args.dataset}: {n_done} cases, free_acc={n_ok}/{n_done} -> {out}")


if __name__ == "__main__":
    asyncio.run(main_async())
