#!/usr/bin/env python
"""Exp-4: does 30B-orchestrated reflection over the SAME uniform-16 frames beat
4B free-form? (Frame-controlled orchestrator-distillation premise test.)

Pipeline per case (frames FIXED to uniform-16, NO retrieval):
  1. 4B (eyes): initial answer + reasoning over the 16 frames  [= free-form].
  2. 30B (text critic): cannot see frames; reads the 4B answer, decides if
     reliable, emits up to 3 targeted visual sub-questions to re-check.
  3. 4B: answers each sub-question over the same 16 frames (fresh visual reads).
  4. 30B: integrates initial + sub-answers, picks the final MCQ option.
Compare final (orchestrated) vs initial (free-form). gain = orch-right ∧
free-wrong (the internalizable reflection signal); lost = free-right ∧ orch-wrong.
"""
from __future__ import annotations

import os as _os
for _k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): _os.environ.pop(_k,None)
_os.environ["NO_PROXY"]="*"; _os.environ["no_proxy"]="*"

import argparse
import asyncio
import collections
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
from dotenv import load_dotenv
from PIL import Image

from app.config import settings
from app.distill.common import read_json, sampler_frame_manifest
from app.distill.filter_consistency import _answer_matches
from app.vqa import LocalVLMBackbone


def _load_frames(manifest):
    frames, ts = [], []
    for item in manifest:
        p = Path(str(item.get("path") or ""))
        if p.exists():
            frames.append(Image.open(p).convert("RGB"))
            ts.append(float(item["timestamp"]))
    return frames, ts


def _orch(messages: list[dict], max_tokens: int = 400) -> str:
    payload = {"model": settings.orchestrator_model_name, "messages": messages,
               "temperature": 0.2, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if settings.orchestrator_api_key:
        headers["Authorization"] = f"Bearer {settings.orchestrator_api_key}"
    with httpx.Client(timeout=180) as c:
        r = c.post(f"{settings.orchestrator_api_base_url.rstrip('/')}/chat/completions",
                   headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _parse_subqs(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x) for x in arr][:3] if isinstance(arr, list) else []
    except Exception:
        return []


async def run_case(backbone, q, ref, frames, ts):
    free = await backbone.answer_question(q, frames, ts)
    critic = _orch([
        {"role": "system", "content":
         "You are a careful visual-reasoning critic. A vision model answered a "
         "multiple-choice question about video frames. You CANNOT see the frames; "
         "you only see its reading. If its answer may be wrong, list up to 3 "
         "specific visual sub-questions to re-check against the frames (counting, "
         "ordering, who-does-what, fine detail). Output ONLY a JSON array of "
         "strings (empty array if the answer is clearly reliable)."},
        {"role": "user", "content": f"QUESTION:\n{q}\n\nVISION MODEL ANSWER:\n{free}"},
    ])
    subqs = _parse_subqs(critic)
    sub_qa = []
    for sq in subqs:
        a = await backbone.answer_question(sq, frames, ts)
        sub_qa.append(f"Q: {sq}\nA: {a}")
    if sub_qa:
        final = _orch([
            {"role": "system", "content":
             "Integrate the vision model's readings and choose the single best "
             "option. Reply with the option letter and the option text only."},
            {"role": "user", "content":
             f"QUESTION:\n{q}\n\nINITIAL READING:\n{free}\n\n"
             f"RE-CHECK Q&A:\n" + "\n\n".join(sub_qa)},
        ])
    else:
        final = free
    return free, final, len(subqs)


async def main_async() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="data/distill/pilot/trajectories")
    ap.add_argument("--output", default="data/distill/pilot/orch_reflection_diag.json")
    args = ap.parse_args()

    backbone = LocalVLMBackbone()
    rows = []
    for path in sorted(Path(args.traj_dir).glob("*.json")):
        t = read_json(path)
        case = t.get("case") or {}
        frames, ts = _load_frames(sampler_frame_manifest(t))
        if not frames:
            continue
        q = str(case.get("question") or ""); ref = str(case.get("reference_answer") or "")
        try:
            free, final, nsub = await run_case(backbone, q, ref, frames, ts)
        except Exception as e:
            print(f"{case.get('case_id')}: ERROR {e}", flush=True); continue
        fok = _answer_matches(q, ref, free); ook = _answer_matches(q, ref, final)
        rows.append({"case_id": case.get("case_id"), "qtype": case.get("question_type"),
                     "free_correct": bool(fok), "orch_correct": bool(ook), "n_subq": nsub})
        print(f"{case.get('case_id')}: free={fok} orch={ook} subq={nsub}", flush=True)

    n = len(rows)
    free = sum(r["free_correct"] for r in rows); orch = sum(r["orch_correct"] for r in rows)
    gain = [r for r in rows if r["orch_correct"] and not r["free_correct"]]
    lost = [r for r in rows if r["free_correct"] and not r["orch_correct"]]
    summary = {"n": n, "free_accuracy": free / n if n else 0,
               "orch_accuracy": orch / n if n else 0,
               "orch_gain_cases": len(gain), "orch_lost_cases": len(lost),
               "net_gain": (orch - free) / n if n else 0,
               "gain_ids": [r["case_id"] for r in gain],
               "gain_by_qtype": dict(collections.Counter(r["qtype"] for r in gain))}
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
