#!/usr/bin/env python
"""CLEVRER reasoning-headroom gap: free-form vs 30B-orchestrated reflection over
the SAME uniform frames, read directly from cases.jsonl (no agent run needed —
uniform frames come from cache). The decisive "does reasoning-bound data have
internalizable headroom" test. Works for whichever VLM the env points LOCAL_VLM at.
"""
from __future__ import annotations

import os as _os
for _k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"): _os.environ.pop(_k,None)
_os.environ["NO_PROXY"]="*"; _os.environ["no_proxy"]="*"

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from PIL import Image

from app.cache import get_video_status
from app.distill.frames import uniform_frame_manifest
from app.distill.filter_consistency import _answer_matches
from app.vqa import LocalVLMBackbone
from scripts.diag_orch_reflection import run_case


def _load_frames(video_id):
    frames, ts = [], []
    for item in uniform_frame_manifest(video_id):
        p = Path(str(item.get("path") or ""))
        if p.exists():
            frames.append(Image.open(p).convert("RGB"))
            ts.append(float(item["timestamp"]))
    return frames, ts


async def main_async() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/eval/datasets/clevrer_pilot/cases.jsonl")
    ap.add_argument("--output", default="data/distill/clevrer/gap_diag.json")
    ap.add_argument("--tag", default="8b")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases) if l.strip()]
    cases = [c for c in cases if c.get("video_id") and get_video_status(c["video_id"]) == "done"]
    bb = LocalVLMBackbone()
    rows = []
    for c in cases:
        frames, ts = _load_frames(c["video_id"])
        if not frames:
            continue
        q = str(c.get("question") or ""); ref = str(c.get("reference_answer") or "")
        try:
            free, final, nsub = await run_case(bb, q, ref, frames, ts)
        except Exception as e:
            print(f"{c['case_id']}: ERROR {e}", flush=True); continue
        fok = _answer_matches(q, ref, free); ook = _answer_matches(q, ref, final)
        rows.append({"case_id": c["case_id"], "qtype": c.get("question_type"),
                     "free_correct": bool(fok), "orch_correct": bool(ook), "n_subq": nsub})
        print(f"{c['case_id']}: free={fok} orch={ook} subq={nsub} qt={c.get('question_type')}", flush=True)

    n = len(rows)
    free = sum(r["free_correct"] for r in rows); orch = sum(r["orch_correct"] for r in rows)
    gain = [r for r in rows if r["orch_correct"] and not r["free_correct"]]
    lost = [r for r in rows if r["free_correct"] and not r["orch_correct"]]
    qt = collections.defaultdict(lambda: [0, 0, 0])  # n, free, orch
    for r in rows:
        qt[r["qtype"]][0] += 1; qt[r["qtype"]][1] += r["free_correct"]; qt[r["qtype"]][2] += r["orch_correct"]
    summary = {"tag": args.tag, "n": n,
               "free_accuracy": free / n if n else 0, "orch_accuracy": orch / n if n else 0,
               "orch_gain_cases": len(gain), "orch_lost_cases": len(lost),
               "net_gain": (orch - free) / n if n else 0,
               "by_qtype": {k: {"n": v[0], "free": v[1]/v[0], "orch": v[2]/v[0]} for k, v in sorted(qt.items())}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
