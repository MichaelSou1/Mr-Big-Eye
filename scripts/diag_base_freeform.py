#!/usr/bin/env python
"""Diagnostic: 4B free-form accuracy on the uniform sample (no tools, no CoT).

Answers the pilot question: are the strict-dropped cases dropped because the
visual reasoning is hard (4B free-form WRONG → real internalization signal) or
just for agent citation/process reasons (4B free-form already RIGHT → task too
easy on this slice)? Reads the generated trajectories (which carry the uniform
sampler_frames + gold answer) and runs the base 4B once per case.
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


async def main_async() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="data/distill/pilot/trajectories")
    ap.add_argument("--strict-report", default="data/distill/pilot/strict_filter_report.json")
    ap.add_argument("--output", default="data/distill/pilot/base_freeform_diag.json")
    args = ap.parse_args()

    strict_pass = set()
    if Path(args.strict_report).exists():
        rep = read_json(args.strict_report)
        for r in rep.get("results", []):
            if r.get("passed"):
                strict_pass.add(r.get("case_id"))

    backbone = LocalVLMBackbone()
    rows = []
    for path in sorted(Path(args.traj_dir).glob("*.json")):
        traj = read_json(path)
        case = traj.get("case") or {}
        manifest = sampler_frame_manifest(traj)
        frames, ts = _load_frames(manifest)
        if not frames:
            rows.append({"case_id": case.get("case_id"), "skipped": "no_frames"})
            continue
        q = str(case.get("question") or "")
        ref = str(case.get("reference_answer") or "")
        ans = await backbone.answer_question(q, frames, ts)
        ok = _answer_matches(q, ref, ans)
        rows.append({
            "case_id": case.get("case_id"),
            "qtype": case.get("question_type"),
            "free_correct": bool(ok),
            "strict_passed": case.get("case_id") in strict_pass,
            "n_frames": len(frames),
        })
        print(f"{case.get('case_id')}: free_correct={ok} qtype={case.get('question_type')} "
              f"strict={case.get('case_id') in strict_pass}", flush=True)

    scored = [r for r in rows if "free_correct" in r]
    n = len(scored)
    correct = sum(1 for r in scored if r["free_correct"])
    by_strict = {
        "strict_passed": [r for r in scored if r["strict_passed"]],
        "strict_dropped": [r for r in scored if not r["strict_passed"]],
    }
    qtype_acc = collections.defaultdict(lambda: [0, 0])
    for r in scored:
        qtype_acc[r["qtype"]][1] += 1
        qtype_acc[r["qtype"]][0] += int(r["free_correct"])

    summary = {
        "n_scored": n,
        "free_accuracy_all": correct / n if n else 0.0,
        "free_accuracy_strict_passed": (
            sum(r["free_correct"] for r in by_strict["strict_passed"]) / len(by_strict["strict_passed"])
            if by_strict["strict_passed"] else None
        ),
        "free_accuracy_strict_dropped": (
            sum(r["free_correct"] for r in by_strict["strict_dropped"]) / len(by_strict["strict_dropped"])
            if by_strict["strict_dropped"] else None
        ),
        "n_strict_dropped": len(by_strict["strict_dropped"]),
        "free_wrong_count": n - correct,
        "by_qtype": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(qtype_acc.items())},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
