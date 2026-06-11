#!/usr/bin/env python
"""Spec §3.1b — ORACLE-perception upper bound (the cheapest, most direct proof of
"perception is the wall"). For each FREE-WRONG case, remove vision from the answer path:
a STRONG VLM (e.g. 32B) captions the GT-localized hi-res frames into a thorough textual
scene description, then the TEXT reasoner (DeepSeek) answers from (question + caption)
with NO image access.

Read:
  oracle_acc (on free-wrong) ≈ 0   -> reasoning isn't the bottleneck; perception-bound (regime 1)
  oracle_acc (on free-wrong) high   -> reasoning IS reachable; the VLM's eyes are the wall
NExT has no human caption in our trajectories, so the description is a strong-VLM caption of
GT frames => report this as NEAR-ORACLE (audited), per spec.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from PIL import Image

from app.cache import video_cache_dir
from app.distill.common import read_json
from app.distill.eval_common import grade_textaware
from app.distill.methods import orch, vlm_answer

CAPTION_SYS = (
    "Describe these video frames in exhaustive, neutral detail: who/what is present, every "
    "action and its order, counts, spatial relations, and any fine detail (text, color, motion "
    "direction). Do NOT answer any question or guess intent — only report what is visible."
)
ANSWER_SYS = (
    "You are answering a multiple-choice question about a video you CANNOT see. You are given a "
    "detailed eyewitness description of the relevant moments. Reason over it and pick the single "
    "best option. Reply with the option letter and its text only."
)


def _gt_window_frames(traj, pad=1.0, cap=16):
    case = traj["case"]; vid = case["video_id"]
    scenes = case.get("gold_scenes") or []
    gts = case.get("gold_timestamps") or []
    if scenes:
        lo = min(s["start"] for s in scenes) - pad; hi = max(s["end"] for s in scenes) + pad
    elif gts:
        lo, hi = min(gts) - pad, max(gts) + pad
    else:
        return [], []
    dd = video_cache_dir(vid) / "frames_dense"
    frames, ts = [], []
    for f in sorted(dd.glob("t*.jpg")):
        try:
            t = float(f.stem.lstrip("t"))
        except ValueError:
            continue
        if lo <= t <= hi:
            frames.append(Image.open(f).convert("RGB")); ts.append(t)
    # subsample to cap evenly
    if len(frames) > cap:
        step = len(frames) / cap
        idx = [int(i * step) for i in range(cap)]
        frames = [frames[i] for i in idx]; ts = [ts[i] for i in idx]
    return frames, ts


async def main_async() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="data/distill/pilot/trajectories")
    ap.add_argument("--dump", required=True, help="dump_<model>_next.jsonl (for free_correct per case)")
    ap.add_argument("--captioner-base", required=True, help="strong VLM base_url, e.g. http://127.0.0.1:30002/v1")
    ap.add_argument("--captioner-model", required=True, help="e.g. Qwen3-VL-32B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    free_wrong = [r for r in rows if not r.get("free_correct")]
    print(f"=== oracle-perception: {len(free_wrong)} free-wrong cases (captioner={args.captioner_model}) ===",
          flush=True)
    out, recov = [], 0
    for i, r in enumerate(free_wrong):
        cid = r["case_id"]
        p = Path(args.traj_dir) / f"{cid}.json"
        if not p.exists():
            continue
        traj = read_json(p); case = traj["case"]
        q = str(case.get("question") or ""); gold = str(case.get("reference_answer") or "")
        gf, gt = _gt_window_frames(traj)
        if not gf:
            out.append({"case_id": cid, "skip": "no_gt_frames"}); continue
        caption = await vlm_answer(CAPTION_SYS + "\n\nFrames follow.", gf, gt, temp=0.0,
                                   base_url=args.captioner_base, model=args.captioner_model, max_tokens=1024)
        ans = orch([{"role": "system", "content": ANSWER_SYS},
                    {"role": "user", "content": f"QUESTION:\n{q}\n\nEYEWITNESS DESCRIPTION:\n{caption}"}],
                   temp=0.0, max_tokens=1024)
        g = grade_textaware(q, gold, ans)
        recov += int(g["correct"])
        out.append({"case_id": cid, "qtype": case.get("question_type"),
                    "oracle_correct": g["correct"], "caption": caption[:500], "oracle_answer": ans[:200]})
        print(f"  [{i+1}/{len(free_wrong)}] {cid} oracle_correct={g['correct']}", flush=True)

    n = len([o for o in out if "oracle_correct" in o])
    summary = {"n_free_wrong": len(free_wrong), "n_scored": n, "oracle_recovered": recov,
               "oracle_recovery_rate": recov / n if n else 0.0,
               "note": "NEAR-ORACLE: GT-frame caption by strong VLM, not human GT caption"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": out}, ensure_ascii=False, indent=2))
    print("\n=== ORACLE SUMMARY ===\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
