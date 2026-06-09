#!/usr/bin/env python
"""Exp-C: is the 4B's free-wrong due to PERCEPTION (resolution/localization) or
is the model fundamentally unable? Probe the free-wrong cases under better
perception conditions and count how many flip to correct.

Conditions (all free-form, no tools, no CoT):
  base_448   : uniform-16 @ side 448  (recap of the wrong baseline)
  uniform_hi : uniform-16 @ side 640  (native dense res — RESOLUTION lever)
  gt_local   : dense frames inside the gold window @ 640 (LOCALIZATION lever)
  gt_local_hi: gt window frames @ 768 (localization + max upscale)
If gt_local* flips many → the wall is perception/localization (type-2 frame
selection, NOT internalizable). If nothing flips → the 4B fundamentally can't.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from PIL import Image

from app.cache import video_cache_dir
from app.config import settings
from app.distill.common import read_json, sampler_frame_manifest
from app.distill.filter_consistency import _answer_matches
from app.vqa import LocalVLMBackbone


def _open(paths_ts):
    frames, ts = [], []
    for p, t in paths_ts:
        pp = Path(p)
        if pp.exists():
            frames.append(Image.open(pp).convert("RGB"))
            ts.append(float(t))
    return frames, ts


def _uniform(traj):
    return [(item["path"], item["timestamp"]) for item in sampler_frame_manifest(traj)
            if item.get("path")]


def _gt_window(traj, pad=1.0, cap=24):
    case = traj["case"]; vid = case["video_id"]
    scenes = case.get("gold_scenes") or []
    gts = case.get("gold_timestamps") or []
    if scenes:
        lo = min(s["start"] for s in scenes) - pad
        hi = max(s["end"] for s in scenes) + pad
    elif gts:
        lo, hi = min(gts) - pad, max(gts) + pad
    else:
        return []
    dd = video_cache_dir(vid) / "frames_dense"
    out = []
    for f in sorted(dd.glob("t*.jpg")):
        try:
            t = float(f.stem.lstrip("t"))
        except ValueError:
            continue
        if lo <= t <= hi:
            out.append((str(f), t))
    return out[:cap]


async def _answer(bb, q, frames, ts, side):
    settings.vqa_max_image_side = side
    return await bb.answer_question(q, frames, ts)


async def main_async() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="data/distill/pilot/trajectories")
    ap.add_argument("--diag", default="data/distill/pilot/base_freeform_diag.json")
    ap.add_argument("--output", default="data/distill/pilot/perception_headroom_diag.json")
    args = ap.parse_args()

    diag = {r["case_id"]: r for r in read_json(args.diag)["rows"] if "free_correct" in r}
    free_wrong = [c for c, r in diag.items() if not r["free_correct"]]

    bb = LocalVLMBackbone()
    rows = []
    conds = ["base_448", "uniform_hi", "gt_local", "gt_local_hi"]
    for cid in free_wrong:
        p = Path(args.traj_dir) / f"{cid}.json"
        if not p.exists():
            continue
        traj = read_json(p); case = traj["case"]
        q = str(case.get("question") or ""); ref = str(case.get("reference_answer") or "")
        uni = _uniform(traj); gtw = _gt_window(traj)
        uf, ut = _open(uni); gf, gt = _open(gtw)
        res = {"case_id": cid, "qtype": case.get("question_type"), "n_gt_frames": len(gf)}
        a = await _answer(bb, q, uf, ut, 448); res["base_448"] = _answer_matches(q, ref, a)
        a = await _answer(bb, q, uf, ut, 640); res["uniform_hi"] = _answer_matches(q, ref, a)
        if gf:
            a = await _answer(bb, q, gf, gt, 640); res["gt_local"] = _answer_matches(q, ref, a)
            a = await _answer(bb, q, gf, gt, 768); res["gt_local_hi"] = _answer_matches(q, ref, a)
        else:
            res["gt_local"] = res["gt_local_hi"] = None
        rows.append(res)
        print(f"{cid}: " + " ".join(f"{c}={res.get(c)}" for c in conds), flush=True)

    settings.vqa_max_image_side = 448
    n = len(rows)
    summary = {"n_free_wrong": n}
    for c in conds:
        flips = sum(1 for r in rows if r.get(c) is True)
        summary[f"{c}_now_correct"] = flips
        summary[f"{c}_flip_rate"] = flips / n if n else 0.0
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY (of free-wrong cases, how many become correct) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
