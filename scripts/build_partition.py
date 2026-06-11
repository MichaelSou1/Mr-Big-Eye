#!/usr/bin/env python
"""Spec §3.0 — frames-visible partition. Split NExT cases into EVIDENCE_IN (≥1
uniform-sampled frame inside the GT grounding window) vs EVIDENCE_OUT (selection-
limited). The reasoning-headroom result is stated on EVIDENCE_IN; EVIDENCE_OUT is the
type-2/frame-selection slice. Reuses app/distill/frames.py:covers_evidence.

Reads the trajectories (which carry sampler_frames + case.gold_{timestamps,scenes}).
Writes data/distill/analysis/partition_<dataset>.json = {evidence_in:[ids], evidence_out:[ids]}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distill.common import read_json, sampler_frame_manifest
from app.distill.frames import covers_evidence


def partition_nextgqa(traj_dir: Path, tolerance_sec: float = 2.0) -> dict:
    ev_in, ev_out, no_gt = [], [], []
    for p in sorted(traj_dir.glob("*.json")):
        t = read_json(p)
        case = t.get("case") or {}
        cid = case.get("case_id")
        gts = [float(x) for x in case.get("gold_timestamps", []) or []]
        scenes = case.get("gold_scenes", []) or []
        if not gts and not scenes:
            no_gt.append(cid)
            continue
        sampled = [float(it["timestamp"]) for it in sampler_frame_manifest(t)]
        if covers_evidence(sampled, gts, scenes, tolerance_sec=tolerance_sec):
            ev_in.append(cid)
        else:
            ev_out.append(cid)
    return {"evidence_in": ev_in, "evidence_out": ev_out, "no_gt": no_gt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="next", choices=["next"])
    ap.add_argument("--traj-dir", default="data/distill/pilot/trajectories")
    ap.add_argument("--tolerance-sec", type=float, default=2.0)
    ap.add_argument("--out", default="data/distill/analysis/partition_next.json")
    args = ap.parse_args()

    part = partition_nextgqa(Path(args.traj_dir), args.tolerance_sec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(part, ensure_ascii=False, indent=2), encoding="utf-8")
    nin, nout, ngt = len(part["evidence_in"]), len(part["evidence_out"]), len(part["no_gt"])
    tot = nin + nout
    print(f"EVIDENCE_IN={nin}  EVIDENCE_OUT={nout}  no_gt={ngt}  "
          f"(in-rate={nin/tot:.2f} of {tot} GT cases)  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
