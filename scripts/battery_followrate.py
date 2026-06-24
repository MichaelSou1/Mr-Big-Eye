#!/usr/bin/env python
"""WU-3 §4.2 — present-vs-masked re-perception breakdown ("where does the arithmetic happen?").

Reads the battery present/masked JSONs and, for each kept case, classifies the post-corrupt answer:
  snap   = matches gold        (re-read / re-derived the true answer; ignored the corrupted CoT)
  follow = matches injected     (used the corrupted CoT number -> CoT was load-bearing)
  other  = neither             (fabricated a different value -> hallucination)

The contrast across present->masked is the mechanism evidence: with the image present the model
re-reads (snap high, follow~0); masked it cannot re-read, and what rises is mostly `other`
(hallucination), not `follow` -> the load-bearing thing is the IMAGE (perception), not the chain.

Pure CPU; reuses existing battery outputs. Run in env `mbe-up`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distill.eval_common import relaxed_match


def analyze(path: str):
    det = json.load(open(path))["details"]
    n = snap = follow = flip = other = 0
    for r in det:
        ca, ba, gold, inj = r.get("corrupt_ans"), r.get("base_ans"), r.get("gold"), r.get("injected")
        if inj is None or ca is None:
            continue
        n += 1
        if not relaxed_match(ca, ba):
            flip += 1
        if relaxed_match(ca, gold):
            snap += 1
        elif inj and relaxed_match(ca, inj):
            follow += 1
        else:
            other += 1
    return {"n": n, "flip": flip, "snap": snap, "follow": follow, "other": other}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["tabmwp8b", "tabmwp32b"])
    ap.add_argument("--poc-dir", default="data/distill/poc")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    print(f"{'cell/cond':22s}{'n':>5s}{'flip':>8s}{'snap':>8s}{'follow':>8s}{'other':>8s}")
    for cell in args.cells:
        for cond in ("present", "masked"):
            p = f"{args.poc_dir}/battery_{cell}_{cond}.json"
            if not Path(p).exists():
                print(f"{cell+' '+cond:22s}  (missing {p})")
                continue
            a = analyze(p)
            n = a["n"]
            print(f"{cell+' '+cond:22s}{n:>5d}{a['flip']/n:>8.3f}{a['snap']/n:>8.3f}"
                  f"{a['follow']/n:>8.3f}{a['other']/n:>8.3f}")
            rows.append({"cell": cell, "cond": cond, **a,
                         "flip_rate": a["flip"]/n, "snap_rate": a["snap"]/n,
                         "follow_rate": a["follow"]/n, "other_rate": a["other"]/n})
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
