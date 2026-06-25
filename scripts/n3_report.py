#!/usr/bin/env python
"""N3 report — unify snap/follow/other across the three rereadability regimes and
classify where the natural-image (V*-style) pole lands on the bypass / fabricate /
load-bearing axis.

Reads the battery present/masked JSONs (re-classifies the post-corrupt answer per
case, identical rule to scripts/battery_followrate.py) for:
  charts  : cells 8b / 32b          (ChartQA, WU-2)
  tables  : cells tabmwp8b / tabmwp32b (TabMWP, WU-3)
  natural : cells natcount8b / natcount32b (N3)
and adds binomial **Wilson 95% CIs** on follow & other (the natcount n is small, so
the CI is the honest read), base free-form accuracy (when recorded), and an
auto-classified regime label for the natural-present cells.

Pure CPU; one-command regen of the N3 comparison table. Run in env `mbe-up`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distill.eval_common import relaxed_match

CELLS = [
    ("chart  8b",  "8b"),       ("chart  32b", "32b"),
    ("table  8b",  "tabmwp8b"), ("table  32b", "tabmwp32b"),
    ("nat    8b",  "natcount8b"), ("nat    32b", "natcount32b"),
]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def classify(path: str):
    obj = json.load(open(path))
    det = obj["details"]
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
    base_acc = obj.get("summary", {}).get("base_acc")
    return {"n": n, "flip": flip, "snap": snap, "follow": follow, "other": other,
            "base_acc": base_acc}


def regime_label(p_snap, p_follow, p_other):
    """Heuristic state on the rereadability axis (the numbers carry the argument;
    this is just a one-word tag for the table)."""
    if p_snap >= 0.60 and p_follow < 0.10 and p_other < 0.10:
        return "bypass(reread)"
    if p_other >= 0.15 and p_other >= p_follow:
        return "fabricate"
    if p_follow >= 0.15 and p_follow > p_other:
        return "load-bearing"
    return "mixed/narrow"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poc-dir", default="data/distill/poc")
    ap.add_argument("--out-json", default="data/distill/results/n3_regime.json")
    ap.add_argument("--out-md", default="data/distill/results/n3_regime.md")
    args = ap.parse_args()

    rows = []
    hdr = (f"{'cell/cond':16s}{'n':>5s}{'baseAcc':>8s}{'flip':>7s}{'snap':>7s}"
           f"{'follow':>8s}{'[95%CI]':>15s}{'other':>7s}{'[95%CI]':>15s}  regime")
    print(hdr)
    print("-" * len(hdr))
    md = ["# N3 regime table — snap/follow/other across rereadability regimes",
          "",
          "| cell | cond | n | baseAcc | flip | snap | follow | follow 95%CI | other | other 95%CI | regime |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, cell in CELLS:
        for cond in ("present", "masked"):
            p = f"{args.poc_dir}/battery_{cell}_{cond}.json"
            if not Path(p).exists():
                print(f"{label+' '+cond:16s}  (missing)")
                continue
            a = classify(p)
            n = a["n"]
            if n == 0:
                continue
            ps, pf, po = a["snap"]/n, a["follow"]/n, a["other"]/n
            fl = a["flip"]/n
            fci = wilson(a["follow"], n); oci = wilson(a["other"], n)
            ba = a["base_acc"]
            reg = regime_label(ps, pf, po) if cond == "present" else "—"
            ba_s = f"{ba:.3f}" if isinstance(ba, (int, float)) else "—"
            print(f"{label+' '+cond:16s}{n:>5d}{ba_s:>8s}{fl:>7.3f}{ps:>7.3f}"
                  f"{pf:>8.3f}{('['+format(fci[0],'.3f')+','+format(fci[1],'.3f')+']'):>15s}"
                  f"{po:>7.3f}{('['+format(oci[0],'.3f')+','+format(oci[1],'.3f')+']'):>15s}  {reg}")
            md.append(f"| {label.strip()} | {cond} | {n} | {ba_s} | {fl:.3f} | {ps:.3f} | "
                      f"{pf:.3f} | [{fci[0]:.3f},{fci[1]:.3f}] | {po:.3f} | "
                      f"[{oci[0]:.3f},{oci[1]:.3f}] | {reg} |")
            rows.append({"cell": cell, "cond": cond, "n": n, "base_acc": ba,
                         "flip": fl, "snap": ps, "follow": pf, "other": po,
                         "follow_ci": fci, "other_ci": oci, "regime": reg,
                         "counts": {k: a[k] for k in ("snap", "follow", "other", "flip")}})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"\n-> {args.out_json}\n-> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
