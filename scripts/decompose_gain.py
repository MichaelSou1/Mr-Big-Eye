#!/usr/bin/env python
"""WU-3 follow-up — turn "the SFT *gain* is reading, not load-bearing CoT" from an aggregate
inference into a PER-CASE statement, with no new GPU work.

Join two existing artifacts:
  - eval preds  (eval_sft_<cell>_tabmwp_n400.<adapter>.preds.jsonl): per case {cid, correct, base_correct}
                 where base_correct = the PRE-SFT base model's correctness (greedy).
  - battery present (battery_tabmwp<cell>_present.json) details: per SFT-correct kept case
                 {case_id, base_ans (SFT model's own answer), gold, injected, corrupt_ans}.

Every battery-kept case is SFT-correct WITH a CoT by construction. We partition those cases by
the PRE-SFT base correctness into:
  GAINED   = base wrong  -> SFT right   (these ARE the +Δacc cases)
  RETAINED = base right  -> SFT right   (already correct before SFT)
and, for each partition, measure how the SFT model's CoT corruption behaves:
  corrupt_flip = answer changed vs the SFT model's own (uncorrupted) answer
  follow       = answer == the INJECTED wrong number  (=> answer WAS computed from the CoT number)
  snap         = answer == gold                        (=> answer re-derived/re-read, ignores corruption)

If the GAINED partition shows ~0 follow and low flip (statistically like RETAINED), the newly-correct
answers do NOT come from propagating the CoT's arithmetic -> the gain is reading, per case. The
strongest single number is follow_rate on GAINED: a Wilson 95% upper bound near 0 means almost none
of the gained answers were computed from the (now-corrupted) chain.
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


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial rate (k successes / n)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="eval_sft_*.preds.jsonl (has base_correct)")
    ap.add_argument("--battery", required=True, help="battery_*_present.json (has per-case corrupt_ans)")
    ap.add_argument("--label", default="cell")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_correct = {}
    for line in open(args.preds):
        if line.strip():
            r = json.loads(line)
            base_correct[r["cid"]] = bool(r["base_correct"])

    det = json.load(open(args.battery))["details"]

    parts = {"GAINED": [], "RETAINED": [], "UNKNOWN": []}
    for r in det:
        cid = r["case_id"]
        if r.get("injected") is None or r.get("corrupt_ans") is None:
            continue
        bc = base_correct.get(cid)
        key = "UNKNOWN" if bc is None else ("RETAINED" if bc else "GAINED")
        ca, ba, gold, inj = r["corrupt_ans"], r["base_ans"], r["gold"], r["injected"]
        parts[key].append({
            "flip": not relaxed_match(ca, ba),
            "follow": bool(relaxed_match(ca, inj)),
            "snap": bool(relaxed_match(ca, gold)),
        })

    def summ(rows):
        n = len(rows)
        f = sum(x["flip"] for x in rows)
        fo = sum(x["follow"] for x in rows)
        sn = sum(x["snap"] for x in rows)
        return n, f, fo, sn

    print(f"\n=== {args.label}: SFT-gain decomposition (per-case corrupt behavior) ===")
    print(f"{'partition':9s} {'n':>4s} {'flip':>14s} {'follow_injected':>22s} {'snap_to_gold':>16s}")
    out = {"label": args.label, "partitions": {}}
    for key in ("GAINED", "RETAINED", "UNKNOWN"):
        rows = parts[key]
        n, f, fo, sn = summ(rows)
        if n == 0:
            print(f"{key:9s} {0:>4d}  (none)")
            continue
        pf, _, _ = wilson(f, n)
        pfo, lo_fo, hi_fo = wilson(fo, n)
        psn, _, _ = wilson(sn, n)
        print(f"{key:9s} {n:>4d}  {f:>3d} ({pf:.3f})  {fo:>3d} ({pfo:.3f}, 95%≤{hi_fo:.3f})  {sn:>3d} ({psn:.3f})")
        out["partitions"][key] = {"n": n, "flip": f, "flip_rate": pf,
                                  "follow": fo, "follow_rate": pfo, "follow_ci95_hi": hi_fo,
                                  "snap": sn, "snap_rate": psn}
    # headline statement
    g = out["partitions"].get("GAINED")
    if g:
        print(f"\n>>> GAINED (base-wrong -> SFT-right, n={g['n']}): "
              f"follow_injected={g['follow']}/{g['n']} (rate={g['follow_rate']:.3f}, 95% upper {g['follow_ci95_hi']:.3f}), "
              f"corrupt_flip={g['flip_rate']:.3f}, snap_to_gold={g['snap_rate']:.3f}")
        print(">>> If follow≈0 and flip low on GAINED, the +Δacc answers are NOT computed from the CoT "
              "number -> the gain is reading, per case.")
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
