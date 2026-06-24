#!/usr/bin/env python
"""Spec §2 — power & sample-size sanity. Given the free_form baseline accuracy from the
result store (§1), tabulate the MINIMUM DETECTABLE NET (paired, 80% power, alpha=0.05)
at the observed n and at n = 200/300/500. Converts "we saw nothing" into "we were
powered to see X% and saw nothing." Stdlib + the eval_stats machinery only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distill.eval_stats import min_detectable_net
from app.distill.seed_runner import load_results


def free_acc_for(rows, model_id, dataset):
    """free_form baseline accuracy + n for a cell, at the LARGEST available n.

    The WU-1 scaled (n=400) runs share (model_id, dataset) keys with the original n=60 runs,
    so we report the highest-n run per cell — the post-upgrade power numbers — averaging
    free_acc only over the free_form rows at that n (more than one if a seed was re-run)."""
    free = [r for r in rows
            if r["model_id"] == model_id and r["dataset"] == dataset and r["method"] == "free_form"]
    if not free:
        return (None, None)
    n_max = max(r["n"] for r in free)
    accs = [r["free_acc"] for r in free if r["n"] == n_max]
    return (sum(accs) / len(accs), n_max)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/distill/results/results.jsonl")
    ap.add_argument("--out", default="data/distill/results/power_table.json")
    ap.add_argument("--ns", type=int, nargs="+", default=[200, 300, 500])
    args = ap.parse_args()

    rows = load_results(Path(args.results))
    cells = sorted({(r["model_id"], r["dataset"]) for r in rows})
    table = []
    print(f"{'model':7s}{'dataset':9s}{'p_free':>8s}{'n_obs':>6s}"
          f"{'mde@n_obs':>11s}" + "".join(f"{'mde@'+str(n):>10s}" for n in args.ns))
    for model_id, dataset in cells:
        p_free, n_obs = free_acc_for(rows, model_id, dataset)
        if p_free is None or n_obs is None:
            continue
        mde_obs = min_detectable_net(p_free, n_obs)
        mde_ns = {n: min_detectable_net(p_free, n) for n in args.ns}
        table.append({"model_id": model_id, "dataset": dataset, "p_free": p_free,
                      "n_obs": n_obs, "mde_at_n_obs": mde_obs,
                      "mde_at_n": {str(k): v for k, v in mde_ns.items()}})
        print(f"{model_id:7s}{dataset:9s}{p_free:>8.3f}{n_obs:>6d}{mde_obs:>11.3f}"
              + "".join(f"{mde_ns[n]:>10.3f}" for n in args.ns))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"power": "80%", "alpha": 0.05, "cells": table},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")
    print("Read: at n_obs we can detect a net >= mde@n_obs with 80% power; "
          "observed |net| below it => 'powered to see X, saw nothing'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
