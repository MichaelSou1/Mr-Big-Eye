#!/usr/bin/env python
"""Spec §8 — regenerate ALL paper tables from the result store (no hand-copied numbers).

Reads data/distill/results/results.jsonl and produces:
  (1) MASTER TABLE: per (dataset, model, method) -> free_acc, seed-mean net, 95% CI
      (pooled paired-bootstrap over cases AND seeds), verdict {effect|within-variance}.
  (2) THE MAP: per (dataset, model) cell -> regime {1 perception/selection | 2 reasoning},
      annotated with the best method's net±CI (filled from mechanism artifacts if present).
  (3) RETRACTION BOX: any net that was >0 in a single run but crosses 0 multi-seed.

Verdict rule (Spec §1.3): a setting shows an effect iff the pooled 95% CI of net excludes 0.
The pooled bootstrap resamples cases within each seed and seeds across the K runs, folding in
both case-sampling and decoding variance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distill.seed_runner import load_results


def pooled_net_ci(per_seed_rows, B=10000, seed=0):
    """Pool paired per-case nets across seeds. Each row carries rows[].{free,method}_correct.
    Resample (seed, then cases within seed) -> CI that folds case + decoding variance."""
    seed_diffs = []
    for r in per_seed_rows:
        f = np.array([int(x["free_correct"]) for x in r["rows"]], float)
        m = np.array([int(x["method_correct"]) for x in r["rows"]], float)
        seed_diffs.append(m - f)
    if not seed_diffs:
        return None
    point = float(np.mean([d.mean() for d in seed_diffs]))
    rng = np.random.default_rng(seed)
    K = len(seed_diffs)
    boot = np.empty(B)
    for b in range(B):
        si = rng.integers(0, K)
        d = seed_diffs[si]
        idx = rng.integers(0, len(d), size=len(d))
        boot[b] = d[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    seed_means = np.array([d.mean() for d in seed_diffs])
    return {"net": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "seed_std": float(seed_means.std(ddof=1)) if K > 1 else 0.0, "k": K,
            "excludes_0": bool(lo > 0 or hi < 0),
            "gain": int(sum(r["bootstrap"]["gain"] for r in per_seed_rows) / K),
            "lost": int(sum(r["bootstrap"]["lost"] for r in per_seed_rows) / K)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/distill/results/results.jsonl")
    ap.add_argument("--out", default="data/distill/results/tables.json")
    args = ap.parse_args()

    rows = load_results(Path(args.results))
    by = defaultdict(list)
    free_acc = {}
    for r in rows:
        by[(r["dataset"], r["model_id"], r["method"])].append(r)
        if r["method"] == "free_form":
            free_acc[(r["dataset"], r["model_id"])] = r["free_acc"]

    master, retraction = [], []
    print(f"{'dataset':9s}{'model':7s}{'method':22s}{'free':>7s}{'net':>8s}"
          f"{'CI95':>20s}{'σseed':>7s}{'k':>3s}  verdict")
    for (ds, mid, method), rs in sorted(by.items()):
        if method == "free_form":
            continue
        ci = pooled_net_ci(rs)
        if ci is None:
            continue
        fa = free_acc.get((ds, mid), float("nan"))
        verdict = "effect" if ci["excludes_0"] else "within-variance"
        master.append({"dataset": ds, "model_id": mid, "method": method, "free_acc": fa,
                       "net": ci["net"], "ci": [ci["ci_lo"], ci["ci_hi"]], "seed_std": ci["seed_std"],
                       "k": ci["k"], "gain": ci["gain"], "lost": ci["lost"], "verdict": verdict})
        print(f"{ds:9s}{mid:7s}{method:22s}{fa:>7.3f}{ci['net']:>+8.3f}"
              f"  [{ci['ci_lo']:+.3f},{ci['ci_hi']:+.3f}]{ci['seed_std']:>7.3f}{ci['k']:>3d}  {verdict}")
        # retraction: any single seed positive that the pooled CI does not sustain
        per_seed = [r["bootstrap"]["net"] for r in rs]
        if any(s > 0 for s in per_seed) and not ci["excludes_0"]:
            retraction.append({"dataset": ds, "model_id": mid, "method": method,
                               "best_single_seed_net": max(per_seed), "pooled_net": ci["net"],
                               "pooled_ci": [ci["ci_lo"], ci["ci_hi"]]})

    print("\n=== RETRACTION BOX (single-seed positive -> within-variance pooled) ===")
    for r in retraction:
        print(f"  {r['dataset']}/{r['model_id']}/{r['method']}: best-seed net "
              f"{r['best_single_seed_net']:+.3f} -> pooled {r['pooled_net']:+.3f} "
              f"CI[{r['pooled_ci'][0]:+.3f},{r['pooled_ci'][1]:+.3f}] (crosses 0)")
    if not retraction:
        print("  (none yet)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"master": master, "retraction": retraction,
                                          "free_acc": {f"{k[0]}/{k[1]}": v for k, v in free_acc.items()}},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
