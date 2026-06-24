#!/usr/bin/env python
"""Spec §8 — THE MAP. Classify each (dataset, model) cell into regime
{1 perception/selection-bound | 2 reasoning-bound} from:
  - net verdict of the best agentic method (regen_tables / results store),
  - free_form accuracy (perception proxy: does the base read the scene?),
  - perception-probe recovery at GT-localized hi-res (type-2 perception/selection slice),
  - near-oracle recovery (perfect-perception reasoning ceiling on free-wrong cases).

Rule (Spec §3.3):
  Regime 2 (reasoning-bound): free_form perceives well (free_acc high / perception "solved")
    AND agentic net is within-variance AND a reasoning residual remains (oracle recovers
    beyond what better perception does, OR documented arithmetic residual). -> §11 candidate.
  Regime 1 (perception/selection-bound): otherwise — net within-variance and the difficulty
    is perception/selection/label (better perception or oracle recovers little, or free_acc
    shows perception itself is the wall).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distill.seed_runner import load_results

# Mechanism artifacts available per cell (extend as more probes are run).
PERCEPTION = {  # cell -> gt_local hi-res recovery rate on free-wrong
    ("next", "4b"): 0.3125,
    ("next", "8b"): 0.2222,
}
ORACLE = {  # cell -> near-oracle recovery rate on free-wrong
    ("next", "8b"): 0.3333,
}
# Documented mechanism (Exp-8 per-case) where no fresh probe exists.
NOTE = {
    ("chartqa", "32b"): "perception solved (.80); residual 12 fails = multi-step arithmetic",
    ("chartqa", "8b"): "fails on reading values/rankings (perception)",
    ("chartqa", "4b"): "fails on reading values/rankings (perception)",
    # TabMWP (WU-3): rendered tables are highly re-readable, so table reading is solved even at
    # small scale (free_acc high); the residual difficulty is the multi-step arithmetic word
    # problem. -> reasoning-bound (regime 2) wherever free_acc clears the perception bar.
    ("tabmwp", "32b"): "table reading solved; residual = multi-step arithmetic word problem",
    ("tabmwp", "8b"): "table reading solved; residual = multi-step arithmetic word problem",
    ("tabmwp", "4b"): "table reading solved; residual = multi-step arithmetic word problem",
    ("next", "4b"): "reads scene, misses brief/specific action; ~40% label-ambiguous",
    ("next", "8b"): "perception/temporal localization wall; oracle recovers only 1/3",
    ("clevrer", "*"): "pure perception/tracking wall (~chance all scales)",
}
PERCEPTION_SOLVED_ACC = 0.75   # free_acc proxy that the base reads the scene


def best_agentic(master, ds, mid):
    """Most favorable agentic net for the cell, from the pooled master table."""
    cands = [m for m in master if m["dataset"] == ds and m["model_id"] == mid
             and m["method"] in ("self_reflect", "orch_reflect_blind", "orch_reflect_sighted")]
    if not cands:
        return None
    return max(cands, key=lambda m: m["net"])


def classify(free_acc, net_verdict, perc, orac, note):
    perception_solved = free_acc is not None and free_acc >= PERCEPTION_SOLVED_ACC
    reasoning_residual = (orac is not None and perc is not None and orac > perc + 0.05) \
        or (note and "arithmetic" in note)
    if perception_solved and reasoning_residual and net_verdict != "effect+":
        return 2, "reasoning-bound (perception solved; residual = multi-step reasoning)"
    return 1, "perception/selection-bound (no internalizable reasoning headroom)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/distill/results/results.jsonl")
    ap.add_argument("--tables", default="data/distill/results/tables.json")
    ap.add_argument("--out", default="data/distill/results/map.json")
    args = ap.parse_args()

    rows = load_results(Path(args.results))
    tbl = json.load(open(args.tables)) if Path(args.tables).exists() else {"master": []}
    free_acc = {tuple(k.split("/")): v for k, v in tbl.get("free_acc", {}).items()}
    # master rows carry the pooled CI verdict per (ds, model, method)
    pooled = {(m["dataset"], m["model_id"], m["method"]): m for m in tbl.get("master", [])}

    cells = sorted({(r["dataset"], r["model_id"]) for r in rows})
    out = []
    print(f"{'dataset':9s}{'model':6s}{'free':>6s}{'best_net':>10s}{'verdict':>17s}"
          f"{'perc%':>7s}{'orac%':>7s}  regime")
    for ds, mid in cells:
        fa = free_acc.get((ds, mid))
        ba = best_agentic(tbl.get("master", []), ds, mid)
        verdict, bnet, bmethod = "n/a", None, None
        if ba is not None:
            bnet, bmethod, verdict = ba["net"], ba["method"], ba["verdict"]
        perc = PERCEPTION.get((ds, mid))
        orac = ORACLE.get((ds, mid))
        note = NOTE.get((ds, mid)) or NOTE.get((ds, "*"))
        regime, why = classify(fa, verdict, perc, orac, note)
        out.append({"dataset": ds, "model_id": mid, "free_acc": fa, "best_method": bmethod,
                    "best_net": bnet, "net_verdict": verdict, "perception_recovery": perc,
                    "oracle_recovery": orac, "regime": regime, "rationale": why, "note": note})
        print(f"{ds:9s}{mid:6s}{(fa if fa is not None else float('nan')):>6.2f}"
              f"{(bnet if bnet is not None else float('nan')):>+10.3f}{verdict:>17s}"
              f"{(perc if perc is not None else float('nan')):>7.2f}"
              f"{(orac if orac is not None else float('nan')):>7.2f}  R{regime}")
    Path(args.out).write_text(json.dumps({"cells": out}, ensure_ascii=False, indent=2))
    print(f"\n-> {args.out}")
    r1 = [c for c in out if c['regime'] == 1]; r2 = [c for c in out if c['regime'] == 2]
    print(f"Regime 1 (perception/selection-bound): {[(c['dataset'],c['model_id']) for c in r1]}")
    print(f"Regime 2 (reasoning-bound, §11 candidates): {[(c['dataset'],c['model_id']) for c in r2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
