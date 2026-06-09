#!/usr/bin/env python
"""Join all per-case experiment records into analysis CSVs (NExT / CLEVRER / ChartQA)
and surface key per-case patterns. Correctness-level (the diagnostics saved
right/wrong, not answer text)."""
from __future__ import annotations
import csv, json, glob, os, sys
from pathlib import Path

ROOT = Path("/home/gpus/Mr-Big-Eye-internalization")
OUT = ROOT / "data/distill/analysis"
OUT.mkdir(parents=True, exist_ok=True)


def load_rows(p):
    if not p:
        return []
    f = ROOT / p
    if not f.exists() or not f.is_file():
        return []
    d = json.load(open(f))
    return d.get("rows", [])


def by_case(rows, key="case_id"):
    return {r.get(key): r for r in rows if r.get(key) is not None}


def fresh(p):  # ignore stale/garbage files (the 32B crash wrote free=1.0 to all)
    rows = load_rows(p)
    if not rows:
        return {}
    # crude staleness guard: a real run isn't all-correct
    fr = [r for r in rows if "free_correct" in r]
    if fr and all(r["free_correct"] for r in fr):
        return {}
    return by_case(rows)


# ---------- NExT master ----------
traj = {}
for t in glob.glob(str(ROOT / "data/distill/pilot/trajectories/*.json")):
    d = json.load(open(t))
    c = d.get("case", {})
    traj[c.get("case_id")] = {
        "qtype": c.get("question_type"),
        "question": (c.get("question") or "").split("\n")[0][:90],
        "gold": c.get("reference_answer"),
        "n_gold_scenes": len(c.get("gold_scenes") or []),
    }

models = {
    "4b": {"free": "data/distill/pilot/base_freeform_diag.json",
           "selfreflect": "data/distill/pilot/reflection_gap_diag.json",
           "orch": "data/distill/pilot/orch_reflection_diag.json",
           "perc": "data/distill/pilot/perception_headroom_diag.json"},
    "8b": {"free": "data/distill/pilot/base_freeform_diag_8b.json",
           "selfreflect": "data/distill/pilot/reflection_gap_diag_8b.json",
           "orch": "data/distill/pilot/orch_reflection_diag_8b.json",
           "perc": "data/distill/pilot/perception_headroom_diag_8b.json"},
    "30ba3b": {"orch": "data/distill/pilot/orch_reflection_diag_30ba3b.json"},
    "32b": {"orch": "data/distill/pilot/orch_reflection_diag_32b.json"},
}
freecache = {m: by_case(load_rows(srcs.get("free", ""))) for m, srcs in models.items()}
orchcache = {m: fresh(srcs.get("orch", "")) for m, srcs in models.items()}
selfcache = {m: by_case(load_rows(srcs.get("selfreflect", ""))) for m, srcs in models.items()}
perccache = {m: by_case(load_rows(srcs.get("perc", ""))) for m, srcs in models.items()}

cols = ["case_id", "qtype", "question", "gold", "n_gold_scenes"]
for m in models:
    cols += [f"{m}_free", f"{m}_orch"]
cols += ["4b_selfreflect", "8b_selfreflect",
         "4b_perc_gtlocal", "8b_perc_gtlocal"]  # did GT-localized frames fix the free-wrong?

rows = []
for cid, meta in sorted(traj.items()):
    row = {"case_id": cid, **meta}
    for m in models:
        fr = freecache.get(m, {}).get(cid) or orchcache.get(m, {}).get(cid)
        row[f"{m}_free"] = fr.get("free_correct") if fr else ""
        oc = orchcache.get(m, {}).get(cid)
        row[f"{m}_orch"] = oc.get("orch_correct") if oc else ""
    row["4b_selfreflect"] = (selfcache["4b"].get(cid) or {}).get("reflect_correct", "")
    row["8b_selfreflect"] = (selfcache["8b"].get(cid) or {}).get("reflect_correct", "")
    row["4b_perc_gtlocal"] = (perccache["4b"].get(cid) or {}).get("gt_local", "")
    row["8b_perc_gtlocal"] = (perccache["8b"].get(cid) or {}).get("gt_local", "")
    rows.append(row)

with open(OUT / "next_percase.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"wrote next_percase.csv ({len(rows)} cases)")

# ---------- CLEVRER ----------
clev_cases = {}
for line in open(ROOT / "data/eval/datasets/clevrer_pilot/cases.jsonl"):
    if line.strip():
        c = json.loads(line)
        clev_cases[c["case_id"]] = {"qtype": c.get("question_type"),
                                    "question": (c.get("question") or "").split("\n")[0][:90],
                                    "gold": c.get("reference_answer")}
clev_models = {"8b": "data/distill/clevrer/gap_diag_8b.json",
               "30ba3b": "data/distill/clevrer/gap_diag_30ba3b.json",
               "32b": "data/distill/clevrer/gap_diag_32b.json"}
clev_cache = {m: by_case(load_rows(p)) for m, p in clev_models.items()}
ccols = ["case_id", "qtype", "question", "gold"] + [f"{m}_{k}" for m in clev_models for k in ("free", "orch")]
crows = []
for cid, meta in sorted(clev_cases.items()):
    if not any(cid in clev_cache[m] for m in clev_models):
        continue
    row = {"case_id": cid, **meta}
    for m in clev_models:
        r = clev_cache[m].get(cid)
        row[f"{m}_free"] = r.get("free_correct") if r else ""
        row[f"{m}_orch"] = r.get("orch_correct") if r else ""
    crows.append(row)
with open(OUT / "clevrer_percase.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ccols, extrasaction="ignore"); w.writeheader(); w.writerows(crows)
print(f"wrote clevrer_percase.csv ({len(crows)} cases)")

# ---------- auto insights (NExT) ----------
def acc(m, k):
    vals = [r[f"{m}_{k}"] for r in rows if isinstance(r.get(f"{m}_{k}"), bool)]
    return f"{sum(vals)}/{len(vals)}" if vals else "-"

print("\n=== NExT free-form accuracy by model ===")
for m in models:
    print(f"  {m:7s} free={acc(m,'free')}  orch={acc(m,'orch')}")
print("\n=== cases where a BIGGER model fixes what 8B free-form missed ===")
for r in rows:
    if r.get("8b_free") is False:
        fixers = [m for m in ("30ba3b","32b") if r.get(f"{m}_free") is True]
        if fixers:
            print(f"  {r['case_id']} [{r['qtype']}] fixed-by-free:{fixers}  Q:{r['question'][:55]}")
print("\n=== cases where orchestrated reflection FLIPS (per model: gain+ / lost-) ===")
for m in ("4b","8b","30ba3b"):
    g=[r['case_id'] for r in rows if r.get(f"{m}_orch") is True and r.get(f"{m}_free") is False]
    l=[r['case_id'] for r in rows if r.get(f"{m}_orch") is False and r.get(f"{m}_free") is True]
    print(f"  {m}: gain({len(g)})={g}  lost({len(l)})={l}")
