#!/usr/bin/env python
"""Spec §1 — the variance gate. free_form (greedy, 1 run) vs self_reflect / orch_blind
(K seeds, temp>0) on NExT-GQA CLEAN ∩ EVIDENCE_IN, with paired bootstrap CIs. Writes to
the result store (app/distill/seed_runner) and prints the per-setting table + verdict.

Reuses: app/distill/methods.py (the methods w/ temp/seed), seed_runner, eval_stats.
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

from app.config import settings
from app.distill.common import read_json, sampler_frame_manifest
from app.distill.methods import METHOD_FNS, make_orch_sighted
from app.distill.seed_runner import run_method_over_seeds


def _load_frames(traj):
    frames, ts = [], []
    for it in sampler_frame_manifest(traj):
        p = Path(str(it.get("path") or ""))
        if p.exists():
            frames.append(Image.open(p).convert("RGB"))
            ts.append(float(it["timestamp"]))
    return frames, ts


def load_cases(traj_dir, keep_ids):
    cases = []
    for p in sorted(Path(traj_dir).glob("*.json")):
        t = read_json(p)
        c = t.get("case") or {}
        cid = c.get("case_id")
        if keep_ids is not None and cid not in keep_ids:
            continue
        frames, ts = _load_frames(t)
        if not frames:
            continue
        cases.append({"case_id": cid, "question": str(c.get("question") or ""),
                      "gold": str(c.get("reference_answer") or ""), "_frames": frames, "_ts": ts})
    return cases


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="label for the served VLM, e.g. 8b")
    ap.add_argument("--traj-dir", default="data/distill/pilot/trajectories")
    ap.add_argument("--partition", default="data/distill/analysis/partition_next.json")
    ap.add_argument("--clean", default=None, help="label_audit json with clean_ids")
    ap.add_argument("--methods", nargs="+", default=["self_reflect", "orch_reflect_blind"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--critic-base", default=None, help="sighted-critic VLM base_url (enables orch_reflect_sighted)")
    ap.add_argument("--critic-model", default=None, help="sighted-critic VLM served-model-name")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel cases per seed (orch throughput lever)")
    ap.add_argument("--no-free-form", action="store_true", help="skip free_form baseline (already in store)")
    args = ap.parse_args()

    # §4: register the sighted critic if a 2nd VLM endpoint is given.
    if args.critic_base and "orch_reflect_sighted" in args.methods:
        METHOD_FNS["orch_reflect_sighted"] = make_orch_sighted(args.critic_base, args.critic_model)

    keep = None
    if Path(args.partition).exists():
        keep = set(json.load(open(args.partition))["evidence_in"])
    if args.clean and Path(args.clean).exists():
        clean = set(json.load(open(args.clean))["clean_ids"])
        keep = clean if keep is None else (keep & clean)
    cases = load_cases(args.traj_dir, keep)
    print(f"=== variance gate: model={args.model_id} n={len(cases)} cases "
          f"(EVIDENCE_IN{'∩CLEAN' if args.clean else ''}) ===", flush=True)

    def wrap(name):
        base = METHOD_FNS[name]
        def fn(case, seed):
            return asyncio.run(base(case, case["_frames"], case["_ts"], seed))
        return fn

    decode = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 512}
    # free_form baseline: greedy, 1 seed (skip when re-running a method on an already-scored split)
    if not args.no_free_form:
        run_method_over_seeds(dataset="next", model_id=args.model_id, method="free_form",
                              cases=cases, method_fn=wrap("free_form"), seeds=[0],
                              decode={"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
                              n_frames=16, concurrency=args.concurrency)
    results = []
    for m in args.methods:
        r = run_method_over_seeds(dataset="next", model_id=args.model_id, method=m, cases=cases,
                                  method_fn=wrap(m), seeds=list(range(args.seeds)),
                                  decode=decode, n_frames=16, concurrency=args.concurrency)
        results.append(r)
    print("\n=== SUMMARY (NExT CLEAN∩EVIDENCE_IN) ===")
    for r in results:
        print(f"  {r['method']:22s} net_mean={r['net_mean']:+.3f} ± seedstd {r['net_std']:.3f} "
              f"(k={r['k']})  verdict={r['verdict']}  per_seed={[f'{x:+.2f}' for x in r['per_seed_nets']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
