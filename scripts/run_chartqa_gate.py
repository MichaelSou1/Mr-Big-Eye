#!/usr/bin/env python
"""Spec §5 main B — ChartQA variance gate (the regime-crossing dataset). Same protocol
as run_variance_gate.py but for static charts: a single chart image is fed as a 1-frame
"video". free_form (greedy) vs self_reflect / orch_reflect_blind (K seeds). Grading is
relaxed-numeric (5% tol) via eval_common.grade_textaware (open-ended path). Writes to the
shared result store. ChartQA is perception-bound at 4B/8B and (per Exp-8) reasoning-bound
at 32B — this gate tests whether that crossing survives multi-seed.
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

from app.distill.methods import METHOD_FNS, make_orch_sighted
from app.distill.seed_runner import run_method_over_seeds


def load_cases(dump_path, img_dir, keep_ids):
    """Each dump row: {case_id '<tag>-N', question, gold}. Image = <img_dir>/<tag>_N.png
    (prefix derived from the case_id, so chartqa-N -> chartqa_N.png and tabmwp-N -> tabmwp_N.png)."""
    rows = [json.loads(l) for l in open(dump_path) if l.strip()]
    cases = []
    for r in rows:
        cid = r.get("case_id", "")
        if keep_ids is not None and cid not in keep_ids:
            continue
        idx = cid.rsplit("-", 1)[-1]
        prefix = cid.rsplit("-", 1)[0] or "chartqa"
        img = Path(img_dir) / f"{prefix}_{idx}.png"
        if not img.exists():
            continue
        cases.append({"case_id": cid, "question": str(r.get("question") or ""),
                      "gold": str(r.get("gold") or ""),
                      "_frames": [Image.open(img).convert("RGB")], "_ts": [0.0]})
    return cases


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--dataset", default="chartqa",
                    help="dataset tag written to the result store (chartqa | tabmwp)")
    ap.add_argument("--dump", default="data/distill/analysis/dump_8b_chartqa.jsonl",
                    help="source of cases (question+gold); image resolved from img-dir")
    ap.add_argument("--img-dir", default="/home/gpus/mbe_data/chartqa_images")
    ap.add_argument("--clean", default=None, help="label_audit json with clean_ids")
    ap.add_argument("--methods", nargs="+", default=["self_reflect", "orch_reflect_blind"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--no-free-form", action="store_true")
    ap.add_argument("--critic-base", default=None)
    ap.add_argument("--critic-model", default=None)
    args = ap.parse_args()
    if args.critic_base and "orch_reflect_sighted" in args.methods:
        METHOD_FNS["orch_reflect_sighted"] = make_orch_sighted(args.critic_base, args.critic_model)

    keep = None
    if args.clean and Path(args.clean).exists():
        keep = set(json.load(open(args.clean))["clean_ids"])
    cases = load_cases(args.dump, args.img_dir, keep)
    print(f"=== {args.dataset} gate: model={args.model_id} n={len(cases)} cases"
          f"{' (CLEAN)' if args.clean else ''} ===", flush=True)

    def wrap(name):
        base = METHOD_FNS[name]
        def fn(case, seed):
            return asyncio.run(base(case, case["_frames"], case["_ts"], seed))
        return fn

    decode = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 512}
    if not args.no_free_form:
        run_method_over_seeds(dataset=args.dataset, model_id=args.model_id, method="free_form",
                              cases=cases, method_fn=wrap("free_form"), seeds=[0],
                              decode={"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
                              n_frames=1, concurrency=args.concurrency)
    results = []
    for m in args.methods:
        results.append(run_method_over_seeds(dataset=args.dataset, model_id=args.model_id, method=m,
                                             cases=cases, method_fn=wrap(m), seeds=list(range(args.seeds)),
                                             decode=decode, n_frames=1, concurrency=args.concurrency))
    print(f"\n=== SUMMARY ({args.dataset}) ===")
    for r in results:
        print(f"  {r['method']:22s} net_mean={r['net_mean']:+.3f} ± {r['net_std']:.3f} "
              f"(k={r['k']}) verdict={r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
