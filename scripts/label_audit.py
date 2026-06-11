#!/usr/bin/env python
"""Spec §0.5.2 — LLM-judge label-ambiguity audit. For each case (from the dumps) the
30B text judge sees the question+options+gold and the model's free-form reading (as
scene evidence/rationale) and verdicts {clean | ambiguous | wrong_gold}. Outputs the
CLEAN case-id set + label-noise rate (with a bootstrap CI). Hand-spot-check 30 after.

Judge is text-only (Spec §0.5.2: "given question, options, gold, a short rationale").
Needs the orchestrator (30B) served at settings.orchestrator_api_base_url.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from app.distill.methods import orch
from app.distill.eval_stats import paired_bootstrap_net  # reuse bootstrap for a proportion CI

# Text-only SEMANTIC ambiguity. We deliberately do NOT show the model's reading: a judge
# given a (frequently wrong) VLM description conflates perception-error with label-error and
# wildly over-declares "wrong gold". Without the video the judge cannot rule the gold wrong,
# so it only assesses whether the OPTIONS are semantically separable and the gold is the
# unique best phrasing. This isolates the defensible label-ambiguity signal (grazing≈feeding).
JUDGE_SYS = (
    "You audit a multiple-choice question's OPTIONS for ambiguity. You CANNOT see the video. "
    "Given the question, options, and which option is gold, decide whether the options are "
    "cleanly separable so that, for SOME plausible video, the gold is the single best answer "
    "(clean) — or whether two or more options are near-synonyms / overlapping so that they "
    "could BOTH be correct for the same video (ambiguous), e.g. 'grazing' vs 'feeding', "
    "'excited' vs 'enjoying'. Most items are clean. Judge ONLY option separability, not which "
    'is true. Output ONLY JSON: {"verdict": "clean|ambiguous", "reason": "<short>"}.'
)


def judge_case(question, gold, free_answer, seed=0) -> dict:
    out = orch([{"role": "system", "content": JUDGE_SYS},
               {"role": "user", "content":
                f"QUESTION + OPTIONS:\n{question}\n\nGOLD OPTION: {gold}"}],
               temp=0.0, seed=seed, max_tokens=1024)  # reasoning-model headroom
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return {"verdict": "clean", "reason": "parse_fail", "raw": out[:120]}
    try:
        d = json.loads(m.group(0))
        v = str(d.get("verdict", "clean")).strip().lower()
        if v not in ("clean", "ambiguous"):
            v = "clean"
        return {"verdict": v, "reason": str(d.get("reason", ""))[:160]}
    except Exception:
        return {"verdict": "clean", "reason": "json_fail", "raw": out[:120]}


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="data/distill/analysis/dump_<model>_<ds>.jsonl")
    ap.add_argument("--out", required=True, help="data/distill/analysis/label_audit_<ds>.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    audited, clean_ids = [], []
    counts = {"clean": 0, "ambiguous": 0, "wrong_gold": 0}
    for i, r in enumerate(rows):
        v = judge_case(r.get("question", ""), str(r.get("gold", "")), r.get("free_answer", ""))
        counts[v["verdict"]] += 1
        if v["verdict"] == "clean":
            clean_ids.append(r.get("case_id"))
        audited.append({"case_id": r.get("case_id"), "qtype": r.get("qtype"),
                        "gold": r.get("gold"), "verdict": v["verdict"], "reason": v["reason"]})
        if (i + 1) % 20 == 0:
            print(f"  judged {i+1}/{len(rows)}", flush=True)

    n = len(rows)
    clean_flags = [1 if a["verdict"] == "clean" else 0 for a in audited]
    # noise rate = 1 - clean_rate; CI via bootstrap of the clean proportion
    boot = paired_bootstrap_net([0] * n, clean_flags)   # net = clean_rate (free=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": {"n": n, "counts": counts,
                                "clean_rate": counts["clean"] / n if n else 0,
                                "noise_rate": 1 - counts["clean"] / n if n else 0,
                                "clean_rate_ci": [boot["ci_lo"], boot["ci_hi"]]},
                    "clean_ids": clean_ids, "rows": audited}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"n={n} clean={counts['clean']} ambiguous={counts['ambiguous']} "
          f"wrong_gold={counts['wrong_gold']} | noise_rate="
          f"{1-counts['clean']/n:.2f} clean_CI[{boot['ci_lo']:.2f},{boot['ci_hi']:.2f}] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
