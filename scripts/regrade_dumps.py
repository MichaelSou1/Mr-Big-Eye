#!/usr/bin/env python
"""Spec §0.5.1 — re-grade the existing per-case dumps with the TEXT-AWARE grader and
report letter-luck. Runs entirely on data/distill/analysis/dump_*.jsonl (no GPU, no
model calls): the dumps already store the full free/final answer TEXT.

Output: per (model, dataset) — letter-only acc vs text-aware acc, and the count of
letter-luck cases (letter matched gold but the prose endorsed a different option).
Writes data/distill/analysis/regrade_<model>_<dataset>.jsonl with per-case verdicts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distill.eval_common import grade_textaware

MODELS = ["4b", "8b", "30ba3b", "32b"]
DATASETS = ["next", "clevrer", "chartqa"]


def regrade_file(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    out = []
    free_letter = free_text = orch_letter = orch_text = 0
    free_luck = orch_luck = 0
    for r in rows:
        q, gold = r.get("question", ""), str(r.get("gold", ""))
        fg = grade_textaware(q, gold, r.get("free_answer", ""))
        og = grade_textaware(q, gold, r.get("final_answer", ""))
        free_letter += int(bool(r.get("free_correct")))
        orch_letter += int(bool(r.get("orch_correct")))
        free_text += int(fg["correct"]); orch_text += int(og["correct"])
        free_luck += int(fg.get("letter_luck", False)); orch_luck += int(og.get("letter_luck", False))
        out.append({"case_id": r.get("case_id"), "qtype": r.get("qtype"),
                    "free_correct_letter": bool(r.get("free_correct")), "free_correct_text": fg["correct"],
                    "free_letter_luck": fg.get("letter_luck", False),
                    "orch_correct_letter": bool(r.get("orch_correct")), "orch_correct_text": og["correct"],
                    "orch_letter_luck": og.get("letter_luck", False)})
    n = len(rows)
    summary = {"n": n,
               "free_acc_letter": free_letter / n if n else 0, "free_acc_text": free_text / n if n else 0,
               "orch_acc_letter": orch_letter / n if n else 0, "orch_acc_text": orch_text / n if n else 0,
               "free_letter_luck": free_luck, "orch_letter_luck": orch_luck}
    return out, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="data/distill/analysis")
    args = ap.parse_args()
    ad = Path(args.analysis_dir)
    print(f"{'model':7s}{'dataset':9s}{'n':>4s}{'free.L':>8s}{'free.T':>8s}{'Δfree':>7s}"
          f"{'orch.L':>8s}{'orch.T':>8s}{'f.luck':>7s}{'o.luck':>7s}")
    grand = {}
    for m in MODELS:
        for ds in DATASETS:
            f = ad / f"dump_{m}_{ds}.jsonl"
            if not f.exists():
                continue
            rows, s = regrade_file(f)
            (ad / f"regrade_{m}_{ds}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            print(f"{m:7s}{ds:9s}{s['n']:>4d}{s['free_acc_letter']:>8.3f}{s['free_acc_text']:>8.3f}"
                  f"{s['free_acc_text']-s['free_acc_letter']:>+7.3f}{s['orch_acc_letter']:>8.3f}"
                  f"{s['orch_acc_text']:>8.3f}{s['free_letter_luck']:>7d}{s['orch_letter_luck']:>7d}")
            grand[f"{m}_{ds}"] = s
    (ad / "regrade_summary.json").write_text(json.dumps(grand, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
