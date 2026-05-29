#!/usr/bin/env python
"""Phase A acceptance check (spec §4.3).

Reads a raw-trajectory JSONL and reports field completeness, judge pass rate,
guard distribution, message/tool-call length, and the real tier-1 yield — the
number that decides whether to scale data collection now.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.distill_filter import classify_tier
from app.distill_trajectory import recomputed_guards


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tool_calls_in(messages: list[dict]) -> int:
    return sum(len(m.get("tool_calls") or []) for m in messages if m.get("role") == "assistant")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--expected", type=int, default=None, help="Expected #cases for the >=90%% check.")
    parser.add_argument("--out", default="data/distillation/phase_a_report.md")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.trajectories))
    n = len(rows)
    # Normalize guards to current logic (recompute from message stream).
    for r in rows:
        r["guards_triggered"] = recomputed_guards(r)

    missing_fields = Counter()
    for r in rows:
        if not r.get("messages"):
            missing_fields["messages"] += 1
        if not r.get("system_prompt"):
            missing_fields["system_prompt"] += 1
        if "guards_triggered" not in r:
            missing_fields["guards_triggered"] += 1
        if "judge_correct" not in r:
            missing_fields["judge_correct"] += 1

    judged = [r for r in rows if r.get("judge_correct") is not None]
    judge_correct = sum(1 for r in judged if r.get("judge_correct"))
    guard_dist = Counter()
    for r in rows:
        for g in r.get("guards_triggered") or []:
            guard_dist[g] += 1
    terminated = Counter(str(r.get("agent_terminated")) for r in rows)
    tiers = Counter(classify_tier(r) for r in rows)
    tool_calls = [_tool_calls_in(r.get("messages") or []) for r in rows]
    msg_lens = [len(r.get("messages") or []) for r in rows]

    expected = args.expected or n
    coverage_ok = n >= 0.9 * expected if expected else True
    fields_ok = not missing_fields
    tier1_yield = (tiers.get("tier_1", 0) / n) if n else 0.0

    lines = [
        "# Phase A report — teacher trajectory collection",
        "",
        f"- trajectories: **{n}** (expected {expected}, coverage_ok={coverage_ok})",
        f"- field completeness ok: **{fields_ok}**" + (f"  missing={dict(missing_fields)}" if missing_fields else ""),
        f"- judge correct rate: **{judge_correct}/{len(judged)} = {judge_correct/len(judged):.3f}**" if judged else "- judge correct rate: n/a (no judge)",
        f"- **tier-1 yield: {tiers.get('tier_1',0)}/{n} = {tier1_yield:.3f}**  (tier_2={tiers.get('tier_2',0)}, tier_3={tiers.get('tier_3',0)})",
        f"- mean tool_calls/case: **{mean(tool_calls):.2f}**" if tool_calls else "- mean tool_calls/case: n/a",
        f"  (p_min={min(tool_calls)}, max={max(tool_calls)})" if tool_calls else "",
        f"- mean messages/case: **{mean(msg_lens):.2f}** (max={max(msg_lens)})" if msg_lens else "",
        f"- guard distribution: `{dict(guard_dist)}`",
        f"- agent_terminated distribution: `{dict(terminated)}`",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(l for l in lines if l != "") + "\n", encoding="utf-8")
    print("\n".join(l for l in lines if l != ""))
    print(f"\nwrote {out}")
    return 0 if (coverage_ok and fields_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
