#!/usr/bin/env python
"""Phase D: compare student (distilled) vs baseline (teacher) eval runs (spec §7.3).

Each input is a harness report JSON (run_meta/summary/results). Aggregates
pass_rate and tool-calls/case as mean±std across seeds for each side and reports
the delta, so the §7.4 acceptance gates can be checked.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path


def _load(paths: list[str]) -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def _pass_rate(report: dict) -> float | None:
    return (report.get("summary") or {}).get("pass_rate")


def _tool_calls_per_case(report: dict) -> float | None:
    results = report.get("results") or []
    counts = [len(r.get("agent_actions") or []) for r in results]
    return (sum(counts) / len(counts)) if counts else None


def _agg(reports: list[dict], fn):
    vals = [v for v in (fn(r) for r in reports) if v is not None]
    if not vals:
        return None
    mean = stats.mean(vals)
    sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": sd, "n": len(vals), "values": vals}


def _fmt(a) -> str:
    return f"{a['mean']:.3f} ± {a['std']:.3f} (n={a['n']})" if a else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", nargs="+", required=True, help="Student run report JSON(s).")
    parser.add_argument("--baseline", nargs="+", required=True, help="Baseline run report JSON(s).")
    parser.add_argument("--out", default="data/distillation/phase_d_report.md")
    args = parser.parse_args()

    student = _load(args.student)
    baseline = _load(args.baseline)

    s_pass = _agg(student, _pass_rate)
    b_pass = _agg(baseline, _pass_rate)
    s_tc = _agg(student, _tool_calls_per_case)
    b_tc = _agg(baseline, _tool_calls_per_case)
    delta_pp = (s_pass["mean"] - b_pass["mean"]) * 100 if (s_pass and b_pass) else None

    def _models(reports):
        return sorted({(r.get("run_meta") or {}).get("orchestrator_model_name") for r in reports})

    lines = [
        "# Phase D report — student vs baseline",
        "",
        f"- student runs: {len(student)}  models={_models(student)}",
        f"- baseline runs: {len(baseline)}  models={_models(baseline)}",
        "",
        "| metric | student | baseline |",
        "|---|---|---|",
        f"| pass_rate | {_fmt(s_pass)} | {_fmt(b_pass)} |",
        f"| tool_calls/case | {_fmt(s_tc)} | {_fmt(b_tc)} |",
        "",
        f"- **pass_rate delta (student - baseline): {delta_pp:+.2f} pp**" if delta_pp is not None else "- pass_rate delta: n/a",
    ]
    if s_pass and b_pass:
        gate3 = abs(delta_pp) <= 3.0
        gate5 = abs(delta_pp) <= 5.0
        lines.append(f"- acceptance: |delta| ≤ 3pp (3-seed): **{gate3}**; ≤ 5pp (single): **{gate5}**")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
