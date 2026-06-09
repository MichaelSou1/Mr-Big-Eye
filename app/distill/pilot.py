from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from app.distill.filter_consistency import filter_cot_dir
from app.distill.filter_strict import filter_directory
from app.distill.generate import generate_trajectories, select_cases
from app.distill.rewrite import rewrite_from_report
from app.eval_harness import load_cases


def _decision(retention: float) -> str:
    if retention < 0.30:
        return "STOP_SHRINK_TASK_SUBSET"
    if retention < 0.50:
        return "CAUTION_AUTHOR_REVIEW"
    return "PROCEED_TO_SCALE"


async def main_async() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Phase 0-4 50-case distillation pilot.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--work-dir", default="data/distill/pilot")
    parser.add_argument("--per-case-delay-sec", type=float, default=0.0)
    parser.add_argument(
        "--dry-run-rewrite",
        action="store_true",
        help="Use placeholder CoT to test plumbing without REWRITER_*.",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    trajectory_dir = work_dir / "trajectories"
    cot_dir = work_dir / "cot"
    strict_report_path = work_dir / "strict_filter_report.json"
    consistency_report_path = work_dir / "consistency_filter_report.json"

    cases = select_cases(load_cases(args.cases), sample=args.n, seed=args.seed)
    await generate_trajectories(
        cases=cases,
        output_dir=trajectory_dir,
        per_case_delay_sec=args.per_case_delay_sec,
    )
    strict_report = filter_directory(
        trajectory_dir=trajectory_dir,
        output=strict_report_path,
        cases_path=args.cases,
    )
    rewrite_report = rewrite_from_report(
        strict_report_path=strict_report_path,
        output_dir=cot_dir,
        dry_run=args.dry_run_rewrite,
    )
    consistency_report = await filter_cot_dir(
        cot_dir=cot_dir,
        output=consistency_report_path,
    )
    retention = float(consistency_report["summary"]["retention_rate"])
    report = {
        "n_requested": args.n,
        "n_selected": len(cases),
        "strict": strict_report["summary"],
        "rewrite": rewrite_report["summary"],
        "consistency": consistency_report["summary"],
        "decision": _decision(retention),
        "gate_text": (
            "retention <30% => STOP/Shrink; 30-50% => caution; >=50% => proceed to scale"
        ),
    }
    (work_dir / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if retention >= 0.30 else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
