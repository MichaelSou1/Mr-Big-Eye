from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.distill.common import (
    TEXT_EVIDENCE_TOOLS,
    read_json,
    sampler_frame_manifest,
    trajectory_to_prediction,
    write_json,
)
from app.distill.frames import covers_evidence
from app.eval_harness import EvalCase, evaluate_case, load_cases


MAX_DISTILL_TOOL_CALLS = 8


def _case_from_trajectory(trajectory: dict[str, Any]) -> EvalCase:
    case = trajectory.get("case") or {}
    return EvalCase(
        case_id=str(case.get("case_id") or ""),
        video_id=str(case.get("video_id") or ""),
        question=str(case.get("question") or ""),
        gold_timestamps=[float(value) for value in case.get("gold_timestamps", []) or []],
        gold_scenes=[
            {"start": float(item["start"]), "end": float(item["end"])}
            for item in case.get("gold_scenes", []) or []
            if isinstance(item, dict) and "start" in item and "end" in item
        ],
        reference_answer=str(case.get("reference_answer") or ""),
        required_keywords=[str(value) for value in case.get("required_keywords", []) or []],
        forbidden_keywords=[str(value) for value in case.get("forbidden_keywords", []) or []],
        question_type=str(case.get("question_type") or ""),
    )


def _load_case_index(cases_path: str | Path | None) -> dict[str, EvalCase]:
    if not cases_path:
        return {}
    return {case.case_id: case for case in load_cases(cases_path)}


def _tool_names(trajectory: dict[str, Any]) -> list[str]:
    return [
        str(step.get("tool_name") or "")
        for step in trajectory.get("tool_steps", []) or []
        if step.get("tool_name")
    ]


def _has_tool_loop(tool_names: list[str]) -> bool:
    seen: dict[str, int] = {}
    for name in tool_names:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] >= 4:
            return True
    return False


def _scope_filter(trajectory: dict[str, Any]) -> tuple[bool, str]:
    tool_names = set(_tool_names(trajectory))
    state = trajectory.get("state") or {}
    has_text_state = bool(state.get("retrieved_transcripts") or state.get("retrieved_slides"))
    if tool_names & TEXT_EVIDENCE_TOOLS or has_text_state:
        return False, "out_of_scope_text_or_audio"
    return True, ""


def _evidence_coverage(trajectory: dict[str, Any], case: EvalCase) -> tuple[bool, str]:
    """Drop cases the fixed uniform sample fails to cover (evidence not visible).

    Cases with no grounding GT (e.g. Video-MME supplement) cannot be falsified, so
    they pass with a flag — the perception-routing capability they would need is
    explicitly out of scope, and they already bypass grounding-dependent probes.
    """
    if not case.gold_timestamps and not case.gold_scenes:
        return True, "no_grounding_gt"
    sampled = [float(item["timestamp"]) for item in sampler_frame_manifest(trajectory)]
    if covers_evidence(sampled, case.gold_timestamps, case.gold_scenes):
        return True, ""
    return False, "evidence_not_in_uniform_sample"


def strict_filter_trajectory(
    trajectory: dict[str, Any],
    *,
    case_override: EvalCase | None = None,
    tolerance_sec: float = 2.0,
) -> dict[str, Any]:
    case = case_override or _case_from_trajectory(trajectory)
    prediction = trajectory_to_prediction(trajectory)
    result = evaluate_case(
        case,
        prediction,
        tolerance_sec=tolerance_sec,
        distill_strict=True,
    )
    tool_names = _tool_names(trajectory)
    scope_ok, scope_reason = _scope_filter(trajectory)
    coverage_ok, coverage_reason = _evidence_coverage(trajectory, case)
    reasons: list[str] = []
    if not result["passed"]:
        reasons.extend(result.get("failure_tags", []) or ["strict_eval_failed"])
    if not scope_ok:
        reasons.append(scope_reason)
    if not coverage_ok:
        reasons.append(coverage_reason)
    state = trajectory.get("state") or {}
    if state.get("agent_terminated") == "cap" or len(tool_names) > int(
        settings.orchestrator_max_tool_calls or MAX_DISTILL_TOOL_CALLS
    ):
        reasons.append("tool_cap_hit")
    if _has_tool_loop(tool_names):
        reasons.append("tool_loop_detected")
    grounding_report = state.get("grounding_report") or {}
    if grounding_report and grounding_report.get("grounded") is False:
        reasons.append("grounding_report_failed")
    passed = not reasons
    return {
        "case_id": case.case_id,
        "video_id": case.video_id,
        "passed": passed,
        "drop_reasons": sorted(set(reasons)),
        "scope_ok": scope_ok,
        "coverage_ok": coverage_ok,
        "no_grounding_gt": coverage_reason == "no_grounding_gt",
        "strict_eval": result,
        "tool_count": len(tool_names),
        "tool_names": tool_names,
    }


def filter_directory(
    *,
    trajectory_dir: Path,
    output: Path,
    cases_path: str | Path | None = None,
    tolerance_sec: float = 2.0,
) -> dict[str, Any]:
    case_index = _load_case_index(cases_path)
    results: list[dict[str, Any]] = []
    passed_paths: list[str] = []
    for path in sorted(trajectory_dir.glob("*.json")):
        trajectory = read_json(path)
        case_id = str((trajectory.get("case") or {}).get("case_id") or path.stem)
        result = strict_filter_trajectory(
            trajectory,
            case_override=case_index.get(case_id),
            tolerance_sec=tolerance_sec,
        )
        result["trajectory_path"] = str(path)
        results.append(result)
        if result["passed"]:
            passed_paths.append(str(path))
    summary = {
        "total": len(results),
        "passed": len(passed_paths),
        "pass_rate": len(passed_paths) / len(results) if results else 0.0,
        "out_of_scope": sum(
            1 for item in results
            if "out_of_scope_text_or_audio" in item.get("drop_reasons", [])
        ),
        "evidence_uncovered": sum(
            1 for item in results
            if "evidence_not_in_uniform_sample" in item.get("drop_reasons", [])
        ),
        "no_grounding_gt": sum(1 for item in results if item.get("no_grounding_gt")),
        "bad_process": sum(
            1 for item in results
            if item.get("drop_reasons")
            and not ({"out_of_scope_text_or_audio", "evidence_not_in_uniform_sample"}
                     & set(item.get("drop_reasons", [])))
        ),
    }
    report = {"summary": summary, "passed_trajectories": passed_paths, "results": results}
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly filter distillation trajectories.")
    parser.add_argument("--trajectory-dir", default="data/distill/trajectories")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--output", default="data/distill/strict_filter_report.json")
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    args = parser.parse_args()

    report = filter_directory(
        trajectory_dir=Path(args.trajectory_dir),
        output=Path(args.output),
        cases_path=args.cases,
        tolerance_sec=args.tolerance_sec,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
