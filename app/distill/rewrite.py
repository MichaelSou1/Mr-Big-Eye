from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.distill.common import (
    read_json,
    sampler_frame_manifest,
    shown_frame_timestamps,
    stable_hash,
    write_json,
)
from app.eval_harness import FRAME_MARKER_RE, SLIDE_MARKER_RE, TRANSCRIPT_MARKER_RE


FORBIDDEN_TOOL_TOKENS = (
    "retrieve_video_evidence",
    "retrieve_transcript_evidence",
    "search_transcript_keyword",
    "retrieve_slide_evidence",
    "align_audiovisual_evidence",
    "build_timeline",
    "retrieve_hypothesis_evidence",
    "segment_focus",
    "expand_temporal_evidence",
    "stitched_verify",
    "assess_evidence_sufficiency",
    "answer_with_evidence",
    "verify_grounding",
    "search_user_memories",
    "retriev",
    "resolver",
    "fts",
    "rrf",
    "rank",
    "i searched",
    "tool",
)
FORBIDDEN_NUMERIC_PATTERNS = (
    re.compile(r"\bscore\s*[:=]?\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"\bmatch\s+score\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"\bconfidence\s*[:=]?\s*\d+(?:\.\d+)?", re.IGNORECASE),
)


@dataclass
class RewriteValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    frame_markers: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "frame_markers": list(self.frame_markers),
        }


def snap_markers_to_shown(cot: str, shown_frames: list[float], *, max_snap_sec: float = 2.5) -> str:
    """Rewrite each [FRAME:t=X] to the nearest shown (uniform-sample) timestamp.

    The rewriter often cites timestamps that leaked from the agent's retrieved
    frames (present in the trajectory context) rather than the uniform sample we
    actually feed the model. Snapping to the nearest shown frame within
    ``max_snap_sec`` makes the stored CoT cite REAL shown frames (correct for SFT
    and the §7.3c claimed-timestamp probe); markers farther than ``max_snap_sec``
    from any shown frame are left as-is so validation rejects genuine
    hallucinations.
    """
    if not shown_frames:
        return cot

    def _replace(match: "re.Match[str]") -> str:
        value = float(match.group(1))
        nearest = min(shown_frames, key=lambda s: abs(s - value))
        if abs(nearest - value) <= max_snap_sec:
            return f"[FRAME:t={nearest:.1f}]"
        return match.group(0)

    return FRAME_MARKER_RE.sub(_replace, cot or "")


def validate_cot(cot: str, shown_frames: list[float], *, tolerance_sec: float = 0.65) -> RewriteValidation:
    errors: list[str] = []
    text = cot or ""
    lowered = text.lower()
    for token in FORBIDDEN_TOOL_TOKENS:
        if token in lowered:
            errors.append(f"forbidden_token:{token}")
    for pattern in FORBIDDEN_NUMERIC_PATTERNS:
        if pattern.search(text):
            errors.append(f"forbidden_numeric:{pattern.pattern}")
    if TRANSCRIPT_MARKER_RE.search(text):
        errors.append("forbidden_transcript_marker")
    if SLIDE_MARKER_RE.search(text):
        errors.append("forbidden_slide_marker")
    markers = [float(match.group(1)) for match in FRAME_MARKER_RE.finditer(text)]
    if not markers:
        errors.append("missing_frame_marker")
    for marker in markers:
        nearest = min((abs(marker - shown) for shown in shown_frames), default=None)
        if nearest is None or nearest > tolerance_sec:
            errors.append(f"unknown_frame_marker:{marker:.1f}")
    return RewriteValidation(ok=not errors, errors=sorted(set(errors)), frame_markers=markers)


def _rewrite_prompt(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    case = trajectory.get("case") or {}
    state = trajectory.get("state") or {}
    compact = {
        "case": case,
        "shown_frames": sampler_frame_manifest(trajectory),
        "tool_steps": trajectory.get("tool_steps", []),
        "observer_notes": state.get("observer_notes", []),
        "draft_answer": state.get("draft_answer", ""),
        "grounding_report": state.get("grounding_report", {}),
        "subject_registry": state.get("subject_registry", []),
        "candidate_timeline": state.get("candidate_timeline", []),
        "audiovisual_candidate_matrix": state.get("audiovisual_candidate_matrix", []),
    }
    system = (
        "You rewrite agent trajectories into concise first-person visual Chain-of-Thought "
        "for a single-forward VLM. The target model sees only the provided frames once. "
        "Keep only visual reasoning that is directly inferable from pixels and anchored "
        "to [FRAME:t=...] markers present in shown_frames. Never mention tools, retrieval, "
        "search, resolver decisions, scores, ranks, transcript, OCR, slides, or audio. "
        "Strip assess/verify/meta content. End with the verified final answer."
    )
    allowed_ts = [round(float(item["timestamp"]), 1) for item in compact["shown_frames"]
                  if isinstance(item, dict) and "timestamp" in item]
    user = (
        "Rewrite this trajectory into JSON with keys cot and answer. "
        "Every [FRAME:t=X] marker MUST use a timestamp taken EXACTLY from this "
        f"allowed list (pick the closest one, do not invent values): {allowed_ts}\n\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_rewriter_json(text: str) -> dict[str, str]:
    snippet = text.strip()
    if snippet.startswith("```"):
        snippet = snippet.strip("`")
        if snippet.lower().startswith("json"):
            snippet = snippet[4:]
        snippet = snippet.strip()
    if not snippet.startswith("{"):
        match = re.search(r"\{.*\}", snippet, re.DOTALL)
        if match:
            snippet = match.group(0)
    data = json.loads(snippet)
    return {
        "cot": str(data.get("cot") or ""),
        "answer": str(data.get("answer") or ""),
    }


def call_rewriter(trajectory: dict[str, Any]) -> dict[str, str]:
    if not settings.rewriter_api_base_url or not settings.rewriter_model_name:
        raise RuntimeError("REWRITER_API_BASE_URL and REWRITER_MODEL_NAME are required.")
    payload = {
        "model": settings.rewriter_model_name,
        "messages": _rewrite_prompt(trajectory),
        "temperature": 0.0,
        "max_tokens": 1200,
    }
    headers = {"Content-Type": "application/json"}
    if settings.rewriter_api_key:
        headers["Authorization"] = f"Bearer {settings.rewriter_api_key}"
    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{settings.rewriter_api_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
    text = body["choices"][0]["message"]["content"]
    return _parse_rewriter_json(text)


def rewrite_trajectory(
    trajectory: dict[str, Any],
    *,
    dry_run_cot: str | None = None,
) -> dict[str, Any]:
    shown_frames = shown_frame_timestamps(trajectory)
    if dry_run_cot is None:
        rewritten = call_rewriter(trajectory)
    else:
        rewritten = {
            "cot": dry_run_cot,
            "answer": str((trajectory.get("state") or {}).get("draft_answer") or ""),
        }
    # Snap cited markers to the actual uniform-sample frames before validation:
    # the rewriter tends to echo agent-retrieved timestamps from the trajectory
    # context rather than the shown set.
    rewritten["cot"] = snap_markers_to_shown(rewritten["cot"], shown_frames)
    validation = validate_cot(rewritten["cot"], shown_frames)
    case = trajectory.get("case") or {}
    state = trajectory.get("state") or {}
    return {
        "schema_version": "cot_v1",
        "video_id": str(case.get("video_id") or ""),
        "case_id": str(case.get("case_id") or ""),
        "question": str(case.get("question") or ""),
        "shown_frames": shown_frames,
        "frame_paths": [
            str(item.get("path") or "")
            for item in sampler_frame_manifest(trajectory)
            if isinstance(item, dict)
        ],
        "cot": rewritten["cot"],
        "answer": rewritten["answer"] or str(state.get("draft_answer") or ""),
        # Grounding GT carried through for the §7.3c claimed-timestamp probe and
        # any GT-based reward; absent for grounding-less supplements (Video-MME).
        "gold_timestamps": [float(value) for value in case.get("gold_timestamps", []) or []],
        "gold_scenes": [
            {"start": float(item["start"]), "end": float(item["end"])}
            for item in case.get("gold_scenes", []) or []
            if isinstance(item, dict) and "start" in item and "end" in item
        ],
        "source_traj_hash": trajectory.get("source_traj_hash") or stable_hash(trajectory),
        "validation_report": validation.as_dict(),
    }


def rewrite_from_report(
    *,
    strict_report_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    strict_report = read_json(strict_report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for trajectory_path in strict_report.get("passed_trajectories", []) or []:
        trajectory = read_json(trajectory_path)
        dry_cot = None
        if dry_run:
            frames = shown_frame_timestamps(trajectory)
            marker = f"[FRAME:t={frames[0]:.1f}]" if frames else "[FRAME:t=0.0]"
            dry_cot = f"I inspect the visible frame evidence {marker}. Therefore the answer follows."
        cot_payload = rewrite_trajectory(trajectory, dry_run_cot=dry_cot)
        path = output_dir / f"{cot_payload['case_id']}.json"
        write_json(path, cot_payload)
        results.append(
            {
                "case_id": cot_payload["case_id"],
                "output_path": str(path),
                "passed": bool(cot_payload["validation_report"]["ok"]),
                "errors": cot_payload["validation_report"]["errors"],
            }
        )
    summary = {
        "total": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "pass_rate": (
            sum(1 for item in results if item["passed"]) / len(results)
            if results else 0.0
        ),
    }
    return {"summary": summary, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite strict trajectories into visual CoT.")
    parser.add_argument("--strict-report", default="data/distill/strict_filter_report.json")
    parser.add_argument("--output-dir", default="data/distill/cot")
    parser.add_argument("--dry-run", action="store_true", help="Generate placeholder CoT for pipeline tests.")
    args = parser.parse_args()

    report = rewrite_from_report(
        strict_report_path=Path(args.strict_report),
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
