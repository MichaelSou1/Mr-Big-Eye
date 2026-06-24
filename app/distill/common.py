from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
except ModuleNotFoundError:  # langchain not installed in the upgrade env (mbe-up)
    # The trajectory helpers below isinstance()-check these and already fall back to a
    # message.type string, so absent classes (sentinels that match nothing) are harmless.
    # The ChartQA/TabMWP gate harness only uses read_json/write_json from this module.
    class _AbsentMessage:  # sentinel: isinstance(x, _AbsentMessage) is always False
        pass

    AIMessage = HumanMessage = ToolMessage = _AbsentMessage  # type: ignore[assignment,misc]

from app.cache import video_cache_dir
from app.distill import DISTILL_SCHEMA_VERSION, TRAIN_MODALITY
from app.distill.frames import uniform_frame_manifest


TEXT_EVIDENCE_TOOLS = {
    "retrieve_transcript_evidence",
    "search_transcript_keyword",
    "retrieve_slide_evidence",
    "align_audiovisual_evidence",
}
TYPE1_TOOLS = {"segment_focus", "stitched_verify", "answer_with_evidence"}
TYPE2_TOOLS = {
    "retrieve_video_evidence",
    "retrieve_transcript_evidence",
    "search_transcript_keyword",
    "retrieve_slide_evidence",
    "align_audiovisual_evidence",
    "build_timeline",
    "retrieve_hypothesis_evidence",
    "expand_temporal_evidence",
    "assess_evidence_sufficiency",
    "verify_grounding",
    "search_user_memories",
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                items.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return items


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return len(rows)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def dense_frame_path(video_id: str, timestamp: float) -> str:
    # Lazy import: app.preprocess pulls the video stack (chromadb/decord/scenedetect)
    # that the upgrade env (mbe-up) deliberately omits. Only the video-trajectory path
    # reaches here, never the ChartQA/TabMWP gate harness.
    from app.preprocess import dense_frame_filename

    name = dense_frame_filename(float(timestamp))
    path = video_cache_dir(video_id) / "frames_dense" / name
    return str(path) if path.exists() else ""


def frame_manifest(video_id: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    seen: set[float] = set()
    for item in state.get("retrieved_frames", []) or []:
        if not isinstance(item, dict) or "timestamp" not in item:
            continue
        timestamp = round(float(item["timestamp"]), 1)
        if timestamp in seen:
            continue
        seen.add(timestamp)
        frames.append(
            {
                "timestamp": timestamp,
                "source": str(item.get("source") or ""),
                "path": dense_frame_path(video_id, timestamp),
            }
        )
    return sorted(frames, key=lambda item: float(item["timestamp"]))


def message_to_dict(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": str(getattr(message, "type", "") or message.__class__.__name__),
        "content": str(getattr(message, "content", "") or ""),
    }
    if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
        payload["role"] = "user"
    elif isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
        payload["role"] = "assistant"
        payload["tool_calls"] = list(getattr(message, "tool_calls", []) or [])
    elif isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool":
        payload["role"] = "tool"
        payload["tool_call_id"] = str(getattr(message, "tool_call_id", "") or "")
        payload["name"] = str(getattr(message, "name", "") or "")
        payload["payload"] = parse_tool_payload(message)
    return payload


def parse_tool_payload(message: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(getattr(message, "content", "") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_tool_steps(messages: list[Any]) -> list[dict[str, Any]]:
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not (isinstance(message, AIMessage) or getattr(message, "type", "") == "ai"):
            continue
        for call in getattr(message, "tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            if call_id:
                tool_calls_by_id[call_id] = {
                    "tool_name": str(call.get("name") or ""),
                    "tool_args": call.get("args") or {},
                    "tool_call_id": call_id,
                }

    steps: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not (isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool"):
            continue
        call_id = str(getattr(message, "tool_call_id", "") or "")
        payload = parse_tool_payload(message)
        call = tool_calls_by_id.get(call_id, {})
        tool_name = str(payload.get("tool") or call.get("tool_name") or getattr(message, "name", "") or "")
        steps.append(
            {
                "index": index,
                "tool_name": tool_name,
                "tool_args": call.get("tool_args") or {},
                "tool_result": payload,
                "tool_call_id": call_id,
                "internalizability": "type1" if tool_name in TYPE1_TOOLS else "type2",
            }
        )
    return steps


def trajectory_payload(
    *,
    case: Any,
    state: dict[str, Any],
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    messages = list(state.get("messages", []) or [])
    frames = frame_manifest(str(case.video_id), state)
    # The uniform sample is the frame set used downstream (CoT/consistency/SFT);
    # retrieved_frames are kept only as agent provenance.
    sampler_frames = uniform_frame_manifest(str(case.video_id))
    payload = {
        "schema_version": DISTILL_SCHEMA_VERSION,
        "train_modality": TRAIN_MODALITY,
        "case": {
            "case_id": case.case_id,
            "video_id": case.video_id,
            "question": case.question,
            "reference_answer": case.reference_answer,
            "required_keywords": list(case.required_keywords),
            "forbidden_keywords": list(case.forbidden_keywords),
            "gold_timestamps": list(case.gold_timestamps),
            "gold_scenes": list(case.gold_scenes),
            "question_type": case.question_type,
        },
        "run_meta": run_meta,
        "messages": [message_to_dict(message) for message in messages],
        "tool_steps": extract_tool_steps(messages),
        "state": {
            "retrieved_frames": frames,
            "sampler_frames": sampler_frames,
            "retrieved_scene_hits": state.get("retrieved_scene_hits", []) or [],
            "retrieved_transcripts": state.get("retrieved_transcripts", []) or [],
            "retrieved_slides": state.get("retrieved_slides", []) or [],
            "retrieval_plan": state.get("retrieval_plan", {}) or {},
            "timeline": state.get("timeline", []) or [],
            "candidate_timeline": state.get("candidate_timeline", []) or [],
            "audiovisual_candidate_matrix": state.get("audiovisual_candidate_matrix", []) or [],
            "hypotheses": state.get("hypotheses", []) or [],
            "evidence_sufficiency": state.get("evidence_sufficiency", {}) or {},
            "draft_answer": state.get("draft_answer", "") or "",
            "observer_notes": state.get("observer_notes", []) or [],
            "grounding_report": state.get("grounding_report", {}) or {},
            "subject_registry": state.get("subject_registry", []) or [],
            "agent_terminated": state.get("agent_terminated"),
        },
    }
    payload["source_traj_hash"] = stable_hash(
        {
            "case": payload["case"],
            "tool_steps": payload["tool_steps"],
            "state": payload["state"],
            "run_meta": payload["run_meta"],
        }
    )
    return payload


def trajectory_to_prediction(trajectory: dict[str, Any]):
    from app.eval_harness import EvalPrediction

    case = trajectory.get("case") or {}
    state = trajectory.get("state") or {}
    return EvalPrediction(
        case_id=str(case.get("case_id") or ""),
        retrieved_timestamps=[
            float(item["timestamp"])
            for item in state.get("retrieved_frames", []) or []
            if isinstance(item, dict) and "timestamp" in item
        ],
        scene_hits=[
            dict(item)
            for item in state.get("retrieved_scene_hits", []) or []
            if isinstance(item, dict)
        ],
        retrieved_transcripts=[
            dict(item)
            for item in state.get("retrieved_transcripts", []) or []
            if isinstance(item, dict)
        ],
        retrieved_slides=[
            dict(item)
            for item in state.get("retrieved_slides", []) or []
            if isinstance(item, dict)
        ],
        answer=str(state.get("draft_answer") or _last_final_ai_text(trajectory.get("messages", []))),
        agent_actions=[
            str(step.get("tool_name"))
            for step in trajectory.get("tool_steps", []) or []
            if step.get("tool_name")
        ],
        evidence_sufficiency=dict(state.get("evidence_sufficiency", {}) or {}),
        grounding_report=dict(state.get("grounding_report", {}) or {}),
    )


def _last_final_ai_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return str(message.get("content") or "")
    return ""


def sampler_frame_manifest(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """The uniform-sample frame manifest the CoT/SFT/consistency stages consume."""
    state = trajectory.get("state") or {}
    manifest = state.get("sampler_frames")
    if manifest is None:
        # Backward-compat for trajectories produced before the uniform sampler.
        manifest = state.get("retrieved_frames") or []
    return [item for item in manifest if isinstance(item, dict) and "timestamp" in item]


def shown_frame_timestamps(trajectory: dict[str, Any]) -> list[float]:
    return [float(item["timestamp"]) for item in sampler_frame_manifest(trajectory)]
