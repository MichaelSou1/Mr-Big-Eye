from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from langgraph.types import Command
from PIL import Image
from pydantic import BeforeValidator, Field

from app import memory
from app.cache import load_meta, video_cache_dir
from app.config import settings
from app.text_assets import nearby_text, search_keyword, search_text
from app.vqa import (
    ANSWER_WITH_EVIDENCE_PROMPT,
    SEGMENT_FOCUS_PROMPT,
    STITCHED_VERIFY_PROMPT,
    answer_question,
)


QUESTION_TYPES = {
    "overview",
    "event_location",
    "temporal_order",
    "counting",
    "comparison",
    "visual_detail",
    "text_ocr",
    "existence",
    "general",
}

RETRIEVAL_PROFILES = {
    "focused",
    "balanced",
    "broad",
    "temporal",
    "detail",
    "negative_check",
}

TEMPORAL_QUESTION_TYPES = {"temporal_order", "counting", "comparison"}
NEGATIVE_MARKERS = (
    "no ",
    "not ",
    "none",
    "absent",
    "doesn't",
    "didn't",
    "cannot see",
    "can't see",
    "没有",
    "不存在",
    "未看到",
    "看不到",
    "没看到",
    "无法看到",
)
NEGATIVE_SCOPE_MARKERS = (
    "checked",
    "available",
    "provided",
    "retrieved",
    "evidence",
    "frames",
    "已检查",
    "当前",
    "现有",
    "提供",
    "检索",
    "证据",
    "关键帧",
)
VISUAL_CLAIM_MARKERS = (
    "shows",
    "shown",
    "visible",
    "appears",
    "looks",
    "wearing",
    "holding",
    "standing",
    "walking",
    "running",
    "moving",
    "object",
    "person",
    "scene",
    "frame",
    "看到",
    "画面",
    "显示",
    "出现",
    "人物",
    "人",
    "物体",
    "颜色",
    "穿",
    "拿",
    "站",
    "走",
    "跑",
    "移动",
    "动作",
    "镜头",
)
FRAME_MARKER_RE = re.compile(r"\[FRAME:t=([0-9]+(?:\.[0-9]+)?)\]")
TRANSCRIPT_MARKER_RE = re.compile(r"\[TRANSCRIPT:t=([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\]")
SLIDE_MARKER_RE = re.compile(r"\[SLIDE:t=([0-9]+(?:\.[0-9]+)?)\]")
DENSE_FRAME_RE = re.compile(r"^t([0-9]+(?:\.[0-9]+)?)\.jpg$")
SUBJECT_REGISTRY_MAX = 15
SEGMENT_FOCUS_MAX_FRAMES = 12
STITCHED_VERIFY_MAX_WINDOWS = 4
STITCHED_VERIFY_MAX_FRAMES = 24

OBSERVER_NOTE_FOR_ORCHESTRATOR = (
    "This is an Observer sub-call's intermediate observation, NOT the final user answer. "
    "Do NOT emit the `observation` field to the user. You MUST call `answer_with_evidence` "
    "(then `verify_grounding`) before producing the final answer. For MCQ questions, "
    "the final answer MUST be in the form 'The correct answer is X) <option>'."
)


def _coerce_float_list(value: Any) -> list[float] | None:
    """Accept list[float], JSON-encoded list, or comma-separated string from tool callers."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [chunk.strip() for chunk in text.strip("[]").split(",") if chunk.strip()]
        value = parsed
    if not isinstance(value, (list, tuple)):
        raise ValueError("timestamps must be a list of numbers")
    out: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            out.append(float(item))
        elif isinstance(item, str) and item.strip():
            out.append(float(item.strip()))
        else:
            raise ValueError("timestamps entries must be numeric")
    return out


TimestampList = Annotated[list[float] | None, BeforeValidator(_coerce_float_list)]


def merge_subject_deltas(
    registry: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for subject in registry or []:
        if not isinstance(subject, dict) or not subject.get("id"):
            continue
        sid = str(subject["id"])
        by_id[sid] = {
            **subject,
            "id": sid,
            "label": subject.get("label") or sid,
            "attributes": list(subject.get("attributes") or []),
            "evidence_frames": list(subject.get("evidence_frames") or []),
        }

    for delta in deltas or []:
        if not isinstance(delta, dict):
            continue
        op = str(delta.get("op") or "").strip().lower()
        sid = str(delta.get("id") or "").strip()
        if not sid:
            continue
        if op == "add":
            first_seen = _optional_float(delta.get("first_seen_t"))
            if sid not in by_id:
                by_id[sid] = {
                    "id": sid,
                    "label": delta.get("label") or sid,
                    "first_seen_t": first_seen,
                    "last_seen_t": _optional_float(delta.get("last_seen_t")) or first_seen,
                    "attributes": _string_list(delta.get("attributes")),
                    "evidence_frames": _float_list(delta.get("evidence_frames")),
                }
            else:
                entry = by_id[sid]
                _merge_subject_entry(
                    entry,
                    attributes=_string_list(delta.get("attributes")),
                    evidence_frames=_float_list(delta.get("evidence_frames")),
                    last_seen_t=_optional_float(delta.get("last_seen_t")) or first_seen,
                )
                if first_seen is not None:
                    current_first = _optional_float(entry.get("first_seen_t"))
                    entry["first_seen_t"] = (
                        min(current_first, first_seen) if current_first is not None else first_seen
                    )
        elif op == "update" and sid in by_id:
            _merge_subject_entry(
                by_id[sid],
                attributes=_string_list(delta.get("attributes_add")),
                evidence_frames=_float_list(delta.get("evidence_frames_add")),
                last_seen_t=_optional_float(delta.get("last_seen_t")),
            )

    merged = list(by_id.values())
    merged.sort(key=lambda subject: _optional_float(subject.get("last_seen_t")) or 0.0, reverse=True)
    return merged[:SUBJECT_REGISTRY_MAX]


def parse_subject_deltas(answer_text: str) -> tuple[str, list[dict[str, Any]]]:
    lines = (answer_text or "").splitlines()
    delta_index: int | None = None
    prefix = "SUBJECT_DELTAS:"
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip().startswith(prefix):
            delta_index = index
            break
    if delta_index is None:
        return (answer_text or "").strip(), []

    raw_json = lines[delta_index].strip()[len(prefix) :].strip()
    parsed: Any | None = None
    remove_from_index = False
    for candidate, remove_rest in (
        (raw_json, False),
        ("\n".join([raw_json, *lines[delta_index + 1 :]]).strip(), True),
    ):
        try:
            parsed = json.loads(candidate)
            remove_from_index = remove_rest
            break
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if parsed is None:
        # Malformed/truncated JSON: still strip from the SUBJECT_DELTAS line forward
        # so the leftover JSON fragment never leaks into the user-facing answer.
        return "\n".join(lines[:delta_index]).strip(), []
    deltas = parsed.get("deltas") if isinstance(parsed, dict) else None
    if not isinstance(deltas, list):
        return "\n".join(lines[:delta_index]).strip(), []

    if remove_from_index:
        clean_lines = lines[:delta_index]
    else:
        clean_lines = [line for index, line in enumerate(lines) if index != delta_index]
    return "\n".join(clean_lines).strip(), [delta for delta in deltas if isinstance(delta, dict)]


def _merge_subject_entry(
    entry: dict[str, Any],
    *,
    attributes: list[str],
    evidence_frames: list[float],
    last_seen_t: float | None,
) -> None:
    entry["attributes"] = sorted(set(_string_list(entry.get("attributes"))) | set(attributes))
    entry["evidence_frames"] = sorted(set(_float_list(entry.get("evidence_frames"))) | set(evidence_frames))
    if last_seen_t is not None:
        current_last = _optional_float(entry.get("last_seen_t"))
        entry["last_seen_t"] = max(current_last or 0.0, last_seen_t)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        number = _optional_float(item)
        if number is not None:
            out.append(round(number, 1))
    return out


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


@dataclass(frozen=True)
class RetrievalPlan:
    question_type: str
    retrieval_profile: str
    top_n_scenes: int
    top_k_frames: int
    planner_notes: str


@tool
async def retrieve_video_evidence(
    question: Annotated[
        str,
        Field(description="The user's video question, rewritten only if needed for retrieval."),
    ],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    question_type: Annotated[
        str,
        Field(
            description=(
                "Planner label. Use one of: overview, event_location, temporal_order, "
                "counting, comparison, visual_detail, text_ocr, existence, general."
            )
        ),
    ] = "general",
    retrieval_profile: Annotated[
        str,
        Field(
            description=(
                "Retrieval strategy: focused, balanced, broad, temporal, detail, "
                "or negative_check."
            )
        ),
    ] = "balanced",
    top_n_scenes: Annotated[int | None, Field(description="Optional scene recall override.")] = None,
    top_k_frames: Annotated[int | None, Field(description="Optional frame recall override.")] = None,
    planner_notes: Annotated[str | None, Field(description="Short reason for this retrieval plan.")] = None,
) -> Command:
    """Retrieve visual evidence without answering yet."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})

    plan = _resolve_retrieval_plan(
        question_type=question_type,
        retrieval_profile=retrieval_profile,
        top_n_scenes=top_n_scenes,
        top_k_frames=top_k_frames,
        planner_notes=planner_notes,
    )
    result = _retrieve_video(
        str(video_id),
        question,
        top_n_scenes=plan.top_n_scenes,
        top_k_frames=plan.top_k_frames,
    )
    new_frames = _frame_payloads_from_result(result, source="retrieval")
    frames = _merge_frame_payloads(state.get("retrieved_frames", []), new_frames)
    scene_hits = _merge_scene_hits(state.get("retrieved_scene_hits", []), result.scene_hits)
    payload = {
        "tool": "retrieve_video_evidence",
        "frames": _public_frame_refs(frames),
        "scene_hits": scene_hits,
        "retrieval_plan": _plan_payload(plan),
        "next": "assess_evidence_sufficiency",
    }
    return _command(
        tool_call_id,
        payload,
        update={
            "retrieved_frames": frames,
            "retrieved_scene_hits": scene_hits,
            "retrieval_plan": _plan_payload(plan),
        },
    )


@tool
async def retrieve_transcript_evidence(
    query: Annotated[str, Field(description="The speech/transcript search query.")],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    top_k: Annotated[int, Field(description="Number of transcript chunks to return.")] = 5,
) -> Command:
    """Retrieve speech transcript evidence for audio-heavy or joint questions."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})
    hits = [hit.as_dict() for hit in search_text(str(video_id), query, kind="transcript", top_k=top_k)]
    evidence = _merge_text_evidence(state.get("retrieved_transcripts", []), hits)
    payload = {
        "tool": "retrieve_transcript_evidence",
        "evidence": hits,
        "next": "answer_with_evidence",
    }
    return _command(tool_call_id, payload, update={"retrieved_transcripts": evidence})


@tool
async def search_transcript_keyword(
    keyword: Annotated[str, Field(description="Exact keyword or term to find in the transcript.")],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    top_k: Annotated[int, Field(description="Maximum keyword hits to return.")] = 10,
) -> Command:
    """Find exact transcript keyword mentions such as REINFORCE or A2C."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})
    hits = [hit.as_dict() for hit in search_keyword(str(video_id), keyword, kind="transcript", top_k=top_k)]
    evidence = _merge_text_evidence(state.get("retrieved_transcripts", []), hits)
    payload = {
        "tool": "search_transcript_keyword",
        "evidence": hits,
        "next": "align_audiovisual_evidence" if hits else "retrieve_transcript_evidence",
    }
    return _command(tool_call_id, payload, update={"retrieved_transcripts": evidence})


@tool
async def retrieve_slide_evidence(
    query: Annotated[str, Field(description="The slide/PPT/OCR search query.")],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    top_k: Annotated[int, Field(description="Number of slide OCR chunks to return.")] = 5,
) -> Command:
    """Retrieve OCR text from slides/whiteboard frames."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})
    hits = [hit.as_dict() for hit in search_text(str(video_id), query, kind="slide", top_k=top_k)]
    evidence = _merge_text_evidence(state.get("retrieved_slides", []), hits)
    payload = {
        "tool": "retrieve_slide_evidence",
        "evidence": hits,
        "next": "answer_with_evidence",
    }
    return _command(tool_call_id, payload, update={"retrieved_slides": evidence})


@tool
async def align_audiovisual_evidence(
    timestamp: Annotated[float, Field(description="Center timestamp in seconds.")],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    window_sec: Annotated[float, Field(description="Seconds before and after timestamp.")] = 8.0,
    max_frames: Annotated[int, Field(description="Maximum nearby frames to add.")] = 12,
) -> Command:
    """Return frames plus transcript/slide evidence from the same time window."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})
    frames_added = _load_dense_payloads(
        str(video_id),
        [float(timestamp)],
        window_sec=window_sec,
        max_frames=max_frames,
        source="audiovisual_align",
    )
    frames = _merge_frame_payloads(state.get("retrieved_frames", []), frames_added)
    transcripts = [hit.as_dict() for hit in nearby_text(str(video_id), timestamp, window_sec=window_sec, kind="transcript")]
    slides = [hit.as_dict() for hit in nearby_text(str(video_id), timestamp, window_sec=window_sec, kind="slide")]
    transcript_evidence = _merge_text_evidence(state.get("retrieved_transcripts", []), transcripts)
    slide_evidence = _merge_text_evidence(state.get("retrieved_slides", []), slides)
    payload = {
        "tool": "align_audiovisual_evidence",
        "timestamp": float(timestamp),
        "window_sec": float(window_sec),
        "frames": _public_frame_refs(frames_added),
        "transcripts": transcripts,
        "slides": slides,
        "next": "answer_with_evidence",
    }
    return _command(
        tool_call_id,
        payload,
        update={
            "retrieved_frames": frames,
            "retrieved_transcripts": transcript_evidence,
            "retrieved_slides": slide_evidence,
        },
    )


@tool
async def build_timeline(
    question: Annotated[str, Field(description="The user's temporal video question.")],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    expand: Annotated[
        bool,
        Field(description="Whether to add nearby dense frames around current evidence timestamps."),
    ] = True,
    window_sec: Annotated[
        float,
        Field(description="Seconds before and after each timestamp to inspect."),
    ] = 3.0,
    max_frames: Annotated[int | None, Field(description="Maximum extra frames to add.")] = None,
) -> Command:
    """Build a lightweight timeline for order/counting/comparison questions."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})

    frames = list(state.get("retrieved_frames", []))
    added: list[dict[str, Any]] = []
    if expand:
        targets = [float(item["timestamp"]) for item in frames if "timestamp" in item]
        added = _load_dense_payloads(
            str(video_id),
            targets,
            window_sec=window_sec,
            max_frames=max_frames or settings.planner_max_top_k_frames,
            source="timeline_expand",
        )
        frames = _merge_frame_payloads(frames, added)

    timeline = _timeline_entries(state.get("retrieved_scene_hits", []), frames)
    payload = {
        "tool": "build_timeline",
        "question": question,
        "added_frames": _public_frame_refs(added),
        "timeline": timeline,
        "next": "assess_evidence_sufficiency",
    }
    return _command(
        tool_call_id,
        payload,
        update={"retrieved_frames": frames, "timeline": timeline},
    )


@tool
async def retrieve_hypothesis_evidence(
    question: Annotated[str, Field(description="The user's original question.")],
    hypothesis: Annotated[
        str,
        Field(description="A concrete visual hypothesis to support or refute."),
    ],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    retrieval_profile: Annotated[
        str,
        Field(description="Usually detail, temporal, broad, or negative_check."),
    ] = "detail",
    top_n_scenes: Annotated[int | None, Field(description="Optional scene recall override.")] = None,
    top_k_frames: Annotated[int | None, Field(description="Optional frame recall override.")] = None,
) -> Command:
    """Retrieve extra evidence targeted at a specific hypothesis."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})

    plan = _resolve_retrieval_plan(
        question_type="general",
        retrieval_profile=retrieval_profile,
        top_n_scenes=top_n_scenes,
        top_k_frames=top_k_frames,
        planner_notes=f"hypothesis: {hypothesis}",
    )
    query = f"{question}\nHypothesis to verify visually: {hypothesis}"
    result = _retrieve_video(
        str(video_id),
        query,
        top_n_scenes=plan.top_n_scenes,
        top_k_frames=plan.top_k_frames,
    )
    new_frames = _frame_payloads_from_result(
        result,
        source="hypothesis_retrieval",
        hypothesis=hypothesis,
    )
    frames = _merge_frame_payloads(state.get("retrieved_frames", []), new_frames)
    scene_hits = _merge_scene_hits(state.get("retrieved_scene_hits", []), result.scene_hits)
    hypotheses = [
        *state.get("hypotheses", []),
        {"hypothesis": hypothesis, "status": "evidence_retrieved"},
    ]
    payload = {
        "tool": "retrieve_hypothesis_evidence",
        "hypothesis": hypothesis,
        "frames": _public_frame_refs(new_frames),
        "scene_hits": result.scene_hits,
        "retrieval_plan": _plan_payload(plan),
        "next": "assess_evidence_sufficiency",
    }
    return _command(
        tool_call_id,
        payload,
        update={
            "retrieved_frames": frames,
            "retrieved_scene_hits": scene_hits,
            "hypotheses": hypotheses,
        },
    )


@tool
async def segment_focus(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    question: Annotated[str, Field(description="原始用户问题。")],
    center_t: Annotated[float, Field(description="窗口中心时间（秒）。")],
    half_window_sec: Annotated[float, Field(description="窗口半宽，默认 4 秒。")] = 4.0,
    fps: Annotated[float, Field(description="抽帧 fps，默认 1.0；最多 12 帧。")] = 1.0,
) -> Command:
    """Densely sample one short window for fine visual detail (≤12 frames)."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})

    window = _segment_focus_window(str(video_id), center_t, half_window_sec)
    duration = max(0.0, window["end"] - window["start"])
    frame_cap = min(
        SEGMENT_FOCUS_MAX_FRAMES,
        max(1, math.ceil(duration * max(0.1, float(fps or 1.0))) + 1),
    )
    added = _load_dense_payloads(
        str(video_id),
        [(window["start"] + window["end"]) / 2.0],
        window_sec=duration / 2.0,
        max_frames=frame_cap,
        source="segment_focus",
    )
    added = _merge_frame_payloads([], added, limit=SEGMENT_FOCUS_MAX_FRAMES)
    if not added:
        return _command(
            tool_call_id,
            {
                "tool": "segment_focus",
                "window": {**window, "frame_count": 0},
                "observation": "I could not load dense frames for the requested window.",
                "subject_deltas": [],
                "required_next_action": "retrieve_video_evidence",
                "note_for_orchestrator": OBSERVER_NOTE_FOR_ORCHESTRATOR,
            },
        )

    frames, timestamps = _payloads_as_images(added)
    registry = state.get("subject_registry", []) or []
    answer = await answer_question(
        question,
        frames,
        timestamps,
        history=None,
        system_prompt=SEGMENT_FOCUS_PROMPT,
        subject_registry=registry,
    )
    clean_answer, deltas = parse_subject_deltas(answer)
    merged_registry = merge_subject_deltas(registry, deltas)
    all_frames = _merge_frame_payloads(state.get("retrieved_frames", []), added)
    payload = {
        "tool": "segment_focus",
        "window": {**window, "frame_count": len(added)},
        "observation": clean_answer,
        "subject_deltas": deltas,
        "required_next_action": "answer_with_evidence",
        "note_for_orchestrator": OBSERVER_NOTE_FOR_ORCHESTRATOR,
    }
    return _command(
        tool_call_id,
        payload,
        update={
            "retrieved_frames": all_frames,
            "draft_answer": clean_answer,
            "subject_registry": merged_registry,
        },
    )


@tool
async def expand_temporal_evidence(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    timestamps: Annotated[
        TimestampList,
        Field(description="Evidence timestamps to expand around. Defaults to current evidence."),
    ] = None,
    window_sec: Annotated[
        float,
        Field(description="Seconds before and after each timestamp to inspect."),
    ] = 4.0,
    max_frames: Annotated[int | None, Field(description="Maximum frames to add.")] = None,
) -> Command:
    """(legacy; prefer segment_focus) Add nearby dense frames around candidate moments."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})

    targets = timestamps or [
        float(item["timestamp"])
        for item in state.get("retrieved_frames", [])
        if "timestamp" in item
    ]
    added = _load_dense_payloads(
        str(video_id),
        targets,
        window_sec=window_sec,
        max_frames=max_frames or settings.planner_max_top_k_frames,
        source="temporal_expand",
    )
    frames = _merge_frame_payloads(state.get("retrieved_frames", []), added)
    payload = {
        "tool": "expand_temporal_evidence",
        "expanded_around": [round(float(item), 1) for item in targets],
        "added_frames": _public_frame_refs(added),
        "next": "assess_evidence_sufficiency",
    }
    return _command(tool_call_id, payload, update={"retrieved_frames": frames})


@tool
async def stitched_verify(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    question: Annotated[str, Field(description="原始用户问题。")],
    windows: Annotated[
        list[dict[str, float]],
        Field(
            description=(
                "待对比的时间窗列表，例如 "
                '[{"start":10.0,"end":15.0}, {"start":30.0,"end":34.0}]。最多 4 段。'
            )
        ),
    ],
    fps_per_window: Annotated[
        float, Field(description="每段抽帧 fps，默认 1.0；总帧数自动 cap 在 24。")
    ] = 1.0,
) -> Command:
    """Compare/synthesize evidence across 2-4 disjoint time windows."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})

    normalized_windows, warning = _normalize_stitched_windows(windows)
    if not normalized_windows:
        return _command(
            tool_call_id,
            {
                "tool": "stitched_verify",
                "windows": [],
                "observation": "No valid time windows were provided.",
                "subject_deltas": [],
                "required_next_action": "retrieve_video_evidence",
                "note_for_orchestrator": OBSERVER_NOTE_FOR_ORCHESTRATOR,
            },
        )

    added, window_summaries = _load_stitched_window_payloads(
        str(video_id),
        normalized_windows,
        fps_per_window=max(0.1, float(fps_per_window or 1.0)),
        max_frames=STITCHED_VERIFY_MAX_FRAMES,
    )
    if not added:
        payload = {
            "tool": "stitched_verify",
            "windows": window_summaries,
            "observation": "I could not load dense frames for the requested windows.",
            "subject_deltas": [],
            "required_next_action": "retrieve_video_evidence",
            "note_for_orchestrator": OBSERVER_NOTE_FOR_ORCHESTRATOR,
        }
        if warning:
            payload["warning"] = warning
        return _command(tool_call_id, payload)

    frames, timestamps = _payloads_as_images(added)
    registry = state.get("subject_registry", []) or []
    answer = await answer_question(
        question,
        frames,
        timestamps,
        history=None,
        system_prompt=STITCHED_VERIFY_PROMPT,
        subject_registry=registry,
    )
    clean_answer, deltas = parse_subject_deltas(answer)
    merged_registry = merge_subject_deltas(registry, deltas)
    all_frames = _merge_frame_payloads(state.get("retrieved_frames", []), added)
    payload = {
        "tool": "stitched_verify",
        "windows": window_summaries,
        "observation": clean_answer,
        "subject_deltas": deltas,
        "required_next_action": "answer_with_evidence",
        "note_for_orchestrator": OBSERVER_NOTE_FOR_ORCHESTRATOR,
    }
    if warning:
        payload["warning"] = warning
    return _command(
        tool_call_id,
        payload,
        update={
            "retrieved_frames": all_frames,
            "draft_answer": clean_answer,
            "subject_registry": merged_registry,
        },
    )


@tool
async def assess_evidence_sufficiency(
    question: Annotated[str, Field(description="The user's video question.")],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    question_type: Annotated[
        str,
        Field(description="Same question type used by retrieve_video_evidence."),
    ] = "general",
    retrieval_profile: Annotated[
        str | None,
        Field(description="The profile already used, if known."),
    ] = None,
) -> Command:
    """Decide whether the current evidence is enough before answering."""
    plan = state.get("retrieval_plan", {}) or {}
    qtype = _normalize_choice(question_type or plan.get("question_type"), QUESTION_TYPES, "general")
    profile = _normalize_choice(
        retrieval_profile or plan.get("retrieval_profile"),
        RETRIEVAL_PROFILES,
        "balanced",
    )
    report = _sufficiency_report(
        question=question,
        question_type=qtype,
        retrieval_profile=profile,
        state=state,
    )
    return _command(
        tool_call_id,
        {"tool": "assess_evidence_sufficiency", **report},
        update={"evidence_sufficiency": report},
    )


@tool
async def answer_with_evidence(
    question: Annotated[str, Field(description="The user's original video question.")],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    answer_mode: Annotated[
        str,
        Field(description="Use direct, cautious, temporal, or negative_cautious."),
    ] = "direct",
) -> Command:
    """Answer using the current evidence set only."""
    frames, timestamps = _state_frames_as_images(state)
    text_evidence = _state_text_evidence(state)
    if not frames and not text_evidence:
        payload = {
            "tool": "answer_with_evidence",
            "answer": "I need retrieved video, transcript, or slide evidence before I can answer that.",
            "error": "no_evidence",
        }
        return _command(tool_call_id, payload, update={"draft_answer": payload["answer"]})

    history = _history_for_vqa(state.get("messages", []))
    protocol_question = _question_with_answer_protocol(question, answer_mode, state)
    registry = state.get("subject_registry", []) or []
    answer = await answer_question(
        protocol_question,
        frames,
        timestamps,
        history,
        system_prompt=ANSWER_WITH_EVIDENCE_PROMPT,
        subject_registry=registry,
        text_evidence=text_evidence,
    )
    clean_answer, deltas = parse_subject_deltas(answer)
    cleaned = (clean_answer or "").strip()
    # Reject answers that look truncated mid-byte by the VLM (typical "length" finish):
    # they end with the Unicode replacement character. Keeping such garbage would
    # overwrite a longer, well-formed prior draft.
    if cleaned.endswith("�") or "�" in cleaned[-4:]:
        cleaned = ""
    if not cleaned:
        prior_draft = str(state.get("draft_answer") or "").strip()
        payload = {
            "tool": "answer_with_evidence",
            "answer": prior_draft,
            "error": "empty_vlm_response",
            "next": "final_answer" if prior_draft else "verify_grounding",
            "message": (
                "The VLM returned an empty answer. Use the previous draft as the "
                "final answer if it is acceptable; do not call answer_with_evidence "
                "again with the same frames."
                if prior_draft
                else "The VLM returned an empty answer. Refine retrieval or "
                "rephrase the question before retrying."
            ),
        }
        return _command(tool_call_id, payload)
    report = _grounding_report(cleaned, timestamps, state)
    payload = {
        "tool": "answer_with_evidence",
        "answer": cleaned,
        "subject_deltas": deltas,
        "grounding_report": report,
        "next": "verify_grounding",
    }
    return _command(
        tool_call_id,
        payload,
        update={
            "draft_answer": cleaned,
            "grounding_report": report,
            "subject_registry": merge_subject_deltas(registry, deltas),
        },
    )


@tool
async def verify_grounding(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    answer: Annotated[
        str | None,
        Field(description="Draft answer to verify. Defaults to the latest answer_with_evidence output."),
    ] = None,
) -> Command:
    """Check citation validity and negative-answer caution before final response."""
    draft = answer or str(state.get("draft_answer") or "")
    timestamps = [
        float(item["timestamp"])
        for item in state.get("retrieved_frames", [])
        if "timestamp" in item
    ]
    report = _grounding_report(draft, timestamps, state)
    payload = {
        "tool": "verify_grounding",
        "answer": draft,
        "grounding_report": report,
        "next": "final_answer" if report["grounded"] else report["recommended_next_action"],
    }
    return _command(tool_call_id, payload, update={"grounding_report": report})


@tool
async def multimodal_vqa(
    question: Annotated[
        str,
        Field(description="Legacy shortcut: retrieve, answer, and return frames in one call."),
    ],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    question_type: Annotated[str, Field(description="Planner question type.")] = "general",
    retrieval_profile: Annotated[str, Field(description="Planner retrieval profile.")] = "balanced",
    top_n_scenes: Annotated[int | None, Field(description="Optional scene recall override.")] = None,
    top_k_frames: Annotated[int | None, Field(description="Optional frame recall override.")] = None,
    planner_notes: Annotated[str | None, Field(description="Short reason for this retrieval plan.")] = None,
) -> Command:
    """Backward-compatible one-shot video QA tool; prefer the finer tools."""
    video_id = state.get("video_id")
    if not video_id:
        return _command(tool_call_id, {"error": "No video is attached to this session."})

    plan = _resolve_retrieval_plan(
        question_type=question_type,
        retrieval_profile=retrieval_profile,
        top_n_scenes=top_n_scenes,
        top_k_frames=top_k_frames,
        planner_notes=planner_notes,
    )
    result = _retrieve_video(
        str(video_id),
        question,
        top_n_scenes=plan.top_n_scenes,
        top_k_frames=plan.top_k_frames,
    )
    history = _history_for_vqa(state.get("messages", []))
    answer = await answer_question(
        _question_with_answer_protocol(question, "direct", {"retrieval_plan": _plan_payload(plan)}),
        result.frames,
        result.timestamps,
        history,
    )
    frame_payloads = _frame_payloads_from_result(result, source="legacy_multimodal_vqa")
    report = _grounding_report(answer, result.timestamps, {"retrieval_plan": _plan_payload(plan)})
    payload = {
        "tool": "multimodal_vqa",
        "answer": answer,
        "frames": _public_frame_refs(frame_payloads),
        "scene_hits": result.scene_hits,
        "retrieval_plan": _plan_payload(plan),
        "grounding_report": report,
    }
    return _command(
        tool_call_id,
        payload,
        update={
            "draft_answer": answer,
            "retrieved_frames": frame_payloads,
            "retrieved_scene_hits": result.scene_hits,
            "retrieval_plan": _plan_payload(plan),
            "grounding_report": report,
        },
    )


@tool
async def search_user_memories(
    query: str,
    state: Annotated[dict[str, Any], InjectedState],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Search persistent memories for the current user and return relevant facts."""
    user_id = str(state.get("user_id") or "")
    if not user_id:
        return ""
    return await memory.memory_context(
        store=store,
        user_id=user_id,
        query=query,
        limit=settings.langmem_query_limit,
    )


TOOLS = [
    retrieve_video_evidence,
    retrieve_transcript_evidence,
    search_transcript_keyword,
    retrieve_slide_evidence,
    align_audiovisual_evidence,
    build_timeline,
    retrieve_hypothesis_evidence,
    segment_focus,
    expand_temporal_evidence,
    stitched_verify,
    assess_evidence_sufficiency,
    answer_with_evidence,
    verify_grounding,
    search_user_memories,
]


def _resolve_retrieval_plan(
    *,
    question_type: str | None,
    retrieval_profile: str | None,
    top_n_scenes: int | None,
    top_k_frames: int | None,
    planner_notes: str | None,
) -> RetrievalPlan:
    qtype = _normalize_choice(question_type, QUESTION_TYPES, "general")
    profile = _normalize_choice(retrieval_profile, RETRIEVAL_PROFILES, "balanced")
    default_top_n, default_top_k = _profile_defaults(profile)
    max_top_n = max(1, int(settings.planner_max_top_n_scenes))
    max_top_k = max(1, int(settings.planner_max_top_k_frames))
    return RetrievalPlan(
        question_type=qtype,
        retrieval_profile=profile,
        top_n_scenes=_clamp_int(top_n_scenes, default_top_n, 1, max_top_n),
        top_k_frames=_clamp_int(top_k_frames, default_top_k, 1, max_top_k),
        planner_notes=(planner_notes or "")[:240],
    )


def _profile_defaults(profile: str) -> tuple[int, int]:
    base_n = max(1, int(settings.top_n_scenes))
    base_k = max(1, int(settings.top_k_frames))
    if profile == "focused":
        return min(base_n, 3), min(base_k, 8)
    if profile == "broad":
        return base_n * 2, base_k * 2
    if profile == "temporal":
        return base_n * 2, max(base_k, int(base_k * 1.5))
    if profile == "detail":
        return base_n, base_k * 2
    if profile == "negative_check":
        return base_n * 3, base_k * 2
    return base_n, base_k


def _normalize_choice(value: str | None, allowed: set[str], default: str) -> str:
    normalized = (value or "").strip().lower()
    return normalized if normalized in allowed else default


def _clamp_int(value: int | None, default: int, minimum: int, maximum: int) -> int:
    candidate = default if value is None else value
    try:
        number = int(candidate)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _retrieve_video(
    video_id: str,
    question: str,
    *,
    top_n_scenes: int | None = None,
    top_k_frames: int | None = None,
):
    from app.retrieval import two_stage_retrieve

    return two_stage_retrieve(
        video_id,
        question,
        top_n_scenes=top_n_scenes,
        top_k_frames=top_k_frames,
    )


def _command(
    tool_call_id: str,
    payload: dict[str, Any],
    *,
    update: dict[str, Any] | None = None,
) -> Command:
    return Command(
        update={
            **(update or {}),
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def _plan_payload(plan: RetrievalPlan) -> dict[str, Any]:
    return {
        "question_type": plan.question_type,
        "retrieval_profile": plan.retrieval_profile,
        "top_n_scenes": plan.top_n_scenes,
        "top_k_frames": plan.top_k_frames,
        "planner_notes": plan.planner_notes,
    }


def _frame_payloads_from_result(
    result: Any,
    *,
    source: str,
    hypothesis: str | None = None,
) -> list[dict[str, Any]]:
    frames = []
    for frame, timestamp in zip(result.frames, result.timestamps, strict=False):
        item = {
            "timestamp": float(timestamp),
            "image_b64": _image_to_b64(frame),
            "source": source,
        }
        if hypothesis:
            item["hypothesis"] = hypothesis
        frames.append(item)
    return frames


def _merge_frame_payloads(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    limit = limit or max(settings.planner_max_top_k_frames, settings.vqa_max_frames)
    by_key: dict[float, dict[str, Any]] = {}
    for item in [*existing, *new]:
        if "timestamp" not in item:
            continue
        key = round(float(item["timestamp"]), 1)
        by_key.setdefault(key, {**item, "timestamp": key})
    frames = sorted(by_key.values(), key=lambda item: float(item["timestamp"]))
    if len(frames) <= limit:
        return frames
    indices = _evenly_spaced_indices(len(frames), limit)
    return [frames[index] for index in indices]


def _merge_scene_hits(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[float, float], dict[str, Any]] = {}
    for item in [*existing, *new]:
        try:
            key = (round(float(item["start"]), 2), round(float(item["end"]), 2))
        except (KeyError, TypeError, ValueError):
            continue
        by_key.setdefault(key, dict(item))
    return sorted(by_key.values(), key=lambda item: float(item.get("start", 0.0)))


def _merge_text_evidence(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, float, float, str], dict[str, Any]] = {}
    for item in [*(existing or []), *(new or [])]:
        try:
            kind = str(item.get("kind") or "")
            t_start = round(float(item.get("t_start", 0.0)), 2)
            t_end = round(float(item.get("t_end", t_start)), 2)
            text = str(item.get("text") or "").strip()
        except (TypeError, ValueError):
            continue
        if not kind or not text:
            continue
        key = (kind, t_start, t_end, text[:80])
        current = by_key.get(key)
        if current is None or float(item.get("score", 0.0) or 0.0) > float(current.get("score", 0.0) or 0.0):
            by_key[key] = {**item, "t_start": t_start, "t_end": t_end}
    merged = sorted(
        by_key.values(),
        key=lambda item: (
            str(item.get("kind") or ""),
            float(item.get("t_start", 0.0)),
            float(item.get("t_end", 0.0)),
        ),
    )
    return merged[:limit]


def _state_text_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *_merge_text_evidence([], state.get("retrieved_transcripts", []) or []),
        *_merge_text_evidence([], state.get("retrieved_slides", []) or []),
    ]


def _public_frame_refs(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: item[key]
            for key in ("timestamp", "source", "hypothesis")
            if key in item
        }
        for item in frames
    ]


def _timeline_entries(
    scene_hits: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for scene in scene_hits:
        entries.append(
            {
                "type": "scene",
                "start": float(scene.get("start", 0.0)),
                "end": float(scene.get("end", 0.0)),
                "caption": scene.get("caption", ""),
            }
        )
    for frame in frames:
        entries.append(
            {
                "type": "frame",
                "timestamp": float(frame.get("timestamp", 0.0)),
                "source": frame.get("source", "evidence"),
            }
        )
    return sorted(entries, key=lambda item: float(item.get("timestamp", item.get("start", 0.0))))


def _load_dense_payloads(
    video_id: str,
    timestamps: list[float],
    *,
    window_sec: float,
    max_frames: int,
    source: str,
) -> list[dict[str, Any]]:
    if not timestamps:
        return []
    cache_dir = video_cache_dir(video_id)
    frames_dir = cache_dir / "frames_dense"
    available = _available_dense_timestamps(frames_dir)
    selected: set[float] = set()
    for target in timestamps:
        start = max(0.0, float(target) - max(0.0, window_sec))
        end = float(target) + max(0.0, window_sec)
        for timestamp in available:
            if start <= timestamp <= end:
                selected.add(timestamp)
    ordered = sorted(selected)
    if len(ordered) > max_frames:
        ordered = [ordered[index] for index in _evenly_spaced_indices(len(ordered), max_frames)]

    payloads: list[dict[str, Any]] = []
    for timestamp in ordered:
        path = frames_dir / _dense_frame_filename(timestamp)
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        payloads.append(
            {
                "timestamp": float(timestamp),
                "image_b64": _image_to_b64(image),
                "source": source,
            }
        )
    return payloads


def _normalize_stitched_windows(
    windows: list[dict[str, float]],
) -> tuple[list[dict[str, float]], str | None]:
    normalized: list[dict[str, float]] = []
    for window in windows or []:
        if not isinstance(window, dict):
            continue
        start = _optional_float(window.get("start"))
        end = _optional_float(window.get("end"))
        if start is None or end is None:
            continue
        start = max(0.0, start)
        end = max(0.0, end)
        if end < start:
            start, end = end, start
        normalized.append({"start": round(start, 2), "end": round(end, 2)})
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    warning = None
    if len(normalized) > STITCHED_VERIFY_MAX_WINDOWS:
        normalized = normalized[:STITCHED_VERIFY_MAX_WINDOWS]
        warning = "truncated to 4 windows"
    return normalized, warning


def _segment_focus_window(
    video_id: str,
    center_t: float,
    half_window_sec: float,
) -> dict[str, float]:
    half = max(0.0, float(half_window_sec or 0.0))
    center = max(0.0, float(center_t or 0.0))
    start = max(0.0, center - half)
    end = center + half
    meta = _safe_meta(video_id)
    duration = _optional_float(meta.get("duration"))
    if duration is not None and duration > 0:
        end = min(duration, end)
        start = min(start, end)
    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "center_t": round((start + end) / 2.0, 2),
    }


def _load_stitched_window_payloads(
    video_id: str,
    windows: list[dict[str, float]],
    *,
    fps_per_window: float,
    max_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_payloads: list[dict[str, Any]] = []
    for window in windows:
        start = float(window["start"])
        end = float(window["end"])
        duration = max(0.0, end - start)
        per_window_cap = max(1, math.ceil(duration * fps_per_window) + 1)
        center = (start + end) / 2.0
        half_window = duration / 2.0
        all_payloads.extend(
            _load_dense_payloads(
                video_id,
                [center],
                window_sec=half_window,
                max_frames=min(max_frames, per_window_cap),
                source="stitched_verify",
            )
        )

    selected = _merge_frame_payloads([], all_payloads, limit=max_frames)
    selected_timestamps = [
        float(item["timestamp"])
        for item in selected
        if "timestamp" in item
    ]
    summaries = []
    for window in windows:
        start = float(window["start"])
        end = float(window["end"])
        summaries.append(
            {
                "start": start,
                "end": end,
                "frame_count": sum(
                    1 for timestamp in selected_timestamps if start <= timestamp <= end
                ),
            }
        )
    return selected, summaries


def _available_dense_timestamps(frames_dir) -> list[float]:
    if not frames_dir.exists():
        return []
    timestamps: list[float] = []
    for path in frames_dir.glob("t*.jpg"):
        match = DENSE_FRAME_RE.match(path.name)
        if match:
            timestamps.append(round(float(match.group(1)), 1))
    return sorted(set(timestamps))


def _dense_frame_filename(timestamp: float) -> str:
    return f"t{timestamp:06.1f}.jpg"


def _evenly_spaced_indices(length: int, limit: int) -> list[int]:
    if limit <= 0:
        return []
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [length // 2]
    step = (length - 1) / (limit - 1)
    indices = []
    for i in range(limit):
        index = min(length - 1, round(i * step))
        if index not in indices:
            indices.append(index)
    index = 0
    while len(indices) < limit and index < length:
        if index not in indices:
            indices.append(index)
        index += 1
    return sorted(indices)


def _sufficiency_report(
    *,
    question: str,
    question_type: str,
    retrieval_profile: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    frames = state.get("retrieved_frames", [])
    scenes = state.get("retrieved_scene_hits", [])
    text_count = len(state.get("retrieved_transcripts", []) or []) + len(state.get("retrieved_slides", []) or [])
    frame_count = len(frames)
    scene_count = len(scenes)
    timestamps = sorted(float(item["timestamp"]) for item in frames if "timestamp" in item)
    span = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
    meta = _safe_meta(str(state.get("video_id") or ""))
    total_scenes = int(meta.get("scene_count") or 0)
    missing: list[str] = []
    action = "answer_with_evidence"

    if frame_count == 0 and text_count == 0:
        missing.append("No visual, transcript, or slide evidence has been retrieved.")
        action = "retrieve_video_evidence"
    if scene_count == 0 and text_count == 0:
        missing.append("No caption-level scene or text evidence has been retrieved.")
        action = "retrieve_video_evidence"
    if question_type in TEMPORAL_QUESTION_TYPES:
        if frame_count < min(8, settings.planner_max_top_k_frames):
            missing.append("Temporal questions need more nearby frames.")
            action = "build_timeline"
        if span < 1.0:
            missing.append("Temporal evidence covers too little time.")
            action = "expand_temporal_evidence"
    if _is_negative_question(question, question_type):
        required_scenes = min(
            total_scenes or settings.planner_max_top_n_scenes,
            max(settings.top_n_scenes * 2, 6),
        )
        required_frames = min(settings.planner_max_top_k_frames, max(settings.top_k_frames * 2, 12))
        if retrieval_profile != "negative_check":
            missing.append("Absence claims require the negative_check retrieval profile.")
            action = "retrieve_video_evidence"
        if scene_count < required_scenes:
            missing.append("Absence claims need broader scene coverage.")
            action = "retrieve_video_evidence"
        if frame_count < required_frames:
            missing.append("Absence claims need broader frame coverage.")
            action = "retrieve_video_evidence"

    sufficient = not missing
    return {
        "sufficient": sufficient,
        "confidence": 0.78 if sufficient else 0.35,
        "question_type": question_type,
        "retrieval_profile": retrieval_profile,
        "frame_count": frame_count,
        "scene_count": scene_count,
        "text_evidence_count": text_count,
        "time_span_sec": round(span, 2),
        "missing_evidence": missing,
        "recommended_next_action": action,
    }


def _safe_meta(video_id: str) -> dict[str, Any]:
    if not video_id:
        return {}
    try:
        return load_meta(video_id)
    except Exception:
        return {}


def _is_negative_question(question: str, question_type: str) -> bool:
    text = question.strip().lower()
    if question_type == "existence":
        return True
    return any(marker in text for marker in ("any", "no ", "without", "有没有", "是否有", "没有", "不存在"))


def _state_frames_as_images(state: dict[str, Any]) -> tuple[list[Image.Image], list[float]]:
    pairs = []
    for item in state.get("retrieved_frames", []):
        encoded = item.get("image_b64")
        if not encoded or "timestamp" not in item:
            continue
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except Exception:
            continue
        pairs.append((float(item["timestamp"]), image))
    pairs.sort(key=lambda pair: pair[0])
    return [image for _, image in pairs], [timestamp for timestamp, _ in pairs]


def _payloads_as_images(payloads: list[dict[str, Any]]) -> tuple[list[Image.Image], list[float]]:
    pairs = []
    for item in payloads:
        encoded = item.get("image_b64")
        if not encoded or "timestamp" not in item:
            continue
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except Exception:
            continue
        pairs.append((float(item["timestamp"]), image))
    pairs.sort(key=lambda pair: pair[0])
    return [image for _, image in pairs], [timestamp for timestamp, _ in pairs]


def _question_with_answer_protocol(
    question: str,
    answer_mode: str,
    state: dict[str, Any],
) -> str:
    sufficiency = state.get("evidence_sufficiency", {}) or {}
    grounding = (
        "Answer using only the provided frames, transcript chunks, and slide/OCR chunks. "
        "Tie every concrete visual claim to [FRAME:t=...] or [SLIDE:t=...] markers, "
        "and every speech/transcript claim to [TRANSCRIPT:t=...] markers using exact shown timestamps. "
        "If the evidence is insufficient, say what cannot be determined."
    )
    negative = (
        " For absence/existence questions, do not make an absolute whole-video "
        "claim unless broad negative_check retrieval was performed. Prefer wording "
        "like 'in the checked frames/evidence, I do not see...'."
    )
    temporal = (
        " For temporal questions, describe the order using timestamps and avoid "
        "inferring motion between frames unless the timeline evidence supports it."
    )
    if answer_mode == "negative_cautious" or not sufficiency.get("sufficient", True):
        grounding += negative
    if answer_mode == "temporal":
        grounding += temporal
    return f"{question}\n\nAnswering protocol: {grounding}"


def _grounding_report(
    answer: str,
    timestamps: list[float],
    state: dict[str, Any],
) -> dict[str, Any]:
    markers = [float(match.group(1)) for match in FRAME_MARKER_RE.finditer(answer or "")]
    invalid_markers = [
        marker
        for marker in markers
        if not _has_nearby_timestamp(marker, timestamps)
    ]
    transcript_markers = [
        (float(match.group(1)), float(match.group(2)))
        for match in TRANSCRIPT_MARKER_RE.finditer(answer or "")
    ]
    slide_markers = [float(match.group(1)) for match in SLIDE_MARKER_RE.finditer(answer or "")]
    invalid_transcript_markers = [
        marker
        for marker in transcript_markers
        if not _has_nearby_text_interval(marker, state.get("retrieved_transcripts", []) or [])
    ]
    invalid_slide_markers = [
        marker
        for marker in slide_markers
        if not _has_nearby_text_timestamp(marker, state.get("retrieved_slides", []) or [])
    ]
    visual_claims = _visual_claim_lines(answer or "")
    uncited_claims = [
        claim
        for claim in visual_claims
        if not _line_has_visual_marker(answer or "", claim)
    ]
    negative = _looks_like_negative_answer(answer or "")
    plan = state.get("retrieval_plan", {}) or {}
    profile = str(plan.get("retrieval_profile") or "")
    warnings: list[str] = []
    recommended = "revise_answer_with_citations"
    if invalid_markers:
        warnings.append("Some [FRAME:t=...] markers do not match retrieved evidence.")
    if invalid_transcript_markers:
        warnings.append("Some [TRANSCRIPT:t=...] markers do not match retrieved transcript evidence.")
    if invalid_slide_markers:
        warnings.append("Some [SLIDE:t=...] markers do not match retrieved slide evidence.")
    if uncited_claims:
        warnings.append("Some visual claim lines are not tied to frame or slide markers.")
    if negative and profile != "negative_check":
        warnings.append("Negative/absence answer was produced without negative_check retrieval.")
        recommended = "retrieve_video_evidence"
    if negative and not _has_negative_scope(answer or ""):
        warnings.append("Negative answer should scope itself to checked evidence.")
    if not (markers or slide_markers) and visual_claims:
        warnings.append("Visual answer has no frame citations.")
    grounded = not warnings
    return {
        "grounded": grounded,
        "valid_markers": len(markers) - len(invalid_markers),
        "invalid_markers": invalid_markers,
        "valid_transcript_markers": len(transcript_markers) - len(invalid_transcript_markers),
        "invalid_transcript_markers": invalid_transcript_markers,
        "valid_slide_markers": len(slide_markers) - len(invalid_slide_markers),
        "invalid_slide_markers": invalid_slide_markers,
        "uncited_claims": uncited_claims[:5],
        "warnings": warnings,
        "recommended_next_action": "final_answer" if grounded else recommended,
    }


def _has_nearby_timestamp(marker: float, timestamps: list[float]) -> bool:
    return any(abs(marker - timestamp) <= 0.65 for timestamp in timestamps)


def _has_nearby_text_interval(marker: tuple[float, float], evidence: list[dict[str, Any]]) -> bool:
    start, end = marker
    if end < start:
        start, end = end, start
    for item in evidence:
        try:
            t_start = float(item.get("t_start", 0.0))
            t_end = float(item.get("t_end", t_start))
        except (TypeError, ValueError):
            continue
        if abs(t_start - start) <= 0.75 and abs(t_end - end) <= 0.75:
            return True
    return False


def _has_nearby_text_timestamp(marker: float, evidence: list[dict[str, Any]]) -> bool:
    for item in evidence:
        try:
            t_start = float(item.get("t_start", 0.0))
            t_end = float(item.get("t_end", t_start))
        except (TypeError, ValueError):
            continue
        if t_start - 0.75 <= marker <= t_end + 0.75:
            return True
    return False


def _visual_claim_lines(answer: str) -> list[str]:
    lines: list[str] = []
    for raw_line in re.split(r"[\n。！？.!?]+", answer):
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in VISUAL_CLAIM_MARKERS):
            lines.append(line)
    return lines


def _line_has_visual_marker(answer: str, claim: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if claim not in line:
            continue
        neighborhood = "\n".join(lines[max(0, index - 1) : index + 2])
        if FRAME_MARKER_RE.search(neighborhood) or SLIDE_MARKER_RE.search(neighborhood):
            return True
    return bool(FRAME_MARKER_RE.search(claim) or SLIDE_MARKER_RE.search(claim))


def _looks_like_negative_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in NEGATIVE_MARKERS)


def _has_negative_scope(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in NEGATIVE_SCOPE_MARKERS)


def _history_for_vqa(messages: list[Any]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages[-10:-1]:
        msg_type = getattr(message, "type", "")
        if msg_type == "human":
            history.append({"role": "user", "content": str(message.content)})
        elif msg_type == "ai":
            content = str(message.content)
            if content:
                history.append({"role": "assistant", "content": content})
    return history


def _image_to_b64(image: Any) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
