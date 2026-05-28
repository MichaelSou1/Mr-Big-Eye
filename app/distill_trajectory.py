"""Trajectory capture helpers for orchestrator distillation (Phase A).

The eval harness already drives the LangGraph agent and keeps the full
``state["messages"]`` list. For distillation we additionally need, per case:

* the message stream serialized to plain dicts (role/content/tool_calls/...),
* the system prompt the orchestrator actually saw (it is re-composed every
  round and is NOT part of ``state["messages"]``), and
* which deterministic guards fired (so Phase B can tier-filter and exclude
  harness-injected forced calls as training targets).

Everything here is pure/post-hoc over the message stream so the live agent
behaviour is untouched. See distillation_spec.md §2.2, §4.1, §5.3.
"""

from __future__ import annotations

import json
from typing import Any

# AIMessage.tool_call ids injected by the harness (graph.py _forced_answer_call /
# _forced_verify_call). These messages are NOT teacher decisions and must be
# excluded as training targets in Phase B.
FORCED_CALL_IDS = {"force_answer_with_evidence", "force_verify_grounding"}

_ROLE_BY_TYPE = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}


def _msg_type(message: Any) -> str:
    return str(getattr(message, "type", "") or "")


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    out: list[dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            out.append(call)
        else:
            out.append(
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(call, "name", None),
                    "args": getattr(call, "args", None),
                }
            )
    return out


def serialize_message(message: Any) -> dict[str, Any]:
    """Serialize a LangChain message to a plain, JSON-safe dict."""
    mtype = _msg_type(message)
    role = _ROLE_BY_TYPE.get(mtype, mtype or "unknown")
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(content)
    record: dict[str, Any] = {"role": role, "content": content}
    mid = getattr(message, "id", None)
    if mid:
        record["id"] = str(mid)
    if mtype == "ai":
        calls = _tool_calls(message)
        if calls:
            record["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("args") or {},
                }
                for call in calls
            ]
    elif mtype == "tool":
        tcid = getattr(message, "tool_call_id", None)
        if tcid:
            record["tool_call_id"] = str(tcid)
        name = getattr(message, "name", None)
        if name:
            record["name"] = str(name)
    return record


def serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [serialize_message(message) for message in messages]


# ---------------------------------------------------------------------------
# Guard inference (operates on serialized message dicts so it is easy to test)
# ---------------------------------------------------------------------------


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(str(record.get("content", "")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _last_human_index(records: list[dict[str, Any]]) -> int:
    idx = -1
    for i, record in enumerate(records):
        if record.get("role") == "user":
            idx = i
    return idx


def _tool_call_count_after_last_human(records: list[dict[str, Any]]) -> int:
    start = _last_human_index(records) + 1
    total = 0
    for record in records[start:]:
        if record.get("role") == "assistant":
            total += len(record.get("tool_calls") or [])
    return total


def _forced_guards(records: list[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for record in records:
        if record.get("role") != "assistant":
            continue
        for call in record.get("tool_calls") or []:
            cid = call.get("id")
            if cid == "force_answer_with_evidence":
                found.add("forced_answer")
            elif cid == "force_verify_grounding":
                found.add("forced_verify")
    return found


def _has_duplicate_tool_content(records: list[dict[str, Any]]) -> bool:
    """A dedup-synthesized ToolMessage reuses an earlier ToolMessage's content
    verbatim, so two tool messages share identical content (graph.py
    _dedup_tool_calls)."""
    seen: set[str] = set()
    for record in records:
        if record.get("role") != "tool":
            continue
        content = str(record.get("content", ""))
        if content in seen:
            return True
        seen.add(content)
    return False


def _verify_grounding_stalled(records: list[dict[str, Any]]) -> bool:
    """Port of graph.py _verify_grounding_stalled over serialized messages."""
    seen: list[str] = []
    for record in reversed(records):
        if record.get("role") != "tool":
            continue
        payload = _payload(record)
        if payload.get("tool") != "verify_grounding":
            continue
        answer = str(payload.get("answer") or "").strip()
        if not answer:
            return False
        seen.append(answer)
        if len(seen) >= 2:
            return seen[0] == seen[1]
    return False


def _final_answer_text(records: list[dict[str, Any]]) -> str:
    for record in reversed(records):
        if record.get("role") == "assistant" and not (record.get("tool_calls") or []):
            content = str(record.get("content", "")).strip()
            if content:
                return content
    return ""


def _last_tool_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in reversed(records):
        if record.get("role") == "tool":
            return _payload(record)
    return {}


def _looks_salvaged(records: list[dict[str, Any]]) -> bool:
    """Final answer is a verbatim copy of an answer_with_evidence draft that was
    NOT emitted via the clean grounded-verify path (graph.py _salvage_draft_answer
    vs _grounded_verify_answer)."""
    final = _final_answer_text(records)
    if not final:
        return False
    drafts = set()
    for record in records:
        if record.get("role") != "tool":
            continue
        payload = _payload(record)
        if payload.get("tool") == "answer_with_evidence" and not payload.get("error"):
            answer = str(payload.get("answer") or "").strip()
            if answer:
                drafts.add(answer)
    if final not in drafts:
        return False
    last = _last_tool_payload(records)
    grounded_emit = (
        last.get("tool") == "verify_grounding"
        and (last.get("grounding_report") or {}).get("grounded") is True
        and str(last.get("answer") or "").strip() == final
    )
    return not grounded_emit


def infer_guards(
    records: list[dict[str, Any]],
    agent_terminated: str | None = None,
    max_tool_calls: int = 8,
) -> list[str]:
    """Infer which deterministic guards fired, from the serialized stream.

    Returns a stable-ordered, de-duplicated list drawn from:
    cap, empty, forced_answer, forced_verify, dedup, stall, salvage.
    """
    guards: list[str] = []
    if agent_terminated == "cap" or _tool_call_count_after_last_human(records) >= max_tool_calls:
        guards.append("cap")
    if agent_terminated == "empty":
        guards.append("empty")
    for forced in sorted(_forced_guards(records)):
        guards.append(forced)
    if _has_duplicate_tool_content(records):
        guards.append("dedup")
    if _verify_grounding_stalled(records):
        guards.append("stall")
    if _looks_salvaged(records):
        guards.append("salvage")
    seen: set[str] = set()
    ordered: list[str] = []
    for guard in guards:
        if guard not in seen:
            seen.add(guard)
            ordered.append(guard)
    return ordered


def render_orchestrator_system_prompt(video_id: str | None) -> str:
    """Re-compose the exact system prompt the orchestrator saw for this case.

    The prompt is rebuilt every round inside the graph and is not stored in
    ``state["messages"]``, so we replicate graph.py's logic (profile included).
    """
    from app.graph import _orchestrator_prompt

    profile = None
    if video_id:
        try:
            from app.cache import compute_video_profile

            prof = compute_video_profile(str(video_id))
            profile = prof.get("profile") if prof else None
        except Exception:  # noqa: BLE001
            profile = None
    return _orchestrator_prompt(has_video=bool(video_id), profile=profile)


def build_trajectory_record(
    *,
    case: Any,
    prediction: Any,
    result: dict[str, Any],
    orchestrator_model: str,
    vlm_model: str,
) -> dict[str, Any]:
    """Assemble one raw-trajectory JSONL row (spec §4.1)."""
    answer_block = result.get("answer") or {}
    judge = answer_block.get("llm_judge") or {}
    has_judge = bool(judge)
    return {
        "case_id": case.case_id,
        "video_id": case.video_id,
        "question": case.question,
        "system_prompt": prediction.system_prompt,
        "messages": prediction.messages,
        "guards_triggered": prediction.guards_triggered,
        "agent_terminated": prediction.agent_terminated,
        "judge_correct": bool(judge.get("correct")) if has_judge else None,
        "judge_reason": judge.get("justification") if has_judge else None,
        "pass_components": {
            "retrieval": (result.get("retrieval") or {}).get("passed"),
            "answer": answer_block.get("passed"),
            "agent": (result.get("agent_loop") or {}).get("passed"),
        },
        "passed": result.get("passed"),
        "orchestrator_model": orchestrator_model,
        "vlm_model": vlm_model,
        "agent_actions": list(prediction.agent_actions),
    }
