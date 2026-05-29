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


def _has_duplicate_tool_call_signature(records: list[dict[str, Any]]) -> bool:
    """The dedup guard fires when a tool_call's (name, args) signature repeats
    (graph.py _tool_signature). Detect the same condition rather than identical
    result content — two distinct calls can legitimately return the same empty
    result without dedup having fired."""
    seen: set[str] = set()
    for record in records:
        if record.get("role") != "assistant":
            continue
        for call in record.get("tool_calls") or []:
            name = call.get("name")
            try:
                args = json.dumps(call.get("args") or {}, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(call.get("args"))
            sig = f"{name}::{args}"
            if sig in seen:
                return True
            seen.add(sig)
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


def infer_guards(
    records: list[dict[str, Any]],
    agent_terminated: str | None = None,
    max_tool_calls: int = 8,
) -> list[str]:
    """Infer which deterministic guards fired, from the serialized stream.

    Detection follows spec §4.1:
    * cap / empty  ← agent_terminated (the only reliable signal; a clean
      trajectory can naturally use the full tool budget and finish, so tool-call
      count is NOT used here).
    * forced_answer / forced_verify  ← harness-injected tool_call ids.
    * dedup  ← a repeated (name, args) tool_call signature.
    * stall  ← the last two verify_grounding answers are identical.

    salvage is intentionally not inferred: it has no post-hoc signal distinct
    from cap/stall, and the salvaged/draft-copy final message is already excluded
    as a training target in app.distill_format. ``max_tool_calls`` is accepted
    for signature stability but unused.
    """
    guards: list[str] = []
    if agent_terminated == "cap":
        guards.append("cap")
    if agent_terminated == "empty":
        guards.append("empty")
    for forced in sorted(_forced_guards(records)):
        guards.append(forced)
    if _has_duplicate_tool_call_signature(records):
        guards.append("dedup")
    if _verify_grounding_stalled(records):
        guards.append("stall")
    seen: set[str] = set()
    ordered: list[str] = []
    for guard in guards:
        if guard not in seen:
            seen.add(guard)
            ordered.append(guard)
    return ordered


def recomputed_guards(traj: dict[str, Any]) -> list[str]:
    """Guards for a stored trajectory, recomputed from its message stream.

    A trajectory's ``guards_triggered`` was computed by whatever infer_guards was
    loaded when the teacher ran; recomputing from the (immutable) ``messages`` +
    ``agent_terminated`` makes tiering deterministic w.r.t. the current logic.
    Falls back to the stored value when messages are absent (old caches).
    """
    messages = traj.get("messages")
    if not messages:
        return list(traj.get("guards_triggered") or [])
    return infer_guards(messages, traj.get("agent_terminated"))


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
