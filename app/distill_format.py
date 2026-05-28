"""Trajectory -> SFT sample formatting (Phase B, spec §5.2-§5.4, §5.7).

Each teacher-generated assistant message becomes one training sample whose
*target* is that message and whose *input* is the exact context the orchestrator
saw before producing it: the re-composed system prompt + the visible message
prefix (app.graph._visible_messages semantics) with tool results base64-truncated.

Harness-injected messages are excluded as targets (spec §5.3):
* forced_answer / forced_verify tool calls (id in FORCED_CALL_IDS), and
* a final bare assistant message that is a verbatim copy of an
  answer_with_evidence draft (salvage / stall / grounded-verify emit).

Output is OpenAI-style messages + tools, which transformers' Qwen chat template
and LLaMA Factory both consume. Only the final assistant message carries loss.
"""

from __future__ import annotations

import json
from typing import Any

from app.distill_trajectory import FORCED_CALL_IDS
from app.distill_truncate import truncate_record


def export_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function schemas for the 14 TOOLS, same source as bind_tools."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from app.tools import TOOLS

    return [convert_to_openai_tool(tool) for tool in TOOLS]


def load_tool_schemas(path: str | None = None) -> list[dict[str, Any]]:
    if path:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return export_tool_schemas()


def visible_filter(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Port of graph._visible_messages over serialized records.

    Drops system messages and, for turns before the last human message, strips
    tool_calls off assistant messages and discards content-less assistant /
    non-human messages. For single-turn eval this keeps the current turn intact.
    """
    last_human = -1
    for idx, record in enumerate(records):
        if record.get("role") == "user":
            last_human = idx
    visible: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        role = record.get("role")
        if idx < last_human:
            if role == "user":
                visible.append(record)
            elif role == "assistant" and str(record.get("content", "")):
                visible.append({"role": "assistant", "content": record["content"]})
            continue
        if role in {"user", "assistant", "tool"}:
            visible.append(record)
    return visible


def _draft_answers(records: list[dict[str, Any]]) -> set[str]:
    drafts: set[str] = set()
    for record in records:
        if record.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(record.get("content", "")))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("tool") == "answer_with_evidence":
            answer = str(payload.get("answer") or "").strip()
            if answer:
                drafts.add(answer)
    return drafts


def _is_forced_call(record: dict[str, Any]) -> bool:
    for call in record.get("tool_calls") or []:
        if (call.get("id") if isinstance(call, dict) else None) in FORCED_CALL_IDS:
            return True
    return False


def is_forced_or_guard(record: dict[str, Any], drafts: set[str]) -> bool:
    """True when this assistant message is harness-injected, not a teacher decision."""
    if record.get("role") != "assistant":
        return True
    if _is_forced_call(record):
        return True
    # Bare final message that merely copies an answer_with_evidence draft
    # (salvage / stall / grounded-verify verbatim emit) — not teacher prose.
    if not (record.get("tool_calls") or []):
        if str(record.get("content", "")).strip() in drafts:
            return True
    return False


def _to_openai(record: dict[str, Any]) -> dict[str, Any]:
    role = record.get("role")
    if role == "assistant":
        out: dict[str, Any] = {"role": "assistant", "content": record.get("content") or ""}
        calls = record.get("tool_calls") or []
        if calls:
            out["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call.get("name"),
                        "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
                    },
                }
                for call in calls
            ]
        return out
    if role == "tool":
        out = {"role": "tool", "content": record.get("content") or ""}
        if record.get("tool_call_id"):
            out["tool_call_id"] = record["tool_call_id"]
        if record.get("name"):
            out["name"] = record["name"]
        return out
    return {"role": role, "content": record.get("content") or ""}


def slice_trajectory(
    traj: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    *,
    truncate: bool = True,
) -> list[dict[str, Any]]:
    """Expand one trajectory into a list of SFT samples (one per teacher step)."""
    system_prompt = str(traj.get("system_prompt") or "")
    records = list(traj.get("messages") or [])
    if not system_prompt or not records:
        return []
    if truncate:
        records = [truncate_record(r) if r.get("role") == "tool" else r for r in records]
    drafts = _draft_answers(records)
    samples: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if record.get("role") != "assistant":
            continue
        if is_forced_or_guard(record, drafts):
            continue
        prefix = [_to_openai(r) for r in visible_filter(records[:idx])]
        target = _to_openai(record)
        messages = [{"role": "system", "content": system_prompt}, *prefix, target]
        samples.append({"messages": messages, "tools": tool_schemas})
    return samples


def target_kind(sample: dict[str, Any]) -> str:
    """'tool_call' if the final assistant message calls tools, else 'final_answer'."""
    last = sample["messages"][-1]
    return "tool_call" if last.get("tool_calls") else "final_answer"


def target_tool_names(sample: dict[str, Any]) -> list[str]:
    last = sample["messages"][-1]
    return [tc["function"]["name"] for tc in (last.get("tool_calls") or [])]
