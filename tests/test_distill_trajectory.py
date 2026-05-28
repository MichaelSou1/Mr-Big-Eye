import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.distill_trajectory import (
    FORCED_CALL_IDS,
    infer_guards,
    serialize_message,
    serialize_messages,
)


def _tool_content(tool: str, **extra) -> str:
    return json.dumps({"tool": tool, **extra}, ensure_ascii=False)


def test_serialize_human_assistant_tool_roundtrip():
    msgs = [
        HumanMessage(content="What happens at the end?", id="h1"),
        AIMessage(
            content="PLAN: retrieve",
            id="a1",
            tool_calls=[{"id": "c1", "name": "retrieve_video_evidence", "args": {"query": "end"}}],
        ),
        ToolMessage(content=_tool_content("retrieve_video_evidence", frames=3), tool_call_id="c1", name="retrieve_video_evidence", id="t1"),
    ]
    out = serialize_messages(msgs)
    assert out[0] == {"role": "user", "content": "What happens at the end?", "id": "h1"}
    assert out[1]["role"] == "assistant"
    assert out[1]["tool_calls"][0]["name"] == "retrieve_video_evidence"
    assert out[1]["tool_calls"][0]["id"] == "c1"
    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "c1"
    assert out[2]["name"] == "retrieve_video_evidence"


def test_serialize_non_string_content_is_json_encoded():
    msg = AIMessage(content=[{"type": "text", "text": "hi"}], id="a")
    out = serialize_message(msg)
    assert isinstance(out["content"], str)
    assert "hi" in out["content"]


def test_clean_grounded_trajectory_has_no_guards():
    answer = "It ends with a sunset. [FRAME:t=12.0]"
    records = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "PLAN", "tool_calls": [{"id": "c1", "name": "retrieve_video_evidence", "args": {}}]},
        {"role": "tool", "content": _tool_content("retrieve_video_evidence", frames=2), "tool_call_id": "c1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "name": "answer_with_evidence", "args": {}}]},
        {"role": "tool", "content": _tool_content("answer_with_evidence", answer=answer, next="verify_grounding"), "tool_call_id": "c2"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c3", "name": "verify_grounding", "args": {}}]},
        {"role": "tool", "content": _tool_content("verify_grounding", answer=answer, grounding_report={"grounded": True}), "tool_call_id": "c3"},
        {"role": "assistant", "content": answer},
    ]
    assert infer_guards(records, None, max_tool_calls=8) == []


def test_cap_guard_from_tool_call_count():
    records = [{"role": "user", "content": "q"}]
    for i in range(8):
        records.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "name": "retrieve_video_evidence", "args": {"i": i}}]})
        records.append({"role": "tool", "content": _tool_content("retrieve_video_evidence", i=i), "tool_call_id": f"c{i}"})
    assert "cap" in infer_guards(records, None, max_tool_calls=8)


def test_cap_and_empty_from_agent_terminated():
    records = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "giving up"}]
    assert "cap" in infer_guards(records, "cap", max_tool_calls=8)
    assert "empty" in infer_guards(records, "empty", max_tool_calls=8)


def test_forced_calls_detected_by_id():
    records = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "force_answer_with_evidence", "name": "answer_with_evidence", "args": {}}]},
        {"role": "tool", "content": _tool_content("answer_with_evidence", answer="A"), "tool_call_id": "force_answer_with_evidence"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "force_verify_grounding", "name": "verify_grounding", "args": {}}]},
        {"role": "tool", "content": _tool_content("verify_grounding", answer="A", grounding_report={"grounded": True}), "tool_call_id": "force_verify_grounding"},
        {"role": "assistant", "content": "A"},
    ]
    guards = infer_guards(records, None, max_tool_calls=8)
    assert "forced_answer" in guards
    assert "forced_verify" in guards
    assert FORCED_CALL_IDS == {"force_answer_with_evidence", "force_verify_grounding"}


def test_dedup_detected_from_duplicate_tool_content():
    dup = _tool_content("retrieve_video_evidence", frames=2)
    records = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "retrieve_video_evidence", "args": {"q": "x"}}]},
        {"role": "tool", "content": dup, "tool_call_id": "c1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "name": "retrieve_video_evidence", "args": {"q": "x"}}]},
        {"role": "tool", "content": dup, "tool_call_id": "c2"},
        {"role": "assistant", "content": "done"},
    ]
    assert "dedup" in infer_guards(records, None, max_tool_calls=8)


def test_stall_detected_from_repeated_verify_answer():
    records = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "verify_grounding", "args": {}}]},
        {"role": "tool", "content": _tool_content("verify_grounding", answer="same draft", grounding_report={"grounded": False}), "tool_call_id": "c1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "name": "verify_grounding", "args": {}}]},
        {"role": "tool", "content": _tool_content("verify_grounding", answer="same draft", grounding_report={"grounded": False}), "tool_call_id": "c2"},
        {"role": "assistant", "content": "final"},
    ]
    assert "stall" in infer_guards(records, None, max_tool_calls=8)


def test_salvage_detected_when_final_copies_draft_without_grounded_verify():
    draft = "Best guess answer. [FRAME:t=3.0]"
    records = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "answer_with_evidence", "args": {}}]},
        {"role": "tool", "content": _tool_content("answer_with_evidence", answer=draft), "tool_call_id": "c1"},
        {"role": "assistant", "content": draft},
    ]
    assert "salvage" in infer_guards(records, None, max_tool_calls=8)
