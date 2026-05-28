import json

from app.distill_format import (
    is_forced_or_guard,
    slice_trajectory,
    target_kind,
    target_tool_names,
    visible_filter,
)

_BIG = "QUJD" * 2000


def _tool(tool, **extra):
    return json.dumps({"tool": tool, **extra}, ensure_ascii=False)


def _trajectory():
    answer = "The answer is A. [FRAME:t=3.0]"
    return {
        "system_prompt": "SYS-PROMPT",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "PLAN", "tool_calls": [{"id": "c1", "name": "retrieve_video_evidence", "args": {"q": "x"}}]},
            {"role": "tool", "content": _tool("retrieve_video_evidence", frames=[{"timestamp": 3.0, "image_b64": _BIG}]), "tool_call_id": "c1", "name": "retrieve_video_evidence"},
            {"role": "assistant", "content": "PLAN2", "tool_calls": [{"id": "c2", "name": "answer_with_evidence", "args": {}}]},
            {"role": "tool", "content": _tool("answer_with_evidence", answer=answer), "tool_call_id": "c2", "name": "answer_with_evidence"},
            # harness-injected forced verify — must be excluded as target
            {"role": "assistant", "content": "", "tool_calls": [{"id": "force_verify_grounding", "name": "verify_grounding", "args": {}}]},
            {"role": "tool", "content": _tool("verify_grounding", answer=answer, grounding_report={"grounded": True}), "tool_call_id": "force_verify_grounding"},
            # final bare message copies the draft — excluded
            {"role": "assistant", "content": answer},
        ],
    }


def test_visible_filter_single_turn_keeps_current_turn():
    recs = _trajectory()["messages"]
    vis = visible_filter(recs[:4])
    assert [r["role"] for r in vis] == ["user", "assistant", "tool", "assistant"]


def test_is_forced_or_guard():
    drafts = {"DRAFT"}
    assert is_forced_or_guard({"role": "assistant", "tool_calls": [{"id": "force_answer_with_evidence"}]}, drafts)
    assert is_forced_or_guard({"role": "assistant", "content": "DRAFT"}, drafts)
    assert not is_forced_or_guard({"role": "assistant", "content": "real", "tool_calls": [{"id": "c1", "name": "retrieve_video_evidence"}]}, drafts)
    assert is_forced_or_guard({"role": "tool", "content": "x"}, drafts)


def test_slice_trajectory_targets_and_format():
    tools = [{"type": "function", "function": {"name": "retrieve_video_evidence"}}]
    samples = slice_trajectory(_trajectory(), tools)
    # two valid teacher decisions: retrieve_video_evidence, answer_with_evidence
    assert len(samples) == 2
    for s in samples:
        assert s["messages"][0] == {"role": "system", "content": "SYS-PROMPT"}
        assert s["messages"][-1]["role"] == "assistant"
        assert s["tools"] is tools
    assert [target_tool_names(s)[0] for s in samples] == ["retrieve_video_evidence", "answer_with_evidence"]
    assert all(target_kind(s) == "tool_call" for s in samples)


def test_slice_trajectory_truncates_tool_results_in_prefix():
    tools = []
    samples = slice_trajectory(_trajectory(), tools)
    # second sample's prefix includes the retrieve tool result; base64 must be gone
    second = samples[1]
    blob = json.dumps(second["messages"])
    assert _BIG not in blob
    assert "<frame_b64_omitted" in blob


def test_genuine_final_answer_is_kept_as_target():
    traj = {
        "system_prompt": "SYS",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "PLAN", "tool_calls": [{"id": "c1", "name": "retrieve_video_evidence", "args": {}}]},
            {"role": "tool", "content": _tool("retrieve_video_evidence"), "tool_call_id": "c1"},
            {"role": "assistant", "content": "A model-written final answer not matching any draft."},
        ],
    }
    samples = slice_trajectory(traj, [])
    kinds = [target_kind(s) for s in samples]
    assert kinds == ["tool_call", "final_answer"]


def test_empty_system_prompt_yields_no_samples():
    traj = {"system_prompt": "", "messages": [{"role": "assistant", "content": "x"}]}
    assert slice_trajectory(traj, []) == []
