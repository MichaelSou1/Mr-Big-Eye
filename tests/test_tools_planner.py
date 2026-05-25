import pytest

from app import tools
from app.config import settings


def test_resolve_retrieval_plan_uses_profile_defaults(monkeypatch):
    monkeypatch.setattr(settings, "top_n_scenes", 5)
    monkeypatch.setattr(settings, "top_k_frames", 12)
    monkeypatch.setattr(settings, "planner_max_top_n_scenes", 12)
    monkeypatch.setattr(settings, "planner_max_top_k_frames", 36)

    plan = tools._resolve_retrieval_plan(
        question_type="counting",
        retrieval_profile="temporal",
        top_n_scenes=None,
        top_k_frames=None,
        planner_notes=None,
    )

    assert plan.question_type == "counting"
    assert plan.retrieval_profile == "temporal"
    assert plan.top_n_scenes == 10
    assert plan.top_k_frames == 18


def test_resolve_retrieval_plan_clamps_overrides(monkeypatch):
    monkeypatch.setattr(settings, "top_n_scenes", 5)
    monkeypatch.setattr(settings, "top_k_frames", 12)
    monkeypatch.setattr(settings, "planner_max_top_n_scenes", 9)
    monkeypatch.setattr(settings, "planner_max_top_k_frames", 20)

    plan = tools._resolve_retrieval_plan(
        question_type="unknown",
        retrieval_profile="negative_check",
        top_n_scenes=99,
        top_k_frames=99,
        planner_notes="x" * 300,
    )

    assert plan.question_type == "general"
    assert plan.retrieval_profile == "negative_check"
    assert plan.top_n_scenes == 9
    assert plan.top_k_frames == 20
    assert len(plan.planner_notes) == 240


def test_sufficiency_requires_negative_check_for_absence_claims(monkeypatch):
    monkeypatch.setattr(settings, "top_n_scenes", 5)
    monkeypatch.setattr(settings, "top_k_frames", 12)
    monkeypatch.setattr(settings, "planner_max_top_k_frames", 36)

    report = tools._sufficiency_report(
        question="Is there a cat in the video?",
        question_type="existence",
        retrieval_profile="balanced",
        state={
            "video_id": "missing-meta",
            "retrieved_frames": [{"timestamp": float(i)} for i in range(4)],
            "retrieved_scene_hits": [{"start": 0.0, "end": 2.0}],
        },
    )

    assert not report["sufficient"]
    assert report["recommended_next_action"] == "retrieve_video_evidence"
    assert any("negative_check" in item for item in report["missing_evidence"])


def test_grounding_report_flags_uncited_visual_claims():
    report = tools._grounding_report(
        "The person is holding a red cup.",
        [1.0, 2.0],
        {"retrieval_plan": {"retrieval_profile": "balanced"}},
    )

    assert not report["grounded"]
    assert report["uncited_claims"]
    assert "Visual answer has no frame citations." in report["warnings"]


def test_grounding_report_accepts_scoped_negative_with_citation():
    report = tools._grounding_report(
        "In the checked frames, I do not see a cat.\n[FRAME:t=1.0]",
        [1.0, 2.0],
        {"retrieval_plan": {"retrieval_profile": "negative_check"}},
    )

    assert report["grounded"]
    assert report["invalid_markers"] == []


def test_coerce_float_list_accepts_native_list():
    assert tools._coerce_float_list([1.0, 2.5]) == [1.0, 2.5]


def test_coerce_float_list_accepts_json_string():
    assert tools._coerce_float_list("[2.0, 3.0]") == [2.0, 3.0]


def test_coerce_float_list_accepts_comma_separated_string():
    assert tools._coerce_float_list("1.5, 2.5 , 3") == [1.5, 2.5, 3.0]


def test_coerce_float_list_handles_none_and_empty():
    assert tools._coerce_float_list(None) is None
    assert tools._coerce_float_list("") is None


def test_coerce_float_list_rejects_non_numeric():
    with pytest.raises(ValueError):
        tools._coerce_float_list(["a", "b"])


@pytest.mark.asyncio
async def test_expand_temporal_evidence_accepts_string_timestamps(monkeypatch):
    monkeypatch.setattr(
        tools, "_load_dense_payloads", lambda *args, **kwargs: []
    )
    tool_call = {
        "name": "expand_temporal_evidence",
        "id": "call_1",
        "type": "tool_call",
        "args": {
            "timestamps": "[2.0, 3.0]",
            "window_sec": 1.5,
            "max_frames": 6,
            "state": {"video_id": "vid", "retrieved_frames": []},
            "tool_call_id": "call_1",
        },
    }
    command = await tools.expand_temporal_evidence.ainvoke(tool_call)
    update = command.update if hasattr(command, "update") else command["update"]
    tool_message = update["messages"][0]
    import json as _json

    payload = _json.loads(tool_message.content)
    assert payload["tool"] == "expand_temporal_evidence"
    assert payload["expanded_around"] == [2.0, 3.0]


@pytest.mark.asyncio
async def test_answer_with_evidence_treats_truncated_replacement_char_as_empty(monkeypatch):
    """Responses that the VLM cut off mid-byte end with �. Reject them so they
    don't clobber a longer, well-formed prior draft."""
    import base64
    import io as _io
    from PIL import Image

    async def fake_truncated(*args, **kwargs):
        return "**1. �"

    monkeypatch.setattr(tools, "answer_question", fake_truncated)

    buf = _io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    prior_draft = "long prior analysis [FRAME:t=1.0]"
    tool_call = {
        "name": "answer_with_evidence",
        "id": "tr",
        "type": "tool_call",
        "args": {
            "question": "q",
            "answer_mode": "direct",
            "state": {
                "video_id": "vid",
                "retrieved_frames": [{"timestamp": 1.0, "image_b64": image_b64}],
                "draft_answer": prior_draft,
                "messages": [],
            },
            "tool_call_id": "tr",
        },
    }
    command = await tools.answer_with_evidence.ainvoke(tool_call)
    update = command.update if hasattr(command, "update") else command["update"]
    assert "draft_answer" not in update
    import json as _json

    payload = _json.loads(update["messages"][0].content)
    assert payload["error"] == "empty_vlm_response"
    assert payload["answer"] == prior_draft


@pytest.mark.asyncio
async def test_answer_with_evidence_preserves_prior_draft_on_empty_response(monkeypatch):
    import base64
    import io as _io
    from PIL import Image

    async def fake_answer_question(*args, **kwargs):
        return "   "

    monkeypatch.setattr(tools, "answer_question", fake_answer_question)

    buf = _io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    prior_draft = "earlier grounded answer [FRAME:t=1.0]"

    tool_call = {
        "name": "answer_with_evidence",
        "id": "call_2",
        "type": "tool_call",
        "args": {
            "question": "what is happening?",
            "answer_mode": "direct",
            "state": {
                "video_id": "vid",
                "retrieved_frames": [{"timestamp": 1.0, "image_b64": image_b64}],
                "draft_answer": prior_draft,
                "messages": [],
            },
            "tool_call_id": "call_2",
        },
    }
    command = await tools.answer_with_evidence.ainvoke(tool_call)
    update = command.update if hasattr(command, "update") else command["update"]
    # Only the tool message should be in the update; draft_answer must not be clobbered.
    assert "draft_answer" not in update
    import json as _json

    payload = _json.loads(update["messages"][0].content)
    assert payload["error"] == "empty_vlm_response"
    assert payload["answer"] == prior_draft
    assert payload["next"] == "final_answer"
