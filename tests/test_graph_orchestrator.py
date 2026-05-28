import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app import graph
from app import tools


class FakeOrchestrator:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        if any(getattr(message, "type", "") == "tool" for message in messages):
            return AIMessage(content="Final answer with [FRAME:t=1.0]")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_video_evidence",
                    "args": {
                        "question": "What is happening?",
                        "question_type": "temporal_order",
                        "retrieval_profile": "temporal",
                        "top_n_scenes": 7,
                        "top_k_frames": 18,
                        "planner_notes": "Need nearby moments.",
                    },
                    "id": "call_vqa",
                }
            ],
        )


class FakeMemoryManager:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, payload, config=None):
        self.calls.append((payload, config))


class FakeStitchedOrchestrator:
    def __init__(self):
        self.invocations = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.invocations += 1
        if any(getattr(message, "type", "") == "tool" for message in messages):
            return AIMessage(content="Final stitched answer [FRAME:t=1.0]")
        if self.invocations > 2:
            return AIMessage(content="Follow-up answer")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "stitched_verify",
                    "args": {
                        "question": "Compare the two moments.",
                        "windows": [
                            {"start": 0.0, "end": 2.0},
                            {"start": 4.0, "end": 6.0},
                        ],
                        "fps_per_window": 1.0,
                    },
                    "id": "call_stitched",
                }
            ],
        )


@pytest.mark.asyncio
async def test_graph_orchestrator_tool_loop_updates_frames(monkeypatch):
    async def fake_answer_question(question, frames, timestamps, history=None):
        return "The clip shows a test frame. [FRAME:t=1.0]"

    class RetrievalResult:
        frames = []
        timestamps = [1.0]
        scene_hits = [{"start": 0.0, "end": 2.0, "caption": "test scene"}]

    captured_plan = {}

    def fake_retrieve(video_id, question, *, top_n_scenes=None, top_k_frames=None):
        from PIL import Image

        captured_plan.update(
            {
                "video_id": video_id,
                "question": question,
                "top_n_scenes": top_n_scenes,
                "top_k_frames": top_k_frames,
            }
        )
        result = RetrievalResult()
        result.frames = [Image.new("RGB", (8, 8), "white")]
        return result

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: FakeOrchestrator())
    monkeypatch.setattr("app.tools.answer_question", fake_answer_question)
    monkeypatch.setattr(tools, "_retrieve_video", fake_retrieve)

    manager = FakeMemoryManager()
    app = graph.build_graph(InMemorySaver(), InMemoryStore(), manager)
    state = await app.ainvoke(
        {
            "messages": [HumanMessage(content="What is happening?")],
            "video_id": "vid001",
            "user_id": "user001",
            "retrieved_frames": [],
            "retrieved_scene_hits": [],
            "retrieval_plan": {},
            "timeline": [],
            "hypotheses": [],
            "evidence_sufficiency": {},
            "draft_answer": "",
            "grounding_report": {},
        },
        config={"configurable": {"thread_id": "thread001"}},
    )

    assert graph._last_ai_text(state["messages"]) == "Final answer with [FRAME:t=1.0]"
    assert state["retrieved_frames"][0]["timestamp"] == 1.0
    assert state["retrieved_scene_hits"][0]["caption"] == "test scene"
    assert captured_plan == {
        "video_id": "vid001",
        "question": "What is happening?",
        "top_n_scenes": 7,
        "top_k_frames": 18,
    }
    assert manager.calls


@pytest.mark.asyncio
async def test_graph_orchestrator_stitched_verify_updates_registry_checkpoint(monkeypatch):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    def fake_load_dense_payloads(video_id, timestamps, *, window_sec, max_frames, source):
        center = float(timestamps[0])
        return [
            {"timestamp": round(center, 1), "image_b64": image_b64, "source": source}
        ]

    async def fake_answer_question(question, frames, timestamps, history=None, **kwargs):
        return (
            "The moments differ. [FRAME:t=1.0]\n"
            'SUBJECT_DELTAS: {"deltas": [{"op": "add", "id": "person_A", '
            '"label": "红衣男子", "first_seen_t": 1.0, '
            '"attributes": ["转身"], "evidence_frames": [1.0]}]}'
        )

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: FakeStitchedOrchestrator())
    monkeypatch.setattr(tools, "_load_dense_payloads", fake_load_dense_payloads)
    monkeypatch.setattr("app.tools.answer_question", fake_answer_question)

    manager = FakeMemoryManager()
    app = graph.build_graph(InMemorySaver(), InMemoryStore(), manager)
    config = {"configurable": {"thread_id": "thread-stitched"}}
    state = await app.ainvoke(
        {
            "messages": [HumanMessage(content="Compare the two moments.")],
            "video_id": "vid001",
            "user_id": "user001",
            "retrieved_frames": [],
            "retrieved_scene_hits": [],
            "retrieval_plan": {},
            "timeline": [],
            "hypotheses": [],
            "evidence_sufficiency": {},
            "draft_answer": "",
            "grounding_report": {},
            "subject_registry": [],
        },
        config=config,
    )
    snapshot = await app.aget_state(config)

    assert graph._last_ai_text(state["messages"]) == "Final stitched answer [FRAME:t=1.0]"
    assert [item["timestamp"] for item in state["retrieved_frames"]] == [1.0, 5.0]
    assert state["subject_registry"][0]["id"] == "person_A"
    assert snapshot.values["subject_registry"][0]["attributes"] == ["转身"]

    state2 = await app.ainvoke(
        {
            "messages": [HumanMessage(content="Who was that person?")],
            "video_id": "vid001",
            "user_id": "user001",
            "retrieved_frames": [],
            "retrieved_scene_hits": [],
            "retrieval_plan": {},
            "timeline": [],
            "hypotheses": [],
            "evidence_sufficiency": {},
            "draft_answer": "",
            "grounding_report": {},
        },
        config=config,
    )

    assert graph._last_ai_text(state2["messages"]) == "Follow-up answer"
    assert state2["subject_registry"][0]["id"] == "person_A"


def test_graph_keeps_compatibility_symbols():
    assert graph._should_use_video("What happens in this video?")
    assert not graph._should_use_video("hello")
    assert callable(graph.tool_node)
    assert callable(graph._route_after_chat)


def test_salvage_draft_answer_prefers_state_draft_when_no_tool_messages():
    state = {"messages": [], "draft_answer": "  saved draft  "}
    assert graph._salvage_draft_answer(state) == "saved draft"


def test_salvage_draft_answer_uses_tool_message():
    import json as _json
    from langchain_core.messages import ToolMessage

    tool_message = ToolMessage(
        content=_json.dumps({"tool": "answer_with_evidence", "answer": "rescued answer"}),
        tool_call_id="abc",
    )
    state = {"messages": [tool_message], "draft_answer": ""}
    assert graph._salvage_draft_answer(state) == "rescued answer"


def test_salvage_draft_answer_picks_longest_across_history():
    """Prefer the longest valid answer so a later short/garbled response doesn't win."""
    import json as _json
    from langchain_core.messages import ToolMessage

    long_msg = ToolMessage(
        content=_json.dumps(
            {"tool": "answer_with_evidence", "answer": "long well-formed analysis with detail"}
        ),
        tool_call_id="a",
    )
    short_msg = ToolMessage(
        content=_json.dumps({"tool": "answer_with_evidence", "answer": "**1. brief"}),
        tool_call_id="b",
    )
    state = {
        "messages": [long_msg, short_msg],
        "draft_answer": "**1. brief",  # last write overwrote the long one
    }
    assert graph._salvage_draft_answer(state) == "long well-formed analysis with detail"


def test_salvage_draft_answer_skips_replacement_character():
    """Answers ending in the Unicode replacement char are VLM-truncated; reject."""
    import json as _json
    from langchain_core.messages import ToolMessage

    truncated = ToolMessage(
        content=_json.dumps({"tool": "answer_with_evidence", "answer": "header\n**1. �"}),
        tool_call_id="t",
    )
    good = ToolMessage(
        content=_json.dumps({"tool": "answer_with_evidence", "answer": "a clean shorter line"}),
        tool_call_id="g",
    )
    state = {"messages": [truncated, good], "draft_answer": ""}
    assert graph._salvage_draft_answer(state) == "a clean shorter line"


def test_salvage_draft_answer_skips_error_payloads():
    import json as _json
    from langchain_core.messages import ToolMessage

    err_msg = ToolMessage(
        content=_json.dumps(
            {"tool": "answer_with_evidence", "answer": "stale draft", "error": "empty_vlm_response"}
        ),
        tool_call_id="e",
    )
    state = {"messages": [err_msg], "draft_answer": ""}
    assert graph._salvage_draft_answer(state) == ""


def test_salvage_draft_answer_returns_empty_when_no_signal():
    assert graph._salvage_draft_answer({"messages": [], "draft_answer": ""}) == ""


def test_salvage_draft_answer_ignores_observer_notes():
    state = {
        "messages": [],
        "draft_answer": "",
        "observer_notes": [
            {"tool": "segment_focus", "observation": "observer detail [FRAME:t=1.0]"}
        ],
    }
    assert graph._salvage_draft_answer(state) == ""


def test_orchestrator_prompt_mcq_and_dedup_rules():
    """Phase C: video-branch prompt must enforce MCQ commit + anti-loop rules."""
    video_prompt = graph._orchestrator_prompt(has_video=True)
    no_video_prompt = graph._orchestrator_prompt(has_video=False)
    assert "MUST commit" in video_prompt
    assert "stitched_verify" in video_prompt
    assert "segment_focus" in video_prompt
    assert "PLAN" in video_prompt
    assert "OBSERVE" in video_prompt
    assert "Do not call the same tool with the same arguments twice" in video_prompt
    assert "Do not call the same tool with the same arguments twice" in no_video_prompt
    assert "answer_with_evidence, then verify_grounding" in video_prompt
    assert "observer notes only" in video_prompt
    # MCQ rule does not apply when there is no video
    assert "MUST commit" not in no_video_prompt


@pytest.mark.asyncio
async def test_orchestrator_dedups_identical_tool_call(monkeypatch):
    """Phase D: a fresh tool_call that matches a prior (name, args) signature
    must NOT re-invoke the live tool — it should be answered from the cached
    ToolMessage and stripped from the AIMessage."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class RepeatingModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_video_evidence",
                        "args": {"question": "What is happening?"},
                        "id": "call_repeat",
                    }
                ],
            )

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: RepeatingModel())

    orchestrator = graph._make_orchestrator()
    prior_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retrieve_video_evidence",
                "args": {"question": "What is happening?"},
                "id": "call_prev",
            }
        ],
    )
    prior_tool = ToolMessage(
        content='{"tool": "retrieve_video_evidence", "timestamps": [1.0]}',
        tool_call_id="call_prev",
    )
    state = {
        "messages": [
            HumanMessage(content="What is happening?"),
            prior_ai,
            prior_tool,
        ],
        "draft_answer": "",
        "user_id": "u",
        "video_id": "v",
    }
    result = await orchestrator(state)
    msgs = result["messages"]
    # Expect: new AIMessage with surviving tool_calls stripped + synthesized ToolMessage
    assert len(msgs) == 2
    assert isinstance(msgs[0], AIMessage)
    assert not (msgs[0].tool_calls or [])
    assert isinstance(msgs[1], ToolMessage)
    assert msgs[1].tool_call_id == "call_repeat"
    assert "timestamps" in msgs[1].content


@pytest.mark.asyncio
async def test_orchestrator_force_terminates_on_stale_verify_grounding(monkeypatch):
    """Phase D: when the last two verify_grounding results agree on the same
    answer, the orchestrator must short-circuit with that answer rather than
    consult the model again."""
    import json as _json
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class NeverCalledModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):  # pragma: no cover
            raise AssertionError("orchestrator must short-circuit on stalled verify_grounding")

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: NeverCalledModel())

    orchestrator = graph._make_orchestrator()
    payload = _json.dumps(
        {"tool": "answer_with_evidence", "answer": "stable answer [FRAME:t=1.0]"}
    )
    verify_payload = _json.dumps(
        {"tool": "verify_grounding", "answer": "stable answer [FRAME:t=1.0]", "grounded": False}
    )
    state = {
        "messages": [
            HumanMessage(content="Q?"),
            AIMessage(content="", tool_calls=[{"name": "answer_with_evidence", "args": {}, "id": "a"}]),
            ToolMessage(content=payload, tool_call_id="a"),
            AIMessage(content="", tool_calls=[{"name": "verify_grounding", "args": {}, "id": "v1"}]),
            ToolMessage(content=verify_payload, tool_call_id="v1"),
            AIMessage(content="", tool_calls=[{"name": "verify_grounding", "args": {}, "id": "v2"}]),
            ToolMessage(content=verify_payload, tool_call_id="v2"),
        ],
        "draft_answer": "",
        "user_id": "u",
        "video_id": "v",
    }
    result = await orchestrator(state)
    assert result["messages"][0].content == "stable answer [FRAME:t=1.0]"


@pytest.mark.asyncio
async def test_orchestrator_cap_signals_unanswered(monkeypatch):
    """Phase D: when the tool-call cap is hit with no salvageable draft, the
    state must surface agent_terminated='cap' alongside the fallback string."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class NeverCalledModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):  # pragma: no cover
            raise AssertionError("orchestrator must not be invoked past cap")

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: NeverCalledModel())
    monkeypatch.setattr(graph.settings, "orchestrator_max_tool_calls", 2)

    orchestrator = graph._make_orchestrator()
    state = {
        "messages": [
            HumanMessage(content="Q?"),
            AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "a"}]),
            ToolMessage(content="{}", tool_call_id="a"),
            AIMessage(content="", tool_calls=[{"name": "y", "args": {}, "id": "b"}]),
            ToolMessage(content="{}", tool_call_id="b"),
        ],
        "draft_answer": "",
        "user_id": "u",
        "video_id": "v",
    }
    result = await orchestrator(state)
    assert result.get("agent_terminated") == "cap"
    assert "could not finish" in result["messages"][0].content


@pytest.mark.asyncio
async def test_orchestrator_cap_forces_answer_when_evidence_exists(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class NeverCalledModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):  # pragma: no cover
            raise AssertionError("orchestrator must not invoke model once cap is hit")

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: NeverCalledModel())
    monkeypatch.setattr(graph.settings, "orchestrator_max_tool_calls", 2)

    orchestrator = graph._make_orchestrator()
    state = {
        "messages": [
            HumanMessage(content="Which option is correct?"),
            AIMessage(content="", tool_calls=[{"name": "retrieve_video_evidence", "args": {}, "id": "a"}]),
            ToolMessage(content='{"tool": "retrieve_video_evidence"}', tool_call_id="a"),
            AIMessage(content="", tool_calls=[{"name": "retrieve_transcript_evidence", "args": {}, "id": "b"}]),
            ToolMessage(content='{"tool": "retrieve_transcript_evidence"}', tool_call_id="b"),
        ],
        "retrieved_frames": [{"timestamp": 1.0, "image_b64": "x"}],
        "draft_answer": "",
        "user_id": "u",
        "video_id": "v",
    }

    result = await orchestrator(state)
    call = result["messages"][0].tool_calls[0]
    assert call["name"] == "answer_with_evidence"
    assert call["args"]["question"] == "Which option is correct?"


@pytest.mark.asyncio
async def test_orchestrator_cap_forces_answer_when_prior_answer_tool_failed(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class NeverCalledModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):  # pragma: no cover
            raise AssertionError("orchestrator must not invoke model once cap is hit")

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: NeverCalledModel())
    monkeypatch.setattr(graph.settings, "orchestrator_max_tool_calls", 2)

    orchestrator = graph._make_orchestrator()
    state = {
        "messages": [
            HumanMessage(content="Which option is correct?"),
            AIMessage(content="", tool_calls=[{"name": "answer_with_evidence", "args": {}, "id": "a"}]),
            ToolMessage(content='{"tool": "answer_with_evidence", "answer": "", "error": "empty_vlm_response"}', tool_call_id="a"),
            AIMessage(content="", tool_calls=[{"name": "retrieve_transcript_evidence", "args": {}, "id": "b"}]),
            ToolMessage(content='{"tool": "retrieve_transcript_evidence"}', tool_call_id="b"),
        ],
        "retrieved_transcripts": [{"text": "evidence"}],
        "draft_answer": "",
        "user_id": "u",
        "video_id": "v",
    }

    result = await orchestrator(state)
    call = result["messages"][0].tool_calls[0]
    assert call["name"] == "answer_with_evidence"


@pytest.mark.asyncio
async def test_orchestrator_forces_answer_after_sufficient_report(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class NeverCalledModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):  # pragma: no cover
            raise AssertionError("sufficient evidence should force answer_with_evidence")

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: NeverCalledModel())

    orchestrator = graph._make_orchestrator()
    state = {
        "messages": [
            HumanMessage(content="What happens?"),
            AIMessage(content="", tool_calls=[{"name": "assess_evidence_sufficiency", "args": {}, "id": "s"}]),
            ToolMessage(
                content='{"tool": "assess_evidence_sufficiency", "sufficient": true}',
                tool_call_id="s",
            ),
        ],
        "retrieved_transcripts": [{"text": "evidence"}],
        "retrieval_plan": {"question_type": "temporal_order"},
        "draft_answer": "",
        "user_id": "u",
        "video_id": "v",
    }

    result = await orchestrator(state)
    call = result["messages"][0].tool_calls[0]
    assert call["name"] == "answer_with_evidence"
    assert call["args"]["answer_mode"] == "temporal"


@pytest.mark.asyncio
async def test_orchestrator_forces_verify_after_answer(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class NeverCalledModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):  # pragma: no cover
            raise AssertionError("answer_with_evidence must be followed by verify_grounding")

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: NeverCalledModel())

    orchestrator = graph._make_orchestrator()
    state = {
        "messages": [
            HumanMessage(content="Which option?"),
            AIMessage(content="", tool_calls=[{"name": "answer_with_evidence", "args": {}, "id": "a"}]),
            ToolMessage(
                content='{"tool": "answer_with_evidence", "answer": "Answer: A) Cats. [FRAME:t=1.0]", "next": "verify_grounding"}',
                tool_call_id="a",
            ),
        ],
        "draft_answer": "Answer: A) Cats. [FRAME:t=1.0]",
        "user_id": "u",
        "video_id": "v",
    }

    result = await orchestrator(state)
    call = result["messages"][0].tool_calls[0]
    assert call["name"] == "verify_grounding"


@pytest.mark.asyncio
async def test_orchestrator_finalizes_grounded_verify(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class NeverCalledModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):  # pragma: no cover
            raise AssertionError("grounded verify should finalize without model call")

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: NeverCalledModel())

    orchestrator = graph._make_orchestrator()
    state = {
        "messages": [
            HumanMessage(content="Which option?"),
            AIMessage(content="", tool_calls=[{"name": "verify_grounding", "args": {}, "id": "v"}]),
            ToolMessage(
                content='{"tool": "verify_grounding", "answer": "Answer: A) Cats. [FRAME:t=1.0]", "grounding_report": {"grounded": true}}',
                tool_call_id="v",
            ),
        ],
        "draft_answer": "Answer: A) Cats. [FRAME:t=1.0]",
        "user_id": "u",
        "video_id": "v",
    }

    result = await orchestrator(state)
    assert result["messages"][0].content == "Answer: A) Cats. [FRAME:t=1.0]"


@pytest.mark.asyncio
async def test_orchestrator_salvages_when_final_ai_content_is_empty(monkeypatch):
    """If the model decides to stop (no tool_calls) but emits empty content,
    fall back to the prior answer_with_evidence draft instead of an empty answer."""
    import json as _json
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class EmptyFinishModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="", tool_calls=[])

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: EmptyFinishModel())

    orchestrator = graph._make_orchestrator()
    tool_payload = _json.dumps(
        {"tool": "answer_with_evidence", "answer": "rescued draft [FRAME:t=1.0]"}
    )
    state = {
        "messages": [
            HumanMessage(content="Q?"),
            AIMessage(content="", tool_calls=[{"name": "answer_with_evidence", "args": {}, "id": "a"}]),
            ToolMessage(content=tool_payload, tool_call_id="a"),
        ],
        "draft_answer": "",
        "user_id": "u",
        "video_id": "v",
    }
    result = await orchestrator(state)
    assert result["messages"][0].content == "rescued draft [FRAME:t=1.0]"


@pytest.mark.asyncio
async def test_orchestrator_fallback_uses_salvaged_draft(monkeypatch):
    """When the tool-call cap is hit, the orchestrator should emit the prior good draft
    instead of the generic 'tried several tool calls' message."""
    import json as _json
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class NeverCalledModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):  # pragma: no cover - should not be called
            raise AssertionError("orchestrator model must not be invoked once cap is hit")

    monkeypatch.setattr(graph, "_orchestrator_model", lambda: NeverCalledModel())
    monkeypatch.setattr(graph.settings, "orchestrator_max_tool_calls", 2)

    orchestrator = graph._make_orchestrator()
    tool_payload = _json.dumps(
        {"tool": "answer_with_evidence", "answer": "good draft [FRAME:t=1.0]"}
    )
    state = {
        "messages": [
            HumanMessage(content="how is my form?"),
            AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "a"}]),
            ToolMessage(content=tool_payload, tool_call_id="a"),
            AIMessage(content="", tool_calls=[{"name": "y", "args": {}, "id": "b"}]),
            ToolMessage(content="{}", tool_call_id="b"),
        ],
        "draft_answer": "",
        "user_id": "u",
        "video_id": "v",
    }
    result = await orchestrator(state)
    assert result["messages"][0].content == "good draft [FRAME:t=1.0]"
