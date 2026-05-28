from __future__ import annotations

import base64
import io
import json
from typing import Annotated, Any, TypedDict

import aiosqlite
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore

from app import memory
from app.config import settings
from app.tools import TOOLS
from app.vqa import answer_question


def _last_write(left: Any, right: Any) -> Any:
    """Reducer for state fields that may receive parallel writes in one step.

    When the orchestrator emits multiple tool_calls in a single round and more
    than one tool updates the same scalar/dict/list field, LangGraph treats it
    as a multi-writer conflict unless the field has a reducer. Keep the right
    (latest) non-None value so the most recent tool wins.
    """
    if right is None:
        return left
    return right


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    video_id: str | None
    user_id: str
    retrieved_frames: Annotated[list[dict[str, Any]], _last_write]
    retrieved_scene_hits: Annotated[list[dict[str, Any]], _last_write]
    retrieved_transcripts: Annotated[list[dict[str, Any]], _last_write]
    retrieved_slides: Annotated[list[dict[str, Any]], _last_write]
    retrieval_plan: Annotated[dict[str, Any], _last_write]
    timeline: Annotated[list[dict[str, Any]], _last_write]
    candidate_timeline: Annotated[list[dict[str, Any]], _last_write]
    audiovisual_candidate_matrix: Annotated[list[dict[str, Any]], _last_write]
    hypotheses: Annotated[list[dict[str, Any]], _last_write]
    evidence_sufficiency: Annotated[dict[str, Any], _last_write]
    draft_answer: Annotated[str, _last_write]
    observer_notes: Annotated[list[dict[str, Any]], _last_write]
    grounding_report: Annotated[dict[str, Any], _last_write]
    subject_registry: Annotated[list[dict[str, Any]], _last_write]
    agent_terminated: str | None


async def build_checkpointer() -> AsyncSqliteSaver:
    path = settings.graph_checkpoint_path or (settings.data_dir / "graph_checkpoints.sqlite3")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver


def build_graph(
    checkpointer: AsyncSqliteSaver,
    store: BaseStore,
    memory_manager: Any | None = None,
):
    memory_manager = memory_manager or memory.build_memory_manager(store)
    graph = StateGraph(GraphState)
    graph.add_node("orchestrator", _make_orchestrator())
    graph.add_node("tool_node", ToolNode(TOOLS, name="tool_node"))
    graph.add_node("memory_write_node", _make_memory_write_node(memory_manager))
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"tool_node": "tool_node", "memory_write_node": "memory_write_node"},
    )
    graph.add_edge("tool_node", "orchestrator")
    graph.add_edge("memory_write_node", END)
    return graph.compile(checkpointer=checkpointer, store=store)


def _make_orchestrator():
    model = _orchestrator_model().bind_tools(TOOLS)

    async def orchestrator(state: GraphState) -> dict[str, Any]:
        question = _last_human_text(state["messages"])
        if not question:
            return {"messages": []}
        if _tool_call_count(state["messages"]) >= settings.orchestrator_max_tool_calls:
            salvaged = _salvage_draft_answer(state)
            if _can_force_answer_with_evidence(state):
                return {"messages": [_forced_answer_call(state, question)]}
            content = salvaged or (
                "I tried several tool calls but could not finish cleanly. "
                "Please rephrase the question or try again."
            )
            out: dict[str, Any] = {"messages": [AIMessage(content=content)]}
            if not salvaged:
                out["agent_terminated"] = "cap"
            return out
        if _verify_grounding_stalled(state["messages"]):
            draft = _salvage_draft_answer(state)
            if draft:
                return {"messages": [AIMessage(content=draft)]}
        grounded = _grounded_verify_answer(state["messages"])
        if grounded:
            return {"messages": [AIMessage(content=grounded)]}
        if _should_verify_after_answer(state):
            return {"messages": [_forced_verify_call()]}
        if _should_reanswer_after_grounding(state):
            return {"messages": [_forced_answer_call(state, question)]}
        if _should_answer_after_sufficiency(state):
            return {"messages": [_forced_answer_call(state, question)]}
        video_id = state.get("video_id")
        profile = None
        if video_id:
            try:
                from app.cache import compute_video_profile

                prof = compute_video_profile(str(video_id))
                profile = prof.get("profile") if prof else None
            except Exception:  # noqa: BLE001
                profile = None
        messages = [
            SystemMessage(content=_orchestrator_prompt(has_video=bool(video_id), profile=profile)),
            *_visible_messages(state["messages"]),
        ]
        response = await model.ainvoke(messages)
        # If the model decides to stop (no tool_calls) but emits empty content,
        # the final answer would be the empty string. Recover by salvaging the
        # best prior answer_with_evidence draft.
        if not (getattr(response, "tool_calls", None) or []):
            text = str(getattr(response, "content", "") or "").strip()
            if not text:
                salvaged = _salvage_draft_answer(state)
                if salvaged:
                    return {"messages": [AIMessage(content=salvaged)]}
                # No draft to salvage (turn-0 empty short-circuit, often seen
                # with GLM-4.7-flash). Re-invoke once with a coercion message
                # so the model commits to calling retrieve_video_evidence.
                coercion = HumanMessage(
                    content=(
                        "Your previous response was empty. You MUST use the available "
                        "tools to answer the user's question. Start by calling "
                        "retrieve_video_evidence; do not reply with an empty message."
                    )
                )
                response = await model.ainvoke(messages + [coercion])
                if not (getattr(response, "tool_calls", None) or []):
                    text2 = str(getattr(response, "content", "") or "").strip()
                    if not text2:
                        return {
                            "messages": [
                                AIMessage(
                                    content=(
                                        "I was unable to produce an answer for this question. "
                                        "Please rephrase or try again."
                                    )
                                )
                            ],
                            "agent_terminated": "empty",
                        }
        deduped = _dedup_tool_calls(response, state["messages"], state)
        return {"messages": deduped}

    return orchestrator


def _tool_signature(name: str, args: Any) -> str:
    try:
        normalized = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        normalized = str(args)
    return f"{name}::{normalized}"


def _prior_tool_results(messages: list[AnyMessage]) -> dict[str, str]:
    """Map tool signature -> cached ToolMessage content for prior calls."""
    # Index ToolMessages by their tool_call_id so we can pair them with the
    # AIMessage tool_call that triggered them.
    tool_by_id: dict[str, ToolMessage] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            tc_id = getattr(message, "tool_call_id", None)
            if tc_id:
                tool_by_id[str(tc_id)] = message
    seen: dict[str, str] = {}
    for message in messages:
        if not (isinstance(message, AIMessage) or getattr(message, "type", "") == "ai"):
            continue
        for call in getattr(message, "tool_calls", []) or []:
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
            if not (call_id and name):
                continue
            matching = tool_by_id.get(str(call_id))
            if matching is None:
                continue
            sig = _tool_signature(str(name), args)
            seen.setdefault(sig, str(matching.content))
    return seen


def _dedup_tool_calls(
    response: AIMessage,
    history: list[AnyMessage],
    state: GraphState,
) -> list[AnyMessage]:
    """Strip duplicate tool_calls from a fresh AI response.

    For each tool_call whose (name, args) signature matches a prior call OR an
    earlier call within the same response, synthesize a ToolMessage with the
    cached content and remove the call from the AIMessage.tool_calls list. This
    short-circuits the LangGraph tool loop without re-issuing the live tool.
    """
    tool_calls = list(getattr(response, "tool_calls", []) or [])
    if not tool_calls:
        return [response]

    prior_seen = _prior_tool_results(history)
    surviving: list[Any] = []
    synthesized: list[ToolMessage] = []
    intra_seen: dict[str, str] = {}

    for call in tool_calls:
        if isinstance(call, dict):
            name = call.get("name")
            args = call.get("args")
            call_id = call.get("id")
        else:
            name = getattr(call, "name", None)
            args = getattr(call, "args", None)
            call_id = getattr(call, "id", None)
        if not (name and call_id):
            surviving.append(call)
            continue
        sig = _tool_signature(str(name), args)
        cached_content = prior_seen.get(sig) or intra_seen.get(sig)
        if cached_content is not None:
            synthesized.append(
                ToolMessage(content=cached_content, tool_call_id=str(call_id), name=str(name))
            )
        else:
            surviving.append(call)
            # If the live tool fires we won't see its result here, but other
            # calls in this same response cluster can still dedup against it.
            intra_seen[sig] = ""

    if synthesized and not surviving:
        # Every call was a duplicate. If we have a draft to emit, terminate the
        # loop with the draft so we don't hand the synthesized messages back to
        # a tools_condition that would route to tool_node with no live calls.
        draft = _salvage_draft_answer(state)
        if draft:
            return [AIMessage(content=draft)]

    new_response = AIMessage(
        content=response.content,
        tool_calls=surviving,
        additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
        response_metadata=getattr(response, "response_metadata", {}) or {},
    )
    return [new_response, *synthesized]


def _can_force_answer_with_evidence(state: GraphState) -> bool:
    return _state_has_evidence(state) and not _has_successful_answer_with_evidence(state["messages"])


def _should_answer_after_sufficiency(state: GraphState) -> bool:
    if not _can_force_answer_with_evidence(state):
        return False
    payload = _last_tool_payload(state["messages"])
    if not payload or payload.get("tool") != "assess_evidence_sufficiency":
        return False
    return bool(payload.get("sufficient")) or payload.get("recommended_next_action") == "answer_with_evidence"


def _should_verify_after_answer(state: GraphState) -> bool:
    payload = _last_tool_payload(state["messages"])
    if not payload or payload.get("tool") != "answer_with_evidence" or payload.get("error"):
        return False
    return payload.get("next") == "verify_grounding"


def _should_reanswer_after_grounding(state: GraphState) -> bool:
    payload = _last_tool_payload(state["messages"])
    if not payload or payload.get("tool") != "verify_grounding":
        return False
    report = payload.get("grounding_report") or {}
    if report.get("grounded") is True:
        return False
    return payload.get("next") in {"revise_answer_with_citations", "answer_with_evidence"}


def _forced_answer_call(state: GraphState, question: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "answer_with_evidence",
                "args": {
                    "question": question,
                    "answer_mode": _answer_mode_for_state(state),
                },
                "id": "force_answer_with_evidence",
            }
        ],
    )


def _forced_verify_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "verify_grounding",
                "args": {},
                "id": "force_verify_grounding",
            }
        ],
    )


def _grounded_verify_answer(messages: list[AnyMessage]) -> str:
    payload = _last_tool_payload(messages)
    if not payload or payload.get("tool") != "verify_grounding":
        return ""
    report = payload.get("grounding_report") or {}
    if report.get("grounded") is True:
        return str(payload.get("answer") or "").strip()
    return ""


def _answer_mode_for_state(state: GraphState) -> str:
    plan = state.get("retrieval_plan", {}) or {}
    qtype = str(plan.get("question_type") or "")
    profile = str(plan.get("retrieval_profile") or "")
    if profile == "negative_check":
        return "negative_cautious"
    if qtype in {"temporal_order", "counting", "comparison"}:
        return "temporal"
    return "direct"


def _state_has_evidence(state: GraphState) -> bool:
    return bool(
        state.get("retrieved_frames")
        or state.get("retrieved_transcripts")
        or state.get("retrieved_slides")
    )


def _tool_was_called(messages: list[AnyMessage], tool_name: str) -> bool:
    for message in messages:
        if not (isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool"):
            continue
        payload = _tool_payload(message)
        if payload.get("tool") == tool_name:
            return True
    return False


def _has_successful_answer_with_evidence(messages: list[AnyMessage]) -> bool:
    for message in messages:
        if not (isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool"):
            continue
        payload = _tool_payload(message)
        if payload.get("tool") == "answer_with_evidence" and payload.get("answer") and not payload.get("error"):
            return True
    return False


def _last_tool_payload(messages: list[AnyMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool":
            return _tool_payload(message)
    return {}


def _tool_payload(message: AnyMessage) -> dict[str, Any]:
    try:
        payload = json.loads(str(message.content))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _verify_grounding_stalled(messages: list[AnyMessage]) -> bool:
    """True when the last two verify_grounding ToolMessages report the same answer.

    Used to break out of a verify_grounding ↔ orchestrator ping-pong where the
    model keeps re-verifying the same draft instead of emitting it.
    """
    seen: list[str] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(str(message.content))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("tool") != "verify_grounding":
            continue
        answer = str(payload.get("answer") or "").strip()
        if not answer:
            return False
        seen.append(answer)
        if len(seen) >= 2:
            return seen[0] == seen[1]
    return False


def _orchestrator_model() -> ChatOpenAI:
    base_url = (settings.orchestrator_api_base_url or settings.vlm_api_base_url).rstrip("/")
    api_key = settings.orchestrator_api_key or settings.vlm_api_key or "EMPTY"
    model_name = settings.orchestrator_model_name or settings.vlm_model_name
    timeout = settings.orchestrator_api_timeout or settings.vlm_api_timeout
    kwargs: dict[str, Any] = {}
    if "api.deepseek.com" in base_url:
        # DeepSeek V4 defaults to thinking mode; multi-turn tool calls then 400
        # because reasoning_content must be threaded back. Disable explicitly.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        temperature=settings.orchestrator_temperature,
        streaming=settings.orchestrator_streaming,
        max_retries=5,
        **kwargs,
    )


def _profile_hint(profile: str | None) -> str:
    """One-line retrieval bias derived from ingest-time modality signals.

    Only visual_dynamic gets an explicit hint: the existing AUDIOVISUAL DUAL-PATH
    RULE already handles lecture/slide content, so a slide_heavy or speech_heavy
    hint adds nothing but prompt weight and was shown to cause tool-budget
    blowups (see eval v5: -2 pass rate vs baseline). For visual_dynamic clips
    the DUAL-PATH RULE would mis-fire — skipping transcript/slide retrieval is
    correct and not covered elsewhere.
    """
    if profile == "visual_dynamic":
        return (
            "PROFILE HINT (auto-derived from ingest signals): this video is VISUAL-DYNAMIC "
            "(action / sports / no narration / no slides). Lean on retrieve_video_evidence "
            "plus segment_focus; skip retrieve_transcript_evidence and retrieve_slide_evidence "
            "unless the question explicitly asks about spoken words or text on screen. "
        )
    return ""


def _orchestrator_prompt(*, has_video: bool, profile: str | None = None) -> str:
    if not has_video:
        return (
            "You are Mr. Big-Eye, a warm and concise video-analysis assistant. "
            "No video is attached. If the user asks about a video, explain that a ready "
            "video is needed. Do not call the same tool with the same arguments twice."
        )
    profile_hint = _profile_hint(profile) if has_video else ""
    return (
        f"{profile_hint}"
        "You are Mr. Big-Eye, a warm and concise video-analysis assistant. "
        "Match the user's language. Treat the attached video as the source of truth.\n"
        "\n"
        "PLANNING: before each tool call, write one short PLAN sentence in Chinese. "
        "Classify the task as overview, event_location, temporal_order, counting, "
        "comparison, visual_detail, text_ocr, existence, or general. Choose focused, "
        "balanced, broad, temporal, detail, or negative_check. Use retrieve_video_evidence "
        "for visual content, retrieve_transcript_evidence for speech, search_transcript_keyword "
        "for exact terms, retrieve_slide_evidence for OCR/PPT, align_audiovisual_evidence "
        "when speech and visuals must be matched, segment_focus for short-window details, "
        "stitched_verify/build_timeline for order/comparison, and retrieve_hypothesis_evidence "
        "for a concrete competing hypothesis.\n"
        "\n"
        "BUDGET: Do not call the same tool with the same arguments twice. After each "
        "tool result, write one short OBSERVE sentence. If a query returns useful evidence, "
        "do not keep searching the same modality with near-synonyms; either test one concrete "
        "hypothesis or move to answering. For lecture/talk/educational/documentary videos, "
        "prefer checking both transcript and slides once because numbers, dates, labels, "
        "and ordered lists often live on slides.\n"
        "\n"
        "STOP DISCIPLINE: final user-facing answers for video questions must come from "
        "answer_with_evidence, then verify_grounding. If assess_evidence_sufficiency says "
        "sufficient=true, immediately call answer_with_evidence next. If verify_grounding "
        "returns grounded=true, emit that draft verbatim as the final answer; do not "
        "re-retrieve, re-expand, or re-answer. segment_focus and stitched_verify observations "
        "are observer notes only, never final answers.\n"
        "\n"
        "MCQ: if the user question contains Candidates: A/B/C/D/E, you MUST commit to one "
        "listed option after evidence lookup. Do not ask for missing options once Candidates "
        "are present.\n"
        "\n"
        "CITATIONS: preserve valid [FRAME:t=...], [TRANSCRIPT:t=A.B-C.D], and [SLIDE:t=...] "
        "markers. Absence/existence claims require negative_check retrieval and scoped wording. "
        "Use search_user_memories only when prior user preferences or context materially help."
    )


def _route_after_orchestrator(state: GraphState) -> str:
    try:
        return "tool_node" if tools_condition(state) == "tools" else "memory_write_node"
    except ValueError:
        return "memory_write_node"


# Backward-compatible name for tests / callers that imported the old route helper.
def _route_after_chat(state: GraphState) -> str:
    return _route_after_orchestrator(state)


# Backward-compatible direct node for tests / callers that imported the old tool node.
async def tool_node(state: GraphState) -> dict[str, Any]:
    question = _last_human_text(state["messages"])
    video_id = state.get("video_id")
    if not question or not video_id:
        return {
            "messages": [AIMessage(content="I need a ready video before I can answer that.")],
            "retrieved_frames": [],
            "retrieved_scene_hits": [],
        }

    from app.retrieval import two_stage_retrieve

    result = two_stage_retrieve(video_id, question)
    history = _history_for_vqa(state["messages"])
    answer = await answer_question(question, result.frames, result.timestamps, history)
    frame_payloads = [
        {"timestamp": timestamp, "image_b64": _image_to_b64(frame)}
        for frame, timestamp in zip(result.frames, result.timestamps, strict=False)
    ]
    return {
        "messages": [AIMessage(content=answer)],
        "retrieved_frames": frame_payloads,
        "retrieved_scene_hits": result.scene_hits,
    }


def _make_memory_write_node(memory_manager: Any):
    async def memory_write_node(state: GraphState) -> dict[str, Any]:
        user_id = state["user_id"]
        question = _last_human_text(state["messages"])
        answer = _last_ai_text(state["messages"])
        if question and answer:
            context_messages: list[AnyMessage] = []
            if state.get("video_id"):
                context_messages.append(
                    SystemMessage(
                        content=(
                            f"This conversation is about analyzed video_id={state['video_id']}. "
                            "Keep useful cross-session facts about videos the user has analyzed."
                        )
                    )
                )
            await memory.write_memories(
                manager=memory_manager,
                messages=[*context_messages, *_visible_messages(state["messages"][-12:])],
                user_id=user_id,
            )
        return {}

    return memory_write_node


def _should_use_video(question: str) -> bool:
    """Return whether an attached video should be consulted for this question."""
    text = question.strip().lower()
    if not text:
        return False
    greeting_markers = (
        "who are you",
        "what can you do",
        "你是谁",
        "你是什么",
        "你能做什么",
        "hello",
        "hi there",
        "hi!",
        "hey",
        "你好",
        "嗨",
        "在吗",
    )
    if any(marker in text for marker in greeting_markers) and len(text) < 40:
        return False
    return True


def _last_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            return str(message.content)
    return ""


def _last_ai_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            if getattr(message, "tool_calls", None):
                continue
            content = str(message.content)
            if content:
                return content
    return ""


def _history_for_vqa(messages: list[AnyMessage]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages[-10:-1]:
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            history.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            if getattr(message, "tool_calls", None):
                continue
            content = str(message.content)
            if content:
                history.append({"role": "assistant", "content": content})
    return history


def messages_from_snapshot(snapshot: Any) -> list[dict[str, str]]:
    values = getattr(snapshot, "values", {}) or {}
    output: list[dict[str, str]] = []
    for message in values.get("messages", []):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            output.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            if getattr(message, "tool_calls", None):
                continue
            content = str(message.content)
            if content:
                output.append({"role": "assistant", "content": content})
    return output


def video_id_from_snapshot(snapshot: Any) -> str | None:
    values = getattr(snapshot, "values", {}) or {}
    value = values.get("video_id")
    return str(value) if value else None


async def sanitize_dangling_tool_calls(graph: Any, config: dict[str, Any]) -> int:
    """Strip a trailing AIMessage(tool_calls) whose tool calls are not fully
    answered by ToolMessages — and any orphan ToolMessages after it.

    Triggered when a previous turn crashed inside tool_node (e.g. provider 5xx,
    pipeline bug). The checkpoint then holds an AIMessage(tool_calls=[...]) with
    missing fulfillments; on next replay DeepSeek/OpenAI rejects the request with
    400 "tool_calls must be followed by tool messages". Returns the number of
    messages removed (0 if state was already clean).
    """
    from langchain_core.messages import RemoveMessage

    snapshot = await graph.aget_state(config)
    values = getattr(snapshot, "values", {}) or {}
    messages: list[Any] = values.get("messages", []) or []
    if not messages:
        return 0

    last_ai_idx = -1
    last_tool_call_ids: set[str] = set()
    for idx, msg in enumerate(messages):
        if getattr(msg, "type", "") == "ai":
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                last_ai_idx = idx
                last_tool_call_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}
    if last_ai_idx < 0:
        return 0

    fulfilled: set[str] = set()
    for msg in messages[last_ai_idx + 1 :]:
        if getattr(msg, "type", "") == "tool":
            tcid = getattr(msg, "tool_call_id", None)
            if tcid:
                fulfilled.add(tcid)

    if last_tool_call_ids.issubset(fulfilled):
        return 0

    to_remove = []
    for msg in messages[last_ai_idx:]:
        mid = getattr(msg, "id", None)
        if mid:
            to_remove.append(RemoveMessage(id=mid))
    if not to_remove:
        return 0
    await graph.aupdate_state(config, {"messages": to_remove})
    return len(to_remove)


def _visible_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    last_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            last_human_index = index

    visible: list[AnyMessage] = []
    for index, message in enumerate(messages):
        msg_type = getattr(message, "type", "")
        if index < last_human_index:
            if msg_type == "human":
                visible.append(message)
            elif msg_type == "ai" and str(message.content):
                # Strip tool_calls from prior-turn assistant messages — the paired
                # ToolMessages get dropped above, so leaving tool_calls intact
                # makes the provider 400 with "insufficient tool messages
                # following tool_calls message" on the next turn.
                visible.append(AIMessage(content=message.content))
            continue
        if msg_type in {"human", "ai", "tool"}:
            visible.append(message)
    return visible


def _salvage_draft_answer(state: GraphState) -> str:
    """Recover the best available draft when the orchestrator hits the tool-call cap.

    Picks the longest non-truncated answer_with_evidence output. Falls back to the
    state's draft_answer only if no tool messages are present (e.g. legacy paths).
    """
    best = ""
    for message in state.get("messages", []):
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(str(message.content))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("tool") != "answer_with_evidence":
            continue
        if payload.get("error"):
            continue
        answer = str(payload.get("answer") or "").strip()
        if answer.endswith("�") or "�" in answer[-4:]:
            continue
        if len(answer) > len(best):
            best = answer
    if best:
        return best
    return str(state.get("draft_answer") or "").strip()


def _tool_call_count(messages: list[AnyMessage]) -> int:
    last_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            last_human_index = index
    total = 0
    for message in messages[last_human_index + 1 :]:
        if isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            total += len(getattr(message, "tool_calls", []) or [])
    return total


def _image_to_b64(image: Any) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
