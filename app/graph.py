from __future__ import annotations

from typing import Annotated, Any, TypedDict

import aiosqlite
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore

from app import memory
from app.config import settings
from app.vqa import answer_question, chat_text


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    video_id: str | None
    user_id: str
    retrieved_frames: list[dict[str, Any]]
    retrieved_scene_hits: list[dict[str, Any]]


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
    graph.add_node("chat_node", _make_chat_node(store))
    graph.add_node("tool_node", tool_node)
    graph.add_node("memory_write_node", _make_memory_write_node(memory_manager))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges(
        "chat_node",
        _route_after_chat,
        {"tool_node": "tool_node", "memory_write_node": "memory_write_node"},
    )
    graph.add_edge("tool_node", "memory_write_node")
    graph.add_edge("memory_write_node", END)
    return graph.compile(checkpointer=checkpointer, store=store)


def _make_chat_node(store: BaseStore):
    async def chat_node(state: GraphState) -> dict[str, Any]:
        question = _last_human_text(state["messages"])
        if not question:
            return {"messages": []}

        if state.get("video_id") and _should_use_video(question):
            return {"messages": []}

        memories = await memory.memory_context(
            store=store,
            user_id=state["user_id"],
            query=question,
            limit=settings.langmem_query_limit,
        )
        prompt = (
            "You are Mr. Big-Eye, a warm and concise video-analysis assistant. "
            "Answer directly when the user is greeting you or asking about prior "
            "context. Match the user's language."
        )
        if memories:
            prompt += "\n\nRelevant user memories:\n" + memories
        messages = [SystemMessage(content=prompt), *state["messages"]]
        answer = await chat_text(messages)
        return {"messages": [AIMessage(content=answer)]}

    return chat_node


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
                messages=[*context_messages, *state["messages"][-8:]],
                user_id=user_id,
            )
        return {}

    return memory_write_node


def _route_after_chat(state: GraphState) -> str:
    question = _last_human_text(state["messages"])
    if state.get("video_id") and question and _should_use_video(question):
        return "tool_node"
    return "memory_write_node"


def _should_use_video(question: str) -> bool:
    text = question.lower()
    direct_chat_markers = (
        "who are you",
        "你是谁",
        "hello",
        "hi",
        "你好",
        "嗨",
    )
    if any(marker in text for marker in direct_chat_markers) and len(text) < 80:
        return False
    video_markers = (
        "video",
        "frame",
        "scene",
        "watch",
        "happen",
        "object",
        "person",
        "minute",
        "second",
        "timestamp",
        "what",
        "where",
        "when",
        "describe",
        "视频",
        "画面",
        "镜头",
        "场景",
        "发生",
        "看到",
        "哪里",
        "什么时候",
        "第几",
        "描述",
    )
    return any(marker in text for marker in video_markers)


def _last_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            return str(message.content)
    return ""


def _last_ai_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            return str(message.content)
    return ""


def _history_for_vqa(messages: list[AnyMessage]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages[-10:-1]:
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            history.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            history.append({"role": "assistant", "content": str(message.content)})
    return history


def messages_from_snapshot(snapshot: Any) -> list[dict[str, str]]:
    values = getattr(snapshot, "values", {}) or {}
    output: list[dict[str, str]] = []
    for message in values.get("messages", []):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            output.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            output.append({"role": "assistant", "content": str(message.content)})
    return output


def video_id_from_snapshot(snapshot: Any) -> str | None:
    values = getattr(snapshot, "values", {}) or {}
    value = values.get("video_id")
    return str(value) if value else None


def _image_to_b64(image) -> str:
    import base64
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
