import asyncio
import base64
import hashlib
import io
import logging
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from PIL import Image

from app import db
from app.cache import get_video_status, set_video_status
from app.config import settings
from app.graph import (
    build_checkpointer,
    build_graph,
    messages_from_snapshot,
    sanitize_dangling_tool_calls,
    _should_use_video,
    video_id_from_snapshot,
)
from app.memory import build_memory_manager, build_memory_store
from app.models import load_all_models
from app.progress import make_progress_callback, publish_progress, sse, stage_label, subscribe_progress
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FramePayload,
    LoginRequest,
    SessionCreateRequest,
    SessionMessagesResponse,
    SessionResponse,
    SessionSummary,
    SessionUpdateRequest,
    StatusResponse,
    UploadResponse,
    UserResponse,
    VideoSummary,
)
from app.usernames import pick_username
from app.vqa import answer_question

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mr. Big-Eye")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.on_event("startup")
async def _startup() -> None:
    (settings.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "cache").mkdir(parents=True, exist_ok=True)
    db.init_db()
    app.state.checkpointer = await build_checkpointer()
    app.state.memory_store = await build_memory_store()
    app.state.memory_manager = build_memory_manager(app.state.memory_store)
    app.state.graph = build_graph(
        app.state.checkpointer,
        app.state.memory_store,
        app.state.memory_manager,
    )
    if settings.load_models_on_startup:
        await asyncio.to_thread(load_all_models)
    else:
        logger.info("LOAD_MODELS_ON_STARTUP=false; retrieval models will load lazily")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    hotwords: str | None = Form(default=None),
    user_id: str | None = Query(default=None),
) -> UploadResponse:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    temp_path = uploads_dir / f"{uuid4().hex}.tmp"

    digest = hashlib.sha256()
    total_size = 0
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    try:
        async with aiofiles.open(temp_path, "wb") as handle:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Upload exceeds {settings.max_upload_size_mb} MB",
                    )
                digest.update(chunk)
                await handle.write(chunk)
        video_id = digest.hexdigest()[:16]
        video_path = uploads_dir / f"{video_id}{suffix}"
        if video_path.exists():
            temp_path.unlink(missing_ok=True)
        else:
            temp_path.rename(video_path)

        from app.preprocess import probe_video_lenient

        meta = await asyncio.to_thread(probe_video_lenient, video_path)
        if float(meta["duration"]) > settings.max_video_duration_sec:
            if video_path.exists():
                video_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Video duration exceeds {settings.max_video_duration_sec} seconds",
            )
        if user_id:
            db.register_video(
                video_id,
                user_id,
                file.filename or video_path.name,
                float(meta["duration"]),
            )

        status = get_video_status(video_id)
        if status == "done":
            return UploadResponse(
                video_id=video_id,
                status="done",
                cached=True,
                stream_url=f"/api/preprocess_stream/{video_id}",
            )
        if status == "running":
            return UploadResponse(
                video_id=video_id,
                status="running",
                cached=True,
                stream_url=f"/api/preprocess_stream/{video_id}",
            )

        set_video_status(video_id, "running")
        publish_progress(video_id, "probe", stage_label("probe"), 0.01)
        asyncio.create_task(_run_preprocess(video_id, video_path, _parse_hotwords(hotwords)))
        return UploadResponse(
            video_id=video_id,
            status="running",
            cached=False,
            stream_url=f"/api/preprocess_stream/{video_id}",
        )
    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        logger.exception("Upload failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/status/{video_id}", response_model=StatusResponse)
async def status(video_id: str) -> StatusResponse:
    return StatusResponse(video_id=video_id, status=get_video_status(video_id))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if req.user_id:
        return await _chat_with_graph(req)
    if not req.video_id:
        raise HTTPException(status_code=400, detail="video_id is required")

    current_status = get_video_status(req.video_id)
    if current_status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Video is not ready: {current_status}",
        )
    try:
        from app.retrieval import two_stage_retrieve

        result = await asyncio.to_thread(two_stage_retrieve, req.video_id, req.question)
        history = [message.model_dump() for message in req.history]
        answer = await answer_question(
            req.question,
            result.frames,
            result.timestamps,
            history,
        )
        frames = [
            FramePayload(timestamp=timestamp, image_b64=_image_to_b64(frame))
            for frame, timestamp in zip(result.frames, result.timestamps, strict=False)
        ]
        return ChatResponse(answer=answer, frames=frames, scene_hits=result.scene_hits)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/login", response_model=UserResponse)
async def login(req: LoginRequest) -> UserResponse:
    username = req.username.strip() if req.username else pick_username(db.list_usernames())
    user = db.create_or_get_user(username)
    return UserResponse(**user)


@app.get("/api/preprocess_stream/{video_id}")
async def preprocess_stream(video_id: str) -> StreamingResponse:
    async def _events():
        status = get_video_status(video_id)
        if status == "absent":
            yield sse("error", {"video_id": video_id, "detail": "Video is not preprocessing"})
            return
        if status == "done":
            yield sse("done", {"video_id": video_id})
            return
        if status.startswith("failed:"):
            yield sse("error", {"video_id": video_id, "detail": status})
            return

        async for event in subscribe_progress(video_id):
            if event.stage == "done":
                yield sse("done", {"video_id": video_id})
                break
            if event.stage == "failed":
                yield sse("error", {"video_id": video_id, "detail": event.detail or event.label})
                break
            yield sse("stage", event.payload())

    return StreamingResponse(_events(), media_type="text/event-stream")


@app.get("/api/sessions", response_model=list[SessionSummary])
async def api_list_sessions(user_id: str) -> list[SessionSummary]:
    _require_user(user_id)
    return [SessionSummary(**item) for item in db.list_sessions(user_id)]


@app.post("/api/sessions", response_model=SessionResponse)
async def api_create_session(req: SessionCreateRequest) -> SessionResponse:
    _require_user(req.user_id)
    session_id = db.create_session(req.user_id, req.video_id, req.title)
    await _seed_graph_session(session_id, req.user_id, req.video_id)
    return SessionResponse(session_id=session_id)


@app.patch("/api/sessions/{session_id}", response_model=SessionResponse)
async def api_update_session(
    session_id: str,
    req: SessionUpdateRequest,
    user_id: str,
) -> SessionResponse:
    _require_session_owner(session_id, user_id)
    updates = req.model_dump(exclude_none=True)
    if updates:
        db.update_session(session_id, **updates)
        snapshot = await app.state.graph.aget_state(_graph_config(session_id))
        values = getattr(snapshot, "values", {}) or {}
        await app.state.graph.aupdate_state(
            _graph_config(session_id),
            {
                "messages": values.get("messages", []),
                "user_id": user_id,
                "video_id": updates.get("video_id", values.get("video_id")),
                "retrieved_frames": values.get("retrieved_frames", []),
                "retrieved_scene_hits": values.get("retrieved_scene_hits", []),
                "retrieved_transcripts": values.get("retrieved_transcripts", []),
                "retrieved_slides": values.get("retrieved_slides", []),
                "retrieval_plan": values.get("retrieval_plan", {}),
                "timeline": values.get("timeline", []),
                "hypotheses": values.get("hypotheses", []),
                "evidence_sufficiency": values.get("evidence_sufficiency", {}),
                "draft_answer": values.get("draft_answer", ""),
                "observer_notes": values.get("observer_notes", []),
                "grounding_report": values.get("grounding_report", {}),
            },
            as_node="memory_write_node",
        )
    return SessionResponse(session_id=session_id)


@app.get("/api/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def api_session_messages(session_id: str, user_id: str) -> SessionMessagesResponse:
    _require_session_owner(session_id, user_id)
    snapshot = await app.state.graph.aget_state(_graph_config(session_id))
    session = db.get_session(session_id) or {}
    return SessionMessagesResponse(
        session_id=session_id,
        video_id=video_id_from_snapshot(snapshot) or session.get("video_id"),
        messages=[ChatMessage(**message) for message in messages_from_snapshot(snapshot)],
    )


@app.get("/api/videos", response_model=list[VideoSummary])
async def api_list_videos(user_id: str) -> list[VideoSummary]:
    _require_user(user_id)
    return [VideoSummary(**item) for item in db.list_videos(user_id)]


@app.get("/api/chat_stream")
async def chat_stream(
    user_id: str,
    question: str,
    session_id: str | None = None,
    video_id: str | None = None,
) -> StreamingResponse:
    async def _events():
        try:
            created_session = False
            sid = session_id
            if sid:
                _require_session_owner(sid, user_id)
            else:
                _require_user(user_id)
                sid = db.create_session(user_id, video_id, _make_title(question))
                await _seed_graph_session(sid, user_id, video_id)
                created_session = True

            session = db.get_session(sid)
            active_video_id = video_id or (session or {}).get("video_id")
            if active_video_id:
                if _should_use_video(question):
                    status = get_video_status(active_video_id)
                    if status != "done":
                        raise HTTPException(status_code=409, detail=f"Video is not ready: {status}")
                db.update_session(sid, video_id=active_video_id)
            if not (session or {}).get("title"):
                db.update_session(sid, title=_make_title(question))

            yield sse(
                "frames",
                {
                    "session_id": sid,
                    "created_session": created_session,
                    "frames": [],
                    "scene_hits": [],
                },
            )

            # Heal any dangling AIMessage(tool_calls) left by a previous crashed
            # turn before the new question hits the LLM provider.
            removed = await sanitize_dangling_tool_calls(
                app.state.graph, _graph_config(sid)
            )
            if removed:
                logger.warning(
                    "Pruned %d dangling tool-call message(s) from session %s",
                    removed,
                    sid,
                )

            answer = ""
            pending_orchestrator_tokens: list[str] = []
            tool_phase_completed = False
            final_state: dict[str, Any] | None = None
            async for event in app.state.graph.astream_events(
                {
                    "messages": [HumanMessage(content=question)],
                    "video_id": active_video_id,
                    "user_id": user_id,
                    "retrieved_frames": [],
                    "retrieved_scene_hits": [],
                    "retrieved_transcripts": [],
                    "retrieved_slides": [],
                    "retrieval_plan": {},
                    "timeline": [],
                    "candidate_timeline": [],
                    "audiovisual_candidate_matrix": [],
                    "hypotheses": [],
                    "evidence_sufficiency": {},
                    "draft_answer": "",
                    "observer_notes": [],
                    "grounding_report": {},
                },
                config=_graph_config(sid),
                version="v2",
            ):
                event_name = event.get("event")
                name = event.get("name")
                data = event.get("data") or {}
                if event_name == "on_chain_stream" and name == "LangGraph":
                    chunk = data.get("chunk") or {}
                    tool_update = chunk.get("tool_node")
                    if tool_update:
                        tool_phase_completed = True
                        pending_orchestrator_tokens = []
                        # LangGraph 1.x emits a list of Command updates when ToolNode
                        # runs several tool calls in parallel; a single call is still a dict.
                        updates = tool_update if isinstance(tool_update, list) else [tool_update]
                        frames: list[Any] = []
                        scene_hits: list[Any] = []
                        transcripts: list[Any] = []
                        slides: list[Any] = []
                        for upd in updates:
                            if not isinstance(upd, dict):
                                continue
                            frames.extend(upd.get("retrieved_frames", []) or [])
                            scene_hits.extend(upd.get("retrieved_scene_hits", []) or [])
                            transcripts.extend(upd.get("retrieved_transcripts", []) or [])
                            slides.extend(upd.get("retrieved_slides", []) or [])
                        if frames or scene_hits or transcripts or slides:
                            yield sse(
                                "frames",
                                {
                                    "session_id": sid,
                                    "created_session": created_session,
                                    "frames": frames,
                                    "scene_hits": scene_hits,
                                    "transcripts": transcripts,
                                    "slides": slides,
                                },
                            )
                elif event_name == "on_chat_model_stream" and name == "ChatOpenAI":
                    metadata = event.get("metadata") or {}
                    if metadata.get("langgraph_node") != "orchestrator":
                        continue
                    chunk = data.get("chunk")
                    text = _message_content_text(getattr(chunk, "content", ""))
                    if text:
                        if tool_phase_completed:
                            answer += text
                            yield sse("token", {"text": text})
                        else:
                            pending_orchestrator_tokens.append(text)
                elif event_name == "on_chain_end" and name == "orchestrator":
                    if tool_phase_completed or not pending_orchestrator_tokens:
                        continue
                    output = data.get("output") or {}
                    node_messages = output.get("messages", [])
                    last_message = node_messages[-1] if node_messages else None
                    if not getattr(last_message, "tool_calls", None):
                        for token in pending_orchestrator_tokens:
                            answer += token
                            yield sse("token", {"text": token})
                    pending_orchestrator_tokens = []
                elif event_name == "on_chain_end" and name == "LangGraph":
                    final_state = data.get("output") or {}

            if not answer and final_state:
                answer = _last_assistant_message(final_state.get("messages", []))
                if answer:
                    yield sse("token", {"text": answer})
            if answer:
                db.update_session(sid)
            yield sse("done", {"session_id": sid})
        except Exception as exc:
            logger.exception("Chat stream failed")
            yield sse("error", {"detail": str(exc)})

    return StreamingResponse(_events(), media_type="text/event-stream")


async def _run_preprocess(video_id: str, video_path: Path, hotwords: list[str] | None = None) -> None:
    try:
        from app.preprocess import preprocess_video

        progress_callback = make_progress_callback(video_id)
        await asyncio.to_thread(
            lambda: asyncio.run(preprocess_video(video_id, video_path, progress_callback, hotwords))
        )
        set_video_status(video_id, "done")
        publish_progress(video_id, "done", stage_label("done"), 1.0)
    except Exception as exc:
        logger.error("Preprocessing failed for %s:\n%s", video_id, traceback.format_exc())
        set_video_status(video_id, f"failed:{exc}")
        publish_progress(video_id, "failed", stage_label("failed"), None, str(exc))


def _image_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


async def _chat_with_graph(req: ChatRequest) -> ChatResponse:
    user = _require_user(req.user_id or "")
    session_id = req.session_id or db.create_session(
        user["user_id"],
        req.video_id,
        _make_title(req.question),
    )
    if req.session_id:
        session = _require_session_owner(req.session_id, user["user_id"])
    else:
        session = db.get_session(session_id) or {}
        await _seed_graph_session(session_id, user["user_id"], req.video_id)

    active_video_id = req.video_id or session.get("video_id")
    if active_video_id:
        status = get_video_status(active_video_id)
        if status != "done":
            raise HTTPException(status_code=409, detail=f"Video is not ready: {status}")
        db.update_session(session_id, video_id=active_video_id)

    removed = await sanitize_dangling_tool_calls(
        app.state.graph, _graph_config(session_id)
    )
    if removed:
        logger.warning(
            "Pruned %d dangling tool-call message(s) from session %s",
            removed,
            session_id,
        )

    state = await app.state.graph.ainvoke(
        {
            "messages": [HumanMessage(content=req.question)],
            "video_id": active_video_id,
            "user_id": user["user_id"],
            "retrieved_frames": [],
            "retrieved_scene_hits": [],
            "retrieved_transcripts": [],
            "retrieved_slides": [],
            "retrieval_plan": {},
            "timeline": [],
            "candidate_timeline": [],
            "audiovisual_candidate_matrix": [],
            "hypotheses": [],
            "evidence_sufficiency": {},
            "draft_answer": "",
            "observer_notes": [],
            "grounding_report": {},
        },
        config=_graph_config(session_id),
    )
    answer = _last_assistant_message(state.get("messages", []))
    db.update_session(session_id)
    return ChatResponse(
        answer=answer,
        frames=[_frame_response(frame) for frame in state.get("retrieved_frames", [])],
        scene_hits=state.get("retrieved_scene_hits", []),
        session_id=session_id,
    )


async def _seed_graph_session(session_id: str, user_id: str, video_id: str | None) -> None:
    with suppress(Exception):
        snapshot = await app.state.graph.aget_state(_graph_config(session_id))
        if getattr(snapshot, "values", None):
            return
    await app.state.graph.aupdate_state(
        _graph_config(session_id),
        {
            "messages": [],
            "user_id": user_id,
            "video_id": video_id,
            "retrieved_frames": [],
            "retrieved_scene_hits": [],
            "retrieved_transcripts": [],
            "retrieved_slides": [],
            "retrieval_plan": {},
            "timeline": [],
            "candidate_timeline": [],
            "audiovisual_candidate_matrix": [],
            "hypotheses": [],
            "evidence_sufficiency": {},
            "draft_answer": "",
            "observer_notes": [],
            "grounding_report": {},
            "subject_registry": [],
        },
        as_node="memory_write_node",
    )


def _graph_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


@app.get("/api/videos/{video_id}/transcripts")
async def api_video_transcripts(video_id: str) -> dict[str, Any]:
    from app.text_assets import load_transcripts

    return {"video_id": video_id, "segments": load_transcripts(video_id)}


@app.get("/api/videos/{video_id}/slides")
async def api_video_slides(video_id: str) -> dict[str, Any]:
    from app.text_assets import load_slides

    return {"video_id": video_id, "slides": load_slides(video_id)}


@app.get("/api/videos/{video_id}/subtitles.vtt")
async def api_video_subtitles(video_id: str) -> FileResponse:
    from app.text_assets import vtt_path

    path = vtt_path(video_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Subtitles not found")
    return FileResponse(path, media_type="text/vtt", filename=f"{video_id}.vtt")


def _frame_response(frame: dict[str, Any]) -> FramePayload:
    return FramePayload(
        timestamp=float(frame["timestamp"]),
        image_b64=str(frame["image_b64"]),
    )


def _parse_hotwords(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = []
    for line in value.replace(",", "\n").splitlines():
        item = line.strip()
        if item:
            items.append(item)
    return items or None


def _require_user(user_id: str) -> dict[str, Any]:
    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Unknown user")
    return user


def _require_session_owner(session_id: str, user_id: str) -> dict[str, Any]:
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    if session["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Session belongs to another user")
    return session


def _last_assistant_message(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
            return str(message.content)
    return ""


def _make_title(question: str) -> str:
    title = " ".join(question.strip().split())
    if not title:
        return "New session"
    max_chars = settings.session_title_max_chars
    return title if len(title) <= max_chars else f"{title[: max_chars - 1]}..."


def _looks_like_video_question(question: str) -> bool:
    return _should_use_video(question)
