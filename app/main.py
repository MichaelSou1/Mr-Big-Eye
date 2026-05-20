import asyncio
import base64
import hashlib
import io
import logging
import traceback
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.cache import get_video_status, set_video_status
from app.config import settings
from app.models import load_all_models
from app.preprocess import _probe, preprocess_video
from app.retrieval import two_stage_retrieve
from app.schemas import ChatRequest, ChatResponse, FramePayload, StatusResponse, UploadResponse
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
    if settings.load_models_on_startup:
        await asyncio.to_thread(load_all_models)
    else:
        logger.info("LOAD_MODELS_ON_STARTUP=false; retrieval models will load lazily")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
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

        meta = await asyncio.to_thread(_probe, video_path)
        if float(meta["duration"]) > settings.max_video_duration_sec:
            if video_path.exists():
                video_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Video duration exceeds {settings.max_video_duration_sec} seconds",
            )

        status = get_video_status(video_id)
        if status == "done":
            return UploadResponse(video_id=video_id, status="done", cached=True)
        if status == "running":
            return UploadResponse(video_id=video_id, status="running", cached=True)

        set_video_status(video_id, "running")
        asyncio.create_task(_run_preprocess(video_id, video_path))
        return UploadResponse(video_id=video_id, status="running", cached=False)
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
    current_status = get_video_status(req.video_id)
    if current_status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Video is not ready: {current_status}",
        )
    try:
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


async def _run_preprocess(video_id: str, video_path: Path) -> None:
    try:
        await asyncio.to_thread(lambda: asyncio.run(preprocess_video(video_id, video_path)))
        set_video_status(video_id, "done")
    except Exception as exc:
        logger.error("Preprocessing failed for %s:\n%s", video_id, traceback.format_exc())
        set_video_status(video_id, f"failed:{exc}")


def _image_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
