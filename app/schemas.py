from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    video_id: str
    status: str
    cached: bool


class StatusResponse(BaseModel):
    video_id: str
    status: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    video_id: str
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


class FramePayload(BaseModel):
    timestamp: float
    image_b64: str


class ChatResponse(BaseModel):
    answer: str
    frames: list[FramePayload]
    scene_hits: list[dict[str, Any]] = Field(default_factory=list)
