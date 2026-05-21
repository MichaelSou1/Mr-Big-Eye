from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    video_id: str
    status: str
    cached: bool
    stream_url: str | None = None


class StatusResponse(BaseModel):
    video_id: str
    status: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    video_id: str | None = None
    question: str
    user_id: str | None = None
    session_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class FramePayload(BaseModel):
    timestamp: float
    image_b64: str


class ChatResponse(BaseModel):
    answer: str
    frames: list[FramePayload]
    scene_hits: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str | None = None


class LoginRequest(BaseModel):
    username: str | None = None


class UserResponse(BaseModel):
    user_id: str
    username: str
    created_at: str


class SessionCreateRequest(BaseModel):
    user_id: str
    video_id: str | None = None
    title: str | None = None


class SessionUpdateRequest(BaseModel):
    video_id: str | None = None
    title: str | None = None


class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    video_id: str | None = None
    title: str | None = None
    video_filename: str | None = None
    created_at: str
    updated_at: str


class SessionResponse(BaseModel):
    session_id: str


class SessionMessagesResponse(BaseModel):
    session_id: str
    video_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)


class VideoSummary(BaseModel):
    video_id: str
    user_id: str
    filename: str
    duration: float
    created_at: str
