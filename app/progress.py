from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from app.config import settings


STAGE_LABELS = {
    "probe": ["正在打开视频卷轴", "Unrolling the video..."],
    "remux": ["正在修复视频封装", "Repairing video container..."],
    "scenes": ["数一数有多少个场景", "Counting scenes..."],
    "captions": ["瞪大眼睛仔细看每个镜头", "Studying each scene..."],
    "asr": ["正在听老师讲了什么", "Listening to the audio..."],
    "slides": ["正在整理 PPT 和板书", "Reading slides and whiteboard..."],
    "indexing": ["把看到的写进小本本", "Taking notes..."],
    "embed": ["给每一帧拍个写真", "Photographing every frame..."],
    "done": ["视频索引完成", "Video index is ready."],
    "failed": ["预处理失败", "Preprocessing failed."],
}


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    label: str
    progress: float | None = None
    detail: str | None = None

    def payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {"stage": self.stage, "label": self.label}
        if self.progress is not None:
            data["progress"] = self.progress
        if self.detail:
            data["detail"] = self.detail
        return data


_SUBSCRIBERS: dict[str, set[asyncio.Queue[ProgressEvent]]] = defaultdict(set)
_LATEST: dict[str, ProgressEvent] = {}


def stage_label(stage: str) -> str:
    choices = STAGE_LABELS.get(stage, [stage, stage])
    index = 0 if settings.progress_lang == "zh" else 1
    return choices[min(index, len(choices) - 1)]


def make_progress_callback(video_id: str) -> Callable[[str, str | None, float | None], None]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    def _callback(stage: str, label: str | None = None, progress: float | None = None) -> None:
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(publish_progress, video_id, stage, label, progress)
        else:
            publish_progress(video_id, stage, label, progress)

    return _callback


def publish_progress(
    video_id: str,
    stage: str,
    label: str | None = None,
    progress: float | None = None,
    detail: str | None = None,
) -> None:
    event = ProgressEvent(
        stage=stage,
        label=label or _jitter_label(stage),
        progress=_clamp_progress(progress),
        detail=detail,
    )
    _LATEST[video_id] = event
    for queue in list(_SUBSCRIBERS.get(video_id, set())):
        queue.put_nowait(event)


async def subscribe_progress(video_id: str):
    queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
    _SUBSCRIBERS[video_id].add(queue)
    latest = _LATEST.get(video_id)
    if latest is not None:
        queue.put_nowait(latest)
    try:
        while True:
            yield await queue.get()
    finally:
        _SUBSCRIBERS[video_id].discard(queue)
        if not _SUBSCRIBERS[video_id]:
            _SUBSCRIBERS.pop(video_id, None)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _jitter_label(stage: str) -> str:
    choices = STAGE_LABELS.get(stage)
    if not choices:
        return stage
    if settings.progress_lang == "zh":
        return choices[0]
    return choices[1] if len(choices) > 1 else choices[0]


def _clamp_progress(progress: float | None) -> float | None:
    if progress is None:
        return None
    return max(0.0, min(1.0, float(progress)))
