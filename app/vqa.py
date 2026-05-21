import base64
import io
import logging
from typing import Any, Iterable

from langchain_core.messages import BaseMessage
from openai import AsyncOpenAI, BadRequestError
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


CAPTION_SYSTEM_PROMPT = (
    "Describe this video frame in one or two concise English sentences focused "
    "on actions, objects, and setting. No speculation. No \"the image shows\" "
    "preamble."
)

QA_SYSTEM_PROMPT = (
    "You are Mr. Big-Eye, a careful video analyst. You will be shown K still "
    "frames sampled from a video, each labeled with its timestamp in seconds. "
    "Answer the user's question using only what is visible in these frames. "
    "When you cite a moment, insert a marker like [FRAME:t=29.7] on its own. "
    "The renderer will replace it with the corresponding thumbnail. Use this "
    "sparingly, 1-3 times per answer. "
    "If the frames are insufficient, say so plainly. Match the user's language."
)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=f"{settings.sglang_endpoint.rstrip('/')}/v1",
        api_key="EMPTY",
        timeout=settings.sglang_timeout,
    )


def _prepare_image(img: Image.Image, max_side: int | None = None) -> Image.Image:
    prepared = img.convert("RGB")
    if max_side and max_side > 0 and max(prepared.size) > max_side:
        prepared = prepared.copy()
        prepared.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return prepared


def _pil_to_data_url(
    img: Image.Image,
    quality: int = 85,
    max_side: int | None = None,
) -> str:
    buffer = io.BytesIO()
    quality = max(1, min(95, quality))
    _prepare_image(img, max_side).save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _select_evidence_frames(
    frames: list[Image.Image],
    timestamps: list[float],
    max_frames: int | None,
) -> tuple[list[Image.Image], list[float]]:
    pairs = list(zip(frames, timestamps, strict=False))
    if not pairs or not max_frames or max_frames <= 0 or len(pairs) <= max_frames:
        return [frame for frame, _ in pairs], [timestamp for _, timestamp in pairs]

    if max_frames == 1:
        selected_indices = [len(pairs) // 2]
    else:
        step = (len(pairs) - 1) / (max_frames - 1)
        selected_indices = []
        for i in range(max_frames):
            index = min(len(pairs) - 1, round(i * step))
            if index not in selected_indices:
                selected_indices.append(index)
        if len(selected_indices) < max_frames:
            selected = set(selected_indices)
            for index in range(len(pairs)):
                if index not in selected:
                    selected_indices.append(index)
                    selected.add(index)
                if len(selected_indices) == max_frames:
                    break
        selected_indices.sort()

    selected_pairs = [pairs[index] for index in selected_indices]
    return (
        [frame for frame, _ in selected_pairs],
        [timestamp for _, timestamp in selected_pairs],
    )


def _build_caption_messages(img: Image.Image) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _pil_to_data_url(img)},
                }
            ],
        },
    ]


def _build_qa_messages(
    question: str,
    frames: list[Image.Image],
    timestamps: list[float],
    history: list[dict[str, Any]] | None = None,
    max_frames: int | None = None,
    max_image_side: int | None = None,
    image_quality: int | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": QA_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)

    evidence_frames, evidence_timestamps = _select_evidence_frames(
        frames,
        timestamps,
        settings.vqa_max_frames if max_frames is None else max_frames,
    )
    image_side = settings.vqa_max_image_side if max_image_side is None else max_image_side
    quality = settings.vqa_image_quality if image_quality is None else image_quality
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Question: {question}\n"
                "Relevant sampled frames follow. Each image is preceded by its timestamp."
            ),
        }
    ]
    for frame, timestamp in zip(evidence_frames, evidence_timestamps, strict=False):
        content.append({"type": "text", "text": f"[t={timestamp:.1f}s]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _pil_to_data_url(
                        frame,
                        quality=quality,
                        max_side=image_side,
                    )
                },
            }
        )
    messages.append({"role": "user", "content": content})
    return messages


def _is_multimodal_prompt_too_long(exc: BadRequestError) -> bool:
    message = str(exc)
    return (
        "Multimodal prompt is too long" in message
        or "origin_input_ids" in message
        or "multimodal tokens" in message
    )


async def generate_caption(image: Image.Image) -> str:
    """Generate a concise English caption for a single frame."""
    response = await _client().chat.completions.create(
        model=settings.sglang_served_model_name,
        messages=_build_caption_messages(image),
        temperature=0.1,
        max_tokens=120,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


async def answer_question(
    question: str,
    frames: list[Image.Image],
    timestamps: list[float],
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Answer a question using sampled keyframes sorted by timestamp."""
    frame_limit = settings.vqa_max_frames if settings.vqa_max_frames > 0 else len(frames)
    image_side = settings.vqa_max_image_side
    attempt = 0
    while True:
        try:
            response = await _client().chat.completions.create(
                model=settings.sglang_served_model_name,
                messages=_build_qa_messages(
                    question,
                    frames,
                    timestamps,
                    history,
                    max_frames=frame_limit,
                    max_image_side=image_side,
                    image_quality=settings.vqa_image_quality,
                ),
                temperature=0.2,
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except BadRequestError as exc:
            if not _is_multimodal_prompt_too_long(exc) or attempt >= 3:
                raise

            next_frame_limit = max(1, min(frame_limit, len(frames)) // 2)
            next_image_side = 320 if image_side <= 0 else max(224, int(image_side * 0.75))
            if next_frame_limit == frame_limit and next_image_side == image_side:
                raise

            logger.warning(
                "VQA prompt exceeded multimodal token budget; retrying with "
                "%s frame(s), max image side %s",
                next_frame_limit,
                next_image_side,
            )
            frame_limit = next_frame_limit
            image_side = next_image_side
            attempt += 1


async def stream_answer_question(
    question: str,
    frames: list[Image.Image],
    timestamps: list[float],
    history: list[dict[str, Any]] | None = None,
):
    """Yield VQA answer tokens using sampled keyframes."""
    frame_limit = settings.vqa_max_frames if settings.vqa_max_frames > 0 else len(frames)
    image_side = settings.vqa_max_image_side
    stream = await _client().chat.completions.create(
        model=settings.sglang_served_model_name,
        messages=_build_qa_messages(
            question,
            frames,
            timestamps,
            history,
            max_frames=frame_limit,
            max_image_side=image_side,
            image_quality=settings.vqa_image_quality,
        ),
        temperature=0.2,
        max_tokens=512,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def chat_text(messages: Iterable[BaseMessage | dict[str, Any]]) -> str:
    """Generate a plain text assistant response for non-video chat turns."""
    response = await _client().chat.completions.create(
        model=settings.sglang_served_model_name,
        messages=[_message_to_openai(message) for message in messages],
        temperature=0.4,
        max_tokens=512,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


async def stream_text(messages: Iterable[BaseMessage | dict[str, Any]]):
    """Yield text deltas from SGLang's OpenAI-compatible streaming API."""
    stream = await _client().chat.completions.create(
        model=settings.sglang_served_model_name,
        messages=[_message_to_openai(message) for message in messages],
        temperature=0.4,
        max_tokens=512,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _message_to_openai(message: BaseMessage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    role = "assistant"
    msg_type = getattr(message, "type", "")
    if msg_type == "human":
        role = "user"
    elif msg_type == "system":
        role = "system"
    return {"role": role, "content": message.content}
