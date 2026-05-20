import base64
import io
from typing import Any

from openai import AsyncOpenAI
from PIL import Image

from app.config import settings


CAPTION_SYSTEM_PROMPT = (
    "Describe this video frame in one or two concise English sentences focused "
    "on actions, objects, and setting. No speculation. No \"the image shows\" "
    "preamble."
)

QA_SYSTEM_PROMPT = (
    "You are Mr. Big-Eye, a careful video analyst. You will be shown K still "
    "frames sampled from a video, each labeled with its timestamp in seconds. "
    "Answer the user's question using only what is visible in these frames. "
    "When you reference a specific moment, cite the timestamp like [t=29.7s]. "
    "If the frames are insufficient, say so plainly. Match the user's language."
)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=f"{settings.sglang_endpoint.rstrip('/')}/v1",
        api_key="EMPTY",
        timeout=settings.sglang_timeout,
    )


def _pil_to_data_url(img: Image.Image, quality: int = 85) -> str:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": QA_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Question: {question}\n"
                "Relevant sampled frames follow. Each image is preceded by its timestamp."
            ),
        }
    ]
    for frame, timestamp in zip(frames, timestamps, strict=False):
        content.append({"type": "text", "text": f"[t={timestamp:.1f}s]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _pil_to_data_url(frame)},
            }
        )
    messages.append({"role": "user", "content": content})
    return messages


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
    response = await _client().chat.completions.create(
        model=settings.sglang_served_model_name,
        messages=_build_qa_messages(question, frames, timestamps, history),
        temperature=0.2,
        max_tokens=512,
    )
    content = response.choices[0].message.content or ""
    return content.strip()
