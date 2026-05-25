"""VLM API client.

Supports two interchangeable request formats so the same business code can
target multiple providers:

- "responses": Volcano Ark / Doubao Responses API
  POST {base_url}/responses
  body: {"model", "input": [{"role", "content": [{"type":"input_text"|"input_image", ...}]}]}

- "chat_completions": OpenAI-compatible Chat Completions
  POST {base_url}/chat/completions  (e.g. Xiaomi MiMo)
  body: {"model", "messages": [{"role", "content": "..." | [...]}]}

The selection is driven by settings.vlm_api_format. Callers use the public
coroutines (generate_caption / answer_question / stream_answer_question)
without caring about the underlying wire format.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from typing import Any, AsyncIterator

import httpx
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
    "The renderer will replace it with the corresponding thumbnail. "
    "Insert at least one [FRAME:t=...] marker for every concrete visual claim. "
    "Two to four markers per answer is typical; only omit them when the answer "
    "is purely conversational (e.g. \"I don't see X in the checked evidence.\"). "
    "If the frames are insufficient, say so plainly. Match the user's language."
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class VLMAPIError(RuntimeError):
    """Raised when the VLM API returns a non-2xx response."""


def _headers() -> dict[str, str]:
    if not settings.vlm_api_key:
        raise VLMAPIError(
            "VLM_API_KEY is empty. Set it in .env (or the environment) to call the VLM provider."
        )
    return {
        "Authorization": f"Bearer {settings.vlm_api_key}",
        "Content-Type": "application/json",
    }


def _endpoint() -> str:
    base = settings.vlm_api_base_url.rstrip("/")
    if settings.vlm_api_format == "responses":
        return f"{base}/responses"
    return f"{base}/chat/completions"


def _client(*, stream: bool = False) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=settings.vlm_api_timeout)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Format-specific payload builders
# ---------------------------------------------------------------------------


def _image_part(url: str) -> dict[str, Any]:
    if settings.vlm_api_format == "responses":
        return {"type": "input_image", "image_url": url}
    return {"type": "image_url", "image_url": {"url": url}}


def _text_part(text: str, *, assistant: bool = False) -> dict[str, Any]:
    if settings.vlm_api_format == "responses":
        return {
            "type": "output_text" if assistant else "input_text",
            "text": text,
        }
    return {"type": "text", "text": text}


def _history_to_input(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert plain {role, content} history to the active wire format."""
    output: list[dict[str, Any]] = []
    for item in history:
        role = item.get("role") or "user"
        text = item.get("content")
        if text is None:
            continue
        if settings.vlm_api_format == "responses":
            item: dict[str, Any] = {
                "role": role,
                "content": [_text_part(str(text), assistant=role == "assistant")],
            }
            if role == "assistant":
                # Volcengine ARK rejects replayed assistant items without status.
                item["type"] = "message"
                item["status"] = "completed"
            output.append(item)
        else:
            output.append({"role": role, "content": str(text)})
    return output


def _build_caption_payload(image: Image.Image) -> dict[str, Any]:
    if settings.vlm_api_format == "responses":
        return {
            "model": settings.vlm_model_name,
            "input": [
                {"role": "system", "content": [_text_part(CAPTION_SYSTEM_PROMPT)]},
                {"role": "user", "content": [_image_part(_pil_to_data_url(image))]},
            ],
            "temperature": 0.1,
            "max_output_tokens": 120,
        }
    return {
        "model": settings.vlm_model_name,
        "messages": [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [_image_part(_pil_to_data_url(image))]},
        ],
        "temperature": 0.1,
        "max_tokens": 120,
    }


def _build_qa_payload(
    question: str,
    frames: list[Image.Image],
    timestamps: list[float],
    history: list[dict[str, Any]] | None = None,
    max_frames: int | None = None,
    max_image_side: int | None = None,
    image_quality: int | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    evidence_frames, evidence_timestamps = _select_evidence_frames(
        frames,
        timestamps,
        settings.vqa_max_frames if max_frames is None else max_frames,
    )
    image_side = settings.vqa_max_image_side if max_image_side is None else max_image_side
    quality = settings.vqa_image_quality if image_quality is None else image_quality

    user_content: list[dict[str, Any]] = [
        _text_part(
            f"Question: {question}\n"
            "Relevant sampled frames follow. Each image is preceded by its timestamp."
        )
    ]
    for frame, timestamp in zip(evidence_frames, evidence_timestamps, strict=False):
        user_content.append(_text_part(f"[t={timestamp:.1f}s]"))
        user_content.append(
            _image_part(_pil_to_data_url(frame, quality=quality, max_side=image_side))
        )

    history_messages = _history_to_input(history or [])

    if settings.vlm_api_format == "responses":
        return {
            "model": settings.vlm_model_name,
            "input": [
                {"role": "system", "content": [_text_part(QA_SYSTEM_PROMPT)]},
                *history_messages,
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_output_tokens": settings.vqa_max_output_tokens,
            "stream": stream,
        }
    return {
        "model": settings.vlm_model_name,
        "messages": [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            *history_messages,
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": settings.vqa_max_output_tokens,
        "stream": stream,
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_text(payload: dict[str, Any]) -> str:
    if settings.vlm_api_format == "responses":
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"].strip()
        chunks: list[str] = []
        for item in payload.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []) or []:
                text = part.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        return "".join(chunks).strip()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        ).strip()
    return str(content or "").strip()


def _streamed_delta(event: dict[str, Any]) -> str | None:
    if settings.vlm_api_format == "responses":
        event_type = event.get("type") or ""
        if event_type.endswith("output_text.delta") or event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                return delta
            if isinstance(delta, dict):
                text = delta.get("text")
                if isinstance(text, str):
                    return text
        return None
    choices = event.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    text = delta.get("content")
    if isinstance(text, str) and text:
        return text
    return None


# ---------------------------------------------------------------------------
# HTTP calls
# ---------------------------------------------------------------------------


MAX_RETRY_TIME = 5  # ModelScope free-tier 429s are frequent; cap ~62s of backoff.
_TRANSIENT_HTTP_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


async def _post_json(payload: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(MAX_RETRY_TIME + 1):
        try:
            async with _client() as client:
                response = await client.post(_endpoint(), headers=_headers(), json=payload)
        except _TRANSIENT_HTTP_ERRORS as exc:
            if attempt >= MAX_RETRY_TIME:
                raise
            backoff = 0.5 * (2 ** attempt)
            logger.warning(
                "VLM POST transient error (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1, MAX_RETRY_TIME + 1, exc, backoff,
            )
            await asyncio.sleep(backoff)
            continue
        # 429 (rate limit) and 5xx are server-side transient; back off and retry.
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt >= MAX_RETRY_TIME:
                raise VLMAPIError(
                    f"VLM API {response.status_code}: {response.text[:500]}"
                )
            backoff = min(30.0, 1.0 * (2 ** attempt))
            # Honor Retry-After if the server told us how long to wait.
            retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
            if retry_after:
                try:
                    backoff = max(backoff, float(retry_after))
                except (TypeError, ValueError):
                    pass
            logger.warning(
                "VLM POST %d (attempt %d/%d); retrying in %.1fs",
                response.status_code, attempt + 1, MAX_RETRY_TIME + 1, backoff,
            )
            await asyncio.sleep(backoff)
            continue
        if response.status_code >= 400:
            raise VLMAPIError(
                f"VLM API {response.status_code}: {response.text[:500]}"
            )
        return response.json()
    raise RuntimeError("unreachable")  # pragma: no cover


async def _stream_sse(payload: dict[str, Any]) -> AsyncIterator[str]:
    for attempt in range(MAX_RETRY_TIME + 1):
        yielded_any = False
        try:
            async with _client(stream=True) as client:
                async with client.stream(
                    "POST",
                    _endpoint(),
                    headers=_headers(),
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise VLMAPIError(
                            f"VLM API {response.status_code}: {body.decode('utf-8', errors='replace')[:500]}"
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith(":"):  # SSE comment / keep-alive
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if not data or data == "[DONE]":
                            if data == "[DONE]":
                                return
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON SSE chunk: %s", data[:120])
                            continue
                        delta = _streamed_delta(event)
                        if delta:
                            yielded_any = True
                            yield delta
                    return
        except _TRANSIENT_HTTP_ERRORS as exc:
            # Don't retry mid-stream — would duplicate tokens to the consumer.
            if yielded_any or attempt >= MAX_RETRY_TIME:
                raise
            backoff = 0.5 * (2 ** attempt)
            logger.warning(
                "VLM stream transient error (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1, MAX_RETRY_TIME + 1, exc, backoff,
            )
            await asyncio.sleep(backoff)


def _is_multimodal_prompt_too_long(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "multimodal prompt is too long" in message
        or "multimodal tokens" in message
        or "too many tokens" in message
        or "context length" in message
        or "input is too long" in message
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_caption(image: Image.Image) -> str:
    """Generate a concise English caption for a single frame."""
    payload = _build_caption_payload(image)
    data = await _post_json(payload)
    return _extract_text(data)


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
            payload = _build_qa_payload(
                question,
                frames,
                timestamps,
                history,
                max_frames=frame_limit,
                max_image_side=image_side,
                image_quality=settings.vqa_image_quality,
                stream=False,
            )
            data = await _post_json(payload)
            text = _extract_text(data)
            if text:
                return text
            # Empty completion: some VLMs (Qwen3-VL, doubao) occasionally return
            # an empty string for valid prompts. Retry once before giving up;
            # callers downstream (D2 salvage) cannot recover an empty draft.
            logger.warning("VLM returned empty answer; retrying once.")
            data = await _post_json(payload)
            return _extract_text(data)
        except VLMAPIError as exc:
            if not _is_multimodal_prompt_too_long(exc) or attempt >= 3:
                raise

            next_frame_limit = max(1, min(frame_limit, len(frames)) // 2)
            next_image_side = 320 if image_side <= 0 else max(224, int(image_side * 0.75))
            if next_frame_limit == frame_limit and next_image_side == image_side:
                raise

            logger.warning(
                "VQA prompt exceeded multimodal budget; retrying with "
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
) -> AsyncIterator[str]:
    """Yield VQA answer tokens using sampled keyframes."""
    payload = _build_qa_payload(
        question,
        frames,
        timestamps,
        history,
        max_frames=settings.vqa_max_frames if settings.vqa_max_frames > 0 else len(frames),
        max_image_side=settings.vqa_max_image_side,
        image_quality=settings.vqa_image_quality,
        stream=True,
    )
    async for delta in _stream_sse(payload):
        yield delta
