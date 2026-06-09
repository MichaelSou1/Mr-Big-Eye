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
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Protocol

import httpx
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


VLMAPIFormat = Literal["responses", "chat_completions"]


@dataclass(frozen=True)
class VLMEndpointConfig:
    api_format: VLMAPIFormat
    base_url: str
    api_key: str
    model_name: str
    timeout: int


class VLMBackbone(Protocol):
    async def generate_caption(self, image: Image.Image) -> str:
        """Generate a concise caption for one frame."""

    async def answer_question(
        self,
        question: str,
        frames: list[Image.Image],
        timestamps: list[float],
        history: list[dict[str, Any]] | None = None,
        *,
        system_prompt: str | None = None,
        subject_registry: list[dict[str, Any]] | None = None,
        text_evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        """Answer from sampled frames."""

    def stream_answer_question(
        self,
        question: str,
        frames: list[Image.Image],
        timestamps: list[float],
        history: list[dict[str, Any]] | None = None,
        *,
        system_prompt: str | None = None,
        subject_registry: list[dict[str, Any]] | None = None,
        text_evidence: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer tokens from sampled frames."""


CAPTION_SYSTEM_PROMPT = (
    "Describe this video frame in one or two concise English sentences focused "
    "on actions, objects, and setting. No speculation. No \"the image shows\" "
    "preamble."
)

QA_SYSTEM_PROMPT = (
    "You are Mr. Big-Eye, a warm, concise, and careful video analyst. You will "
    "be shown K still frames sampled from a video, each labeled with its "
    "timestamp in seconds. Match the user's language.\n"
    "\n"
    "═══ MCQ HARD RULE — read first, applies whenever the question lists "
    "`Candidates:` with A/B/C/D/E options ═══\n"
    "You MUST pick exactly one letter from the listed options and start your "
    "answer with `The correct answer is X) <option text>` (X ∈ A/B/C/D/E). "
    "This is **mandatory even when the visual evidence is weak, partial, "
    "ambiguous, indirect, or seems to contradict every option**. Pick the "
    "**least implausible** option and cite frames for it. A 30%-confident guess "
    "always scores higher than refusing to commit.\n"
    "BANNED on MCQs — using any of the following on an MCQ counts as a failed "
    "answer (zero score):\n"
    "  • \"I cannot determine ...\" / \"I do not see ...\" / \"cannot be "
    "confirmed\" / \"do not show\" / \"not visible in ...\"\n"
    "  • \"insufficient evidence\" / \"the provided frames do not\" / "
    "\"based on the frames I cannot\"\n"
    "  • \"证据不足\" / \"无法确定\" / \"看不到\" / \"看不清\"\n"
    "  • \"the premise is incorrect\" / \"none of the options match\" / "
    "\"none of the above\"\n"
    "Decision procedure for ambiguous MCQs: (1) list which 1-2 options have "
    "*any* visual support, however weak; (2) eliminate the clearly contradicted "
    "ones; (3) among the survivors, commit to the most consistent one; "
    "(4) emit the answer in the mandated format.\n"
    "═══ End MCQ HARD RULE ═══\n"
    "\n"
    "For non-MCQ (open-ended) questions: answer strictly from the provided "
    "frames and any provided transcript/slide evidence; do not speculate about "
    "content outside the supplied evidence; "
    "if evidence is truly insufficient, say \"证据不足\" plainly — but again, "
    "this exception does NOT apply to MCQs.\n"
    "Citation marker protocol is strict: image timestamps are shown as "
    "`[t=X.Xs]` before each image, but final answers must convert them to "
    "`[FRAME:t=X.X]`. Never use bare `[t=Xs]` as a final citation. Every "
    "concrete visual claim must carry a [FRAME:t=X.X] or [SLIDE:t=X.X] "
    "citation; 2-4 markers per answer is typical.\n"
    "Use the exact [TRANSCRIPT:t=A.B-C.D] and [SLIDE:t=X.X] markers printed "
    "in the evidence block when citing speech or OCR/PPT evidence. Do not "
    "invent nearby transcript/slide timestamps.\n"
    "The renderer will replace valid citation markers with inline evidence."
)

SUBJECT_DELTA_PROTOCOL = (
    "\n\n在答案末尾追加一行 JSON（不要任何 markdown 围栏），格式：\n"
    'SUBJECT_DELTAS: {"deltas": [\n'
    '  {"op": "add", "id": "person_A", "label": "红衣男子", '
    '"first_seen_t": 12.3, "attributes": ["持有背包"], '
    '"evidence_frames": [12.3]},\n'
    '  {"op": "update", "id": "person_B", '
    '"attributes_add": ["在跑步"], "last_seen_t": 45.0, '
    '"evidence_frames_add": [45.0]}\n'
    "]}\n"
    '若无主体变化，输出 SUBJECT_DELTAS: {"deltas": []}。'
)

ANSWER_WITH_EVIDENCE_PROMPT = QA_SYSTEM_PROMPT + SUBJECT_DELTA_PROTOCOL

SEGMENT_FOCUS_PROMPT = (
    "你是 Observer 子模块（细致严谨的视频分析助手），**不是最终回答者**。\n"
    "你的输出会作为中间观察传给上游 orchestrator，由它再调 `answer_with_evidence` "
    "产生面向用户的最终答案。**不要尝试直接回答用户的原始问题**（特别是 MCQ 不要选选项，"
    "不要写 'The correct answer is X)' 这种最终答案格式）；只描述本次窗口内看到的视觉细节。\n"
    "你看到的是同一片段内按时间顺序均匀抽取的若干帧，每帧带 [t=Xs] 时间戳。\n"
    "任务：\n"
    "1. 聚焦用户问题所关心的视觉细节，按帧描述变化。\n"
    "2. 严格基于所给帧；不要推断该窗口之外发生的事，也不要想象帧之间的内容。\n"
    "3. 如果细节模糊或证据不足，明确说\"该窗口内看不清/看不到\"。\n"
    "4. 所有可视事实必须给出 [FRAME:t=X.X] 引用；典型 2-4 个。"
    f"{SUBJECT_DELTA_PROTOCOL}"
)

STITCHED_VERIFY_PROMPT = (
    "你是 Observer 子模块（多时刻综合分析专家），**不是最终回答者**。\n"
    "你的输出会作为中间观察传给上游 orchestrator，由它再调 `answer_with_evidence` "
    "产生面向用户的最终答案。**不要尝试直接回答用户的原始问题**（特别是 MCQ 不要选选项，"
    "不要写 'The correct answer is X)' 这种最终答案格式）；只描述各窗口的观察与对比。\n"
    "你看到的是从同一视频不同时间段抽取的若干帧，每帧带 [t=Xs] 时间戳。\n"
    "任务：\n"
    "1. 简要描述每个时间段呈现的内容（按时间顺序）。\n"
    "2. 指出不同时刻之间的联系、变化或对比。\n"
    "3. 严格基于所给帧回答；不要推测帧之间时间间隔内发生了什么。\n"
    "4. 不要做超出帧内容的推断；如果证据不足以回答问题，明确说出来。\n"
    "5. 所有可视事实必须给出 [FRAME:t=X.X] 引用；典型 2-4 个。"
    f"{SUBJECT_DELTA_PROTOCOL}"
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class VLMAPIError(RuntimeError):
    """Raised when the VLM API returns a non-2xx response."""


def _remote_endpoint_config() -> VLMEndpointConfig:
    return VLMEndpointConfig(
        api_format=settings.vlm_api_format,
        base_url=settings.vlm_api_base_url,
        api_key=settings.vlm_api_key,
        model_name=settings.vlm_model_name,
        timeout=settings.vlm_api_timeout,
    )


def _local_endpoint_config() -> VLMEndpointConfig:
    if not settings.local_vlm_base_url or not settings.local_vlm_model_name:
        raise VLMAPIError(
            "LOCAL_VLM_BASE_URL and LOCAL_VLM_MODEL_NAME must be set when "
            "AGENT_VLM_BACKEND=local."
        )
    return VLMEndpointConfig(
        api_format="chat_completions",
        base_url=settings.local_vlm_base_url,
        api_key=settings.local_vlm_api_key or "EMPTY",
        model_name=settings.local_vlm_model_name,
        timeout=settings.vlm_api_timeout,
    )


def _headers(config: VLMEndpointConfig | None = None) -> dict[str, str]:
    endpoint = config or _remote_endpoint_config()
    if not endpoint.api_key:
        raise VLMAPIError(
            "VLM_API_KEY is empty. Set it in .env (or the environment) to call the VLM provider."
        )
    return {
        "Authorization": f"Bearer {endpoint.api_key}",
        "Content-Type": "application/json",
    }


def _endpoint(config: VLMEndpointConfig | None = None) -> str:
    endpoint = config or _remote_endpoint_config()
    base = endpoint.base_url.rstrip("/")
    if endpoint.api_format == "responses":
        return f"{base}/responses"
    return f"{base}/chat/completions"


def _client(*, stream: bool = False, config: VLMEndpointConfig | None = None) -> httpx.AsyncClient:
    endpoint = config or _remote_endpoint_config()
    return httpx.AsyncClient(timeout=endpoint.timeout)


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


def _active_api_format(config: VLMEndpointConfig | None = None) -> VLMAPIFormat:
    return (config or _remote_endpoint_config()).api_format


def _image_part(url: str, config: VLMEndpointConfig | None = None) -> dict[str, Any]:
    if _active_api_format(config) == "responses":
        return {"type": "input_image", "image_url": url}
    return {"type": "image_url", "image_url": {"url": url}}


def _text_part(
    text: str,
    *,
    assistant: bool = False,
    config: VLMEndpointConfig | None = None,
) -> dict[str, Any]:
    if _active_api_format(config) == "responses":
        return {
            "type": "output_text" if assistant else "input_text",
            "text": text,
        }
    return {"type": "text", "text": text}


def _history_to_input(
    history: list[dict[str, Any]],
    config: VLMEndpointConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert plain {role, content} history to the active wire format."""
    output: list[dict[str, Any]] = []
    for item in history:
        role = item.get("role") or "user"
        text = item.get("content")
        if text is None:
            continue
        if _active_api_format(config) == "responses":
            item: dict[str, Any] = {
                "role": role,
                "content": [_text_part(str(text), assistant=role == "assistant", config=config)],
            }
            if role == "assistant":
                # Volcengine ARK rejects replayed assistant items without status.
                item["type"] = "message"
                item["status"] = "completed"
            output.append(item)
        else:
            output.append({"role": role, "content": str(text)})
    return output


def _build_caption_payload(
    image: Image.Image,
    config: VLMEndpointConfig | None = None,
) -> dict[str, Any]:
    endpoint = config or _remote_endpoint_config()
    if endpoint.api_format == "responses":
        return {
            "model": endpoint.model_name,
            "input": [
                {"role": "system", "content": [_text_part(CAPTION_SYSTEM_PROMPT, config=config)]},
                {"role": "user", "content": [_image_part(_pil_to_data_url(image), config=config)]},
            ],
            "temperature": 0.1,
            "max_output_tokens": 120,
        }
    return {
        "model": endpoint.model_name,
        "messages": [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [_image_part(_pil_to_data_url(image), config=config)]},
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
    system_prompt: str | None = None,
    subject_registry: list[dict[str, Any]] | None = None,
    text_evidence: list[dict[str, Any]] | None = None,
    config: VLMEndpointConfig | None = None,
) -> dict[str, Any]:
    endpoint = config or _remote_endpoint_config()
    evidence_frames, evidence_timestamps = _select_evidence_frames(
        frames,
        timestamps,
        settings.vqa_max_frames if max_frames is None else max_frames,
    )
    image_side = settings.vqa_max_image_side if max_image_side is None else max_image_side
    quality = settings.vqa_image_quality if image_quality is None else image_quality
    prompt = system_prompt or QA_SYSTEM_PROMPT
    rendered_question = _question_with_subject_registry(question, subject_registry or [])

    evidence_text = _render_text_evidence(text_evidence or [])
    prompt_text = (
        f"Question: {rendered_question}\n"
        "Relevant sampled frames follow. Each image is preceded by its timestamp."
    )
    if evidence_text:
        prompt_text += (
            "\n\nAdditional transcript/slide evidence follows. Cite it with the exact "
            "[TRANSCRIPT:t=...] or [SLIDE:t=...] marker shown when using it.\n"
            f"{evidence_text}"
        )
    user_content: list[dict[str, Any]] = [_text_part(prompt_text, config=config)]
    for frame, timestamp in zip(evidence_frames, evidence_timestamps, strict=False):
        user_content.append(_text_part(f"[t={timestamp:.1f}s]", config=config))
        user_content.append(
            _image_part(_pil_to_data_url(frame, quality=quality, max_side=image_side), config=config)
        )

    history_messages = _history_to_input(history or [], config=config)

    if endpoint.api_format == "responses":
        return {
            "model": endpoint.model_name,
            "input": [
                {"role": "system", "content": [_text_part(prompt, config=config)]},
                *history_messages,
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_output_tokens": settings.vqa_max_output_tokens,
            "stream": stream,
        }
    return {
        "model": endpoint.model_name,
        "messages": [
            {"role": "system", "content": prompt},
            *history_messages,
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": settings.vqa_max_output_tokens,
        "stream": stream,
    }


def _question_with_subject_registry(
    question: str,
    subject_registry: list[dict[str, Any]],
) -> str:
    entries = [
        entry
        for entry in subject_registry
        if isinstance(entry, dict) and entry.get("id")
    ]
    if not entries:
        return question

    lines = ["【已知主体登记表】（来自此前观察，可直接引用其 id/label 进行消歧）"]
    for entry in entries:
        sid = str(entry.get("id") or "").strip()
        label = str(entry.get("label") or sid).strip()
        first_seen = _format_registry_time(entry.get("first_seen_t"))
        last_seen = _format_registry_time(entry.get("last_seen_t"))
        attributes = ", ".join(str(item) for item in entry.get("attributes") or [])
        if not attributes:
            attributes = "暂无稳定属性"
        lines.append(
            f"- {sid} ({label}, 首次见于 {first_seen}, 最近 {last_seen}): {attributes}"
        )
    return "\n".join([*lines, "", "【用户问题】", question])


def _render_text_evidence(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items[:12]:
        marker = str(item.get("marker") or "").strip()
        text = str(item.get("text") or "").strip()
        if not marker or not text:
            continue
        lines.append(f"{marker} {text}")
    return "\n".join(lines)


def _format_registry_time(value: Any) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "未知"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_text(payload: dict[str, Any], config: VLMEndpointConfig | None = None) -> str:
    if _active_api_format(config) == "responses":
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


def _streamed_delta(
    event: dict[str, Any],
    config: VLMEndpointConfig | None = None,
) -> str | None:
    if _active_api_format(config) == "responses":
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


async def _post_json(
    payload: dict[str, Any],
    config: VLMEndpointConfig | None = None,
) -> dict[str, Any]:
    for attempt in range(MAX_RETRY_TIME + 1):
        try:
            async with _client(config=config) as client:
                response = await client.post(
                    _endpoint(config),
                    headers=_headers(config),
                    json=payload,
                )
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


async def _stream_sse(
    payload: dict[str, Any],
    config: VLMEndpointConfig | None = None,
) -> AsyncIterator[str]:
    for attempt in range(MAX_RETRY_TIME + 1):
        yielded_any = False
        try:
            async with _client(stream=True, config=config) as client:
                async with client.stream(
                    "POST",
                    _endpoint(config),
                    headers=_headers(config),
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
                        delta = _streamed_delta(event, config=config)
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


async def _post_json_for_config(
    payload: dict[str, Any],
    config: VLMEndpointConfig,
) -> dict[str, Any]:
    if config == _remote_endpoint_config():
        return await _post_json(payload)
    return await _post_json(payload, config=config)


async def _stream_sse_for_config(
    payload: dict[str, Any],
    config: VLMEndpointConfig,
) -> AsyncIterator[str]:
    if config == _remote_endpoint_config():
        async for delta in _stream_sse(payload):
            yield delta
        return
    async for delta in _stream_sse(payload, config=config):
        yield delta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RemoteVLMBackbone:
    def __init__(self, config: VLMEndpointConfig | None = None):
        self.config = config or _remote_endpoint_config()

    async def generate_caption(self, image: Image.Image) -> str:
        payload = _build_caption_payload(image, config=self.config)
        data = await _post_json_for_config(payload, self.config)
        return _extract_text(data, config=self.config)

    async def answer_question(
        self,
        question: str,
        frames: list[Image.Image],
        timestamps: list[float],
        history: list[dict[str, Any]] | None = None,
        *,
        system_prompt: str | None = None,
        subject_registry: list[dict[str, Any]] | None = None,
        text_evidence: list[dict[str, Any]] | None = None,
    ) -> str:
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
                    system_prompt=system_prompt,
                    subject_registry=subject_registry,
                    text_evidence=text_evidence,
                    config=self.config,
                )
                data = await _post_json_for_config(payload, self.config)
                text = _extract_text(data, config=self.config)
                if text:
                    return text
                # Empty completion: some VLMs (Qwen3-VL, doubao) occasionally return
                # an empty string for valid prompts. Retry once before giving up.
                logger.warning("VLM returned empty answer; retrying once.")
                data = await _post_json_for_config(payload, self.config)
                return _extract_text(data, config=self.config)
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
        self,
        question: str,
        frames: list[Image.Image],
        timestamps: list[float],
        history: list[dict[str, Any]] | None = None,
        *,
        system_prompt: str | None = None,
        subject_registry: list[dict[str, Any]] | None = None,
        text_evidence: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        payload = _build_qa_payload(
            question,
            frames,
            timestamps,
            history,
            max_frames=settings.vqa_max_frames if settings.vqa_max_frames > 0 else len(frames),
            max_image_side=settings.vqa_max_image_side,
            image_quality=settings.vqa_image_quality,
            stream=True,
            system_prompt=system_prompt,
            subject_registry=subject_registry,
            text_evidence=text_evidence,
            config=self.config,
        )
        async for delta in _stream_sse_for_config(payload, self.config):
            yield delta


class LocalVLMBackbone(RemoteVLMBackbone):
    def __init__(self, config: VLMEndpointConfig | None = None):
        super().__init__(config or _local_endpoint_config())


def get_agent_vlm_backbone() -> VLMBackbone:
    if settings.agent_vlm_backend == "local":
        return LocalVLMBackbone()
    return RemoteVLMBackbone()


def current_agent_vlm_name() -> str:
    if settings.agent_vlm_backend == "local":
        return settings.local_vlm_model_name or ""
    return settings.vlm_model_name


def current_agent_vlm_cache_label() -> str:
    backend = settings.agent_vlm_backend
    model = current_agent_vlm_name()
    if backend == "local":
        base = settings.local_vlm_base_url.rstrip("/")
        return f"{backend}:{model}@{base}"
    return f"{backend}:{model}@{settings.vlm_api_base_url.rstrip('/')}"


async def generate_caption(image: Image.Image) -> str:
    """Generate a concise English caption for a single frame."""
    return await get_agent_vlm_backbone().generate_caption(image)


async def answer_question(
    question: str,
    frames: list[Image.Image],
    timestamps: list[float],
    history: list[dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    subject_registry: list[dict[str, Any]] | None = None,
    text_evidence: list[dict[str, Any]] | None = None,
) -> str:
    """Answer a question using sampled keyframes sorted by timestamp."""
    return await get_agent_vlm_backbone().answer_question(
        question,
        frames,
        timestamps,
        history,
        system_prompt=system_prompt,
        subject_registry=subject_registry,
        text_evidence=text_evidence,
    )


async def stream_answer_question(
    question: str,
    frames: list[Image.Image],
    timestamps: list[float],
    history: list[dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    subject_registry: list[dict[str, Any]] | None = None,
    text_evidence: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Yield VQA answer tokens using sampled keyframes."""
    async for delta in get_agent_vlm_backbone().stream_answer_question(
        question,
        frames,
        timestamps,
        history,
        system_prompt=system_prompt,
        subject_registry=subject_registry,
        text_evidence=text_evidence,
    ):
        yield delta
