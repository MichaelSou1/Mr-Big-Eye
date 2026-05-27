import base64
import io

from PIL import Image

from app.config import settings
from app import vqa
from app.vqa import QA_SYSTEM_PROMPT, _build_qa_payload, _select_evidence_frames


def test_qa_system_prompt_demands_frame_markers():
    """Phase C: prompt must push the VLM to cite frames, not 'use sparingly'."""
    assert "[frame:t=" in QA_SYSTEM_PROMPT.lower()
    assert "Never use bare `[t=Xs]`" in QA_SYSTEM_PROMPT
    assert "exact [TRANSCRIPT:t=A.B-C.D]" in QA_SYSTEM_PROMPT
    assert "证据不足" in QA_SYSTEM_PROMPT
    assert "sparingly" not in QA_SYSTEM_PROMPT.lower()


def test_select_evidence_frames_evenly_samples_sorted_frames():
    frames = [Image.new("RGB", (32, 32), (i, i, i)) for i in range(10)]
    timestamps = [float(i) for i in range(10)]

    selected_frames, selected_timestamps = _select_evidence_frames(
        frames,
        timestamps,
        max_frames=4,
    )

    assert len(selected_frames) == 4
    assert selected_timestamps == [0.0, 3.0, 6.0, 9.0]


def _image_parts(payload: dict) -> list[dict]:
    if settings.vlm_api_format == "responses":
        user_content = payload["input"][-1]["content"]
        return [item for item in user_content if item.get("type") == "input_image"]
    user_content = payload["messages"][-1]["content"]
    return [item for item in user_content if item.get("type") == "image_url"]


def _system_text(payload: dict) -> str:
    if settings.vlm_api_format == "responses":
        return payload["input"][0]["content"][0]["text"]
    return payload["messages"][0]["content"]


def _user_text(payload: dict) -> str:
    if settings.vlm_api_format == "responses":
        parts = payload["input"][-1]["content"]
    else:
        parts = payload["messages"][-1]["content"]
    return "\n".join(item.get("text", "") for item in parts if item.get("type") in {"input_text", "text"})


def _decode_first_image(payload: dict) -> Image.Image:
    parts = _image_parts(payload)
    item = parts[0]
    if settings.vlm_api_format == "responses":
        data_url = item["image_url"]
    else:
        data_url = item["image_url"]["url"]
    encoded = data_url.split(",", maxsplit=1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_build_qa_payload_caps_frame_count_and_image_size():
    frames = [Image.new("RGB", (1200, 800), "white") for _ in range(8)]
    timestamps = [float(i) for i in range(8)]

    payload = _build_qa_payload(
        "What is happening?",
        frames,
        timestamps,
        max_frames=3,
        max_image_side=64,
        image_quality=70,
    )

    assert len(_image_parts(payload)) == 3
    image = _decode_first_image(payload)
    assert max(image.size) <= 64


def test_build_qa_payload_uses_custom_prompt_and_subject_registry():
    payload = _build_qa_payload(
        "What is the person doing?",
        [Image.new("RGB", (32, 32), "white")],
        [12.3],
        system_prompt="CUSTOM OBSERVER",
        subject_registry=[
            {
                "id": "person_A",
                "label": "红衣男子",
                "first_seen_t": 12.3,
                "last_seen_t": 45.6,
                "attributes": ["持有背包", "在跑步"],
            }
        ],
    )

    assert _system_text(payload) == "CUSTOM OBSERVER"
    text = _user_text(payload)
    assert "已知主体登记表" in text
    assert "person_A (红衣男子, 首次见于 12.3s, 最近 45.6s)" in text
    assert "【用户问题】" in text


async def _fake_post_json(payload: dict) -> dict:
    _fake_post_json.payload = payload
    if settings.vlm_api_format == "responses":
        return {"output_text": "ok"}
    return {"choices": [{"message": {"content": "ok"}}]}


def test_answer_question_accepts_custom_system_prompt(monkeypatch):
    async def run():
        monkeypatch.setattr(vqa, "_post_json", _fake_post_json)
        answer = await vqa.answer_question(
            "What is happening?",
            [Image.new("RGB", (32, 32), "white")],
            [1.0],
            history=None,
            system_prompt="CUSTOM SYSTEM",
        )
        assert answer == "ok"
        assert _system_text(_fake_post_json.payload) == "CUSTOM SYSTEM"

    import asyncio

    asyncio.run(run())
