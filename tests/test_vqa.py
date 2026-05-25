import base64
import io

from PIL import Image

from app.config import settings
from app.vqa import QA_SYSTEM_PROMPT, _build_qa_payload, _select_evidence_frames


def test_qa_system_prompt_demands_frame_markers():
    """Phase C: prompt must push the VLM to cite frames, not 'use sparingly'."""
    assert "at least one" in QA_SYSTEM_PROMPT.lower()
    assert "[frame:t=" in QA_SYSTEM_PROMPT.lower()
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
