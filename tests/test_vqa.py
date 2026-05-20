import base64
import io

from PIL import Image

from app.vqa import _build_qa_messages, _select_evidence_frames


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


def test_build_qa_messages_caps_frame_count_and_image_size():
    frames = [Image.new("RGB", (1200, 800), "white") for _ in range(8)]
    timestamps = [float(i) for i in range(8)]

    messages = _build_qa_messages(
        "What is happening?",
        frames,
        timestamps,
        max_frames=3,
        max_image_side=64,
        image_quality=70,
    )

    content = messages[-1]["content"]
    image_items = [item for item in content if item["type"] == "image_url"]

    assert len(image_items) == 3

    data_url = image_items[0]["image_url"]["url"]
    encoded = data_url.split(",", maxsplit=1)[1]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))

    assert max(image.size) <= 64
