from app.main import _message_content_text


def test_message_content_text_accepts_string_and_content_parts():
    assert _message_content_text("hello") == "hello"
    assert (
        _message_content_text(
            [
                {"type": "text", "text": "he"},
                "ll",
                {"type": "text", "text": "o"},
                {"type": "image_url", "image_url": {"url": "ignored"}},
            ]
        )
        == "hello"
    )
