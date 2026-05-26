from app.main import _message_content_text, _parse_hotwords


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


def test_parse_hotwords_accepts_commas_and_lines():
    assert _parse_hotwords("REINFORCE, A2C\nTD target") == [
        "REINFORCE",
        "A2C",
        "TD target",
    ]
    assert _parse_hotwords(" \n ") is None
