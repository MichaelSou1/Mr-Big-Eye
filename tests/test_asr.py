from app.asr import postprocess_text


def test_postprocess_text_strips_emoji_and_normalizes_course_terms():
    text = postprocess_text("rl 😊 a 二 c reinforce td")

    assert text == "RL A2C REINFORCE TD"
