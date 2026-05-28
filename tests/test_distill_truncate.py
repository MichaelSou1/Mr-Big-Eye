import json

from app.distill_truncate import truncate_record, truncate_tool_result

_BIG = "QUJD" * 2000  # >4000 chars, base64-charset


def test_image_b64_replaced_metadata_kept():
    content = json.dumps(
        {
            "tool": "retrieve_video_evidence",
            "frames": [
                {"timestamp": 12.5, "scene_id": "s3", "image_b64": _BIG, "caption": "a cat"}
            ],
        }
    )
    out = truncate_tool_result("retrieve_video_evidence", content)
    data = json.loads(out)  # still valid JSON
    frame = data["frames"][0]
    assert frame["image_b64"] == "<frame_b64_omitted: t=12.5, scene=s3>"
    assert frame["timestamp"] == 12.5
    assert frame["caption"] == "a cat"
    assert _BIG not in out


def test_answer_with_evidence_preserved_verbatim():
    content = json.dumps({"tool": "answer_with_evidence", "answer": "X " + _BIG})
    assert truncate_tool_result("answer_with_evidence", content) == content
    # also preserved when identified by payload tool, not arg name
    assert truncate_tool_result(None, content) == content


def test_verify_grounding_preserved():
    content = json.dumps({"tool": "verify_grounding", "answer": "ok", "grounding_report": {"grounded": True}})
    assert truncate_tool_result("verify_grounding", content) == content


def test_non_json_returned_unchanged():
    assert truncate_tool_result("retrieve_video_evidence", "not json") == "not json"


def test_oversized_blob_omitted():
    content = json.dumps({"tool": "x", "blob": _BIG})
    out = json.loads(truncate_tool_result("x", content))
    assert out["blob"].startswith("<omitted_b64: len=")


def test_truncate_record_uses_payload_tool_name():
    rec = {"role": "tool", "content": json.dumps({"tool": "retrieve_video_evidence", "image_b64": _BIG})}
    out = truncate_record(rec)
    assert "<frame_b64_omitted" in out["content"]
    # non-tool records pass through
    asst = {"role": "assistant", "content": "hi"}
    assert truncate_record(asst) is asst
