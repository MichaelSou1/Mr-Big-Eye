"""Base64 truncation for tool results before sampling (Phase B, spec §5.5).

Most retrieval tools already publish frame *refs* (timestamp/source/hypothesis)
via tools._public_frame_refs, so ToolMessage payloads rarely carry image bytes.
This module is a defensive net: it strips any image base64 that slips through
while preserving all surrounding metadata, and guards against any other oversized
blob that would blow up sequence length.

answer_with_evidence / verify_grounding results are preserved verbatim — they
carry no images and their full text is the teacher's reasoning we want to keep.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Keys whose value is image bytes we replace with a metadata-only placeholder.
IMAGE_KEYS = {"image_b64", "b64", "frame_b64", "thumbnail_b64", "image_bytes"}
# Tool results to keep fully intact.
KEEP_FULL_TOOLS = {"answer_with_evidence", "verify_grounding"}

_B64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_BLOB_LIMIT = 4000  # any base64-looking string longer than this is omitted


def _placeholder(parent: dict[str, Any]) -> str:
    timestamp = parent.get("timestamp")
    scene = parent.get("scene_id", parent.get("scene"))
    return f"<frame_b64_omitted: t={timestamp}, scene={scene}>"


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        scrubbed: dict[str, Any] = {}
        for key, value in obj.items():
            if key in IMAGE_KEYS and isinstance(value, str):
                scrubbed[key] = _placeholder(obj)
            else:
                scrubbed[key] = _scrub(value)
        return scrubbed
    if isinstance(obj, list):
        return [_scrub(item) for item in obj]
    if isinstance(obj, str) and len(obj) > _BLOB_LIMIT and _B64_RE.match(obj):
        return f"<omitted_b64: len={len(obj)}>"
    return obj


def truncate_tool_result(name: str | None, content: str) -> str:
    """Return ``content`` with any image base64 replaced by metadata placeholders.

    Non-JSON content and KEEP_FULL_TOOLS results are returned unchanged. Output
    is always valid JSON when the input was valid JSON.
    """
    if name in KEEP_FULL_TOOLS:
        return content
    try:
        data = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return content
    # answer_with_evidence / verify_grounding may also be identified by payload.
    if isinstance(data, dict) and data.get("tool") in KEEP_FULL_TOOLS:
        return content
    return json.dumps(_scrub(data), ensure_ascii=False)


def truncate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Truncate a serialized tool message dict in place-safe fashion."""
    if record.get("role") != "tool":
        return record
    name = record.get("name")
    if not name:
        try:
            payload = json.loads(str(record.get("content", "")))
            if isinstance(payload, dict):
                name = payload.get("tool")
        except (TypeError, ValueError, json.JSONDecodeError):
            name = None
    new = dict(record)
    new["content"] = truncate_tool_result(name, str(record.get("content", "")))
    return new
