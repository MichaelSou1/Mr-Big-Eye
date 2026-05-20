import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import settings

_STATUS: dict[str, str] = {}


def video_id_from_file(path: Path) -> str:
    """SHA256 of file bytes, return first 16 hex chars."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def video_cache_dir(video_id: str) -> Path:
    """Return cache directory without creating it."""
    return settings.data_dir / "cache" / video_id


def ensure_cache_dirs(video_id: str) -> Path:
    """Create root cache dir and all Phase 1 artifact subdirs."""
    root = video_cache_dir(video_id)
    for child in ("frames_scene", "frames_dense", "caption_index", "frame_index"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def get_video_status(video_id: str) -> str:
    """Return absent, running, done, or failed:<msg>."""
    status = _STATUS.get(video_id)
    if status:
        return status
    if (video_cache_dir(video_id) / ".done").exists():
        return "done"
    return "absent"


def set_video_status(video_id: str, status: str) -> None:
    """Set in-memory status and write marker for completed preprocessing."""
    _STATUS[video_id] = status
    if status == "done":
        cache_dir = ensure_cache_dirs(video_id)
        (cache_dir / ".done").touch()


def load_meta(video_id: str) -> dict[str, Any]:
    """Load meta.json for a preprocessed video."""
    with (video_cache_dir(video_id) / "meta.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_meta(video_id: str, meta: dict[str, Any]) -> None:
    """Write meta.json for a video."""
    cache_dir = ensure_cache_dirs(video_id)
    with (cache_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
