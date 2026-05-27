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
    for child in (
        "frames_scene",
        "frames_dense",
        "caption_index",
        "frame_index",
        "transcript_index",
        "slide_index",
        "frames_slides",
    ):
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


def compute_video_profile(video_id: str) -> dict[str, Any]:
    """Derive modality profile from ingest artifacts.

    Reads meta.json (and best-effort slides.jsonl for OCR char density) to
    produce per-minute densities and a coarse profile label used by the
    orchestrator to bias retrieval. Returns an empty dict on any failure so
    callers can treat absence as "no profile guidance".
    """
    try:
        meta = load_meta(video_id)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    duration = float(meta.get("duration") or 0.0)
    if duration < 1.0:
        return {}
    minutes = duration / 60.0
    slide_count = int(meta.get("slide_count") or 0)
    scene_count = int(meta.get("scene_count") or 0)
    tx_seg_count = int(meta.get("transcript_segment_count") or 0)
    has_transcript = bool(meta.get("has_transcript"))
    has_slides = bool(meta.get("has_slides")) and slide_count > 0

    ocr_chars = 0
    if has_slides:
        slides_path = video_cache_dir(video_id) / "slides.jsonl"
        if slides_path.exists():
            try:
                with slides_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        text = rec.get("text") or ""
                        ocr_chars += len(text)
            except (json.JSONDecodeError, OSError):
                ocr_chars = 0

    slide_density = slide_count / minutes
    scene_density = scene_count / minutes
    tx_seg_density = tx_seg_count / minutes
    ocr_density = ocr_chars / minutes

    # Coarse classification. Thresholds chosen from sampled cache stats:
    # slide_heavy lectures sit at 3-12 slides/min, vlogs/podcasts at 0,
    # action/sports content tends to scene_density >= 10/min with low text.
    if slide_density >= 2.0 and ocr_density >= 100.0:
        profile = "slide_heavy"
    elif has_transcript and slide_density < 1.0 and tx_seg_density >= 5.0:
        profile = "speech_heavy"
    elif slide_density < 0.5 and (not has_transcript or tx_seg_density < 2.0) and scene_density >= 3.0:
        profile = "visual_dynamic"
    else:
        profile = "mixed"

    return {
        "profile": profile,
        "duration_sec": round(duration, 1),
        "slide_density_per_min": round(slide_density, 2),
        "scene_density_per_min": round(scene_density, 2),
        "transcript_segment_density_per_min": round(tx_seg_density, 2),
        "ocr_chars_per_min": round(ocr_density, 1),
        "has_transcript": has_transcript,
        "has_slides": has_slides,
    }
