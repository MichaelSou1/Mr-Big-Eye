"""Batch ingest of the 50 Video-MME videos into our preprocess pipeline.

Reads eval/audiovisual/video_manifest.json (built by build_videomme_eval.py),
runs preprocess_video() for each entry that isn't already in 'done' state.

Run via:
    /home/user/miniconda3/envs/mbe-phase2/bin/python -m scripts.ingest_videomme

GPU-heavy: ~5 min/video × 50 ≈ 4h end-to-end.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/home/user/Mr-Big-Eye")
sys.path.insert(0, str(ROOT))

from app.cache import get_video_status  # noqa: E402
from app.preprocess import preprocess_video  # noqa: E402

UPLOADS = ROOT / "data" / "uploads"
MANIFEST_PATH = ROOT / "eval" / "audiovisual" / "video_manifest.json"


async def ingest_one(video_id: str, source: Path) -> tuple[str, str, float]:
    """Copy source to uploads/{video_id}.mp4 then preprocess. Returns (video_id, status, elapsed)."""
    if get_video_status(video_id) == "done":
        return video_id, "already-done", 0.0
    target = UPLOADS / f"{video_id}.mp4"
    if not target.exists():
        shutil.copy2(source, target)
    t = time.time()
    try:
        await preprocess_video(video_id, target, progress_callback=None)
        return video_id, "done", time.time() - t
    except Exception as exc:  # noqa: BLE001
        return video_id, f"FAILED: {type(exc).__name__}: {exc}", time.time() - t


async def main() -> int:
    with MANIFEST_PATH.open() as f:
        manifest: dict[str, dict] = json.load(f)
    entries = list(manifest.values())
    print(f"ingesting {len(entries)} videos…", flush=True)
    for i, entry in enumerate(entries, 1):
        vid = entry["video_id"]
        src = Path(entry["source_path"])
        vid_short, status, dt = await ingest_one(vid, src)
        print(f"[{i:2d}/{len(entries)}] {vid_short}  {status}  ({dt:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
