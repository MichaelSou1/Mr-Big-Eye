"""Pre-stage every video referenced by video_manifest.json into data/uploads/.

Why: ingest_videomme.py copies source -> uploads/ on first encounter. That copy
requires the original `source_path` to be accessible inside the container.
Doing the staging on the host *before* shipping data/ to a remote server means
the container only needs uploads/ mounted — no need to drag along
data/videomme_videos/ / data/cinepile_videos/ / etc.

Run once on the host:
    python -m scripts.stage_uploads
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MBE_ROOT", "/home/user/Mr-Big-Eye"))
MANIFEST_PATH = ROOT / "eval" / "audiovisual" / "video_manifest.json"
UPLOADS = ROOT / "data" / "uploads"


def main() -> int:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open() as f:
        manifest: dict[str, dict] = json.load(f)

    staged = skipped = missing = 0
    for key, entry in manifest.items():
        vid = entry["video_id"]
        src = Path(entry["source_path"])
        dst = UPLOADS / f"{vid}.mp4"
        if dst.exists():
            skipped += 1
            continue
        if not src.exists():
            print(f"  MISSING source for {key}: {src}", flush=True)
            missing += 1
            continue
        shutil.copy2(src, dst)
        staged += 1
        print(f"  staged {key} -> {dst.name}", flush=True)

    print(
        f"\nstaged={staged} already_present={skipped} missing={missing} "
        f"total_uploads={len(list(UPLOADS.glob('*.mp4')))}",
        flush=True,
    )
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
