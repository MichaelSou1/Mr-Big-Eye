"""Batch ingest of audiovisual manifest videos into our preprocess pipeline.

Reads eval/audiovisual/video_manifest.json (built by build_videomme_eval.py +
build_supplementary_eval.py), runs preprocess_video() for each entry that
isn't already in 'done' state.

Path resolution: defaults to /home/user/Mr-Big-Eye (host workstation), but
honors MBE_ROOT env var so the same script works inside the Docker image
(WORKDIR /app). Run via:

    python -m scripts.ingest_videomme

Sharding for multi-GPU parallelism: set MBE_INGEST_SHARD=N/M and the script
will only ingest entries whose stable order index satisfies (idx % M == N).
Launch one container per GPU with a distinct N to fan out across cards.

GPU-heavy: ~5 min/video × 60 ≈ 5h end-to-end on a single 3080-class GPU.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("MBE_ROOT", "/home/user/Mr-Big-Eye"))
sys.path.insert(0, str(ROOT))

# Pre-init CUDA on the main thread. preprocess offloads ASR / OCR / encode
# work to a ThreadPoolExecutor, and torch._lazy_init() can SIGSEGV when first
# invoked from a worker thread on certain torch/cuDNN combinations (observed
# with torch 2.5+cu121 inside this image). A single allocation here forces
# initialization before any worker is spawned.
try:
    import torch  # noqa: E402

    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda:0")
        del _
        torch.cuda.synchronize()
except Exception as _exc:  # noqa: BLE001
    print(f"WARNING: CUDA pre-init skipped: {_exc}", flush=True)

from app.cache import get_video_status  # noqa: E402
from app.preprocess import preprocess_video  # noqa: E402

UPLOADS = ROOT / "data" / "uploads"
MANIFEST_PATH = ROOT / "eval" / "audiovisual" / "video_manifest.json"


def parse_shard() -> tuple[int, int]:
    raw = os.environ.get("MBE_INGEST_SHARD", "0/1").strip()
    try:
        n_str, m_str = raw.split("/", 1)
        n, m = int(n_str), int(m_str)
        if m <= 0 or not (0 <= n < m):
            raise ValueError
    except (ValueError, AttributeError):
        print(f"WARNING: bad MBE_INGEST_SHARD={raw!r}, defaulting to 0/1", flush=True)
        return 0, 1
    return n, m


async def ingest_one(video_id: str, source: Path) -> tuple[str, str, float]:
    """Copy source to uploads/{video_id}.mp4 then preprocess. Returns (video_id, status, elapsed)."""
    if get_video_status(video_id) == "done":
        return video_id, "already-done", 0.0
    target = UPLOADS / f"{video_id}.mp4"
    if not target.exists():
        if not source.exists():
            return video_id, f"FAILED: source missing ({source})", 0.0
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
    entries = sorted(manifest.items(), key=lambda kv: kv[0])
    n, m = parse_shard()
    if m > 1:
        entries = [(k, v) for i, (k, v) in enumerate(entries) if i % m == n]
        print(f"shard {n}/{m}: ingesting {len(entries)} of {len(manifest)} videos", flush=True)
    else:
        print(f"ingesting {len(entries)} videos…", flush=True)
    for i, (_key, entry) in enumerate(entries, 1):
        vid = entry["video_id"]
        src = Path(entry["source_path"])
        vid_short, status, dt = await ingest_one(vid, src)
        print(f"[{i:2d}/{len(entries)}] {vid_short}  {status}  ({dt:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
