"""ffprobe every extracted Video-MME mp4 to get real per-video seconds.

Writes data/videomme_src/durations.json = {videoID: seconds}, consumed by
sample_videomme.py --durations-file to pick the SHORTEST videos (cheap ingest).

    python scripts/probe_durations.py

Honors MBE_ROOT (defaults to the repo root).
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("MBE_ROOT", Path(__file__).resolve().parents[1]))
VIDEOS = ROOT / "data" / "videomme_videos"
OUT = ROOT / "data" / "videomme_src" / "durations.json"


def probe(mp4: Path) -> tuple[str, float | None]:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(mp4),
            ],
            capture_output=True, text=True, timeout=60,
        )
        return mp4.stem, float(out.stdout.strip())
    except Exception:
        return mp4.stem, None


def main() -> int:
    mp4s = sorted(VIDEOS.glob("*.mp4"))
    print(f"probing {len(mp4s)} videos in {VIDEOS} …", flush=True)
    durations: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for vid, secs in ex.map(probe, mp4s):
            if secs is not None:
                durations[vid] = round(secs, 1)
    OUT.write_text(json.dumps(durations, indent=2))
    if durations:
        vals = sorted(durations.values())
        print(
            f"wrote {OUT}: {len(durations)} probed | "
            f"min={vals[0]:.0f}s p50={vals[len(vals)//2]:.0f}s max={vals[-1]:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
