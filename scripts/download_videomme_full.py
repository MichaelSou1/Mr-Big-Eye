"""Full download + extract of the Video-MME video corpus from ModelScope.

Video-MME ships its 900 videos as 20 independent zip archives
(videos_chunked_01.zip … _20.zip, ~101 GB total) plus the QA parquet and
subtitles. Unlike the old per-entry HTTP range approach, ModelScope serves the
whole dataset; we pull everything, then extract each chunk into a flat
{videoID}.mp4 layout under data/videomme_videos/.

Resumable: ModelScope skips already-downloaded blobs; extraction skips mp4s
that already exist with the expected size.

    python scripts/download_videomme_full.py [--keep-zips]

Honors MBE_ROOT (defaults to the repo root).
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MODELSCOPE_DOMAIN", "www.modelscope.cn")

ROOT = Path(os.environ.get("MBE_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "data" / "videomme_src"
VIDEOS = ROOT / "data" / "videomme_videos"
DATASET = "lmms-lab/Video-MME"


def download() -> None:
    from modelscope import dataset_snapshot_download

    SRC.mkdir(parents=True, exist_ok=True)
    print(f"downloading dataset {DATASET} → {SRC} (resumable) …", flush=True)
    dataset_snapshot_download(DATASET, local_dir=str(SRC))
    print("download complete.", flush=True)


def extract(keep_zips: bool) -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    zips = sorted(SRC.glob("videos_chunked_*.zip"))
    if not zips:
        print(f"WARNING: no videos_chunked_*.zip under {SRC}", flush=True)
        return
    extracted = 0
    skipped = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".mp4"):
                    continue
                name = Path(info.filename).name  # flatten any internal dirs
                target = VIDEOS / name
                if target.exists() and target.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                with z.open(info) as srcf, target.open("wb") as dstf:
                    while chunk := srcf.read(1 << 20):
                        dstf.write(chunk)
                extracted += 1
        print(
            f"  {zp.name}: extracted={extracted} skipped={skipped}",
            flush=True,
        )
        if not keep_zips:
            zp.unlink()
            print(f"  removed {zp.name} (freed ~{zp.stat().st_size if zp.exists() else 0})", flush=True)
    print(f"\nextract done: {extracted} new, {skipped} already present.", flush=True)
    print(f"videos in {VIDEOS}: {len(list(VIDEOS.glob('*.mp4')))}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-zips",
        action="store_true",
        help="keep the chunk zips after extraction (default: delete to save disk)",
    )
    ap.add_argument("--skip-download", action="store_true", help="extract only")
    args = ap.parse_args()

    if not args.skip_download:
        download()
    extract(keep_zips=args.keep_zips)
    return 0


if __name__ == "__main__":
    sys.exit(main())
