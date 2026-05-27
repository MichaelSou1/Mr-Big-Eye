"""Selectively extract sampled Video-MME videos via per-entry HTTP range GETs.

Approach:
1. For each zip chunk, do ONE remote open to read central directory.
2. For each needed entry: get local-file-header offset + compressed_size from CD.
3. Issue a single HTTP Range GET for the entry's bytes.
4. Parse local file header, decompress (DEFLATE) or pass through (STORED), write to disk.

Avoids fsspec's small-block range overhead which limits throughput to ~0.4 MB/s.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import HfFileSystem  # noqa: E402

CACHE = Path("/home/user/Mr-Big-Eye/data/hf_cache/videomme")
OUT_DIR = Path("/home/user/Mr-Big-Eye/data/videomme_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_ENDPOINT = os.environ["HF_ENDPOINT"]
REPO = "lmms-lab/Video-MME"


def chunk_url(idx: int) -> str:
    return f"{HF_ENDPOINT}/datasets/{REPO}/resolve/main/videos_chunked_{idx:02d}.zip"


def fetch_entry(session: requests.Session, url: str, info: zipfile.ZipInfo) -> bytes:
    """Range-GET the compressed bytes for a zip entry and decompress."""
    start = info.header_offset
    slab_size = 1024 + info.compress_size
    end = start + slab_size - 1
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            r = session.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 300),
            )
            r.raise_for_status()
            buf = r.content
            break
        except (requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError) as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
            continue
    else:
        raise RuntimeError(f"fetch_entry failed after 5 retries: {last_exc}")
    # Parse local file header
    if buf[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad LFH signature for {info.filename}")
    fname_len = struct.unpack("<H", buf[26:28])[0]
    extra_len = struct.unpack("<H", buf[28:30])[0]
    data_start = 30 + fname_len + extra_len
    data_end = data_start + info.compress_size
    if data_end > len(buf):
        # rare: header was bigger than padding. Fetch the missing tail.
        miss_start = start + len(buf)
        miss_end = start + data_end - 1
        r2 = session.get(url, headers={"Range": f"bytes={miss_start}-{miss_end}"}, timeout=120)
        r2.raise_for_status()
        buf += r2.content
    compressed = buf[data_start:data_end]
    if info.compress_type == zipfile.ZIP_STORED:
        return compressed
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(compressed, -zlib.MAX_WBITS)
    raise RuntimeError(f"unsupported compress_type {info.compress_type}")


def extract_one(args: tuple[str, zipfile.ZipInfo, str]) -> tuple[str, float, float]:
    vid, info, url = args
    target = OUT_DIR / f"{vid}.mp4"
    if target.exists() and target.stat().st_size == info.file_size:
        return vid, 0.0, info.file_size / 1e6
    t = time.time()
    with requests.Session() as sess:
        data = fetch_entry(sess, url, info)
    if len(data) != info.file_size:
        raise RuntimeError(
            f"{vid}: decompressed size {len(data)} != expected {info.file_size}"
        )
    target.write_bytes(data)
    dt = time.time() - t
    return vid, dt, info.file_size / 1e6


def main() -> int:
    with (CACHE / "chunk_index.json").open() as f:
        index: dict[str, dict] = json.load(f)

    by_chunk: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for vid, info in index.items():
        by_chunk[info["chunk"]].append((vid, info))

    fs = HfFileSystem(endpoint=HF_ENDPOINT)
    tasks: list[tuple[str, zipfile.ZipInfo, str]] = []

    print(f"reading central directories for {len(by_chunk)} chunks…", flush=True)
    for chunk in sorted(by_chunk):
        path = f"datasets/{REPO}/videos_chunked_{chunk:02d}.zip"
        needed_vids = {vid for vid, _ in by_chunk[chunk]}
        with fs.open(path, "rb") as f:
            z = zipfile.ZipFile(f)
            url = chunk_url(chunk)
            hits = 0
            for info in z.infolist():
                vid = info.filename.split("/")[-1].replace(".mp4", "")
                if vid in needed_vids:
                    tasks.append((vid, info, url))
                    hits += 1
        print(f"  chunk_{chunk:02d}: {hits} entries queued", flush=True)

    print(f"\nfetching {len(tasks)} entries in parallel…", flush=True)
    t0 = time.time()
    total_mb = 0.0
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(extract_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            vid, dt, mb = fut.result()
            total_mb += mb
            done += 1
            speed = mb / dt if dt > 0 else 0
            print(
                f"  + {vid}.mp4 {mb:6.1f}MB in {dt:5.1f}s ({speed:5.1f}MB/s) "
                f"[{done}/{len(tasks)}]",
                flush=True,
            )
    elapsed = time.time() - t0
    print(
        f"\nTOTAL: {done} files, {total_mb/1024:.2f}GB in {elapsed:.1f}s "
        f"({total_mb/elapsed:.1f}MB/s avg)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
