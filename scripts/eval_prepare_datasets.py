#!/usr/bin/env python
"""Chained download -> convert -> cleanup pipeline for eval datasets.

Disk-conscious: each dataset's raw annotations are removed once the sampled
EvalCase JSONL + manifest are written, BEFORE the next dataset's download starts.

Run once after cloning the repo (and again whenever you want to refresh the
sampled fixture). Videos are NOT downloaded — see the printed instructions at
the end for video acquisition.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.eval_datasets import (
    LONGVIDEOBENCH_REPO,
    cleanup_hf_cache,
    cleanup_raw,
    download_longvideobench,
    download_nextgqa,
)


SUPPORTED = ("longvideobench", "nextgqa")
RAW_ROOT = ROOT_DIR / "data" / "eval" / "_raw"

VIDEO_INSTRUCTIONS = """
=== Video acquisition (not auto-downloaded) ===

LongVideoBench videos:
  The dataset only ships videos as 8-part tar (videos.tar.part.{{aa..ah}}).
  Concat + extract is the only path; selective extraction can save disk.

    huggingface-cli download longvideobench/LongVideoBench \\
        --repo-type dataset \\
        --include 'videos.tar.part.*' \\
        --local-dir /scratch/lvb_tar \\
        --endpoint https://hf-mirror.com
    cat /scratch/lvb_tar/videos.tar.part.* > /scratch/lvb_videos.tar
    # selective extract for the cases in manifest.json:
    jq -r '.[].video_path' data/eval/datasets/longvideobench/manifest.json > /tmp/lvb_keep.txt
    tar -xf /scratch/lvb_videos.tar -C /scratch/lvb_videos --files-from /tmp/lvb_keep.txt
    rm /scratch/lvb_videos.tar  # reclaim disk
  Then:
    python scripts/eval_ingest_videos.py \\
      --videos-dir /scratch/lvb_videos \\
      --manifest data/eval/datasets/longvideobench/manifest.json \\
      --cases tests/fixtures/eval_cases_longvideobench.jsonl

NExT-GQA videos:
  Use the Google Drive link from https://github.com/doc-doc/NExT-QA README
  (typically the NExTVideo zip). Use gdown:
    pip install gdown
    gdown --fuzzy 'https://drive.google.com/file/d/<FILE_ID>/view'
    unzip NExTVideo.zip -d /scratch/nextqa_videos
  Then:
    python scripts/eval_ingest_videos.py \\
      --videos-dir /scratch/nextqa_videos \\
      --manifest data/eval/datasets/nextgqa/manifest.json \\
      --cases tests/fixtures/eval_cases_nextgqa.jsonl
"""


def run_subprocess(args: list[str]) -> int:
    return subprocess.run(args, check=False).returncode


def prepare_longvideobench(*, sample: int, seed: int, keep_raw: bool) -> bool:
    raw_dir = RAW_ROOT / "longvideobench"
    try:
        files = download_longvideobench(raw_dir)
    except Exception as exc:
        logging.error("LongVideoBench download failed: %s", exc)
        if not keep_raw:
            cleanup_raw(raw_dir)
        return False
    src = files.get("lvb_val.json")
    if src is None or not src.exists():
        logging.error("LongVideoBench: lvb_val.json missing after download.")
        if not keep_raw:
            cleanup_raw(raw_dir)
        return False
    rc = run_subprocess(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "eval_convert_longvideobench.py"),
            "--source", str(src),
            "--sample", str(sample),
            "--seed", str(seed),
        ]
    )
    if rc != 0:
        logging.error("LongVideoBench converter exited %s", rc)
        if not keep_raw:
            cleanup_raw(raw_dir)
        return False
    if not keep_raw:
        cleanup_raw(raw_dir)
        cleanup_hf_cache(LONGVIDEOBENCH_REPO)
    return True


def prepare_nextgqa(*, sample: int, seed: int, keep_raw: bool) -> bool:
    raw_dir = RAW_ROOT / "nextgqa"
    try:
        download_nextgqa(raw_dir)
    except Exception as exc:
        logging.error("NExT-GQA download failed: %s", exc)
        if not keep_raw:
            cleanup_raw(raw_dir)
        return False
    rc = run_subprocess(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "eval_convert_nextgqa.py"),
            "--source-dir", str(raw_dir),
            "--sample", str(sample),
            "--seed", str(seed),
        ]
    )
    if rc != 0:
        logging.error("NExT-GQA converter exited %s", rc)
        if not keep_raw:
            cleanup_raw(raw_dir)
        return False
    if not keep_raw:
        cleanup_raw(raw_dir)
    return True


def main() -> int:
    load_dotenv(ROOT_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download + sample + cleanup eval datasets.")
    parser.add_argument(
        "--datasets",
        default=",".join(SUPPORTED),
        help=f"Comma-separated subset of {SUPPORTED}.",
    )
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Skip cleanup of data/eval/_raw/<dataset>/ so you can re-run converters.",
    )
    args = parser.parse_args()

    requested = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in requested if name not in SUPPORTED]
    if unknown:
        print(f"ERROR: unknown dataset(s): {unknown}; supported: {SUPPORTED}", file=sys.stderr)
        return 2

    results: dict[str, bool] = {}
    for name in requested:
        if name == "longvideobench":
            results[name] = prepare_longvideobench(
                sample=args.sample, seed=args.seed, keep_raw=args.keep_raw
            )
        elif name == "nextgqa":
            results[name] = prepare_nextgqa(
                sample=args.sample, seed=args.seed, keep_raw=args.keep_raw
            )

    print()
    print("=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")
    print(VIDEO_INSTRUCTIONS)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
