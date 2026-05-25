#!/usr/bin/env python
"""Bind videos in --videos-dir to EvalCase entries via SHA256-based video_id.

For each manifest entry:
  1. Resolve the expected video file under --videos-dir (LVB uses `video_path`,
     NExT-GQA uses `vidor_path`).
  2. Compute video_id via app.cache.video_id_from_file.
  3. If get_video_status(video_id) != "done", call app.preprocess.preprocess_video.
  4. Rewrite the cases JSONL in place, replacing the empty video_id with the
     computed one. Drop cases whose video file is missing, and audit them.

Idempotent — re-run anytime.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from app.cache import get_video_status, video_id_from_file
from app.preprocess import preprocess_video


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


def write_cases_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def resolve_video(entry: dict[str, Any], videos_dir: Path) -> Path | None:
    candidates: list[str] = []
    for key in ("video_path", "vidor_path", "filename"):
        value = entry.get(key)
        if value:
            candidates.append(str(value))
    # Try bare basename + common video extensions as fallbacks.
    # NExT-GQA's map_vid_vidorID.json stores vidor_path like "1015/4336654741"
    # without an extension; the actual files are "4336654741.mp4".
    expanded: list[str] = []
    for value in candidates:
        expanded.append(value)
        expanded.append(Path(value).name)
        if "." not in Path(value).name:
            for ext in (".mp4", ".mkv", ".webm", ".mov"):
                expanded.append(value + ext)
                expanded.append(Path(value).name + ext)
    seen: set[str] = set()
    ordered = [c for c in expanded if not (c in seen or seen.add(c))]
    for candidate in ordered:
        path = videos_dir / candidate
        if path.exists() and path.is_file():
            return path
    # As a last resort, glob by basename anywhere under videos_dir.
    for candidate in ordered:
        matches = list(videos_dir.rglob(Path(candidate).name))
        if matches:
            return matches[0]
    return None


async def main_async() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest dataset videos + bind video_ids.")
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Only compute video_id and rewrite cases JSONL; don't run preprocessing. "
             "Use when videos are already preprocessed under data/cache/.",
    )
    parser.add_argument(
        "--audit",
        default=str(ROOT_DIR / "data" / "eval" / "ingest_skipped.jsonl"),
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).resolve()
    manifest_path = Path(args.manifest)
    cases_path = Path(args.cases)
    audit_path = Path(args.audit)

    if not videos_dir.exists():
        print(f"ERROR: videos-dir does not exist: {videos_dir}", file=sys.stderr)
        return 2
    manifest = load_manifest(manifest_path)
    cases = load_cases_jsonl(cases_path)
    cases_by_id = {c["case_id"]: c for c in cases}

    skipped: list[dict[str, Any]] = []
    processed = 0
    rebound = 0
    for entry in manifest:
        case_id = entry["case_id"]
        case = cases_by_id.get(case_id)
        if case is None:
            skipped.append({"case_id": case_id, "reason": "no matching case in JSONL"})
            continue
        video_file = resolve_video(entry, videos_dir)
        if video_file is None:
            skipped.append({"case_id": case_id, "reason": "video file not found", "manifest": entry})
            continue
        try:
            video_id = video_id_from_file(video_file)
        except Exception as exc:
            skipped.append({"case_id": case_id, "reason": f"hash failed: {exc}"})
            continue
        case["video_id"] = video_id
        rebound += 1
        status = get_video_status(video_id)
        if status == "done":
            logging.info("[%s] cache hit %s", case_id, video_id)
            continue
        if args.skip_preprocess:
            logging.info("[%s] skip preprocess (status=%s)", case_id, status)
            continue
        logging.info("[%s] preprocessing %s -> %s", case_id, video_file, video_id)
        try:
            await preprocess_video(video_id, video_file)
            processed += 1
        except Exception as exc:
            logging.exception("preprocess failed for %s", case_id)
            skipped.append({"case_id": case_id, "reason": f"preprocess failed: {exc}"})

    # Keep cases whose video_id is non-empty. Cases that bound a video_id but
    # had a transient preprocess failure stay in the JSONL so a re-run can pick
    # them up without regenerating the dataset.
    kept_cases = [c for c in cases if c.get("video_id")]
    write_cases_jsonl(cases_path, kept_cases)

    if skipped:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as handle:
            for entry in skipped:
                handle.write(json.dumps(entry, ensure_ascii=False))
                handle.write("\n")

    print(
        f"ingest: rebound={rebound}, preprocessed={processed}, "
        f"skipped={len(skipped)} (audit: {audit_path if skipped else '-'})"
    )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
