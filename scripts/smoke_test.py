#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="tests/fixtures/short_clip.mp4")
    parser.add_argument("--question", default="What is happening in the video?")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{settings.app_port}"
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Missing video fixture: {video_path}", file=sys.stderr)
        return 2

    with httpx.Client(timeout=60) as client:
        try:
            sglang = client.get(f"{settings.sglang_endpoint.rstrip('/')}/v1/models")
            sglang.raise_for_status()
        except Exception as exc:
            print(f"SGLang is not reachable: {exc}", file=sys.stderr)
            return 3

        with video_path.open("rb") as handle:
            upload = client.post(
                f"{base_url}/upload",
                files={"file": (video_path.name, handle, "video/mp4")},
            )
        upload.raise_for_status()
        payload = upload.json()
        video_id = payload["video_id"]
        print(f"video_id={video_id} status={payload['status']}")

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            status = client.get(f"{base_url}/status/{video_id}")
            status.raise_for_status()
            current = status.json()["status"]
            print(f"status={current}")
            if current == "done":
                break
            if current.startswith("failed:"):
                print(current, file=sys.stderr)
                return 4
            time.sleep(2)
        else:
            print("Timed out waiting for preprocessing", file=sys.stderr)
            return 5

        chat = client.post(
            f"{base_url}/chat",
            json={"video_id": video_id, "question": args.question, "history": []},
            timeout=180,
        )
        chat.raise_for_status()
        result = chat.json()
        print(result["answer"])
        print(f"frames={len(result['frames'])}")
        return 0 if result["answer"] and result["frames"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
