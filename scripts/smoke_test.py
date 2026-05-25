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


def _vlm_api_reachable() -> tuple[bool, str]:
    """Best-effort health check: send a HEAD/GET to the API base URL.

    We don't have a portable /v1/models endpoint across providers (Doubao
    Responses API doesn't expose it the same way), so we just confirm the
    host resolves and accepts a request.
    """
    url = settings.vlm_api_base_url.rstrip("/")
    headers = {}
    if settings.vlm_api_key:
        headers["Authorization"] = f"Bearer {settings.vlm_api_key}"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers=headers)
        return True, f"HTTP {resp.status_code}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="tests/fixtures/short_clip.mp4")
    parser.add_argument("--question", default="What is happening in the video?")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--skip-vlm-check",
        action="store_true",
        help="Skip the VLM API reachability check.",
    )
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{settings.app_port}"
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Missing video fixture: {video_path}", file=sys.stderr)
        return 2

    if not args.skip_vlm_check:
        ok, detail = _vlm_api_reachable()
        if not ok:
            print(f"VLM API is not reachable at {settings.vlm_api_base_url}: {detail}", file=sys.stderr)
            return 3
        print(f"vlm_api_base_url={settings.vlm_api_base_url} reachable ({detail})")

    with httpx.Client(timeout=60) as client:
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
