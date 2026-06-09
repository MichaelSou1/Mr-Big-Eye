#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.vqa import LocalVLMBackbone, VLMAPIError


def _demo_frames() -> tuple[list[Image.Image], list[float]]:
    frames: list[Image.Image] = []
    for idx, color in enumerate(("white", "lightblue")):
        image = Image.new("RGB", (320, 220), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((40 + idx * 40, 60, 140 + idx * 40, 160), fill="red")
        draw.text((20, 15), f"demo frame {idx + 1}", fill="black")
        frames.append(image)
    return frames, [0.0, 1.0]


async def main_async() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Smoke-test an existing local OpenAI-compatible VLM endpoint."
    )
    parser.add_argument("--question", default="What color is the square in these frames?")
    args = parser.parse_args()

    if not settings.local_vlm_base_url or not settings.local_vlm_model_name:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "LOCAL_VLM_BASE_URL and LOCAL_VLM_MODEL_NAME are required.",
                    "base_url": settings.local_vlm_base_url,
                    "model": settings.local_vlm_model_name,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    backbone = LocalVLMBackbone()
    frames, timestamps = _demo_frames()
    try:
        caption = await backbone.generate_caption(frames[0])
        answer = await backbone.answer_question(args.question, frames, timestamps)
    except (VLMAPIError, Exception) as exc:  # noqa: BLE001 - smoke tool reports shape/errors.
        print(
            json.dumps(
                {
                    "ok": False,
                    "base_url": settings.local_vlm_base_url,
                    "model": settings.local_vlm_model_name,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "base_url": settings.local_vlm_base_url,
                "model": settings.local_vlm_model_name,
                "caption": caption,
                "answer": answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
