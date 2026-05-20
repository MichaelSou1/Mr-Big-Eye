#!/usr/bin/env python
"""Download Phase 1 models via ModelScope for faster mainland China access."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from modelscope import snapshot_download
from modelscope.hub.api import HubApi

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("download_models")


def _download(
    model_id: str,
    target: Path,
    ignore_patterns: list[str] | None = None,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", model_id, target)
    snapshot_download(
        model_id=model_id,
        local_dir=str(target),
        ignore_file_pattern=ignore_patterns,
    )


def _verify(model_id: str) -> None:
    HubApi().get_model(model_id)
    logger.info("Verified ModelScope model: %s", model_id)


def _targets() -> dict[str, tuple[str, Path, list[str] | None]]:
    return {
        "vlm": (settings.vlm_model_name, settings.vlm_model_local_dir, None),
        "bge": (
            settings.bge_m3_model,
            settings.bge_m3_local_dir,
            ["onnx/*", "imgs/*", "*.jpg", "*.webp", "README.md"],
        ),
        "siglip": (settings.siglip2_modelscope_model, settings.siglip2_local_dir, None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("all", "vlm", "bge", "siglip"),
        default="all",
        help="Download one model family or all Phase 1 models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify ModelScope model IDs without downloading weights.",
    )
    args = parser.parse_args()

    targets = _targets()
    selected = targets if args.only == "all" else {args.only: targets[args.only]}
    for model_id, target, ignore_patterns in selected.values():
        if args.dry_run:
            _verify(model_id)
        else:
            _download(model_id, target, ignore_patterns)


if __name__ == "__main__":
    main()
