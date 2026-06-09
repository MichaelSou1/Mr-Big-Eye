from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from app.distill.common import read_json, stable_hash, write_json


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def cot_dedup_key(payload: dict[str, Any]) -> str:
    return stable_hash(
        {
            "case_id": payload.get("case_id"),
            "video_id": payload.get("video_id"),
            "source_traj_hash": payload.get("source_traj_hash"),
            "shown_frames": [round(float(item), 1) for item in payload.get("shown_frames", [])],
        }
    )


def cot_near_key(payload: dict[str, Any]) -> str:
    return stable_hash(
        {
            "video_id": payload.get("video_id"),
            "question": normalized_text(str(payload.get("question") or "")),
            "answer": normalized_text(str(payload.get("answer") or "")),
            "shown_frames": [round(float(item), 1) for item in payload.get("shown_frames", [])],
        }
    )


def dedup_cot_paths(paths: list[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    exact_seen: set[str] = set()
    near_seen: set[str] = set()
    kept: list[Path] = []
    dropped: list[dict[str, Any]] = []
    for path in sorted(paths):
        payload = read_json(path)
        exact = cot_dedup_key(payload)
        near = cot_near_key(payload)
        reason = ""
        if exact in exact_seen:
            reason = "exact_duplicate"
        elif near in near_seen:
            reason = "near_duplicate"
        if reason:
            dropped.append({"path": str(path), "case_id": payload.get("case_id"), "reason": reason})
            continue
        exact_seen.add(exact)
        near_seen.add(near)
        kept.append(path)
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate rewritten CoT JSON files.")
    parser.add_argument("--cot-dir", default="data/distill/cot")
    parser.add_argument("--output", default="data/distill/dedup_report.json")
    args = parser.parse_args()

    kept, dropped = dedup_cot_paths(list(Path(args.cot_dir).glob("*.json")))
    report = {
        "summary": {
            "total": len(kept) + len(dropped),
            "kept": len(kept),
            "dropped": len(dropped),
        },
        "kept": [str(path) for path in kept],
        "dropped": dropped,
    }
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
