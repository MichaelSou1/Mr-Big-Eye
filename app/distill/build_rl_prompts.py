from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.distill.common import read_json, write_jsonl


def rl_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": payload.get("case_id"),
        "video_id": payload.get("video_id"),
        "images": [path for path in payload.get("frame_paths", []) if path],
        "shown_frames": payload.get("shown_frames", []),
        "prompt": str(payload.get("question") or ""),
        "reference_answer": payload.get("answer"),
        "gold_timestamps": payload.get("gold_timestamps", []),
        "gold_scenes": payload.get("gold_scenes", []),
        "reward": {
            "answer_correctness": True,
            "marker_validity": True,
            "temporal_iou": {
                # Intentionally OFF: under the fixed-frame, no-tool setting the
                # cited [FRAME:t=] anchors are determined by the uniform sampler,
                # not a model action, so an implicit-claim IoU reward (VideoTemp-o3
                # Eq.5-6) is not load-bearing and is hackable. Keep GT here for
                # the §7.3c claimed-timestamp probe instead.
                "enabled": False,
                "sigma": 0.1,
                "lambda": 0.1,
                "reason": "implicit-timestamp IoU not load-bearing under fixed-frame no-tool setting",
            },
        },
        "train_modality": "frames_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RL prompt JSONL from filtered CoT artifacts.")
    parser.add_argument("--consistency-report", required=True)
    parser.add_argument("--output", default="data/distill/rl/rl_prompts_frames_only.jsonl")
    args = parser.parse_args()

    report = read_json(args.consistency_report)
    rows = [rl_row(read_json(path)) for path in report.get("kept_cot", [])]
    rows = [row for row in rows if row["images"]]
    write_jsonl(args.output, rows)
    print(json.dumps({"written": len(rows), "output": args.output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
