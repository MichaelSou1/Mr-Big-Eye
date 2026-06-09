from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.distill.common import read_json, write_jsonl
from app.distill.dedup import dedup_cot_paths
from app.eval_harness import load_cases


def sft_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": payload.get("case_id"),
        "video_id": payload.get("video_id"),
        "images": [path for path in payload.get("frame_paths", []) if path],
        "shown_frames": payload.get("shown_frames", []),
        "messages": [
            {
                "role": "user",
                "content": str(payload.get("question") or ""),
            },
            {
                "role": "assistant",
                "content": (
                    f"{str(payload.get('cot') or '').strip()}\n\n"
                    f"Final answer: {str(payload.get('answer') or '').strip()}"
                ).strip(),
            },
        ],
        "cot": payload.get("cot"),
        "answer": payload.get("answer"),
        "gold_timestamps": payload.get("gold_timestamps", []),
        "gold_scenes": payload.get("gold_scenes", []),
        "source_traj_hash": payload.get("source_traj_hash"),
        "train_modality": "frames_only",
    }


def build_dataset(
    *,
    cot_paths: list[Path],
    output: Path,
    max_samples: int | None = None,
    eval_cases_path: Path | None = None,
) -> dict[str, Any]:
    kept, dropped = dedup_cot_paths(cot_paths)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in kept:
        payload = read_json(path)
        if not bool((payload.get("validation_report") or {}).get("ok")):
            skipped.append({"path": str(path), "reason": "validation_failed"})
            continue
        row = sft_row(payload)
        if not row["images"]:
            skipped.append({"path": str(path), "reason": "missing_frame_paths"})
            continue
        rows.append(row)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    if eval_cases_path is not None:
        train_video_ids = {str(row.get("video_id") or "") for row in rows if row.get("video_id")}
        eval_video_ids = {
            case.video_id for case in load_cases(eval_cases_path)
            if case.video_id
        }
        overlap = train_video_ids & eval_video_ids
        if overlap:
            raise RuntimeError(
                "train/eval video_id leakage detected: "
                + ", ".join(sorted(overlap)[:10])
            )
    write_jsonl(output, rows)
    return {
        "summary": {
            "written": len(rows),
            "dedup_dropped": len(dropped),
            "skipped": len(skipped),
            "output": str(output),
        },
        "skipped": skipped,
        "dedup_dropped": dropped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SFT JSONL from kept CoT artifacts.")
    parser.add_argument("--cot-dir", default="data/distill/cot")
    parser.add_argument("--consistency-report", default=None)
    parser.add_argument("--output", default="data/distill/sft/sft_frames_only.jsonl")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--eval-cases", default=None, help="Optional EvalCase JSONL for video isolation assertion.")
    args = parser.parse_args()

    if args.consistency_report:
        report = read_json(args.consistency_report)
        paths = [Path(path) for path in report.get("kept_cot", [])]
    else:
        paths = list(Path(args.cot_dir).glob("*.json"))
    report = build_dataset(
        cot_paths=paths,
        output=Path(args.output),
        max_samples=args.max_samples,
        eval_cases_path=Path(args.eval_cases) if args.eval_cases else None,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
