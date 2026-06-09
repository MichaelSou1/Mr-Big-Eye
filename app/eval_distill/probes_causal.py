from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.distill.common import read_json, write_json


NUMBER_RE = re.compile(r"\b\d+\b")


def corrupt_first_number(cot: str) -> tuple[str, bool]:
    match = NUMBER_RE.search(cot)
    if not match:
        return cot, False
    value = int(match.group(0))
    replacement = str(value + 2 if value < 8 else max(0, value - 2))
    return f"{cot[:match.start()]}{replacement}{cot[match.end():]}", True


def build_counterfactual_set(cot_dir: Path, output: Path) -> dict:
    rows = []
    skipped = []
    for path in sorted(cot_dir.glob("*.json")):
        payload = read_json(path)
        corrupted, changed = corrupt_first_number(str(payload.get("cot") or ""))
        if not changed:
            skipped.append({"path": str(path), "reason": "no_numeric_intermediate"})
            continue
        rows.append(
            {
                "case_id": payload.get("case_id"),
                "video_id": payload.get("video_id"),
                "question": payload.get("question"),
                "images": [item for item in payload.get("frame_paths", []) if item],
                "original_cot": payload.get("cot"),
                "corrupted_cot": corrupted,
                "reference_answer": payload.get("answer"),
            }
        )
    report = {"summary": {"written": len(rows), "skipped": len(skipped)}, "skipped": skipped}
    write_json(output.with_suffix(".report.json"), report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build causal CoT intervention probe inputs.")
    parser.add_argument("--cot-dir", default="data/distill/cot")
    parser.add_argument("--output", default="data/distill/eval/causal_counterfactual.jsonl")
    args = parser.parse_args()
    report = build_counterfactual_set(Path(args.cot_dir), Path(args.output))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
