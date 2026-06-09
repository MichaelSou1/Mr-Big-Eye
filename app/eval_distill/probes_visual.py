from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.distill.common import read_json, read_jsonl, write_json
from app.distill.frames import covers_evidence
from app.eval_harness import FRAME_MARKER_RE


def check_claimed_timestamps(cot_dir: Path, output: Path) -> dict:
    """Probe 3c (claimed-timestamp vs GT consistency).

    NOTE: OpenAI-compatible serving exposes no attention maps, so this is a
    *claim-consistency* check, not an attention check: where a CoT claims
    "looking at t=X", verify X lands on the NExT-GQA grounding GT. A marker the
    CoT cites but that misses the GT window is template phrasing, not grounded
    looking. Cases without GT are skipped.
    """
    rows = []
    skipped = 0
    for path in sorted(cot_dir.glob("*.json")):
        payload = read_json(path)
        gold_ts = [float(v) for v in payload.get("gold_timestamps", []) or []]
        gold_scenes = payload.get("gold_scenes", []) or []
        if not gold_ts and not gold_scenes:
            skipped += 1
            continue
        markers = [float(m.group(1)) for m in FRAME_MARKER_RE.finditer(str(payload.get("cot") or ""))]
        hit = covers_evidence(markers, gold_ts, gold_scenes) if markers else False
        rows.append({"case_id": payload.get("case_id"), "claimed_markers": markers, "claim_on_gt": hit})
    checked = len(rows)
    on_gt = sum(1 for r in rows if r["claim_on_gt"])
    report = {
        "checked": checked,
        "skipped_no_gt": skipped,
        "claim_on_gt_rate": on_gt / checked if checked else 0.0,
        "results": rows,
    }
    write_json(output, report)
    return {k: v for k, v in report.items() if k != "results"}


def summarize_visual_dependency(results_path: Path, output: Path) -> dict:
    rows = read_jsonl(results_path)
    total = len(rows)
    answer_moved = sum(1 for row in rows if bool(row.get("answer_changed_with_image")))
    grounding_hit = sum(1 for row in rows if bool(row.get("grounding_hit")))
    report = {
        "total": total,
        "answer_followed_image_rate": answer_moved / total if total else 0.0,
        "grounding_hit_rate": grounding_hit / total if total else 0.0,
    }
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Visual-dependency probes: summarize results or check claimed timestamps (3c).")
    parser.add_argument("--results", help="JSONL of 3a/3b probe results (summary mode).")
    parser.add_argument("--check-claims", help="CoT dir to run the 3c claimed-timestamp vs GT check on.")
    parser.add_argument("--output", default="data/distill/eval/visual_summary.json")
    args = parser.parse_args()
    if args.check_claims:
        report = check_claimed_timestamps(Path(args.check_claims), Path(args.output))
    elif args.results:
        report = summarize_visual_dependency(Path(args.results), Path(args.output))
    else:
        parser.error("provide --results (summary) or --check-claims (3c check)")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
