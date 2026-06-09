from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.distill.common import read_jsonl, write_json


def summarize_diversity_curve(results_path: Path, output: Path) -> dict:
    rows = read_jsonl(results_path)
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("train_diversity_bucket") or "unknown"), []).append(row)
    groups = {}
    for bucket, items in sorted(buckets.items()):
        correct = sum(1 for item in items if bool(item.get("ood_correct")))
        groups[bucket] = {
            "total": len(items),
            "ood_accuracy": correct / len(items) if items else 0.0,
        }
    report = {"groups": groups}
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize data diagnostic curves.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="data/distill/eval/data_diagnostics_summary.json")
    args = parser.parse_args()
    report = summarize_diversity_curve(Path(args.results), Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
