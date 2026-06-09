from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.distill.common import read_jsonl, write_jsonl, write_json


def build_shuffled_cot_dataset(sft_path: Path, output: Path, *, seed: int = 17) -> dict:
    """§7.1 mandatory control arm: same (frames, question, answer), wrong CoT.

    Replaces each sample's CoT with a *mismatched* CoT borrowed from a different
    case, keeping the SFT message format. Training an identical LoRA on this set
    yields the shuffled-CoT control: if the real model's OOD decay matches this
    control, it learned a template, not reasoning.
    """
    rows = read_jsonl(sft_path)
    cots = [str(row.get("cot") or "") for row in rows]
    rng = random.Random(seed)
    out_rows: list[dict] = []
    for index, row in enumerate(rows):
        # Pick a CoT from any other case (deterministic, derangement-ish).
        others = [i for i in range(len(rows)) if i != index and cots[i]]
        if not others:
            continue
        donor = cots[rng.choice(others)]
        answer = str(row.get("answer") or "").strip()
        shuffled = {
            **row,
            "cot": donor,
            "cot_source": "shuffled_control",
            "messages": [
                {"role": "user", "content": str((row.get("messages") or [{}])[0].get("content") or "")},
                {"role": "assistant", "content": f"{donor.strip()}\n\nFinal answer: {answer}".strip()},
            ],
        }
        out_rows.append(shuffled)
    write_jsonl(output, out_rows)
    return {"written": len(out_rows), "source": str(sft_path), "output": str(output)}


def summarize_by_tier(results_path: Path, output: Path) -> dict:
    rows = read_jsonl(results_path)
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("ood_tier") or "unknown"), []).append(row)
    groups = {}
    for tier, items in sorted(buckets.items()):
        correct = sum(1 for item in items if bool(item.get("correct")))
        groups[tier] = {
            "total": len(items),
            "correct": correct,
            "accuracy": correct / len(items) if items else 0.0,
        }
    ordered = [groups[key]["accuracy"] for key in sorted(groups)]
    decay_slope = ordered[-1] - ordered[0] if len(ordered) >= 2 else None
    report = {"groups": groups, "decay_slope_last_minus_first": decay_slope}
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Distribution-shift probe: summarize results or build shuffled-CoT control.")
    parser.add_argument("--results", help="JSONL with ood_tier and correct fields (summary mode).")
    parser.add_argument("--build-shuffle-control", help="SFT JSONL to derive the shuffled-CoT control dataset from.")
    parser.add_argument("--output", default="data/distill/eval/distshift_summary.json")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.build_shuffle_control:
        report = build_shuffled_cot_dataset(Path(args.build_shuffle_control), Path(args.output), seed=args.seed)
    elif args.results:
        report = summarize_by_tier(Path(args.results), Path(args.output))
    else:
        parser.error("provide --results (summary) or --build-shuffle-control (control dataset)")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
