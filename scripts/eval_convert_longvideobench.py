#!/usr/bin/env python
"""Convert LongVideoBench lvb_val.json into EvalCase JSONL + manifest.

Stratified sample by duration bucket (<=60s, 60-600s, >600s) so coverage of the
long-video segment is preserved when --sample shrinks the population.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.eval_datasets import stratified_sample, write_json, write_jsonl


CASE_PREFIX = "longvideobench"
DEFAULT_OUTPUT_CASES = ROOT_DIR / "tests" / "fixtures" / "eval_cases_longvideobench.jsonl"
DEFAULT_OUTPUT_MANIFEST = ROOT_DIR / "data" / "eval" / "datasets" / "longvideobench" / "manifest.json"

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "by", "with", "at", "as", "it", "this", "that", "these",
    "those", "from", "into", "than", "then", "they", "their", "there", "here",
}


def duration_bucket(duration: float) -> str:
    if duration <= 60:
        return "short"
    if duration <= 600:
        return "medium"
    return "long"


def keywords_from_answer(answer: str, *, limit: int = 5) -> list[str]:
    tokens = re.findall(r"[\w']+", answer.lower())
    seen: list[str] = []
    for tok in tokens:
        if len(tok) < 3 or tok in STOPWORDS or tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= limit:
            break
    return seen


def format_mcq(question: str, candidates: list[str]) -> str:
    block = [question.strip(), "", "Candidates:"]
    for idx, cand in enumerate(candidates):
        label = string.ascii_uppercase[idx] if idx < 26 else str(idx)
        block.append(f"{label}) {cand}")
    block.append("")
    block.append("Answer the question and reference key frames with [FRAME:t=...] markers.")
    return "\n".join(block)


def convert_item(item: dict) -> tuple[dict, dict] | None:
    item_id = item.get("id")
    question = item.get("question") or ""
    candidates = item.get("candidates") or []
    correct_choice = item.get("correct_choice")
    video_path = item.get("video_path") or ""
    if item_id is None or not question or not candidates or correct_choice is None:
        return None
    try:
        reference = candidates[int(correct_choice)]
    except (IndexError, TypeError, ValueError):
        return None
    case = {
        "case_id": f"{CASE_PREFIX}-{item_id}",
        "video_id": "",
        "question": format_mcq(question, candidates),
        "reference_answer": str(reference),
        "required_keywords": keywords_from_answer(str(reference)),
        "forbidden_keywords": [],
        "gold_timestamps": [],
        "gold_scenes": [],
    }
    duration = float(item.get("duration") or 0)
    manifest_entry = {
        "case_id": case["case_id"],
        "video_path": str(video_path),
        "duration": duration,
        "duration_bucket": duration_bucket(duration),
    }
    return case, manifest_entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert LongVideoBench lvb_val.json to EvalCase JSONL.")
    parser.add_argument("--source", required=True, help="Path to lvb_val.json.")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-cases", default=str(DEFAULT_OUTPUT_CASES))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"ERROR: source not found: {source_path}", file=sys.stderr)
        return 2
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("ERROR: lvb_val.json must be a JSON array.", file=sys.stderr)
        return 2

    converted = []
    for item in raw:
        result = convert_item(item)
        if result is None:
            continue
        case, manifest_entry = result
        converted.append({"case": case, "manifest": manifest_entry})

    if not converted:
        print("ERROR: no valid items converted.", file=sys.stderr)
        return 2

    if args.sample and args.sample < len(converted):
        sampled = stratified_sample(
            converted,
            sample_size=args.sample,
            key=lambda x: x["manifest"]["duration_bucket"],
            seed=args.seed,
        )
    else:
        sampled = converted

    cases = [entry["case"] for entry in sampled]
    manifest = [entry["manifest"] for entry in sampled]

    n_cases = write_jsonl(Path(args.output_cases), cases)
    write_json(Path(args.output_manifest), manifest)
    print(
        f"LongVideoBench: wrote {n_cases} cases to {args.output_cases} "
        f"(manifest: {args.output_manifest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
