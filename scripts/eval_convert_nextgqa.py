#!/usr/bin/env python
"""Convert NExT-GQA annotations (val split) into EvalCase JSONL + manifest.

Sources expected in --source-dir:
  val.csv               — per-question metadata (video_id, qid, question, answer, a0..a4, type)
  gsub_val.json         — per-(video_id, qid) gold time intervals
  map_vid_vidorID.json  — video_id -> raw VidOR filename mapping (for ingest)

Stratified sample by `type` so all reasoning categories are kept proportionally.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.eval_datasets import stratified_sample, write_json, write_jsonl


CASE_PREFIX = "nextgqa"
DEFAULT_OUTPUT_CASES = ROOT_DIR / "tests" / "fixtures" / "eval_cases_nextgqa.jsonl"
DEFAULT_OUTPUT_MANIFEST = ROOT_DIR / "data" / "eval" / "datasets" / "nextgqa" / "manifest.json"

TEMPORAL_TYPES = {"TN", "TC", "TP"}

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "by", "with", "at", "as", "it", "this", "that", "these",
    "those", "from", "into", "than", "then", "they", "their", "there", "here",
    "yes", "no",
}


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
    labels = ["A", "B", "C", "D", "E"]
    block = [question.strip(), "", "Candidates:"]
    for label, cand in zip(labels, candidates):
        if cand:
            block.append(f"{label}) {cand}")
    block.append("")
    block.append("Answer the question and reference key frames with [FRAME:t=...] markers.")
    return "\n".join(block)


def convert_row(
    row: dict[str, str],
    gsub: dict,
    vid_to_vidor: dict[str, str],
) -> tuple[dict, dict] | None:
    qid = row.get("qid")
    video_id = row.get("video_id")
    question = row.get("question") or ""
    answer = row.get("answer") or ""
    qtype = row.get("type") or "misc"
    candidates = [row.get(f"a{i}") or "" for i in range(5)]
    if not qid or not video_id or not question or not answer:
        return None

    video_record = gsub.get(video_id) or {}
    locations = (video_record.get("location") or {}).get(qid) or []
    gold_scenes = []
    gold_timestamps = []
    for interval in locations:
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            continue
        try:
            start = float(interval[0])
            end = float(interval[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        gold_scenes.append({"start": start, "end": end})
        gold_timestamps.append((start + end) / 2.0)

    case = {
        "case_id": f"{CASE_PREFIX}-{video_id}-{qid}",
        "video_id": "",
        "question": format_mcq(question, candidates),
        "reference_answer": str(answer),
        "required_keywords": keywords_from_answer(str(answer)),
        "forbidden_keywords": [],
        "gold_timestamps": gold_timestamps,
        "gold_scenes": gold_scenes,
    }
    if qtype in TEMPORAL_TYPES:
        case["expected_action"] = "build_timeline"

    vidor_id = vid_to_vidor.get(video_id) or ""
    manifest_entry = {
        "case_id": case["case_id"],
        "nextqa_video_id": video_id,
        "vidor_path": vidor_id,  # e.g. "1015/5750463032.mp4" under VidOR root
        "qtype": qtype,
    }
    return case, manifest_entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert NExT-GQA val split to EvalCase JSONL.")
    parser.add_argument("--source-dir", required=True, help="Dir containing val.csv, gsub_val.json, map_vid_vidorID.json.")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-cases", default=str(DEFAULT_OUTPUT_CASES))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    val_csv = source_dir / "val.csv"
    gsub_path = source_dir / "gsub_val.json"
    map_path = source_dir / "map_vid_vidorID.json"
    for path in (val_csv, gsub_path, map_path):
        if not path.exists():
            print(f"ERROR: required file missing: {path}", file=sys.stderr)
            return 2

    gsub = json.loads(gsub_path.read_text(encoding="utf-8"))
    vid_to_vidor = json.loads(map_path.read_text(encoding="utf-8"))

    converted: list[dict] = []
    with val_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            result = convert_row(row, gsub, vid_to_vidor)
            if result is None:
                continue
            case, manifest_entry = result
            converted.append({"case": case, "manifest": manifest_entry})

    if not converted:
        print("ERROR: no valid rows converted.", file=sys.stderr)
        return 2

    if args.sample and args.sample < len(converted):
        sampled = stratified_sample(
            converted,
            sample_size=args.sample,
            key=lambda x: x["manifest"]["qtype"],
            seed=args.seed,
        )
    else:
        sampled = converted

    cases = [entry["case"] for entry in sampled]
    manifest = [entry["manifest"] for entry in sampled]

    n_cases = write_jsonl(Path(args.output_cases), cases)
    write_json(Path(args.output_manifest), manifest)
    print(
        f"NExT-GQA: wrote {n_cases} cases to {args.output_cases} "
        f"(manifest: {args.output_manifest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
