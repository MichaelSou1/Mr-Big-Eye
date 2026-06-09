#!/usr/bin/env python
"""Control-group prediction generator: uniform-sample N frames, single VLM call.

Bypasses BGE/SigLIP retrieval and the LangGraph orchestrator entirely so we
can isolate the contribution of the retrieval+agent stack vs. a naive
"feed evenly-spaced frames to the VLM" baseline.

Output JSONL is consumable by `scripts/eval_harness.py --predictions`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.cache import get_video_status
from app.config import settings
from app.distill.frames import load_uniform_frames as _load_uniform_frames
from app.eval_harness import load_cases
from app.vqa import answer_question


async def main_async() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True, help="Predictions JSONL path.")
    parser.add_argument("--num-frames", type=int, default=settings.vqa_max_frames)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.sample is not None and args.sample > 0:
        ingested = [c for c in cases if c.video_id and get_video_status(c.video_id) == "done"]
        if len(ingested) < args.sample:
            print(
                f"WARNING: requested {args.sample} but only {len(ingested)} ingested; using all.",
                file=sys.stderr,
            )
            cases = ingested
        else:
            rng = random.Random(args.sample_seed)
            cases = rng.sample(ingested, args.sample)
        print(f"sampled {len(cases)} cases (seed={args.sample_seed!r})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as fh:
        for i, case in enumerate(cases, 1):
            frames, timestamps = _load_uniform_frames(case.video_id, args.num_frames)
            if not frames:
                print(f"[skip] {case.case_id}: no frames", file=sys.stderr)
                continue
            try:
                answer = await answer_question(case.question, frames, timestamps, history=None)
            except Exception as exc:
                print(f"[error] {case.case_id}: {exc}", file=sys.stderr)
                answer = ""
            record = {
                "case_id": case.case_id,
                "answer": answer,
                "retrieved_timestamps": timestamps,
                "scene_hits": [],
                "agent_actions": ["uniform_sample_vqa"],
                "evidence_sufficiency": {"control_group": "no_retrieval"},
                "grounding_report": {},
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(cases)}] {case.case_id}: {len(timestamps)} frames -> {len(answer)} chars", flush=True)
    print(f"wrote {args.output}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
