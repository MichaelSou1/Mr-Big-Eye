from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.cache import get_video_status
from app.config import settings
from app.distill.common import trajectory_payload, write_json
from app.eval_fingerprint import AGENT_CODE_VERSION, prompt_fingerprint
from app.eval_harness import EvalCase, load_cases
from app.graph import build_graph
from app.vqa import current_agent_vlm_cache_label


class _NoopMemoryManager:
    async def ainvoke(self, payload, config=None):
        return None


def _initial_state(case: EvalCase) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=case.question)],
        "video_id": case.video_id,
        "user_id": "distill",
        "retrieved_frames": [],
        "retrieved_scene_hits": [],
        "retrieved_transcripts": [],
        "retrieved_slides": [],
        "retrieval_plan": {},
        "timeline": [],
        "candidate_timeline": [],
        "audiovisual_candidate_matrix": [],
        "hypotheses": [],
        "evidence_sufficiency": {},
        "draft_answer": "",
        "observer_notes": [],
        "grounding_report": {},
        "subject_registry": [],
    }


def select_cases(cases: list[EvalCase], *, sample: int | None, seed: int | None) -> list[EvalCase]:
    ready = [
        case for case in cases
        if case.video_id and get_video_status(case.video_id) == "done"
    ]
    if sample is None or sample <= 0 or sample >= len(ready):
        return ready
    rng = random.Random(seed)
    return rng.sample(ready, sample)


async def generate_trajectories(
    *,
    cases: list[EvalCase],
    output_dir: Path,
    per_case_delay_sec: float = 0.0,
) -> list[Path]:
    graph = build_graph(InMemorySaver(), InMemoryStore(), memory_manager=_NoopMemoryManager())
    run_meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "agent_vlm_backend": settings.agent_vlm_backend,
        "agent_vlm": current_agent_vlm_cache_label(),
        "orchestrator_model": settings.orchestrator_model_name or settings.vlm_model_name,
        "orchestrator_base_url": settings.orchestrator_api_base_url or settings.vlm_api_base_url,
        "prompt_fingerprint": prompt_fingerprint(),
        "agent_code_version": AGENT_CODE_VERSION,
        "train_modality": settings.train_modality,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case in cases:
        if not case.video_id or get_video_status(case.video_id) != "done":
            raise RuntimeError(f"Video cache is not ready for case={case.case_id} video_id={case.video_id!r}")
        state = await graph.ainvoke(
            _initial_state(case),
            config={"configurable": {"thread_id": f"distill-{case.case_id}-{uuid4().hex}"}},
        )
        payload = trajectory_payload(case=case, state=state, run_meta=run_meta)
        path = output_dir / f"{case.case_id}.json"
        write_json(path, payload)
        written.append(path)
        print(json.dumps({"case_id": case.case_id, "trajectory": str(path)}, ensure_ascii=False), flush=True)
        if per_case_delay_sec > 0:
            await asyncio.sleep(per_case_delay_sec)
    return written


async def main_async() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate full agent trajectories for distillation.")
    parser.add_argument("--cases", required=True, help="EvalCase JSONL.")
    parser.add_argument("--output-dir", default="data/distill/trajectories")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-case-delay-sec", type=float, default=0.0)
    args = parser.parse_args()

    if settings.train_modality != "frames_only":
        raise RuntimeError("TRAIN_MODALITY must be frames_only for this pipeline.")

    cases = select_cases(load_cases(args.cases), sample=args.sample, seed=args.seed)
    if not cases:
        print("No preprocessed cases selected.")
        return 2
    await generate_trajectories(
        cases=cases,
        output_dir=Path(args.output_dir),
        per_case_delay_sec=args.per_case_delay_sec,
    )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
