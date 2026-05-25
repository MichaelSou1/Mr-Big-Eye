#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.cache import get_video_status
from app.config import settings
from app.eval_harness import (
    EvalPrediction,
    JudgeCache,
    JudgeClient,
    PredictionCache,
    evaluate_case,
    group_by_case_prefix,
    load_cases,
    load_predictions,
    summarize_results,
    write_json,
    write_markdown_report,
)
from app.eval_fingerprint import AGENT_CODE_VERSION, prompt_fingerprint


async def main_async() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Mr. Big-Eye quality harness.")
    parser.add_argument("--cases", required=True, help="JSONL eval cases.")
    parser.add_argument(
        "--predictions",
        help="Optional JSONL predictions. If omitted, run the local graph/retrieval stack.",
    )
    parser.add_argument("--output", default="data/eval/latest_report.json")
    parser.add_argument("--markdown", default=None, help="Markdown report path; defaults to <output>.md")
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--recall-k", type=int, default=None)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit non-zero if overall pass_rate is below this threshold.",
    )
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--judge",
        dest="judge",
        action="store_true",
        default=None,
        help="Enable LLM-as-judge. Uses JUDGE_API_* with VLM_API_* fallback.",
    )
    judge_group.add_argument(
        "--no-judge",
        dest="judge",
        action="store_false",
        help="Disable LLM-as-judge even if JUDGE_API_KEY is set.",
    )
    parser.add_argument(
        "--judge-cache",
        default="data/eval/judge_cache.jsonl",
        help="JSONL cache for judge responses; pass empty string to disable.",
    )
    parser.add_argument(
        "--prediction-cache",
        default="data/eval/prediction_cache.jsonl",
        help="JSONL cache for full agent predictions per case; pass empty string to disable.",
    )
    parser.add_argument(
        "--group-by-prefix",
        dest="group_by_prefix",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-group-by-prefix",
        dest="group_by_prefix",
        action="store_false",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Randomly sample N cases from --cases that have a preprocessed video "
            "(get_video_status == 'done'). Omit to use all cases."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for --sample. Omit for non-deterministic sampling each run.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.sample is not None and args.sample > 0:
        import random

        from app.cache import get_video_status

        ingested = [
            c for c in cases
            if c.video_id and get_video_status(c.video_id) == "done"
        ]
        if len(ingested) < args.sample:
            print(
                f"WARNING: requested --sample {args.sample} but only "
                f"{len(ingested)} cases have a preprocessed video; using all of them.",
                file=sys.stderr,
            )
            cases = ingested
        else:
            rng = random.Random(args.sample_seed)
            cases = rng.sample(ingested, args.sample)
        print(
            f"sampled {len(cases)}/{len(ingested)} ingested cases "
            f"(seed={args.sample_seed!r}): "
            + ", ".join(c.case_id for c in cases[:5])
            + (" ..." if len(cases) > 5 else "")
        )
    if args.predictions:
        predictions = load_predictions(args.predictions)
    else:
        prediction_cache: PredictionCache | None = None
        if args.prediction_cache:
            prediction_cache = PredictionCache(args.prediction_cache)
        predictions = await _run_predictions(cases, prediction_cache)

    judge: JudgeClient | None = None
    judge_cache: JudgeCache | None = None
    want_judge = args.judge if args.judge is not None else bool((settings.judge_api_key or "").strip())
    if want_judge:
        judge = JudgeClient.from_settings(settings)
        if judge is None and args.judge is True:
            print(
                "ERROR: --judge requested but no JUDGE_API_KEY / VLM_API_KEY available.",
                file=sys.stderr,
            )
            return 2
        if judge is not None and args.judge_cache:
            judge_cache = JudgeCache(args.judge_cache)

    group_by = group_by_case_prefix if args.group_by_prefix else None

    results = []
    missing = []
    for case in cases:
        prediction = predictions.get(case.case_id)
        if prediction is None:
            missing.append(case.case_id)
            continue
        results.append(
            evaluate_case(
                case,
                prediction,
                tolerance_sec=args.tolerance_sec,
                recall_k=args.recall_k,
                judge=judge,
                judge_cache=judge_cache,
            )
        )

    summary = summarize_results(results, group_by=group_by)
    run_meta = _build_run_meta(args, cases, results, want_judge=bool(judge))
    report = {
        "run_meta": run_meta,
        "summary": summary,
        "missing_predictions": missing,
        "results": results,
    }
    write_json(args.output, report)
    markdown_path = args.markdown or str(Path(args.output).with_suffix(".md"))
    write_markdown_report(report, markdown_path)
    _append_runs_index(run_meta, summary, results, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")

    if missing:
        print(f"missing predictions: {', '.join(missing)}", file=sys.stderr)
        return 2
    return 1 if report["summary"]["pass_rate"] < args.fail_under else 0


async def _run_predictions(
    cases,
    prediction_cache: PredictionCache | None = None,
) -> dict[str, EvalPrediction]:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from app.graph import build_graph

    app_graph = build_graph(InMemorySaver(), InMemoryStore(), memory_manager=_NoopMemoryManager())
    fingerprint = prompt_fingerprint() if prediction_cache is not None else ""
    # Compose both orchestrator and VLM into the cache key so swapping either
    # one invalidates affected entries (VLM-only swap was previously a silent
    # cache hit because only orchestrator name was keyed).
    orch_name = (settings.orchestrator_model_name or settings.vlm_model_name or "")
    vlm_name = (settings.vlm_model_name or "")
    model_name = f"{orch_name}|vlm={vlm_name}"
    predictions = {}
    for case in cases:
        if get_video_status(case.video_id) != "done":
            raise RuntimeError(
                f"Video cache is not ready for {case.video_id}. "
                "Preprocess it first or provide --predictions."
            )
        if prediction_cache is not None:
            key = PredictionCache.make_key(
                case_id=case.case_id,
                model=model_name,
                prompt_fingerprint=fingerprint,
                video_id=case.video_id,
                agent_code_version=AGENT_CODE_VERSION,
            )
            cached = prediction_cache.get(key)
            if cached is not None:
                print(f"[cache] hit case={case.case_id}", flush=True)
                predictions[case.case_id] = PredictionCache.prediction_from_dict(case.case_id, cached)
                continue
            print(f"[cache] miss case={case.case_id}", flush=True)
        state = await app_graph.ainvoke(
            {
                "messages": [HumanMessage(content=case.question)],
                "video_id": case.video_id,
                "user_id": "eval",
                "retrieved_frames": [],
                "retrieved_scene_hits": [],
                "retrieval_plan": {},
                "timeline": [],
                "hypotheses": [],
                "evidence_sufficiency": {},
                "draft_answer": "",
                "grounding_report": {},
            },
            config={"configurable": {"thread_id": f"eval-{case.case_id}-{uuid4().hex}"}},
        )
        evidence_sufficiency = dict(state.get("evidence_sufficiency", {}) or {})
        agent_terminated = state.get("agent_terminated")
        if agent_terminated:
            evidence_sufficiency["agent_terminated"] = agent_terminated
        prediction = EvalPrediction(
            case_id=case.case_id,
            retrieved_timestamps=[
                float(item["timestamp"])
                for item in state.get("retrieved_frames", [])
                if "timestamp" in item
            ],
            scene_hits=state.get("retrieved_scene_hits", []),
            answer=_last_assistant_message(state.get("messages", [])),
            agent_actions=_agent_actions(state.get("messages", [])),
            evidence_sufficiency=evidence_sufficiency,
            grounding_report=state.get("grounding_report", {}),
        )
        predictions[case.case_id] = prediction
        if prediction_cache is not None:
            prediction_cache.put(key, PredictionCache.prediction_to_dict(prediction))
    return predictions


def _last_assistant_message(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
            return str(message.content)
    return ""


def _agent_actions(messages: list[Any]) -> list[str]:
    actions = []
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        try:
            payload = json.loads(str(message.content))
        except json.JSONDecodeError:
            continue
        tool_name = payload.get("tool")
        if tool_name:
            actions.append(str(tool_name))
    return actions


class _NoopMemoryManager:
    async def ainvoke(self, payload, config=None):
        return None


def _build_run_meta(args, cases, results, *, want_judge: bool) -> dict[str, Any]:
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "vlm_model_name": settings.vlm_model_name,
        "vlm_api_base_url": settings.vlm_api_base_url,
        "orchestrator_model_name": settings.orchestrator_model_name or settings.vlm_model_name,
        "orchestrator_api_base_url": settings.orchestrator_api_base_url or settings.vlm_api_base_url,
        "judge_enabled": want_judge,
        "prompt_fingerprint": prompt_fingerprint(),
        "agent_code_version": AGENT_CODE_VERSION,
        "dataset_path": args.cases,
        "sample_size": args.sample,
        "sample_seed": args.sample_seed,
        "tolerance_sec": args.tolerance_sec,
        "n_cases_total": len(cases),
        "n_cases_predicted": len(results),
    }


def _append_runs_index(
    run_meta: dict[str, Any],
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Append one row per run to data/eval/runs_index.csv for cross-run comparison."""
    judge_correct = sum(
        1
        for r in results
        if bool(((r.get("answer") or {}).get("llm_judge") or {}).get("correct"))
    )
    judge_total = sum(
        1 for r in results if (r.get("answer") or {}).get("llm_judge") is not None
    )
    judge_correct_rate = (judge_correct / judge_total) if judge_total else None
    index_path = Path("data/eval/runs_index.csv")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not index_path.exists()
    fields = [
        "timestamp",
        "vlm_model",
        "orch_model",
        "fingerprint",
        "version",
        "n",
        "seed",
        "pass_rate",
        "answer_pass_rate",
        "judge_correct_rate",
        "recall_mean",
        "ts_dist_mean",
        "output_path",
    ]
    row = {
        "timestamp": run_meta["timestamp"],
        "vlm_model": run_meta["vlm_model_name"],
        "orch_model": run_meta["orchestrator_model_name"],
        "fingerprint": run_meta["prompt_fingerprint"],
        "version": run_meta["agent_code_version"],
        "n": run_meta["n_cases_predicted"],
        "seed": run_meta["sample_seed"],
        "pass_rate": summary.get("pass_rate"),
        "answer_pass_rate": summary.get("answer_pass_rate"),
        "judge_correct_rate": judge_correct_rate,
        "recall_mean": summary.get("retrieval_recall_at_k_mean"),
        "ts_dist_mean": summary.get("timestamp_distance_mean"),
        "output_path": output_path,
    }
    with index_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
