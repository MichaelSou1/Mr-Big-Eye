from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from PIL import Image

import re

from app.distill.common import read_json, write_json
from app.distill.rewrite import validate_cot
from app.mcq import normalize_text, parse_candidates, selected_candidate
from app.vqa import LocalVLMBackbone


def _load_frames(cot_payload: dict[str, Any]) -> tuple[list[Image.Image], list[float]]:
    frames: list[Image.Image] = []
    timestamps: list[float] = []
    paths = list(cot_payload.get("frame_paths") or [])
    times = list(cot_payload.get("shown_frames") or [])
    for path_text, timestamp in zip(paths, times, strict=False):
        path = Path(str(path_text))
        if not path.exists():
            continue
        frames.append(Image.open(path).convert("RGB"))
        timestamps.append(float(timestamp))
    return frames, timestamps


def _answer_matches(question: str, reference: str, answer: str) -> bool:
    candidates = parse_candidates(question)
    if candidates:
        selected = selected_candidate(answer, candidates)
        expected = selected_candidate(reference, candidates)
        if expected is None:
            expected_text = reference.strip().lower().rstrip(".")
            for candidate in candidates:
                if expected_text and expected_text in candidate["text"].lower().rstrip("."):
                    expected = candidate
                    break
        return bool(selected and expected and selected["label"] == expected["label"])
    reference_text = reference.strip().lower()
    answer_text = answer.strip().lower()
    return bool(reference_text and (reference_text in answer_text or answer_text in reference_text))


def strip_answer_from_cot(cot: str, question: str, reference: str) -> str:
    """Remove the final answer/option from a seed CoT so the model cannot copy it.

    Without this, Phase-4 degrades to "can the model echo an answer embedded in
    the provided rationale" — which is near-trivially passable and measures
    nothing. We strip explicit ``Final answer:`` lines, ``answer is X`` phrasing,
    and the chosen MCQ option's label/text so the seed CoT only carries the
    *reasoning*, not the conclusion.
    """
    text = cot or ""
    text = re.sub(r"(?im)^\s*final answer\s*[:：].*$", "", text)
    text = re.sub(r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s+is\b.*$", "", text)
    candidates = parse_candidates(question)
    selected = selected_candidate(reference, candidates) if candidates else None
    if selected is None and candidates:
        selected = selected_candidate(cot, candidates)
    if selected is not None:
        label = re.escape(selected["label"])
        text = re.sub(rf"(?i)\boption\s+{label}\b", "", text)
        text = re.sub(rf"(?m)^\s*{label}\s*[\).].*$", "", text)
        option_norm = normalize_text(selected["text"])
        if option_norm:
            kept = [
                line for line in text.splitlines()
                if option_norm not in normalize_text(line)
            ]
            text = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


async def check_cot_payload(cot_payload: dict[str, Any]) -> dict[str, Any]:
    frames, timestamps = _load_frames(cot_payload)
    if not frames:
        return {
            "case_id": cot_payload.get("case_id"),
            "passed": False,
            "drop_reasons": ["missing_frame_files"],
        }
    question = str(cot_payload.get("question") or "")
    reference = str(cot_payload.get("answer") or "")
    cot = str(cot_payload.get("cot") or "")
    validation = validate_cot(cot, [float(item) for item in cot_payload.get("shown_frames", [])])
    if not validation.ok:
        return {
            "case_id": cot_payload.get("case_id"),
            "passed": False,
            "drop_reasons": validation.errors,
        }

    # Seed the model with the *reasoning* only — the answer is stripped so a pass
    # reflects reachability of the reasoning, not echoing an embedded conclusion.
    seed_cot = strip_answer_from_cot(cot, question, reference)
    backbone = LocalVLMBackbone()
    conditioned_question = (
        f"{question}\n\n"
        "Use this visual reasoning chain as prior context and then give the final answer:\n"
        f"{seed_cot}\n\nFinal answer:"
    )
    consistency_question = (
        f"{question}\n\n"
        "Previously reasoned visual chain:\n"
        f"{seed_cot}\n\n"
        "Re-check the frames and answer consistently if the reasoning is supported."
    )
    free_question = question
    conditioned = await backbone.answer_question(conditioned_question, frames, timestamps)
    consistency = await backbone.answer_question(consistency_question, frames, timestamps)
    free = await backbone.answer_question(free_question, frames, timestamps)

    conditioned_ok = _answer_matches(question, reference, conditioned)
    consistency_ok = _answer_matches(question, reference, consistency)
    free_ok = _answer_matches(question, reference, free)
    # Keep only CoTs the 4B can ACT ON to reach the answer (conditioned + stable)
    # AND that it does NOT already get free-form — the latter carry no
    # internalization signal and would inflate retention.
    passed = conditioned_ok and consistency_ok and not free_ok
    reasons: list[str] = []
    if not conditioned_ok:
        reasons.append("conditioned_answer_wrong")
    if not consistency_ok:
        reasons.append("consistency_answer_wrong")
    if conditioned_ok and consistency_ok and free_ok:
        reasons.append("base_already_correct_no_signal")
    return {
        "case_id": cot_payload.get("case_id"),
        "video_id": cot_payload.get("video_id"),
        "passed": passed,
        "drop_reasons": reasons,
        "conditioned_answer": conditioned,
        "consistency_answer": consistency,
        "free_answer": free,
        "conditioned_correct": conditioned_ok,
        "consistency_correct": consistency_ok,
        "free_correct": free_ok,
        "signal_gain": bool(conditioned_ok and not free_ok),
    }


async def filter_cot_dir(*, cot_dir: Path, output: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    kept: list[str] = []
    for path in sorted(cot_dir.glob("*.json")):
        payload = read_json(path)
        result = await check_cot_payload(payload)
        result["cot_path"] = str(path)
        results.append(result)
        if result["passed"]:
            kept.append(str(path))
    total = len(results)
    signal_gain = sum(1 for item in results if item.get("signal_gain"))
    free_correct = sum(1 for item in results if item.get("free_correct"))
    summary = {
        "total": total,
        "passed": len(kept),
        # retention_rate now reflects the strict gate (conditioned+stable AND
        # base-free-wrong), so it is the honest internalizable-signal metric.
        "retention_rate": len(kept) / total if total else 0.0,
        "signal_gain_rate": signal_gain / total if total else 0.0,
        "base_free_correct": free_correct,
        "base_free_accuracy": free_correct / total if total else 0.0,
    }
    report = {"summary": summary, "kept_cot": kept, "results": results}
    write_json(output, report)
    return report


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Filter rewritten CoT by local 4B consistency.")
    parser.add_argument("--cot-dir", default="data/distill/cot")
    parser.add_argument("--output", default="data/distill/consistency_filter_report.json")
    args = parser.parse_args()

    report = await filter_cot_dir(cot_dir=Path(args.cot_dir), output=Path(args.output))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0 if report["summary"]["passed"] else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
