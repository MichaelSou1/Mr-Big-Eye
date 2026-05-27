from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


logger = logging.getLogger(__name__)


FRAME_MARKER_RE = re.compile(r"\[FRAME:t=([0-9]+(?:\.[0-9]+)?)\]")
TRANSCRIPT_MARKER_RE = re.compile(
    r"\[TRANSCRIPT:t=([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\]"
)
SLIDE_MARKER_RE = re.compile(r"\[SLIDE:t=([0-9]+(?:\.[0-9]+)?)\]")
UNCERTAINTY_MARKERS = (
    "not enough",
    "insufficient",
    "cannot determine",
    "can't determine",
    "unclear",
    "not visible",
    "checked frames",
    "checked evidence",
    "do not see",
    "don't see",
    "无法确定",
    "不能确定",
    "不确定",
    "证据不足",
    "看不清",
    "未能确认",
)


@dataclass
class EvalCase:
    case_id: str
    video_id: str
    question: str
    gold_timestamps: list[float] = field(default_factory=list)
    gold_scenes: list[dict[str, float]] = field(default_factory=list)
    reference_answer: str = ""
    required_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    expected_action: str | None = None
    requires_uncertainty: bool = False
    needs_expansion: bool = False
    modality_tag: str = ""
    question_type: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    expected_citation_min: int | None = None
    expected_citation_kinds: list[str] = field(default_factory=list)


@dataclass
class EvalPrediction:
    case_id: str
    retrieved_timestamps: list[float] = field(default_factory=list)
    scene_hits: list[dict[str, float]] = field(default_factory=list)
    retrieved_transcripts: list[dict[str, Any]] = field(default_factory=list)
    retrieved_slides: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    agent_actions: list[str] = field(default_factory=list)
    evidence_sufficiency: dict[str, Any] = field(default_factory=dict)
    grounding_report: dict[str, Any] = field(default_factory=dict)


def load_cases(path: str | Path) -> list[EvalCase]:
    return [parse_case(item) for item in _read_jsonl(path)]


def load_predictions(path: str | Path) -> dict[str, EvalPrediction]:
    return {prediction.case_id: prediction for prediction in map(parse_prediction, _read_jsonl(path))}


def parse_case(item: dict[str, Any]) -> EvalCase:
    case_id = item.get("case_id") or item.get("question_id")
    if not case_id:
        raise KeyError("case_id")
    expected_citation_kinds = _parse_citation_kinds(item.get("expected_citation_kinds", []))
    expected_citation_min = item.get("expected_citation_min")
    if expected_citation_min is None and expected_citation_kinds:
        expected_citation_min = 1
    expected_keywords = [str(value) for value in item.get("expected_keywords", [])]
    required_keywords = [str(value) for value in item.get("required_keywords", [])]
    return EvalCase(
        case_id=str(case_id),
        video_id=str(item.get("video_id") or ""),
        question=str(item["question"]),
        gold_timestamps=[float(value) for value in item.get("gold_timestamps", [])],
        gold_scenes=[
            {"start": float(scene["start"]), "end": float(scene["end"])}
            for scene in item.get("gold_scenes", [])
        ],
        reference_answer=str(item.get("reference_answer") or ""),
        required_keywords=required_keywords or expected_keywords,
        forbidden_keywords=[str(value) for value in item.get("forbidden_keywords", [])],
        expected_action=item.get("expected_action"),
        requires_uncertainty=bool(item.get("requires_uncertainty", False)),
        needs_expansion=bool(item.get("needs_expansion", False)),
        modality_tag=str(item.get("modality_tag") or "").strip().lower(),
        question_type=str(item.get("question_type") or "").strip().lower(),
        expected_keywords=expected_keywords,
        expected_citation_min=(
            int(expected_citation_min) if expected_citation_min is not None else None
        ),
        expected_citation_kinds=expected_citation_kinds,
    )


def parse_prediction(item: dict[str, Any]) -> EvalPrediction:
    return EvalPrediction(
        case_id=str(item["case_id"]),
        retrieved_timestamps=[float(value) for value in item.get("retrieved_timestamps", [])],
        scene_hits=[
            {"start": float(scene["start"]), "end": float(scene["end"])}
            for scene in item.get("scene_hits", [])
            if "start" in scene and "end" in scene
        ],
        retrieved_transcripts=[
            dict(value) for value in item.get("retrieved_transcripts", [])
            if isinstance(value, dict)
        ],
        retrieved_slides=[
            dict(value) for value in item.get("retrieved_slides", [])
            if isinstance(value, dict)
        ],
        answer=str(item.get("answer") or ""),
        agent_actions=[str(value) for value in item.get("agent_actions", [])],
        evidence_sufficiency=dict(item.get("evidence_sufficiency", {}) or {}),
        grounding_report=dict(item.get("grounding_report", {}) or {}),
    )


def evaluate_case(
    case: EvalCase,
    prediction: EvalPrediction,
    *,
    tolerance_sec: float = 2.0,
    recall_k: int | None = None,
    judge: "JudgeClient | None" = None,
    judge_cache: "JudgeCache | None" = None,
) -> dict[str, Any]:
    retrieval = evaluate_retrieval(
        gold_timestamps=case.gold_timestamps,
        retrieved_timestamps=prediction.retrieved_timestamps,
        gold_scenes=case.gold_scenes,
        scene_hits=prediction.scene_hits,
        tolerance_sec=tolerance_sec,
        recall_k=recall_k,
    )
    audiovisual_case = _is_audiovisual_case(case)
    if audiovisual_case:
        answer = evaluate_audiovisual_answer(
            answer=prediction.answer,
            expected_keywords=case.expected_keywords or case.required_keywords,
            forbidden_keywords=case.forbidden_keywords,
            expected_citation_min=case.expected_citation_min,
            expected_citation_kinds=case.expected_citation_kinds,
            requires_uncertainty=case.requires_uncertainty,
        )
        answer["citation_soft_waived"] = False
    else:
        answer = evaluate_answer(
            answer=prediction.answer,
            retrieved_timestamps=prediction.retrieved_timestamps,
            required_keywords=case.required_keywords,
            forbidden_keywords=case.forbidden_keywords,
            requires_uncertainty=case.requires_uncertainty,
        )
        answer["citation_soft_waived"] = False
    if judge is not None:
        judge_result = evaluate_answer_llm(case, prediction, judge=judge, cache=judge_cache)
        answer["llm_judge"] = judge_result
        # Soft gate: when judge accepts, drop the keyword AND the citation
        # requirements (they were instrumentation, not correctness). Still
        # require hallucination_free + uncertainty_ok regardless.
        judge_pass = bool(judge_result.get("correct"))
        if judge_pass:
            answer["citation_soft_waived"] = not answer["citation_correct"]
            answer["passed"] = answer["hallucination_free"] and answer["uncertainty_ok"]
        else:
            # Strict path: judge disagreed, fall back to keyword+citation gate.
            keyword_pass = answer["answered"] and answer["hallucination_free"]
            answer["passed"] = (
                keyword_pass
                and answer["citation_correct"]
                and answer["uncertainty_ok"]
            )
    agent = evaluate_agent_loop(
        agent_actions=prediction.agent_actions,
        expected_action=case.expected_action,
        needs_expansion=case.needs_expansion,
        evidence_sufficiency=prediction.evidence_sufficiency,
    )
    # Soft-waive agent_loop the same way we soft-waive citation: when the judge
    # has accepted the answer, the prescribed-tool path was instrumentation,
    # not correctness. Outcome > process. Keep the strict signal for forensics.
    agent["soft_waived"] = False
    if judge is not None and bool((answer.get("llm_judge") or {}).get("correct")):
        if agent["passed"] is False:
            agent["soft_waived"] = True
    # Soft-waive retrieval the same way we soft-waive citation and agent: when
    # the judge accepts the answer, a strict retrieval-gate FAIL is
    # instrumentation, not correctness. Keep the strict signal in the JSON.
    retrieval["soft_waived"] = False
    if judge is not None and bool((answer.get("llm_judge") or {}).get("correct")):
        if retrieval["passed"] is False:
            retrieval["soft_waived"] = True
    # retrieval["passed"] may be None when the case has no gold timestamps
    # AND no gold scenes (LVB-style MCQs). Treat None as n/a (not a fail).
    retrieval_ok = retrieval["passed"] is not False or retrieval["soft_waived"]
    agent_ok = agent["passed"] is not False or agent["soft_waived"]
    passed = retrieval_ok and answer["passed"] and agent_ok
    return {
        "case_id": case.case_id,
        "video_id": case.video_id,
        "question": case.question,
        "passed": passed,
        "retrieval": retrieval,
        "answer": answer,
        "agent_loop": agent,
        "prediction_text": prediction.answer,
        "retrieved_timestamps": list(prediction.retrieved_timestamps),
        "agent_actions": list(prediction.agent_actions),
    }


def evaluate_retrieval(
    *,
    gold_timestamps: list[float],
    retrieved_timestamps: list[float],
    gold_scenes: list[dict[str, float]],
    scene_hits: list[dict[str, float]],
    tolerance_sec: float,
    recall_k: int | None,
) -> dict[str, Any]:
    retrieved = retrieved_timestamps[:recall_k] if recall_k else retrieved_timestamps
    matched = []
    distances = []
    for gold in gold_timestamps:
        distance = _nearest_distance(gold, retrieved)
        if distance is not None:
            distances.append(distance)
        matched.append(distance is not None and distance <= tolerance_sec)

    recall = (sum(matched) / len(gold_timestamps)) if gold_timestamps else None
    timestamp_distance = mean(distances) if distances else None
    scene_accuracy = _scene_hit_accuracy(gold_scenes, scene_hits)
    if not gold_timestamps and not gold_scenes:
        passed: bool | None = None  # n/a — no retrieval ground truth on this case
    else:
        passed = True
        if gold_timestamps:
            passed = passed and bool(recall is not None and recall >= 1.0)
        if gold_scenes:
            passed = passed and bool(scene_accuracy is not None and scene_accuracy >= 1.0)
    return {
        "passed": passed,
        "recall_at_k": recall,
        "timestamp_distance": timestamp_distance,
        "scene_hit_accuracy": scene_accuracy,
        "gold_count": len(gold_timestamps),
        "retrieved_count": len(retrieved),
        "matched_gold": matched,
    }


def evaluate_answer(
    *,
    answer: str,
    retrieved_timestamps: list[float],
    required_keywords: list[str],
    forbidden_keywords: list[str],
    requires_uncertainty: bool,
) -> dict[str, Any]:
    text = answer.lower()
    required_hits = {
        keyword: keyword.lower() in text
        for keyword in required_keywords
    }
    forbidden_hits = [
        keyword for keyword in forbidden_keywords if keyword.lower() in text
    ]
    markers = extract_frame_markers(answer)
    invalid_markers = [
        marker
        for marker in markers
        if _marker_distance(marker, retrieved_timestamps) is None
        or _marker_distance(marker, retrieved_timestamps) > 0.65
    ]
    uncertainty_ok = True
    if requires_uncertainty:
        uncertainty_ok = any(marker in text for marker in UNCERTAINTY_MARKERS)
    answered = bool(answer.strip()) and all(required_hits.values())
    hallucination_free = not forbidden_hits
    citations_ok = bool(markers) and not invalid_markers if retrieved_timestamps else not markers
    passed = answered and hallucination_free and citations_ok and uncertainty_ok
    return {
        "passed": passed,
        "answered": answered,
        "hallucination_free": hallucination_free,
        "citation_correct": citations_ok,
        "uncertainty_ok": uncertainty_ok,
        "required_keyword_hits": required_hits,
        "forbidden_keyword_hits": forbidden_hits,
        "frame_markers": markers,
        "invalid_markers": invalid_markers,
    }


def evaluate_audiovisual_answer(
    *,
    answer: str,
    expected_keywords: list[str],
    forbidden_keywords: list[str],
    expected_citation_min: int | None,
    expected_citation_kinds: list[str],
    requires_uncertainty: bool = False,
) -> dict[str, Any]:
    """Deterministic Phase-E answer score for audiovisual cases.

    The rubric is intentionally simple and reproducible: all expected keywords
    must appear, forbidden terms must not appear, and the answer must include
    enough citation markers of the expected modalities.
    """
    text = answer.lower()
    keyword_hits = {
        keyword: keyword.lower() in text
        for keyword in expected_keywords
    }
    forbidden_hits = [
        keyword for keyword in forbidden_keywords if keyword.lower() in text
    ]
    citations = extract_evidence_markers(answer)
    expected_kinds = _parse_citation_kinds(expected_citation_kinds)
    citation_min = max(0, int(expected_citation_min or 0))
    citation_counts = {
        "frame": 0,
        "transcript": 0,
        "slide": 0,
    }
    for citation in citations:
        kind = str(citation.get("kind") or "")
        if kind in citation_counts:
            citation_counts[kind] += 1
    citation_kind_coverage = {
        kind: (
            (citation_counts["frame"] + citation_counts["slide"] > 0)
            if kind == "frame_or_slide"
            else citation_counts.get(kind, 0) > 0
        )
        for kind in expected_kinds
    }
    citation_count_ok = sum(citation_counts.values()) >= citation_min
    citation_kinds_ok = all(citation_kind_coverage.values())
    uncertainty_ok = True
    if requires_uncertainty:
        uncertainty_ok = any(marker in text for marker in UNCERTAINTY_MARKERS)
    answered = bool(answer.strip()) and all(keyword_hits.values())
    hallucination_free = not forbidden_hits
    citation_correct = citation_count_ok and citation_kinds_ok
    passed = answered and hallucination_free and citation_correct and uncertainty_ok
    return {
        "passed": passed,
        "answered": answered,
        "hallucination_free": hallucination_free,
        "citation_correct": citation_correct,
        "uncertainty_ok": uncertainty_ok,
        "expected_keyword_hits": keyword_hits,
        "required_keyword_hits": keyword_hits,
        "forbidden_keyword_hits": forbidden_hits,
        "citation_markers": citations,
        "citation_counts": citation_counts,
        "expected_citation_min": citation_min,
        "expected_citation_kinds": expected_kinds,
        "citation_count_ok": citation_count_ok,
        "citation_kind_coverage": citation_kind_coverage,
    }


def evaluate_agent_loop(
    *,
    agent_actions: list[str],
    expected_action: str | None,
    needs_expansion: bool,
    evidence_sufficiency: dict[str, Any],
) -> dict[str, Any]:
    action_set = set(agent_actions)
    expected_ok = True if not expected_action else expected_action in action_set
    expansion_actions = {"expand_temporal_evidence", "build_timeline", "retrieve_hypothesis_evidence"}
    expansion_ok = True if not needs_expansion else bool(action_set & expansion_actions)
    should_check_sufficiency = bool(expected_action or needs_expansion or evidence_sufficiency)
    sufficiency_ok = True
    if should_check_sufficiency:
        sufficiency_ok = "assess_evidence_sufficiency" in action_set or bool(evidence_sufficiency)
    if evidence_sufficiency.get("sufficient") is False:
        recommended = evidence_sufficiency.get("recommended_next_action")
        if recommended:
            sufficiency_ok = sufficiency_ok and recommended in action_set
    passed = expected_ok and expansion_ok and sufficiency_ok
    return {
        "passed": passed,
        "expected_action_ok": expected_ok,
        "expansion_ok": expansion_ok,
        "sufficiency_checked": sufficiency_ok,
        "actions": agent_actions,
    }


def summarize_results(
    case_results: list[dict[str, Any]],
    *,
    group_by: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    summary = _summary_block(case_results)
    if group_by is not None:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for item in case_results:
            key = group_by(item)
            buckets.setdefault(key, []).append(item)
        summary["groups"] = {
            name: _summary_block(items) for name, items in sorted(buckets.items())
        }
    return summary


def _summary_block(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for item in case_results if item["passed"])
    retrieval_recalls = [
        item["retrieval"]["recall_at_k"]
        for item in case_results
        if item["retrieval"]["recall_at_k"] is not None
    ]
    timestamp_distances = [
        item["retrieval"]["timestamp_distance"]
        for item in case_results
        if item["retrieval"]["timestamp_distance"] is not None
    ]
    judge_scores = [
        item["answer"]["llm_judge"]["score"]
        for item in case_results
        if isinstance(item["answer"].get("llm_judge"), dict)
        and isinstance(item["answer"]["llm_judge"].get("score"), (int, float))
    ]
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "retrieval_recall_at_k_mean": mean(retrieval_recalls) if retrieval_recalls else None,
        "timestamp_distance_mean": mean(timestamp_distances) if timestamp_distances else None,
        "answer_pass_rate": _section_pass_rate(case_results, "answer"),
        "agent_loop_pass_rate": _section_pass_rate(case_results, "agent_loop"),
        "retrieval_pass_rate": _section_pass_rate(case_results, "retrieval"),
        "llm_judge_score_mean": mean(judge_scores) if judge_scores else None,
        "llm_judge_count": len(judge_scores),
    }


def group_by_case_prefix(item: dict[str, Any]) -> str:
    """Default grouper: text before the first '-' in case_id, or 'misc'."""
    case_id = str(item.get("case_id", "")) or "misc"
    return case_id.split("-", 1)[0] or "misc"


def extract_frame_markers(answer: str) -> list[float]:
    return [float(match.group(1)) for match in FRAME_MARKER_RE.finditer(answer)]


def extract_evidence_markers(answer: str) -> list[dict[str, Any]]:
    markers: list[tuple[int, dict[str, Any]]] = []
    for match in FRAME_MARKER_RE.finditer(answer or ""):
        timestamp = float(match.group(1))
        markers.append((
            match.start(),
            {
                "kind": "frame",
                "t_start": timestamp,
                "t_end": timestamp,
                "raw": match.group(0),
            },
        ))
    for match in TRANSCRIPT_MARKER_RE.finditer(answer or ""):
        start = float(match.group(1))
        end = float(match.group(2))
        if end < start:
            start, end = end, start
        markers.append((
            match.start(),
            {
                "kind": "transcript",
                "t_start": start,
                "t_end": end,
                "raw": match.group(0),
            },
        ))
    for match in SLIDE_MARKER_RE.finditer(answer or ""):
        timestamp = float(match.group(1))
        markers.append((
            match.start(),
            {
                "kind": "slide",
                "t_start": timestamp,
                "t_end": timestamp,
                "raw": match.group(0),
            },
        ))
    return [payload for _, payload in sorted(markers, key=lambda item: item[0])]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                items.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return items


def _nearest_distance(target: float, timestamps: Iterable[float]) -> float | None:
    distances = [abs(float(target) - float(timestamp)) for timestamp in timestamps]
    return min(distances) if distances else None


def _marker_distance(target: float, timestamps: Iterable[float]) -> float | None:
    return _nearest_distance(target, timestamps)


def _is_audiovisual_case(case: EvalCase) -> bool:
    return bool(
        case.modality_tag
        or case.expected_keywords
        or case.expected_citation_kinds
        or case.expected_citation_min is not None
    )


def _parse_citation_kinds(values: Any) -> list[str]:
    aliases = {
        "frame": "frame",
        "frames": "frame",
        "visual": "frame",
        "frame_or_slide": "frame_or_slide",
        "visual_evidence": "frame_or_slide",
        "transcript": "transcript",
        "transcripts": "transcript",
        "audio": "transcript",
        "speech": "transcript",
        "slide": "slide",
        "slides": "slide",
        "ppt": "slide",
        "ocr": "slide",
    }
    if isinstance(values, str):
        raw_values = [part.strip() for part in values.split(",")]
    else:
        raw_values = list(values or [])
    kinds: list[str] = []
    for value in raw_values:
        key = str(value).strip().lower()
        kind = aliases.get(key)
        if kind and kind not in kinds:
            kinds.append(kind)
    return kinds


def _scene_hit_accuracy(
    gold_scenes: list[dict[str, float]],
    scene_hits: list[dict[str, float]],
) -> float | None:
    if not gold_scenes:
        return None
    matched = 0
    for gold in gold_scenes:
        if any(_intervals_overlap(gold, hit) for hit in scene_hits):
            matched += 1
    return matched / len(gold_scenes)


def _intervals_overlap(left: dict[str, float], right: dict[str, float]) -> bool:
    return max(float(left["start"]), float(right["start"])) <= min(
        float(left["end"]),
        float(right["end"]),
    )


def _section_pass_rate(case_results: list[dict[str, Any]], section: str) -> float | None:
    """Pass rate over cases where this section is applicable.

    Cases whose section reports ``passed is None`` are treated as n/a — excluded
    from both numerator and denominator. Returns None if every case is n/a.
    """
    applicable = [
        item for item in case_results
        if (item.get(section) or {}).get("passed") is not None
    ]
    if not applicable:
        return None
    return sum(1 for item in applicable if item[section]["passed"]) / len(applicable)


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = (
    "You grade a video-QA model's free-form answer against a reference answer. "
    "The question may be in English or Chinese; treat both equally. Score on "
    "semantic equivalence, not surface wording. A frame citation like [FRAME:t=12.3] "
    "is metadata and should NOT lower your score. Return STRICT JSON with keys: "
    '"correct" (boolean), "score" (integer 0-5: 0=wrong, 3=partial, 5=fully correct), '
    '"justification" (one short sentence). Do not add any text outside the JSON.'
)


@dataclass
class JudgeClient:
    """Lightweight chat-completions client for LLM-as-judge.

    Construct via from_settings() to honor the JUDGE_* -> VLM_API_* fallback chain.
    """

    base_url: str
    api_key: str
    model: str
    timeout: int = 120
    temperature: float = 0.0
    max_output_tokens: int = 512
    source: str = "judge"  # "judge" or "vlm_fallback" — surfaced in logs

    @classmethod
    def from_settings(cls, settings: Any) -> "JudgeClient | None":
        key = (settings.judge_api_key or "").strip()
        if key:
            base = (settings.judge_api_base_url or "").strip() or (settings.vlm_api_base_url or "").strip()
            model = (settings.judge_model_name or "").strip() or (settings.vlm_model_name or "").strip()
            if not base or not model:
                logger.warning("JUDGE_API_KEY set but JUDGE_API_BASE_URL/MODEL missing; judge disabled.")
                return None
            return cls(
                base_url=base.rstrip("/"),
                api_key=key,
                model=model,
                timeout=int(settings.judge_api_timeout or 120),
                temperature=float(settings.judge_temperature or 0.0),
                max_output_tokens=int(settings.judge_max_output_tokens or 512),
                source="judge",
            )
        vlm_key = (settings.vlm_api_key or "").strip()
        if vlm_key:
            logger.warning(
                "JUDGE_API_KEY is empty; falling back to VLM_API_* for the judge. "
                "Self-judging bias is likely — set JUDGE_API_KEY to a different "
                "provider/model before publishing eval numbers."
            )
            return cls(
                base_url=(settings.vlm_api_base_url or "").rstrip("/"),
                api_key=vlm_key,
                model=settings.vlm_model_name,
                timeout=int(settings.vlm_api_timeout or 120),
                temperature=float(settings.judge_temperature or 0.0),
                max_output_tokens=int(settings.judge_max_output_tokens or 512),
                source="vlm_fallback",
            )
        logger.info("Neither JUDGE_API_KEY nor VLM_API_KEY is set; LLM judge disabled.")
        return None

    def grade(self, *, question: str, reference: str, answer: str) -> dict[str, Any]:
        import httpx

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference or '(no reference provided)'}\n\n"
            f"Model answer:\n{answer or '(empty)'}\n\n"
            "Reply with the JSON object only."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            return {"correct": False, "score": None, "justification": "", "error": f"http: {exc}"}
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            return {"correct": False, "score": None, "justification": "", "error": f"shape: {exc}"}
        return _parse_judge_json(text)


def _parse_judge_json(text: str) -> dict[str, Any]:
    snippet = text.strip()
    # Strip ``` fences if the model adds them.
    if snippet.startswith("```"):
        snippet = snippet.strip("`")
        if snippet.lower().startswith("json"):
            snippet = snippet[4:]
        snippet = snippet.strip()
    # Try to find the first balanced JSON object if there's chatter around it.
    if not snippet.startswith("{"):
        match = re.search(r"\{.*\}", snippet, re.DOTALL)
        if match:
            snippet = match.group(0)
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError as exc:
        return {"correct": False, "score": None, "justification": text[:200], "error": f"json: {exc}"}
    correct = bool(data.get("correct"))
    score_raw = data.get("score")
    score: float | None
    if isinstance(score_raw, bool):
        score = None
    elif isinstance(score_raw, (int, float)):
        score = float(score_raw)
    else:
        score = None
    return {
        "correct": correct,
        "score": score,
        "justification": str(data.get("justification") or ""),
    }


class JudgeCache:
    """Append-only JSONL disk cache so reruns don't burn tokens."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = record.get("key")
                    if key:
                        self._entries[str(key)] = record.get("value", {})

    @staticmethod
    def _key(case_id: str, model: str, answer: str) -> str:
        digest = hashlib.sha1(f"{case_id}\x00{model}\x00{answer}".encode("utf-8")).hexdigest()
        return digest

    def get(self, case_id: str, model: str, answer: str) -> dict[str, Any] | None:
        return self._entries.get(self._key(case_id, model, answer))

    def put(self, case_id: str, model: str, answer: str, value: dict[str, Any]) -> None:
        key = self._key(case_id, model, answer)
        self._entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False))
            handle.write("\n")


class PredictionCache:
    """Append-only JSONL cache for full agent predictions per case.

    Key bundles (case_id, orchestrator_model, prompt_fingerprint, video_id,
    agent_code_version) so prompt / model / agent-code edits invalidate
    naturally. Value is a full EvalPrediction-shaped dict.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = record.get("key")
                    if key:
                        self._entries[str(key)] = record.get("value", {})

    @staticmethod
    def make_key(
        *,
        case_id: str,
        model: str,
        prompt_fingerprint: str,
        video_id: str,
        agent_code_version: str,
    ) -> str:
        payload = "\x00".join(
            [case_id, model, prompt_fingerprint, video_id, agent_code_version]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False))
            handle.write("\n")

    @staticmethod
    def prediction_to_dict(prediction: "EvalPrediction") -> dict[str, Any]:
        return {
            "case_id": prediction.case_id,
            "retrieved_timestamps": list(prediction.retrieved_timestamps),
            "scene_hits": [dict(scene) for scene in prediction.scene_hits],
            "retrieved_transcripts": [dict(item) for item in prediction.retrieved_transcripts],
            "retrieved_slides": [dict(item) for item in prediction.retrieved_slides],
            "answer": prediction.answer,
            "agent_actions": list(prediction.agent_actions),
            "evidence_sufficiency": dict(prediction.evidence_sufficiency),
            "grounding_report": dict(prediction.grounding_report),
        }

    @staticmethod
    def prediction_from_dict(case_id: str, data: dict[str, Any]) -> "EvalPrediction":
        return EvalPrediction(
            case_id=case_id,
            retrieved_timestamps=[float(v) for v in data.get("retrieved_timestamps", [])],
            scene_hits=[
                {"start": float(s["start"]), "end": float(s["end"])}
                for s in data.get("scene_hits", [])
                if "start" in s and "end" in s
            ],
            retrieved_transcripts=[
                dict(item) for item in data.get("retrieved_transcripts", [])
                if isinstance(item, dict)
            ],
            retrieved_slides=[
                dict(item) for item in data.get("retrieved_slides", [])
                if isinstance(item, dict)
            ],
            answer=str(data.get("answer") or ""),
            agent_actions=[str(v) for v in data.get("agent_actions", [])],
            evidence_sufficiency=dict(data.get("evidence_sufficiency", {}) or {}),
            grounding_report=dict(data.get("grounding_report", {}) or {}),
        )


def evaluate_answer_llm(
    case: EvalCase,
    prediction: EvalPrediction,
    *,
    judge: JudgeClient,
    cache: JudgeCache | None = None,
) -> dict[str, Any]:
    answer = prediction.answer or ""
    if not case.reference_answer:
        return {
            "correct": False,
            "score": None,
            "justification": "no reference_answer on case",
            "skipped": True,
        }
    if cache is not None:
        cached = cache.get(case.case_id, judge.model, answer)
        if cached is not None:
            return {**cached, "cached": True}
    result = judge.grade(
        question=case.question,
        reference=case.reference_answer,
        answer=answer,
    )
    if cache is not None and "error" not in result:
        cache.put(case.case_id, judge.model, answer, result)
    return result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(
    report: dict[str, Any],
    path: str | Path,
    *,
    worst_n: int = 10,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary", {})
    groups = summary.get("groups", {}) or {}
    results = report.get("results", []) or []
    lines: list[str] = []
    lines.append("# Mr. Big-Eye eval report")
    lines.append("")
    lines.append("## Global summary")
    lines.append("")
    lines.extend(_summary_markdown_table(summary))
    if groups:
        lines.append("")
        lines.append("## Per-group summary")
        lines.append("")
        lines.append(_groups_markdown_table(groups))
    failures = [item for item in results if not item.get("passed")]
    if failures:
        lines.append("")
        lines.append(f"## Worst {min(worst_n, len(failures))} failures")
        lines.append("")
        lines.append(_failures_markdown_table(failures[:worst_n]))
    missing = report.get("missing_predictions") or []
    if missing:
        lines.append("")
        lines.append("## Missing predictions")
        lines.append("")
        for case_id in missing:
            lines.append(f"- `{case_id}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_markdown_table(summary: dict[str, Any]) -> list[str]:
    keys = [
        ("total", "Total"),
        ("passed", "Passed"),
        ("pass_rate", "Pass rate"),
        ("retrieval_pass_rate", "Retrieval pass"),
        ("answer_pass_rate", "Answer pass"),
        ("agent_loop_pass_rate", "Agent loop pass"),
        ("retrieval_recall_at_k_mean", "Recall@k mean"),
        ("timestamp_distance_mean", "Timestamp dist mean"),
        ("llm_judge_score_mean", "LLM judge score mean"),
        ("llm_judge_count", "LLM judge count"),
    ]
    rows = ["| metric | value |", "| --- | --- |"]
    for key, label in keys:
        if key in summary:
            rows.append(f"| {label} | {_fmt(summary[key])} |")
    return rows


def _groups_markdown_table(groups: dict[str, dict[str, Any]]) -> str:
    headers = [
        "group",
        "total",
        "pass_rate",
        "retrieval",
        "answer",
        "agent",
        "recall@k",
        "ts_dist",
        "judge_score",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, block in groups.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(block.get("total")),
                    _fmt(block.get("pass_rate")),
                    _fmt(block.get("retrieval_pass_rate")),
                    _fmt(block.get("answer_pass_rate")),
                    _fmt(block.get("agent_loop_pass_rate")),
                    _fmt(block.get("retrieval_recall_at_k_mean")),
                    _fmt(block.get("timestamp_distance_mean")),
                    _fmt(block.get("llm_judge_score_mean")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _failures_markdown_table(failures: list[dict[str, Any]]) -> str:
    lines = [
        "| case_id | failing section(s) | question |",
        "| --- | --- | --- |",
    ]
    for item in failures:
        sections = []
        for name in ("retrieval", "answer", "agent_loop"):
            block = item.get(name) or {}
            if block.get("passed") is False:
                sections.append(name)
        question = (item.get("question") or "").replace("|", "\\|").replace("\n", " ")
        if len(question) > 120:
            question = question[:117] + "..."
        lines.append(
            f"| `{item.get('case_id', '')}` | {', '.join(sections) or '-'} | {question} |"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


# ---------------------------------------------------------------------------
# Module CLI: python -m app.eval_harness --dataset audiovisual --n 20
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


async def _main_async(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run deterministic audiovisual QA evals.")
    parser.add_argument(
        "--dataset",
        choices=["audiovisual"],
        help="Named eval dataset. Currently supports the Phase-E audiovisual schema.",
    )
    parser.add_argument("--cases", help="JSONL cases. Overrides the dataset default path.")
    parser.add_argument(
        "--predictions",
        help="Optional JSONL predictions. If omitted, run the local graph against ingested videos.",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of cases to run from the top of the file.")
    parser.add_argument("--output", default="data/eval/audiovisual_report.json")
    parser.add_argument("--markdown", default=None, help="Markdown report path; defaults to <output>.md")
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--recall-k", type=int, default=None)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit non-zero if overall pass_rate is below this threshold.",
    )
    parser.add_argument(
        "--prediction-cache",
        default="data/eval/audiovisual_prediction_cache.jsonl",
        help="JSONL cache for local graph predictions; pass empty string to disable.",
    )
    parser.add_argument(
        "--per-case-delay-sec",
        type=float,
        default=0.0,
        help="Sleep this many seconds between local graph cases.",
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
    args = parser.parse_args(argv)

    if not args.dataset and not args.cases:
        parser.error("provide --dataset audiovisual or --cases")

    cases_path = Path(args.cases) if args.cases else _default_dataset_cases_path(args.dataset)
    cases = load_cases(cases_path)
    if args.n is not None and args.n > 0:
        cases = cases[: args.n]

    if args.predictions:
        predictions = load_predictions(args.predictions)
    else:
        prediction_cache = PredictionCache(args.prediction_cache) if args.prediction_cache else None
        predictions = await _run_local_predictions(
            cases,
            prediction_cache=prediction_cache,
            per_case_delay_sec=args.per_case_delay_sec,
        )

    from app.config import settings as _settings
    judge = JudgeClient.from_settings(_settings)
    judge_cache = JudgeCache("data/eval/judge_cache.jsonl") if judge is not None else None
    if judge is not None:
        logger.info(
            "LLM judge enabled: model=%s source=%s base=%s",
            judge.model, judge.source, judge.base_url,
        )

    results: list[dict[str, Any]] = []
    missing: list[str] = []
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

    group_by = group_by_case_prefix if args.group_by_prefix else None
    summary = summarize_results(results, group_by=group_by)
    report = {
        "run_meta": _cli_run_meta(args, cases_path, cases, results),
        "summary": summary,
        "missing_predictions": missing,
        "results": results,
    }
    write_json(args.output, report)
    markdown_path = args.markdown or str(Path(args.output).with_suffix(".md"))
    write_markdown_report(report, markdown_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")
    if missing:
        print(f"missing predictions: {', '.join(missing)}", file=sys.stderr)
        return 2
    return 1 if summary["pass_rate"] < args.fail_under else 0


def _default_dataset_cases_path(dataset: str | None) -> Path:
    if dataset != "audiovisual":
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    candidates = [
        Path("eval/audiovisual/questions.jsonl"),
        Path("tests/fixtures/eval_cases_audiovisual_seed.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


async def _run_local_predictions(
    cases: list[EvalCase],
    *,
    prediction_cache: PredictionCache | None,
    per_case_delay_sec: float,
) -> dict[str, EvalPrediction]:
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from app.cache import get_video_status
    from app.config import settings
    from app.eval_fingerprint import AGENT_CODE_VERSION, prompt_fingerprint
    from app.graph import build_graph

    app_graph = build_graph(InMemorySaver(), InMemoryStore(), memory_manager=_NoopMemoryManager())
    fingerprint = prompt_fingerprint() if prediction_cache is not None else ""
    orch_name = (settings.orchestrator_model_name or settings.vlm_model_name or "")
    vlm_name = (settings.vlm_model_name or "")
    model_name = f"{orch_name}|vlm={vlm_name}"
    predictions: dict[str, EvalPrediction] = {}

    for case in cases:
        if not case.video_id or get_video_status(case.video_id) != "done":
            raise RuntimeError(
                f"Video cache is not ready for {case.video_id or '(empty video_id)'}. "
                "Preprocess it first or pass --predictions."
            )
        cache_key: str | None = None
        if prediction_cache is not None:
            cache_key = PredictionCache.make_key(
                case_id=case.case_id,
                model=model_name,
                prompt_fingerprint=fingerprint,
                video_id=case.video_id,
                agent_code_version=AGENT_CODE_VERSION,
            )
            cached = prediction_cache.get(cache_key)
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
                "retrieved_transcripts": [],
                "retrieved_slides": [],
                "retrieval_plan": {},
                "timeline": [],
                "hypotheses": [],
                "evidence_sufficiency": {},
                "draft_answer": "",
                "observer_notes": [],
                "grounding_report": {},
            },
            config={"configurable": {"thread_id": f"eval-{case.case_id}"}},
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
            retrieved_transcripts=[
                dict(item) for item in state.get("retrieved_transcripts", [])
                if isinstance(item, dict)
            ],
            retrieved_slides=[
                dict(item) for item in state.get("retrieved_slides", [])
                if isinstance(item, dict)
            ],
            answer=_last_assistant_message(state.get("messages", [])),
            agent_actions=_agent_actions(state.get("messages", [])),
            evidence_sufficiency=evidence_sufficiency,
            grounding_report=state.get("grounding_report", {}),
        )
        predictions[case.case_id] = prediction
        if prediction_cache is not None and cache_key is not None:
            prediction_cache.put(cache_key, PredictionCache.prediction_to_dict(prediction))
        if per_case_delay_sec > 0:
            await asyncio.sleep(per_case_delay_sec)
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
    async def ainvoke(self, payload: Any, config: Any = None) -> None:
        return None


def _cli_run_meta(
    args: argparse.Namespace,
    cases_path: Path,
    cases: list[EvalCase],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    from app.config import settings
    from app.eval_fingerprint import AGENT_CODE_VERSION, prompt_fingerprint

    return {
        "dataset": args.dataset or "custom",
        "dataset_path": str(cases_path),
        "n": args.n,
        "n_cases_total": len(cases),
        "n_cases_predicted": len(results),
        "vlm_model_name": settings.vlm_model_name,
        "orchestrator_model_name": settings.orchestrator_model_name or settings.vlm_model_name,
        "prompt_fingerprint": prompt_fingerprint(),
        "agent_code_version": AGENT_CODE_VERSION,
        "tolerance_sec": args.tolerance_sec,
        "recall_k": args.recall_k,
        "judge_enabled": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
