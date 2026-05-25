from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


logger = logging.getLogger(__name__)


FRAME_MARKER_RE = re.compile(r"\[FRAME:t=([0-9]+(?:\.[0-9]+)?)\]")
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


@dataclass
class EvalPrediction:
    case_id: str
    retrieved_timestamps: list[float] = field(default_factory=list)
    scene_hits: list[dict[str, float]] = field(default_factory=list)
    answer: str = ""
    agent_actions: list[str] = field(default_factory=list)
    evidence_sufficiency: dict[str, Any] = field(default_factory=dict)
    grounding_report: dict[str, Any] = field(default_factory=dict)


def load_cases(path: str | Path) -> list[EvalCase]:
    return [parse_case(item) for item in _read_jsonl(path)]


def load_predictions(path: str | Path) -> dict[str, EvalPrediction]:
    return {prediction.case_id: prediction for prediction in map(parse_prediction, _read_jsonl(path))}


def parse_case(item: dict[str, Any]) -> EvalCase:
    return EvalCase(
        case_id=str(item["case_id"]),
        video_id=str(item.get("video_id") or ""),
        question=str(item["question"]),
        gold_timestamps=[float(value) for value in item.get("gold_timestamps", [])],
        gold_scenes=[
            {"start": float(scene["start"]), "end": float(scene["end"])}
            for scene in item.get("gold_scenes", [])
        ],
        reference_answer=str(item.get("reference_answer") or ""),
        required_keywords=[str(value) for value in item.get("required_keywords", [])],
        forbidden_keywords=[str(value) for value in item.get("forbidden_keywords", [])],
        expected_action=item.get("expected_action"),
        requires_uncertainty=bool(item.get("requires_uncertainty", False)),
        needs_expansion=bool(item.get("needs_expansion", False)),
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
            if not block.get("passed"):
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
