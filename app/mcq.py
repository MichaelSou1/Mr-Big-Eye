from __future__ import annotations

import re
from typing import Any


MCQ_OPTION_RE = re.compile(
    r"(?ms)^\s*([A-E])[\).]\s+(.+?)(?=^\s*[A-E][\).]\s+|\Z)"
)
EVENT_ITEM_RE = re.compile(
    r"(?ms)^\s*\(([a-z])\)\s+(.+?)(?=^\s*\([a-z]\)\s+|\n\s*Candidates:|\Z)"
)


def parse_candidates(question: str) -> list[dict[str, str]]:
    """Parse Candidates blocks formatted as A) option text."""
    if "Candidates:" not in (question or ""):
        return []
    block = (question or "").split("Candidates:", 1)[1]
    block = re.split(
        r"\n\s*\n\s*(?:Answer the question|Reference key frames|请回答|回答问题)",
        block,
        maxsplit=1,
    )[0]
    out: list[dict[str, str]] = []
    for match in MCQ_OPTION_RE.finditer(block):
        text = _clean_option_text(match.group(2))
        if text:
            out.append({"label": match.group(1).upper(), "text": text})
    return out


def parse_labeled_events(question: str) -> list[dict[str, str]]:
    """Parse temporal event bullets such as '(a) Sunken cities.'."""
    prefix = (question or "").split("Candidates:", 1)[0]
    out: list[dict[str, str]] = []
    for match in EVENT_ITEM_RE.finditer(prefix):
        text = _clean_option_text(match.group(2))
        if text:
            out.append({"label": match.group(1).lower(), "text": text})
    return out


def selected_candidate(answer: str, candidates: list[dict[str, str]]) -> dict[str, str] | None:
    """Return the MCQ candidate selected in an answer, if one is explicit."""
    if not candidates:
        return None
    labels = {item["label"].upper(): item for item in candidates}
    match = re.search(r"\b(?:answer|correct answer|选项|答案)\s*[:：]?\s*\**([A-E])\**\s*[\).]", answer or "", re.I)
    if not match:
        match = re.search(r"\b([A-E])\s*[\).]\s+", answer or "")
    if match:
        return labels.get(match.group(1).upper())
    normalized_answer = normalize_text(answer)
    for item in candidates:
        option = normalize_text(item["text"])
        if option and option in normalized_answer:
            return item
    return None


def resolve_temporal_option(
    question: str,
    candidate_timeline: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve '(a)(b)(c)' style MCQ options from candidate first timestamps."""
    if not candidate_timeline:
        return None
    found = [
        item for item in candidate_timeline
        if item.get("first_timestamp") is not None and item.get("status") == "found"
    ]
    if len(found) != len(candidate_timeline):
        return None
    found.sort(key=lambda item: float(item.get("first_timestamp", 0.0)))
    order = [str(item.get("label") or "").lower() for item in found if item.get("label")]
    if len(order) != len(candidate_timeline):
        return None
    for candidate in parse_candidates(question):
        sequence = parse_order_sequence(candidate["text"])
        if sequence and sequence == order:
            return {**candidate, "order": order}
    return None


def parse_order_sequence(text: str) -> list[str]:
    return [match.group(1).lower() for match in re.finditer(r"\(([a-z])\)", text or "", re.I)]


def score_mcq_options(question: str, evidence: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    """Lightweight lexical support table for factual MCQs."""
    candidates = parse_candidates(question)
    if not candidates:
        return []
    evidence_text = evidence if isinstance(evidence, str) else " ".join(
        str(item.get("text") or item.get("caption") or "") for item in evidence or []
    )
    normalized = normalize_text(evidence_text)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        tokens = content_tokens(candidate["text"])
        support = sum(1 for token in tokens if token in normalized)
        rows.append(
            {
                "label": candidate["label"],
                "text": candidate["text"],
                "support_score": support,
                "status": "support" if support >= min(2, max(1, len(tokens))) else "unknown",
            }
        )
    return rows


def detect_option_contradiction(
    answer: str,
    candidates: list[dict[str, str]],
) -> dict[str, Any] | None:
    selected = selected_candidate(answer, candidates)
    if selected is None:
        return None
    labels = "|".join(re.escape(item["label"]) for item in candidates)
    label_mentions = re.findall(rf"\b({labels})\s*[\).]", answer or "", flags=re.I)
    if len({item.upper() for item in label_mentions}) > 1:
        return {"selected": selected, "reason": "multiple_candidate_labels"}
    selected_tokens = set(content_tokens(selected["text"]))
    refute_markers = (
        "not",
        "does not",
        "doesn't",
        "contradict",
        "contradicted",
        "false",
        "invalid",
        "not true",
        "not supported",
        "does not explicitly",
        "无法支持",
        "不支持",
        "错误",
    )
    for sentence in re.split(r"[\n。！？.!?]+", answer or ""):
        lowered = sentence.lower()
        if not any(marker in lowered for marker in refute_markers):
            continue
        normalized_sentence = normalize_text(sentence)
        token_hits = sum(1 for token in selected_tokens if token in normalized_sentence)
        mentions_label = re.search(rf"\b{re.escape(selected['label'])}\b", sentence, re.I)
        if mentions_label or token_hits >= min(2, len(selected_tokens)):
            return {"selected": selected, "reason": "selected_option_refuted", "sentence": sentence.strip()}
    return None


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value: Any) -> list[str]:
    normalized = normalize_text(value)
    return [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in {"the", "and", "for", "with", "from", "that", "this"}
    ]


def text_contains_option(text: str, option_text: str) -> bool:
    normalized_text = normalize_text(text)
    tokens = content_tokens(option_text)
    if not tokens:
        return False
    return all(token in normalized_text for token in tokens[:6])


def _clean_option_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().rstrip()
