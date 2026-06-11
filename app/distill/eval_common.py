"""Shared evaluation vocabulary + a TEXT-AWARE grader (Spec §0, §0.5).

The headline numbers are at the ±2% scale, while letter-only MCQ grading lets
"letter-luck" cases (the answer explicitly cites option X's letter but its prose
describes a different option) score as correct. This module provides a grader that
demotes those cases — conservatively: a case is flipped to incorrect ONLY when the
cited letter matches gold yet the answer's content clearly endorses a *different*
option. It reuses app/mcq.py so it stays consistent with the rest of the pipeline.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from app.distill.filter_consistency import _answer_matches
from app.eval_fingerprint import AGENT_CODE_VERSION, prompt_fingerprint
from app.mcq import (
    normalize_text,
    parse_candidates,
    selected_candidate,
    text_contains_option,
)

# Spec §0: the five comparable methods.
METHODS = (
    "free_form",
    "self_reflect",
    "orch_reflect_blind",
    "orch_reflect_sighted",
    "agent_retrieval",
)
DEFAULT_N_FRAMES = 16


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def _gold_candidate(gold: str, cands: list[dict[str, str]]) -> dict[str, str] | None:
    """Map a gold answer string to its candidate (bare letter → letter → text)."""
    g = str(gold).strip()
    # bare single letter, e.g. "B" or "B." or "(B)"
    bare = re.fullmatch(r"\(?([A-Ea-e])\)?\.?", g)
    if bare:
        lab = bare.group(1).upper()
        for cand in cands:
            if cand["label"].upper() == lab:
                return cand
    c = selected_candidate(gold, cands)
    if c is not None:
        return c
    gnorm = normalize_text(gold).rstrip(".")
    for cand in cands:
        cn = normalize_text(cand["text"]).rstrip(".")
        if gnorm and (gnorm in cn or cn in gnorm):
            return cand
    return None


def relaxed_match(pred: str, gold: str) -> bool:
    """Open-ended (e.g. ChartQA): 5% tolerance for numbers, normalized substring for text."""
    g = str(gold).strip()
    nums = re.findall(r"-?\d+\.?\d*", str(pred).replace(",", ""))
    try:
        gv = float(g.replace(",", "").replace("%", ""))
        return any(abs(float(p) - gv) <= abs(gv) * 0.05 + 1e-6 for p in nums)
    except ValueError:
        gn = normalize_text(g)
        return bool(gn) and gn in normalize_text(pred)


def grade_textaware(question: str, gold: str, answer: str) -> dict[str, Any]:
    """Returns {correct, letter_luck, mcq, ...}. Text-aware MCQ grading (Spec §0.5.1)."""
    cands = parse_candidates(question)
    if not cands:
        return {"correct": relaxed_match(answer, gold), "letter_luck": False, "mcq": False}

    gold_c = _gold_candidate(gold, cands)
    if gold_c is None:
        return {"correct": _answer_matches(question, gold, answer), "letter_luck": False,
                "mcq": True, "gold_resolved": False}
    sel = selected_candidate(answer, cands)
    letter_ok = bool(sel and sel["label"] == gold_c["label"])

    gold_text_present = text_contains_option(answer, gold_c["text"])
    other_text_present = any(
        c["label"] != gold_c["label"] and text_contains_option(answer, c["text"])
        for c in cands
    )
    # letter says gold, but the prose endorses a different option only.
    letter_luck = bool(letter_ok and not gold_text_present and other_text_present)
    correct = bool(letter_ok and not letter_luck)
    return {
        "correct": correct,
        "letter_luck": letter_luck,
        "letter_ok": bool(letter_ok),
        "gold_text_present": bool(gold_text_present),
        "mcq": True,
        "gold_resolved": True,
    }


# ---------------------------------------------------------------------------
# Per-run config fingerprint (Spec §0 determinism rule)
# ---------------------------------------------------------------------------
def config_fingerprint(
    *,
    dataset: str,
    split_hash: str,
    model_id: str,
    method: str,
    n_frames: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """A reproducible fingerprint dict for the result store. `fp` is a short hash of all of it."""
    payload = {
        "dataset": dataset,
        "split_hash": split_hash,
        "model_id": model_id,
        "method": method,
        "n_frames": n_frames,
        "decode": {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
        "seed": seed,
        "prompt_fp": prompt_fingerprint(),
        "code_version": AGENT_CODE_VERSION,
    }
    blob = repr(sorted(payload.items())).encode("utf-8")
    payload["fp"] = hashlib.sha1(blob).hexdigest()[:12]
    return payload


def case_set_hash(case_ids: list[str]) -> str:
    """Stable hash of a case-id set (for split_hash)."""
    blob = "\n".join(sorted(case_ids)).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
