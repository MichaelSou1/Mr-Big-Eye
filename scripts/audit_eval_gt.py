"""Audit Video-MME eval ground truth against the actual ingested video evidence.

Motivation: vme-377-2 asks for the recommended shoe-cleaner brand with options
SNEAKER LAB / JASON MARKK / LEATHER HONEY / CREP PROTECT and a gold answer of
SNEAKER LAB, but the ingested video's OCR + transcript only ever mention KIWI.
That is a wrong-video / mislabeled case, not an agent failure. This script finds
the same class of problem across the whole set so we don't optimize on a broken
metric.

Heuristic (high precision, text-readable cases only): for questions whose answer
should be readable on screen or spoken (OCR/text questions, or short named-entity
answers), the gold option's text should appear in the video's OCR/transcript/
caption evidence. If it is wholly absent, the case is suspect. If a *distractor*
option appears while the gold one does not, that is a stronger signal.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "audiovisual" / "questions.jsonl"
MANIFEST = ROOT / "eval" / "audiovisual" / "video_manifest.json"
CACHE = ROOT / "data" / "cache"
V22B = ROOT / "data" / "eval" / "audiovisual_report_v22b.json"

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "is", "are",
    "was", "were", "be", "and", "or", "that", "this", "it", "its", "by", "as",
    "from", "than", "all", "any", "no", "not", "one", "two", "three", "yes",
}


def norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9一-鿿]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_option(text: str) -> str:
    return re.sub(r"^[a-e][\.\)]\s*", "", str(text).strip(), flags=re.I).strip().rstrip(".")


def load_evidence(video_id: str) -> tuple[str, str]:
    """Return (normalized_text, spaceless_text) from slides + transcript + captions."""
    parts: list[str] = []
    d = CACHE / video_id
    for fn in ("slides.jsonl", "transcripts.jsonl", "captions.jsonl"):
        p = d / fn
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("text", "ocr", "caption", "document"):
                if o.get(key):
                    parts.append(str(o[key]))
    blob = norm(" ".join(parts))
    return blob, blob.replace(" ", "")


def phrase_present(phrase: str, evidence: str, evidence_nospace: str) -> bool:
    p = norm(phrase)
    if not p:
        return False
    if p in evidence:
        return True
    ns = p.replace(" ", "")
    return len(ns) >= 5 and ns in evidence_nospace


def is_text_readable(case: dict) -> bool:
    qtype = (case.get("question_type") or "").lower()
    kinds = case.get("expected_citation_kinds") or []
    if "ocr" in qtype or qtype == "text_ocr":
        return True
    if kinds == ["slide"]:
        return True
    q = (case.get("question") or "").lower()
    return any(k in q for k in (
        "brand", "logo", "name", "title", "text", "written", "sign", "label",
        "word", "number", "letter", "spelled", "displayed",
    ))


def main() -> int:
    cases = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8") if l.strip()]
    v22b = {}
    if V22B.exists():
        for rec in json.load(V22B.open(encoding="utf-8")).get("results", []):
            v22b[rec["case_id"]] = rec.get("passed")

    none_present: list[dict] = []      # gold absent, no option present
    distractor_present: list[dict] = []  # gold absent but a distractor present (strong)
    not_checkable = 0
    checked = 0

    for case in cases:
        if not is_text_readable(case):
            continue
        options = case.get("source_meta", {}).get("options") or []
        gold = strip_option(case.get("reference_answer") or "")
        gold_tokens = [t for t in norm(gold).split() if t not in STOPWORDS and len(t) > 1]
        # Only enforce on short, named-entity-style answers.
        if not gold or len(gold_tokens) > 5:
            not_checkable += 1
            continue
        ev, ev_ns = load_evidence(case["video_id"])
        if not ev:
            not_checkable += 1
            continue
        checked += 1
        gold_in = phrase_present(gold, ev, ev_ns)
        if gold_in:
            continue
        present_distractors = [
            strip_option(opt) for opt in options
            if strip_option(opt).lower() != gold.lower()
            and phrase_present(strip_option(opt), ev, ev_ns)
        ]
        row = {
            "case_id": case["question_id"],
            "video_id": case["video_id"],
            "youtube_id": case.get("source_meta", {}).get("youtube_id"),
            "question": (case.get("question") or "").split("\n")[0][:90],
            "gold": gold,
            "present_distractors": present_distractors,
            "v22b_passed": v22b.get(case["question_id"]),
        }
        if present_distractors:
            distractor_present.append(row)
        else:
            none_present.append(row)

    def dump(title: str, rows: list[dict]) -> None:
        print(f"\n{'='*72}\n{title}  (n={len(rows)})\n{'='*72}")
        for r in sorted(rows, key=lambda x: (x["v22b_passed"] is not False, x["case_id"])):
            mark = "FAIL" if r["v22b_passed"] is False else ("pass" if r["v22b_passed"] else " -- ")
            print(f"[{mark}] {r['case_id']:<10} yt={r['youtube_id']}  gold={r['gold']!r}")
            print(f"        Q: {r['question']}")
            if r["present_distractors"]:
                print(f"        distractor present in video: {r['present_distractors']}")

    print(f"text-readable cases checked={checked}, not_checkable(skipped)={not_checkable}")
    dump("STRONG SUSPECT: gold absent, a DISTRACTOR option present in video", distractor_present)
    dump("SUSPECT: gold answer text absent from all video evidence", none_present)
    # how many suspects were v22b failures (i.e. polluting the metric)
    susp = distractor_present + none_present
    fails = [r for r in susp if r["v22b_passed"] is False]
    print(f"\nSUMMARY: {len(susp)} suspect cases; {len(fails)} of them are v22b FAILS "
          f"(metric pollution): {[r['case_id'] for r in fails]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
