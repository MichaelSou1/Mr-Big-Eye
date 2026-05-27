"""Convert sampled Video-MME questions to our EvalCase JSONL schema.

Inputs:
    data/hf_cache/videomme/test.parquet  — full Video-MME QA
    data/hf_cache/videomme/sampled_videos.json — list of 50 sampled YouTube videoIDs
    data/videomme_videos/{videoID}.mp4 — downloaded video files (for video_id hashing)

Output:
    eval/audiovisual/questions.jsonl — 150 EvalCase rows (50 videos × 3 questions)
    eval/audiovisual/video_manifest.json — videoID -> our hash-derived video_id mapping
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

CACHE = Path("/home/user/Mr-Big-Eye/data/hf_cache/videomme")
VIDEOS = Path("/home/user/Mr-Big-Eye/data/videomme_videos")
EVAL_DIR = Path("/home/user/Mr-Big-Eye/eval/audiovisual")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Video-MME task_type -> our modality_tag
TASK_TYPE_TO_MODALITY = {
    "Object Recognition": "visual",
    "Object Reasoning": "visual",
    "Action Recognition": "visual",
    "Action Reasoning": "joint",
    "Counting Problem": "visual",
    "Attribute Perception": "visual",
    "OCR Problems": "visual",
    "Spatial Reasoning": "visual",
    "Spatial Perception": "visual",
    "Temporal Reasoning": "joint",
    "Temporal Perception": "joint",
    "Information Synopsis": "overview",
}

STOPWORDS = {
    "the","a","an","of","in","on","at","to","for","with","is","are","was","were","be",
    "and","or","that","this","these","those","it","its","by","as","from","into","than",
    "all","any","both","each","every","most","some","no","not","one","two","three",
    "kinds","kind","same","number","largest","smallest","more","less","several",
}


def format_candidates(options: list[str]) -> str:
    lines = ["Candidates:"]
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        text = re.sub(r"^[A-E][\.\)]\s*", "", str(option)).strip()
        lines.append(f"{letter}) {text}")
    return "\n".join(lines)


def question_with_candidates(question: str, options: list[str]) -> str:
    text = str(question).strip()
    if "Candidates:" in text:
        return text
    return f"{text}\n\n{format_candidates(options)}"


def expected_citation_kinds(task_type: str, modality: str) -> list[str]:
    if modality == "audio":
        return ["transcript"]
    if modality == "overview":
        return ["transcript"]
    if task_type == "OCR Problems":
        return ["slide"]
    if modality == "joint":
        return ["transcript", "frame_or_slide"]
    return ["frame_or_slide"]


def extract_keywords(option_text: str, max_n: int = 3) -> list[str]:
    """Pull discriminating content tokens from an MCQ option's text."""
    # Strip leading "A. " / "(A) " etc.
    text = re.sub(r"^[A-D][\.\)]\s*", "", option_text).strip().rstrip(".")
    # Special case: sequence labels like "(a)(b)(c)" — keep as a single literal keyword
    if re.fullmatch(r"(\([a-zA-Z0-9]\)\s*)+", text.strip()):
        return [re.sub(r"\s+", "", text)]
    # Tokenize on word boundaries
    tokens = re.findall(r"[A-Za-z一-鿿][A-Za-z0-9一-鿿'-]*", text)
    kept: list[str] = []
    seen = set()
    for tok in tokens:
        if len(tok) < 2:  # drop single-char tokens (a/b/c sequence labels, articles)
            continue
        low = tok.lower()
        if low in STOPWORDS:
            continue
        if low in seen:
            continue
        seen.add(low)
        kept.append(tok)
        if len(kept) >= max_n:
            break
    if not kept:  # fallback: whole option text
        kept = [text]
    return kept


def compute_video_id(mp4_path: Path) -> str:
    h = hashlib.sha256()
    with mp4_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    df = pd.read_parquet(CACHE / "test.parquet")
    with (CACHE / "sampled_videos.json").open() as f:
        sampled = set(json.load(f))

    rows = df[df["videoID"].isin(sampled)].copy()
    # Sanity: expect 3 questions per sampled video = 150
    print(
        f"sampled videos: {len(sampled)}, matched rows: {len(rows)} "
        f"(expect {len(sampled)*3})",
        flush=True,
    )

    # Build YouTube videoID -> our content-hash video_id
    manifest: dict[str, dict] = {}
    missing = []
    for yt_id in sorted(sampled):
        mp4 = VIDEOS / f"{yt_id}.mp4"
        if not mp4.exists():
            missing.append(yt_id)
            continue
        vid_hash = compute_video_id(mp4)
        manifest[yt_id] = {
            "video_id": vid_hash,
            "youtube_id": yt_id,
            "source_path": str(mp4),
            "size_bytes": mp4.stat().st_size,
        }
    if missing:
        print(f"WARNING: {len(missing)} videos not yet downloaded: {missing[:5]}…", flush=True)
        print("Run download script first; then re-run this converter.", flush=True)
        if len(missing) == len(sampled):
            return 1

    # Write manifest
    with (EVAL_DIR / "video_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote video_manifest.json ({len(manifest)} entries)", flush=True)

    # Build eval cases
    cases: list[dict] = []
    mod_count: dict[str, int] = {}
    for _, r in rows.iterrows():
        yt_id = r["videoID"]
        if yt_id not in manifest:
            continue  # video missing locally; skip
        opts = list(r["options"])
        correct_letter = r["answer"].strip().upper()
        # Map letter -> option text
        correct_idx = ord(correct_letter) - ord("A")
        if correct_idx < 0 or correct_idx >= len(opts):
            print(f"  skip bad answer letter {correct_letter} for {r['question_id']}", flush=True)
            continue
        correct_opt = opts[correct_idx]
        keywords = extract_keywords(correct_opt)
        task_type = r["task_type"]
        modality = TASK_TYPE_TO_MODALITY.get(task_type, "joint")
        mod_count[modality] = mod_count.get(modality, 0) + 1
        case = {
            "question_id": f"vme-{r['question_id']}",
            "video_id": manifest[yt_id]["video_id"],
            "question": question_with_candidates(r["question"], opts),
            "modality_tag": modality,
            "question_type": task_type.lower(),
            "expected_keywords": keywords,
            "expected_citation_min": 1,
            "expected_citation_kinds": expected_citation_kinds(task_type, modality),
            "reference_answer": re.sub(r"^[A-D][\.\)]\s*", "", correct_opt).strip(),
            "source": "Video-MME",
            "source_meta": {
                "youtube_id": yt_id,
                "domain": r["domain"],
                "sub_category": r["sub_category"],
                "options": opts,
                "correct_letter": correct_letter,
            },
        }
        cases.append(case)

    cases.sort(key=lambda c: c["question_id"])

    out = EVAL_DIR / "questions.jsonl"
    with out.open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nwrote {out} ({len(cases)} cases)", flush=True)
    print(f"modality distribution: {mod_count}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
