"""Build supplementary EvalCase rows from CinePile + Perception Test + AVQA.

Each dataset:
- Samples a small number of videos (matching original allocation)
- Downloads/copies videos into data/uploads/{content_hash}.mp4
- Converts native QA schema to our EvalCase format
- Appends to eval/audiovisual/questions.jsonl
- Updates eval/audiovisual/video_manifest.json

Run only after Video-MME 150-case JSONL is in place.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/user/Mr-Big-Eye")
DATA = ROOT / "data"
EVAL_DIR = ROOT / "eval" / "audiovisual"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
MANIFEST_PATH = EVAL_DIR / "video_manifest.json"
UPLOADS = DATA / "uploads"

PROXY = "http://127.0.0.1:7890"

STOPWORDS = {
    "the","a","an","of","in","on","at","to","for","with","is","are","was","were","be",
    "and","or","that","this","these","those","it","its","by","as","from","into","than",
    "all","any","both","each","every","most","some","no","not","one","two","three",
}


def extract_keywords(text: str, max_n: int = 3) -> list[str]:
    text = text.strip().rstrip(".").rstrip("?").rstrip("!")
    if re.fullmatch(r"(\([a-zA-Z0-9]\)\s*)+", text):
        return [re.sub(r"\s+", "", text)]
    tokens = re.findall(r"[A-Za-z一-鿿][A-Za-z0-9一-鿿'-]*", text)
    kept: list[str] = []
    seen = set()
    for tok in tokens:
        if len(tok) < 2:
            continue
        low = tok.lower()
        if low in STOPWORDS or low in seen:
            continue
        seen.add(low)
        kept.append(tok)
        if len(kept) >= max_n:
            break
    return kept or [text]


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def install_video(source: Path, dest_name: str | None = None) -> tuple[str, Path]:
    """Hash source, copy to uploads/{hash}.{ext}, return (video_id, target_path)."""
    video_id = hash_file(source)
    suffix = source.suffix or ".mp4"
    target = UPLOADS / f"{video_id}{suffix}"
    if not target.exists():
        shutil.copy2(source, target)
    return video_id, target


FFMPEG_BIN = f"{sys.prefix}/bin/ffmpeg"
NODE_BIN = "/usr/bin/node"


def yt_dlp(url: str, out_path: Path, time_range: tuple[float, float] | None = None) -> bool:
    """yt-dlp wrapper with proxy + optional clip cutting via --download-sections."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    cmd = [
        f"{sys.prefix}/bin/yt-dlp",
        "--proxy", PROXY,
        "-f", "best[ext=mp4]/best",
        "--no-playlist",
        "--ffmpeg-location", FFMPEG_BIN,
        "--js-runtimes", f"node:{NODE_BIN}",
        "-o", str(out_path),
        url,
    ]
    if time_range is not None:
        a, b = time_range
        cmd.insert(-1, "--download-sections")
        cmd.insert(-1, f"*{a}-{b}")
    env = {**os.environ, "https_proxy": PROXY, "http_proxy": PROXY}
    print(f"  yt-dlp: {url}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if res.returncode != 0:
        print(f"  FAIL: {res.stderr[-400:]}", flush=True)
        return False
    return out_path.exists()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open() as f:
            return json.load(f)
    return {}


def save_manifest(m: dict) -> None:
    with MANIFEST_PATH.open("w") as f:
        json.dump(m, f, indent=2, sort_keys=True)


def append_cases(cases: list[dict]) -> None:
    with QUESTIONS_PATH.open("a", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def existing_case_ids() -> set[str]:
    if not QUESTIONS_PATH.exists():
        return set()
    out = set()
    with QUESTIONS_PATH.open() as f:
        for line in f:
            try:
                out.add(json.loads(line)["question_id"])
            except (KeyError, json.JSONDecodeError):
                pass
    return out


# ============================================================================
# Perception Test: pick 4 of 8 sample videos, ~16 MCQ questions, visual_heavy
# ============================================================================
def process_perception_test(manifest: dict, existing_ids: set[str]) -> list[dict]:
    ann = json.load((DATA / "perception_test/sample_annotations/sample.json").open())
    video_dir = DATA / "perception_test/sample_videos/videos"
    # Pick 4 videos with the most mc_question entries
    scored = [
        (vname, len(meta.get("mc_question", [])), meta) for vname, meta in ann.items()
    ]
    scored.sort(key=lambda x: -x[1])
    picks = scored[:4]
    cases = []
    for vname, qn, meta in picks:
        src = video_dir / f"{vname}.mp4"
        if not src.exists():
            print(f"  PT video missing: {src}", flush=True)
            continue
        vid, _target = install_video(src)
        manifest[f"pt_{vname}"] = {
            "video_id": vid,
            "source_dataset": "Perception-Test",
            "source_video": vname,
            "source_path": str(src),
        }
        for q in meta["mc_question"]:
            qid = f"pt-{vname}-{q['id']}"
            if qid in existing_ids:
                continue
            correct = q["options"][q["answer_id"]]
            area = q.get("area", "")
            reasoning = q.get("reasoning", "")
            # area=semantics/physics/memory/abstraction → all visual-grounded
            modality = "visual"
            if "audio" in str(q.get("tag", [])).lower() or area == "memory":
                modality = "joint"
            cases.append({
                "question_id": qid,
                "video_id": vid,
                "question": q["question"],
                "modality_tag": modality,
                "question_type": f"pt-{area}-{reasoning}".strip("-"),
                "expected_keywords": extract_keywords(correct),
                "expected_citation_min": 1,
                "expected_citation_kinds": ["frame"] if modality == "visual" else ["transcript", "frame"],
                "reference_answer": correct,
                "source": "Perception-Test",
                "source_meta": {
                    "video_name": vname,
                    "area": area,
                    "reasoning": reasoning,
                    "tag": q.get("tag", []),
                    "options": q["options"],
                    "answer_id": q["answer_id"],
                },
            })
    print(f"PT: {len(cases)} cases across {len(picks)} videos", flush=True)
    return cases


# ============================================================================
# CinePile: pick 4 distinct videos with 5-7 questions each (audio_heavy: dialog)
# ============================================================================
def process_cinepile(manifest: dict, existing_ids: set[str]) -> list[dict]:
    parquet_path = DATA / "hf_cache/cinepile_test.parquet"
    df = pd.read_parquet(parquet_path)
    # CinePile videos have 20-41 questions each. Pick 4 videos and sample 6 questions each.
    vc = df["videoID"].value_counts()
    candidates = vc[vc >= 20].index.tolist()
    random.seed(20260527)
    random.shuffle(candidates)
    chosen = candidates[:8]  # extra in case some YouTube downloads fail
    print(f"CinePile candidates: {len(candidates)}; trying up to {len(chosen)}", flush=True)
    cp_dir = DATA / "cinepile_videos"
    cp_dir.mkdir(exist_ok=True)
    cases = []
    successes = 0
    for ytid in chosen:
        if successes >= 4:
            break
        src_url = f"https://youtube.com/watch?v={ytid}"
        local = cp_dir / f"{ytid}.mp4"
        if not yt_dlp(src_url, local):
            print(f"  skip {ytid}: download failed", flush=True)
            continue
        vid, _ = install_video(local)
        manifest[f"cp_{ytid}"] = {
            "video_id": vid,
            "source_dataset": "CinePile",
            "youtube_id": ytid,
            "source_path": str(local),
        }
        successes += 1
        # Sample 6 questions per video (CinePile has ~30+; we want a manageable subset)
        rows = df[df["videoID"] == ytid].sample(n=min(6, (df["videoID"] == ytid).sum()), random_state=42)
        for _, r in rows.iterrows():
            qid = f"cp-{ytid}-{r.name}"
            if qid in existing_ids:
                continue
            choices = list(r["choices"])
            answer_idx = int(r["answer_key_position"])
            if not (0 <= answer_idx < len(choices)):
                continue
            correct = choices[answer_idx]
            # CinePile questions are dialog/scene-based → audio_heavy or joint
            cat = r["question_category"].replace("\n", " ").strip()
            modality = "audio" if "Character" in cat or "Theme" in cat else "joint"
            cases.append({
                "question_id": qid,
                "video_id": vid,
                "question": r["question"],
                "modality_tag": modality,
                "question_type": cat,
                "expected_keywords": extract_keywords(correct),
                "expected_citation_min": 1,
                "expected_citation_kinds": ["transcript"] if modality == "audio" else ["transcript", "frame"],
                "reference_answer": correct,
                "source": "CinePile",
                "source_meta": {
                    "youtube_id": ytid,
                    "movie_name": r["movie_name"],
                    "year": int(r["year"]) if not pd.isna(r["year"]) else None,
                    "yt_clip_title": r["yt_clip_title"],
                    "choices": choices,
                    "answer_key_position": answer_idx,
                    "visual_reliance": int(r["visual_reliance"]) if not pd.isna(r["visual_reliance"]) else None,
                },
            })
    print(f"CinePile: {len(cases)} cases across {len(chosen)} videos", flush=True)
    return cases


# ============================================================================
# AVQA: pick 2 videos, yt-dlp + 10s clip via --download-sections
# ============================================================================
def process_avqa(manifest: dict, existing_ids: set[str]) -> list[dict]:
    with (DATA / "avqa/val_qa.json").open() as f:
        rows = json.load(f)
    # Group by video_name; pick 2 videos with ≥4 questions
    from collections import Counter
    name_counts = Counter(r["video_name"] for r in rows)
    candidates = [n for n, c in name_counts.items() if c >= 4]
    random.seed(20260527)
    random.shuffle(candidates)
    avqa_dir = DATA / "avqa_videos"
    avqa_dir.mkdir(exist_ok=True)
    cases = []
    chosen = []
    for vname in candidates[:30]:  # try up to 30, take first 2 that succeed
        if len(chosen) >= 2:
            break
        # Parse: "{youtube_id}_{start_sec_6digits}" — start is in seconds
        m = re.match(r"^(.+)_(\d{6})$", vname)
        if not m:
            continue
        yt_id, start_s = m.group(1), int(m.group(2))
        url = f"https://youtube.com/watch?v={yt_id}"
        local = avqa_dir / f"{vname}.mp4"
        if not yt_dlp(url, local, time_range=(start_s, start_s + 10)):
            print(f"  skip {vname}", flush=True)
            continue
        if local.stat().st_size < 50_000:
            print(f"  skip {vname}: file too small", flush=True)
            continue
        vid, _ = install_video(local)
        manifest[f"avqa_{vname}"] = {
            "video_id": vid,
            "source_dataset": "AVQA",
            "video_name": vname,
            "youtube_id": yt_id,
            "start_sec": start_s,
            "source_path": str(local),
        }
        chosen.append(vname)
        for r in rows:
            if r["video_name"] != vname:
                continue
            qid = f"avqa-{vname}-{r['id']}"
            if qid in existing_ids:
                continue
            correct = r["multi_choice"][r["answer"]]
            qrel = r.get("question_relation", "")  # Audio/Visual/Both
            modality = {"Audio": "audio", "Visual": "visual", "Both": "joint"}.get(qrel, "joint")
            cases.append({
                "question_id": qid,
                "video_id": vid,
                "question": r["question_text"],
                "modality_tag": modality,
                "question_type": f"avqa-{r.get('question_type','')}",
                "expected_keywords": extract_keywords(correct),
                "expected_citation_min": 1,
                "expected_citation_kinds": ["transcript"] if modality == "audio" else (
                    ["frame"] if modality == "visual" else ["transcript", "frame"]
                ),
                "reference_answer": correct,
                "source": "AVQA",
                "source_meta": {
                    "video_name": vname,
                    "youtube_id": yt_id,
                    "start_sec": start_s,
                    "question_relation": qrel,
                    "multi_choice": r["multi_choice"],
                    "answer": r["answer"],
                },
            })
    print(f"AVQA: {len(cases)} cases across {len(chosen)} videos", flush=True)
    return cases


def main() -> int:
    manifest = load_manifest()
    existing = existing_case_ids()
    print(f"existing manifest entries: {len(manifest)}, existing case_ids: {len(existing)}", flush=True)

    all_cases: list[dict] = []
    all_cases.extend(process_perception_test(manifest, existing))
    all_cases.extend(process_cinepile(manifest, existing))
    all_cases.extend(process_avqa(manifest, existing))

    if all_cases:
        append_cases(all_cases)
    save_manifest(manifest)
    print(f"\nappended {len(all_cases)} cases; total now {len(existing) + len(all_cases)}", flush=True)

    # Re-tally modality distribution
    from collections import Counter
    mod = Counter()
    with QUESTIONS_PATH.open() as f:
        for line in f:
            mod[json.loads(line).get("modality_tag", "?")] += 1
    print(f"final modality distribution: {dict(mod)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
