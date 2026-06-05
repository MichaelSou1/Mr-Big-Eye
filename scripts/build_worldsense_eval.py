"""Convert the WorldSense omnimodal QA set into our EvalCase JSONL + manifest,
and carve a joint/audio-weighted training-enrichment pool plus a disjoint
audio-visual held-out eval set.

WorldSense (honglyhly/WorldSense, CC BY-NC 4.0) is an MCQ benchmark whose
questions strongly couple audio + vision, so it backfills the `joint`/`audio`
signal that medium-only Video-MME can't (see docs/distill_hard_rebalance_spec.md
and docs/distill_v2_prep_status.md). We use it ONLY as training enrichment +
a separate audio-visual heldout — never to eval the distilled orchestrator on
data it was trained on.

Inputs (download via scripts/download_worldsense.py):
    data/worldsense_src/worldsense_qa.json  — {video_id: {..., task0, task1, ...}}
    data/worldsense_videos/<video_id>.mp4   — extracted raw videos

Outputs (default --out-dir eval/audiovisual/worldsense):
    questions.jsonl        — every EvalCase whose mp4 is present
    video_manifest.json    — youtube/native id -> content-hash video_id
and (default --distill-dir data/distillation/worldsense):
    train_cases.jsonl      — joint/audio-weighted training-enrichment cases
    heldout_cases.jsonl    — disjoint audio-visual held-out eval cases
    teacher_shards/cases_{0..3}.jsonl  — train cases sharded by video

    conda run -n mbe-ingest python scripts/build_worldsense_eval.py \\
        --heldout-videos 40 --train-videos 150

Honors MBE_ROOT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("MBE_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "data" / "worldsense_src"
VIDEOS = ROOT / "data" / "worldsense_videos"

# WorldSense task_type -> our modality_tag. Audio-explicit tasks need the audio
# track; temporal/emotion tasks couple audio-visual reasoning over time (joint);
# the rest are single-frame-ish visual perception.
AUDIO_TASKS = {
    "Audio Source Localization", "Audio Recognition", "Audio Counting", "Audio Change",
}
JOINT_TASKS = {
    "Event Sorting", "Temporal Localization", "Temporal Prediction", "Object State Change",
    "Action Counting", "Causal Reasoning", "Emotion Change", "Human Emotions",
    "Video Emotions", "Event Recognition",
}
MODALITIES = ("joint", "audio", "visual", "overview")

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "is", "are", "was",
    "were", "be", "and", "or", "that", "this", "these", "those", "it", "its", "by", "as",
    "from", "into", "than", "all", "any", "both", "each", "every", "most", "some", "no",
    "not", "one", "two", "three", "same", "number", "more", "less", "several",
}


def modality_of(task_type: str) -> str:
    if task_type in AUDIO_TASKS:
        return "audio"
    if task_type in JOINT_TASKS:
        return "joint"
    return "visual"


def expected_citation_kinds(modality: str) -> list[str]:
    if modality in ("audio", "overview"):
        return ["transcript"]
    if modality == "joint":
        return ["transcript", "frame_or_slide"]
    return ["frame_or_slide"]


def strip_letter(text: str) -> str:
    return re.sub(r"^[A-E][\.\)]\s*", "", str(text)).strip()


def format_candidates(options: list[str]) -> str:
    lines = ["Candidates:"]
    for i, opt in enumerate(options):
        lines.append(f"{chr(ord('A') + i)}) {strip_letter(opt)}")
    return "\n".join(lines)


def extract_keywords(option_text: str, max_n: int = 3) -> list[str]:
    text = strip_letter(option_text).rstrip(".")
    tokens = re.findall(r"[A-Za-z一-鿿][A-Za-z0-9一-鿿'-]*", text)
    kept: list[str] = []
    seen: set[str] = set()
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


def compute_video_id(mp4: Path) -> str:
    h = hashlib.sha256()
    with mp4.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def mix_distance(counts: Counter, target: dict[str, float]) -> float:
    total = sum(counts.values()) or 1
    return sum(abs(counts.get(m, 0) / total - target.get(m, 0.0)) for m in MODALITIES)


def greedy_select(pool, n, per_video_mods, base, target):
    chosen, remaining, running = [], list(pool), Counter(base)
    for _ in range(min(n, len(remaining))):
        best_v, best_key = None, None
        for v in remaining:
            trial = running + Counter(per_video_mods[v])
            key = (mix_distance(trial, target), v)
            if best_key is None or key < best_key:
                best_key, best_v = key, v
        chosen.append(best_v)
        running += Counter(per_video_mods[best_v])
        remaining.remove(best_v)
    return chosen


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa-file", default=str(SRC / "worldsense_qa.json"))
    ap.add_argument("--videos-dir", default=str(VIDEOS))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "audiovisual" / "worldsense"))
    ap.add_argument("--distill-dir", default=str(ROOT / "data" / "distillation" / "worldsense"))
    ap.add_argument("--heldout-videos", type=int, default=40,
                    help="# audio/joint-rich videos reserved for the audio-visual heldout.")
    ap.add_argument("--train-videos", type=int, default=150,
                    help="# joint/audio-weighted videos for training enrichment.")
    ap.add_argument("--joint-frac", type=float, default=0.45)
    ap.add_argument("--audio-frac", type=float, default=0.25)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qa = json.loads(Path(args.qa_file).read_text())
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Hash mp4s -> manifest; build all cases for videos present locally.
    manifest: dict[str, dict] = {}
    cases: list[dict] = []
    per_video_mods: dict[str, list[str]] = defaultdict(list)  # keyed by content hash
    missing = 0
    for native_id, rec in qa.items():
        mp4 = videos_dir / f"{native_id}.mp4"
        if not mp4.exists():
            missing += 1
            continue
        vid = compute_video_id(mp4)
        manifest[native_id] = {
            "video_id": vid, "native_id": native_id, "source_path": str(mp4),
            "size_bytes": mp4.stat().st_size,
        }
        for key in sorted(k for k in rec if k.startswith("task")):
            t = rec[key]
            opts = list(t["candidates"])
            letter = str(t["answer"]).strip().upper()
            idx = ord(letter) - ord("A")
            if idx < 0 or idx >= len(opts):
                continue
            modality = modality_of(t.get("task_type", ""))
            per_video_mods[vid].append(modality)
            cases.append({
                "question_id": f"worldsense-{native_id}-{key}",
                "video_id": vid,
                "question": f"{str(t['question']).strip()}\n\n{format_candidates(opts)}",
                "modality_tag": modality,
                "question_type": str(t.get("task_type", "")).lower(),
                "expected_keywords": extract_keywords(opts[idx]),
                "expected_citation_min": 1,
                "expected_citation_kinds": expected_citation_kinds(modality),
                "reference_answer": strip_letter(opts[idx]),
                "source": "WorldSense",
                "source_meta": {
                    "native_id": native_id,
                    "domain": rec.get("domain"),
                    "sub_category": rec.get("sub_category"),
                    "audio_class": rec.get("audio_class"),
                    "duration": rec.get("duration"),
                    "task_domain": t.get("task_domain"),
                    "task_type": t.get("task_type"),
                    "options": opts,
                    "correct_letter": letter,
                },
            })

    cases.sort(key=lambda c: c["question_id"])
    (out_dir / "video_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(out_dir / "questions.jsonl", cases)
    print(f"videos with mp4: {len(manifest)}  missing: {missing}  cases: {len(cases)}")
    print(f"modality (all): {dict(Counter(c['modality_tag'] for c in cases))}")

    if not manifest:
        print("no mp4s yet — run scripts/download_worldsense.py + unzip first.")
        return 0

    # 2) Split videos: audio-rich heldout, then joint/audio-weighted train pool.
    target = {
        "joint": args.joint_frac, "audio": args.audio_frac,
        "visual": max(0.0, 1.0 - args.joint_frac - args.audio_frac), "overview": 0.0,
    }
    all_vids = sorted(per_video_mods)
    # heldout target: maximise audio share for an audio-visual stress eval.
    heldout = greedy_select(
        all_vids, args.heldout_videos, per_video_mods, Counter(),
        {"audio": 0.5, "joint": 0.35, "visual": 0.15, "overview": 0.0},
    )
    pool = [v for v in all_vids if v not in set(heldout)]
    train = greedy_select(pool, args.train_videos, per_video_mods, Counter(), target)

    train_set, heldout_set = set(train), set(heldout)
    assert not (train_set & heldout_set)
    train_cases = [c for c in cases if c["video_id"] in train_set]
    heldout_cases = [c for c in cases if c["video_id"] in heldout_set]

    distill = Path(args.distill_dir)
    write_jsonl(distill / "train_cases.jsonl", train_cases)
    write_jsonl(distill / "heldout_cases.jsonl", heldout_cases)
    # shard train cases by video (a video's questions stay in one shard)
    by_video: dict[str, list] = defaultdict(list)
    for c in train_cases:
        by_video[c["video_id"]].append(c)
    shards = [[] for _ in range(args.shards)]
    for i, vid in enumerate(sorted(by_video)):
        shards[i % args.shards].extend(by_video[vid])
    for i, sh in enumerate(shards):
        write_jsonl(distill / "teacher_shards" / f"cases_{i}.jsonl", sh)

    def mods(rows):
        return dict(Counter(c["modality_tag"] for c in rows))
    print(f"\nTRAIN enrichment : {len(train_cases)} cases / {len(train_set)} videos  {mods(train_cases)}")
    print(f"AUDIO-V heldout  : {len(heldout_cases)} cases / {len(heldout_set)} videos  {mods(heldout_cases)}")
    print(f"teacher shards   : {[len(s) for s in shards]} -> {distill/'teacher_shards'}")
    print(f"wrote {out_dir/'questions.jsonl'}, {distill/'train_cases.jsonl'}, {distill/'heldout_cases.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
