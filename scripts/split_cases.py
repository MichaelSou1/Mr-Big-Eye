#!/usr/bin/env python
"""Deterministic video-disjoint train/held-out split for an EvalCase JSONL.

Training source = NExT-GQA (full) is the §13.4 decision, but §7.1 needs a clean
IID held-out point to define the IID→OOD decay slope. We carve a video-disjoint
held-out split (so SFT never sees a held-out video): held-out NExT-GQA = IID
eval; cross-dataset sets (Video-MME vision subset / Perception Test) = L1 OOD.

The split keys on ``video_id`` (not case_id) via a stable hash, so all questions
of a video land on the same side. Asserts the two video sets are disjoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _bucket(video_id: str) -> int:
    digest = hashlib.md5(video_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def split_cases(rows: list[dict], heldout_pct: int) -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    heldout: list[dict] = []
    for row in rows:
        video_id = str(row.get("video_id") or "")
        if video_id and _bucket(video_id) < heldout_pct:
            heldout.append(row)
        else:
            train.append(row)
    return train, heldout


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Input EvalCase JSONL.")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--heldout-pct", type=int, default=15, help="Percent of videos held out (default 15).")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.cases))
    train, heldout = split_cases(rows, args.heldout_pct)

    train_videos = {str(r.get("video_id") or "") for r in train if r.get("video_id")}
    heldout_videos = {str(r.get("video_id") or "") for r in heldout if r.get("video_id")}
    overlap = train_videos & heldout_videos
    if overlap:
        print(f"ERROR: video overlap between train and held-out: {sorted(overlap)[:10]}", file=sys.stderr)
        return 1

    _write_jsonl(Path(args.train_out), train)
    _write_jsonl(Path(args.heldout_out), heldout)
    print(
        json.dumps(
            {
                "train_cases": len(train),
                "heldout_cases": len(heldout),
                "train_videos": len(train_videos),
                "heldout_videos": len(heldout_videos),
                "heldout_pct": args.heldout_pct,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
