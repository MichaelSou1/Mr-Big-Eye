#!/usr/bin/env python
"""Phase B: raw trajectories -> SFT train/val sets (spec §5).

Pipeline: tier-filter -> video-level split (no video crosses train/val, §3.4)
-> slice each trajectory into per-step samples (forced/guard targets excluded,
base64 truncated) -> write sharegpt-style JSONL + shared tool_schemas.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.distill_filter import classify_tier
from app.distill_format import (
    export_tool_schemas,
    slice_trajectory,
    target_kind,
    target_tool_names,
)
from app.eval_fingerprint import AGENT_CODE_VERSION


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _split_videos(video_ids: list[str], val_ratio: float, val_videos: int | None, seed: int):
    ordered = sorted(set(video_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = val_videos if val_videos is not None else max(1, round(len(ordered) * val_ratio))
    n_val = min(n_val, len(ordered) - 1) if len(ordered) > 1 else 0
    val = set(ordered[:n_val])
    train = set(ordered[n_val:])
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True, help="Raw trajectory JSONL from Phase A.")
    parser.add_argument("--out-dir", default="data/distillation")
    parser.add_argument("--include-tier-2", action="store_true", help="Also use tier_2 trajectories.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-videos", type=int, default=None, help="Explicit #videos held out for val.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-truncate", action="store_true", help="Skip base64 truncation (debug).")
    args = parser.parse_args()

    trajs = _read_jsonl(Path(args.trajectories))
    tier_counts: Counter = Counter()
    kept: list[dict] = []
    allowed = {"tier_1"} | ({"tier_2"} if args.include_tier_2 else set())
    for traj in trajs:
        tier = classify_tier(traj)
        tier_counts[tier] += 1
        if tier in allowed and traj.get("messages") and traj.get("system_prompt"):
            kept.append(traj)

    video_ids = [str(t.get("video_id") or "") for t in kept]
    train_videos, val_videos = _split_videos(video_ids, args.val_ratio, args.val_videos, args.seed)

    tools = export_tool_schemas()
    train_samples: list[dict] = []
    val_samples: list[dict] = []
    per_video_samples: dict[str, int] = defaultdict(int)
    for traj in kept:
        vid = str(traj.get("video_id") or "")
        samples = slice_trajectory(traj, tools, truncate=not args.no_truncate)
        per_video_samples[vid] += len(samples)
        (val_samples if vid in val_videos else train_samples).extend(samples)

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "train.jsonl", train_samples)
    _write_jsonl(out_dir / "val.jsonl", val_samples)
    (out_dir / "tool_schemas.json").write_text(
        json.dumps({"agent_code_version": AGENT_CODE_VERSION, "n_tools": len(tools), "tools": tools}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def _kind_dist(samples):
        return dict(Counter(target_kind(s) for s in samples))

    def _tool_dist(samples):
        c: Counter = Counter()
        for s in samples:
            for name in target_tool_names(s):
                c[name] += 1
        return dict(c)

    report = {
        "agent_code_version": AGENT_CODE_VERSION,
        "trajectories_total": len(trajs),
        "tier_counts": dict(tier_counts),
        "include_tier_2": args.include_tier_2,
        "kept_trajectories": len(kept),
        "videos_total": len(set(video_ids)),
        "train_videos": len(train_videos),
        "val_videos": len(val_videos),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_target_kind": _kind_dist(train_samples),
        "val_target_kind": _kind_dist(val_samples),
        "train_target_tools": _tool_dist(train_samples),
        "val_target_tools": _tool_dist(val_samples),
    }
    (out_dir / "build_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir/'train.jsonl'} ({len(train_samples)}), {out_dir/'val.jsonl'} ({len(val_samples)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
