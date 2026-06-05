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
from app.distill_trajectory import recomputed_guards
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


def _split_videos(
    video_ids: list[str],
    val_ratio: float,
    val_videos: int | None,
    seed: int,
    forced_holdout: set[str] | None = None,
):
    # forced_holdout videos always land in val (never trained on), so a model can
    # be evaluated on a pre-existing held-out set for an apples-to-apples compare.
    forced = (forced_holdout or set()) & set(video_ids)
    ordered = sorted(set(video_ids) - forced)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = val_videos if val_videos is not None else max(1, round(len(set(video_ids)) * val_ratio))
    n_val = max(0, n_val - len(forced))
    n_val = min(n_val, len(ordered) - 1) if ordered else 0
    val = forced | set(ordered[:n_val])
    train = set(ordered[n_val:])
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True, help="Raw trajectory JSONL from Phase A.")
    parser.add_argument("--out-dir", default="data/distillation")
    parser.add_argument(
        "--cases",
        default=None,
        help="Original eval cases JSONL; if given, write eval_heldout.jsonl with the "
        "cases for the held-out (val) videos for the Phase D comparison (spec §3.4).",
    )
    parser.add_argument("--include-tier-2", action="store_true", help="Also use tier_2 trajectories.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-videos", type=int, default=None, help="Explicit #videos held out for val.")
    parser.add_argument(
        "--holdout-videos",
        default=None,
        help="Path to a JSONL of eval cases (or a file of video_ids) whose videos must "
        "be forced into val/held-out (never trained on) — for matched-set comparison.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap #train samples (spec C1: keep ~1500-1700). If the build overshoots, "
        "drop whole videos (deterministic shuffle by --seed) until under the cap, so no "
        "trajectory is split across the cap. Val is untouched.",
    )
    parser.add_argument("--no-truncate", action="store_true", help="Skip base64 truncation (debug).")
    args = parser.parse_args()

    trajs = _read_jsonl(Path(args.trajectories))
    tier_counts: Counter = Counter()
    kept: list[dict] = []
    allowed = {"tier_1"} | ({"tier_2"} if args.include_tier_2 else set())
    for traj in trajs:
        # Recompute guards from the message stream so tiering uses current logic
        # regardless of which infer_guards version captured the trajectory.
        traj["guards_triggered"] = recomputed_guards(traj)
        tier = classify_tier(traj)
        tier_counts[tier] += 1
        if tier in allowed and traj.get("messages") and traj.get("system_prompt"):
            kept.append(traj)

    video_ids = [str(t.get("video_id") or "") for t in kept]
    forced_holdout: set[str] = set()
    if args.holdout_videos:
        for row in _read_jsonl(Path(args.holdout_videos)):
            vid = str(row.get("video_id") or "").strip()
            if vid:
                forced_holdout.add(vid)
        print(f"forcing {len(forced_holdout)} videos into held-out (never trained)")
    train_videos, val_videos = _split_videos(
        video_ids, args.val_ratio, args.val_videos, args.seed, forced_holdout
    )

    tools = export_tool_schemas()
    val_samples: list[dict] = []
    per_video_samples: dict[str, int] = defaultdict(int)
    train_by_video: dict[str, list[dict]] = defaultdict(list)
    for traj in kept:
        vid = str(traj.get("video_id") or "")
        samples = slice_trajectory(traj, tools, truncate=not args.no_truncate)
        per_video_samples[vid] += len(samples)
        if vid in val_videos:
            val_samples.extend(samples)
        else:
            train_by_video[vid].extend(samples)

    # Honor C1 (~1500-1700 train samples): if over --max-train-samples, drop whole
    # videos (deterministic shuffle) until under the cap. Never split a trajectory.
    dropped_videos = 0
    if args.max_train_samples is not None:
        order = sorted(train_by_video)
        random.Random(args.seed).shuffle(order)
        kept_train: dict[str, list[dict]] = {}
        running = 0
        for vid in order:
            n = len(train_by_video[vid])
            if running + n > args.max_train_samples and kept_train:
                dropped_videos += 1
                continue
            kept_train[vid] = train_by_video[vid]
            running += n
        train_by_video = kept_train
    train_videos = set(train_by_video)
    train_samples: list[dict] = [s for vid in train_by_video for s in train_by_video[vid]]

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "train.jsonl", train_samples)
    _write_jsonl(out_dir / "val.jsonl", val_samples)

    heldout_n = 0
    if args.cases:
        cases = _read_jsonl(Path(args.cases))
        heldout = [c for c in cases if str(c.get("video_id") or "") in val_videos]
        _write_jsonl(out_dir / "eval_heldout.jsonl", heldout)
        heldout_n = len(heldout)
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
        "max_train_samples": args.max_train_samples,
        "dropped_videos_over_cap": dropped_videos,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "eval_heldout_cases": heldout_n,
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
