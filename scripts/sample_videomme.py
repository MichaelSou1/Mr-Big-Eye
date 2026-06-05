"""Sample a target set of Video-MME videos for distillation, extending the set
already in eval/audiovisual/video_manifest.json.

Keeps every Video-MME video we have already ingested (so prior trajectories and
caches stay valid) and adds new videos to reach --target, stratified across the
three duration buckets (short / medium / long) and the six domains. The long
bucket is the expensive one to ingest (videos up to ~1h), so --long-frac caps
its share of the *newly added* videos.

Output:
    data/videomme_src/sampled_videos.json — JSON list of YouTube videoIDs,
    consumed by build_videomme_eval.py.

    python scripts/sample_videomme.py --target 250 --long-frac 0.2

--- Modality-weighted (v2 hard-rebalance, spec docs/distill_hard_rebalance_spec.md) ---
With --modality-weighted the candidate pool (unused videos in --buckets) is
selected greedily to push the *question-modality* mix of the kept set toward
--joint-frac / --overview-frac (over-weighting `joint` audio-visual/temporal
questions), instead of shortest-first. It carves a held-out eval set (seeded
from already-ingested videos via --heldout-seed-yt, plus --heldout-new new
videos) that is disjoint from training, and writes separate train / heldout /
to-ingest video lists:

    python scripts/sample_videomme.py --modality-weighted --buckets medium \\
        --reuse-yt /tmp/reuse_yt.json --heldout-seed-yt /tmp/heldout_seed_yt.json \\
        --new-train 120 --heldout-new 14 --joint-frac 0.35 --overview-frac 0.15 \\
        --durations-file data/videomme_src/durations.json \\
        --train-out data/videomme_src/v2_train.json \\
        --heldout-out data/videomme_src/v2_heldout_new.json \\
        --ingest-out data/videomme_src/v2_to_ingest.json \\
        --out data/videomme_src/v2_sampled.json

Honors MBE_ROOT (defaults to the repo root).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("MBE_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / "data" / "videomme_src"
MANIFEST = ROOT / "eval" / "audiovisual" / "video_manifest.json"
OUT = SRC / "sampled_videos.json"

# Video-MME task_type -> our modality_tag (must match build_videomme_eval.py).
TASK_TYPE_TO_MODALITY = {
    "Action Reasoning": "joint",
    "Temporal Reasoning": "joint",
    "Temporal Perception": "joint",
    "Information Synopsis": "overview",
}
MODALITIES = ("joint", "visual", "overview")


def _modality(task_type: str) -> str:
    return TASK_TYPE_TO_MODALITY.get(task_type, "visual")


def _load_json_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    return set(json.loads(Path(path).read_text()))


def _mix_distance(counts: Counter, target: dict[str, float]) -> float:
    """L1 distance between a question-modality Counter's fractions and target."""
    total = sum(counts.values()) or 1
    return sum(abs(counts.get(m, 0) / total - target.get(m, 0.0)) for m in MODALITIES)


def _greedy_select(
    pool: list[str],
    n: int,
    per_video_mods: dict[str, list[str]],
    base_counts: Counter,
    target: dict[str, float],
    secs: dict[str, float],
) -> list[str]:
    """Greedily pick n videos from pool so the kept question-modality mix
    (base_counts + picks) approaches `target`. Tie-break toward shorter videos
    (cheaper ingest). O(n * |pool|) — fine for a few hundred candidates."""
    chosen: list[str] = []
    remaining = list(pool)
    running = Counter(base_counts)
    for _ in range(min(n, len(remaining))):
        best_v, best_key = None, None
        for v in remaining:
            trial = running + Counter(per_video_mods.get(v, []))
            key = (_mix_distance(trial, target), secs.get(v, float("inf")), v)
            if best_key is None or key < best_key:
                best_key, best_v = key, v
        chosen.append(best_v)
        running += Counter(per_video_mods.get(best_v, []))
        remaining.remove(best_v)
    return chosen


def load_parquet() -> pd.DataFrame:
    direct = SRC / "test.parquet"
    if direct.exists():
        return pd.read_parquet(direct)
    cands = sorted(SRC.rglob("test-*.parquet"))
    if not cands:
        raise FileNotFoundError(f"no Video-MME parquet under {SRC}")
    return pd.read_parquet(cands[0])


def existing_videomme_ids(all_ids: set[str]) -> set[str]:
    if not MANIFEST.exists():
        return set()
    manifest = json.loads(MANIFEST.read_text())
    # manifest keys are YouTube IDs; keep only those that belong to Video-MME
    return {yt for yt in manifest if yt in all_ids}


def run_modality_weighted(args, df, vids, all_ids, dur_of, dom_of, secs, allowed) -> int:
    """v2 hard-rebalance selection: pick joint-rich training videos and a disjoint
    modality-balanced held-out eval set; write train / heldout / to-ingest lists."""
    target = {
        "joint": args.joint_frac,
        "overview": args.overview_frac,
        "visual": max(0.0, 1.0 - args.joint_frac - args.overview_frac),
    }
    per_video_mods: dict[str, list[str]] = (
        df.assign(modality=df["task_type"].map(_modality))
        .groupby("videoID")["modality"].apply(list).to_dict()
    )
    reuse = _load_json_ids(args.reuse_yt) & all_ids
    heldout_seed = _load_json_ids(args.heldout_seed_yt) & all_ids

    downloaded = {p.stem for p in (ROOT / "data" / "videomme_videos").glob("*.mp4")}
    # Candidate pool: unused (not reuse, not heldout-seed) videos in the allowed
    # buckets, optionally local-only and within the duration cap.
    pool = sorted(
        v for v in all_ids - reuse - heldout_seed
        if dur_of[v] in allowed
        and (not args.downloaded_only or v in downloaded)
        and (args.max_seconds is None or secs.get(v, float("inf")) <= args.max_seconds)
    )

    def mods_counts(ids) -> Counter:
        c: Counter = Counter()
        for v in ids:
            c.update(per_video_mods.get(v, []))
        return c

    # 1) Held-out eval set: add --heldout-new videos so seed+new approaches target.
    heldout_new = _greedy_select(
        pool, args.heldout_new, per_video_mods, mods_counts(heldout_seed), target, secs
    )
    pool = [v for v in pool if v not in set(heldout_new)]
    # 2) Training additions: push reuse+new toward target (over-weight joint).
    train_new = _greedy_select(
        pool, args.new_train, per_video_mods, mods_counts(reuse), target, secs
    )

    to_ingest = sorted(set(train_new) | set(heldout_new))
    sampled = sorted(reuse | set(train_new) | heldout_seed | set(heldout_new))

    def _write(path: str | None, ids) -> None:
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(sorted(ids), indent=2))

    _write(args.train_out, train_new)
    _write(args.heldout_out, heldout_new)
    _write(args.ingest_out, to_ingest)
    _write(args.out or str(OUT), sampled)

    def _frac(c: Counter) -> dict[str, str]:
        t = sum(c.values()) or 1
        return {m: f"{c.get(m,0)}({c.get(m,0)/t:.0%})" for m in MODALITIES}

    print("=== v2 modality-weighted selection ===")
    print(f"target mix          : {target}")
    print(f"reuse train videos  : {len(reuse)}  mix {_frac(mods_counts(reuse))}")
    print(f"new train videos    : {len(train_new)}  mix {_frac(mods_counts(train_new))}")
    print(f"TRAIN combined       : {len(reuse)+len(train_new)} videos  "
          f"mix {_frac(mods_counts(reuse | set(train_new)))}")
    print(f"heldout seed videos : {len(heldout_seed)}  mix {_frac(mods_counts(heldout_seed))}")
    print(f"heldout new videos  : {len(heldout_new)}  mix {_frac(mods_counts(heldout_new))}")
    print(f"HELDOUT combined     : {len(heldout_seed)+len(heldout_new)} videos  "
          f"mix {_frac(mods_counts(heldout_seed | set(heldout_new)))}")
    print(f"NEW to ingest        : {len(to_ingest)} videos "
          f"(train_new {len(train_new)} + heldout_new {len(heldout_new)})")
    proj = (len(reuse) + len(train_new)) * 9.1  # pilot ≈ 9.1 samples/medium video (448/49)
    print(f"projected train_samples ≈ {proj:.0f} (pilot ≈9.1/medium video; verify at build)")
    if secs and to_ingest:
        s = sorted(secs.get(v, 0.0) for v in to_ingest)
        print(f"new-video seconds   : min={s[0]:.0f} max={s[-1]:.0f} "
              f"mean={sum(s)/len(s):.0f} total={sum(s)/60:.0f}min")
    print(f"domain of TRAIN     : {dict(Counter(dom_of[v] for v in (reuse | set(train_new))))}")
    print(f"wrote sampled={args.out or OUT} ({len(sampled)} ids); "
          f"train_new={args.train_out} heldout_new={args.heldout_out} ingest={args.ingest_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=250, help="total videos to end up with")
    ap.add_argument(
        "--buckets",
        default="short,medium,long",
        help="comma list of duration buckets newly added videos may come from "
        "(use 'short' to keep ingest cheap)",
    )
    ap.add_argument(
        "--durations-file",
        default=None,
        help="optional JSON {videoID: seconds} (ffprobe). When given, pick the "
        "SHORTEST candidates within the allowed buckets instead of random.",
    )
    ap.add_argument("--seed", type=int, default=42)
    # --- modality-weighted v2 mode (spec docs/distill_hard_rebalance_spec.md) ---
    ap.add_argument("--modality-weighted", action="store_true",
                    help="Greedily select added videos to hit --joint-frac/--overview-frac.")
    ap.add_argument("--joint-frac", type=float, default=0.35)
    ap.add_argument("--overview-frac", type=float, default=0.15)
    ap.add_argument("--new-train", type=int, default=120, help="# new training videos to add (v2).")
    ap.add_argument("--heldout-new", type=int, default=14, help="# new held-out eval videos (v2).")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Exclude candidates longer than this many seconds (needs --durations-file).")
    ap.add_argument("--reuse-yt", default=None, help="JSON list of YouTube IDs already trained (reuse).")
    ap.add_argument("--heldout-seed-yt", default=None,
                    help="JSON list of YouTube IDs that seed the held-out set (already ingested).")
    ap.add_argument("--downloaded-only", action="store_true",
                    help="Restrict the candidate pool to videos with a local mp4.")
    ap.add_argument("--train-out", default=None, help="Write NEW train YouTube IDs here (v2).")
    ap.add_argument("--heldout-out", default=None, help="Write NEW held-out YouTube IDs here (v2).")
    ap.add_argument("--ingest-out", default=None,
                    help="Write all NEW videos needing ingest (train-new + heldout-new) here (v2).")
    ap.add_argument("--out", default=None, help="Override the sampled_videos.json output path (v2).")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    allowed = {b.strip() for b in args.buckets.split(",") if b.strip()}
    secs = {}
    if args.durations_file:
        secs = {k: float(v) for k, v in json.loads(Path(args.durations_file).read_text()).items()}

    df = load_parquet()
    # one row per video (duration/domain are constant within a videoID)
    vids = df.drop_duplicates("videoID")[["videoID", "duration", "domain"]]
    all_ids = set(vids["videoID"])
    dur_of = dict(zip(vids["videoID"], vids["duration"]))
    dom_of = dict(zip(vids["videoID"], vids["domain"]))

    if args.modality_weighted:
        return run_modality_weighted(args, df, vids, all_ids, dur_of, dom_of, secs, allowed)

    keep = existing_videomme_ids(all_ids)
    n_add = max(0, args.target - len(keep))

    # candidate pool: unused videos restricted to the allowed duration buckets
    pool = [v for v in sorted(all_ids - keep) if dur_of[v] in allowed]
    if secs:
        # shortest first (ffprobe seconds); videos with no probe go last
        pool.sort(key=lambda v: secs.get(v, float("inf")))
    else:
        rng.shuffle(pool)

    if len(pool) < n_add:
        print(
            f"WARNING: only {len(pool)} candidates in buckets {allowed}, "
            f"need {n_add}; taking all available."
        )
    added = pool[:n_add]
    sampled = sorted(keep | set(added))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sampled, indent=2))

    def dist(ids) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for v in ids:
            d[dur_of[v]] += 1
        return dict(d)

    print(f"kept (already have): {len(keep)}  added: {len(added)}  total: {len(sampled)}")
    print(f"duration of total : {dist(sampled)}")
    print(f"duration of added : {dist(added)}")
    if secs and added:
        s = sorted(secs.get(v, float("inf")) for v in added)
        s = [x for x in s if x != float("inf")]
        if s:
            tot = sum(s)
            print(
                f"added seconds     : min={s[0]:.0f}s max={s[-1]:.0f}s "
                f"mean={tot/len(s):.0f}s total={tot/60:.0f}min"
            )
    doms: dict[str, int] = defaultdict(int)
    for v in sampled:
        doms[dom_of[v]] += 1
    print(f"domain of total   : {dict(doms)}")
    print(f"wrote {OUT} ({len(sampled)} videoIDs, ~{len(sampled)*3} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
