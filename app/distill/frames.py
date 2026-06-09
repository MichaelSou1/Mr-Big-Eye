from __future__ import annotations

"""Fixed, query-agnostic uniform frame sampler for the no-tool single-forward
study.

This is the deterministic preprocessing function that supplies frames at
TRAINING, CoT rewriting, consistency filtering, and inference. It is NOT a
per-query tool loop (no retrieval, no crop) so "no tool at inference" stays
honest, and because it is identical across base/internalized conditions it
holds frame *selection* constant — the confound we explicitly refuse to
internalize (see RESEARCH_SPEC §0.1 / §1 scope).
"""

from pathlib import Path
from typing import Any

from PIL import Image

from app.cache import load_meta, video_cache_dir
from app.config import settings


def sampler_frame_budget() -> int:
    """Frames per case for the fixed uniform sampler (configurable)."""
    return int(getattr(settings, "distill_sampler_frames", 16) or 16)


def uniform_timestamps(duration: float, n: int) -> list[float]:
    """Evenly spaced timestamps that avoid the exact start/end of the clip."""
    if n <= 0 or duration <= 0:
        return []
    if n == 1:
        return [round(duration / 2.0, 1)]
    step = duration / (n + 1)
    return [round(step * (i + 1), 1) for i in range(n)]


def nearest_dense_frame(cache_root: Path, target: float) -> tuple[Path, float] | None:
    """Pick the dense-sampled frame closest to ``target`` seconds."""
    dense_dir = cache_root / "frames_dense"
    if not dense_dir.is_dir():
        return None
    best: tuple[Path, float] | None = None
    best_dist = float("inf")
    for path in dense_dir.glob("t*.jpg"):
        try:
            ts = float(path.stem.lstrip("t"))
        except ValueError:
            continue
        dist = abs(ts - target)
        if dist < best_dist:
            best_dist = dist
            best = (path, ts)
    return best


def uniform_frame_manifest(video_id: str, n: int | None = None) -> list[dict[str, Any]]:
    """Return ``[{timestamp, path}]`` for the uniform sample without opening images.

    Timestamps are snapped to the nearest on-disk dense frame; duplicate frames
    (when the budget exceeds the number of available dense frames) are dropped.
    """
    budget = sampler_frame_budget() if n is None else int(n)
    meta = load_meta(video_id)
    duration = float(meta.get("duration") or 0.0)
    cache_root = video_cache_dir(video_id)
    manifest: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for target in uniform_timestamps(duration, budget):
        picked = nearest_dense_frame(cache_root, target)
        if picked is None:
            continue
        path, actual_ts = picked
        path_str = str(path)
        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)
        manifest.append({"timestamp": round(float(actual_ts), 1), "source": "uniform", "path": path_str})
    return sorted(manifest, key=lambda item: float(item["timestamp"]))


def load_uniform_frames(video_id: str, n: int | None = None) -> tuple[list[Image.Image], list[float]]:
    """Open the uniform-sample frames as PIL images + their timestamps."""
    frames: list[Image.Image] = []
    timestamps: list[float] = []
    for item in uniform_frame_manifest(video_id, n):
        path = Path(str(item["path"]))
        if not path.exists():
            continue
        frames.append(Image.open(path).convert("RGB"))
        timestamps.append(float(item["timestamp"]))
    return frames, timestamps


def covers_evidence(
    sampled_timestamps: list[float],
    gold_timestamps: list[float],
    gold_scenes: list[dict[str, float]],
    *,
    tolerance_sec: float = 2.0,
) -> bool:
    """True iff the uniform sample lands on the evidence window.

    A sample hits when some sampled timestamp is within ``tolerance_sec`` of a
    gold point timestamp, or falls inside a gold scene ``[start, end]`` (with the
    same tolerance padding). When a case carries no grounding GT at all, returns
    True (coverage cannot be falsified; such cases are flagged upstream).
    """
    if not gold_timestamps and not gold_scenes:
        return True
    if not sampled_timestamps:
        return False
    for sample in sampled_timestamps:
        for gold in gold_timestamps:
            if abs(float(sample) - float(gold)) <= tolerance_sec:
                return True
        for scene in gold_scenes:
            start = float(scene.get("start", 0.0)) - tolerance_sec
            end = float(scene.get("end", 0.0)) + tolerance_sec
            if start <= float(sample) <= end:
                return True
    return False
