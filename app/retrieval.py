from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from PIL import Image

from app.cache import video_cache_dir
from app.config import settings
from app.models import get_bge, get_siglip, release_bge, release_siglip
from app.preprocess import dense_frame_filename


@dataclass
class RetrievalResult:
    frames: list[Image.Image]
    timestamps: list[float]
    scene_hits: list[dict[str, Any]]


def two_stage_retrieve(
    video_id: str,
    question: str,
    top_n_scenes: int | None = None,
    top_k_frames: int | None = None,
) -> RetrievalResult:
    """Retrieve keyframes with caption filtering followed by visual search."""
    top_n = top_n_scenes or settings.top_n_scenes
    top_k = top_k_frames or settings.top_k_frames
    cache_dir = video_cache_dir(video_id)

    scene_hits = _query_caption_index(cache_dir, question, top_n)
    time_ranges = [(float(hit["start"]), float(hit["end"])) for hit in scene_hits]
    try:
        frame_hits = _query_frame_index(cache_dir, question, time_ranges, top_k)
    finally:
        if settings.unload_models_after_use:
            release_siglip()

    frames: list[Image.Image] = []
    timestamps: list[float] = []
    for hit in sorted(frame_hits, key=lambda item: float(item["timestamp"])):
        timestamp = float(hit["timestamp"])
        path = cache_dir / "frames_dense" / str(hit.get("filename") or dense_frame_filename(timestamp))
        if path.exists():
            frames.append(Image.open(path).convert("RGB"))
            timestamps.append(timestamp)
    return RetrievalResult(frames=frames, timestamps=timestamps, scene_hits=scene_hits)


def _query_caption_index(cache_dir: Path, question: str, top_n: int) -> list[dict[str, Any]]:
    try:
        collection = chromadb.PersistentClient(path=str(cache_dir / "caption_index")).get_collection(
            "captions"
        )
    except chromadb.errors.NotFoundError:
        # Video was marked .done but caption index never built (typically a video
        # whose preprocess crashed mid-way before chromadb persistence and was
        # later flagged done manually). Treat as "no scene hits found" so the
        # orchestrator can fall back to dense frame retrieval instead of crashing.
        return []
    try:
        query_embedding = get_bge().encode_text([question])[0].tolist()
    finally:
        if settings.unload_models_after_use:
            release_bge()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for metadata, document, distance in zip(metadatas, documents, distances, strict=False):
        item = dict(metadata or {})
        item["caption"] = item.get("caption") or document
        item["score"] = 1.0 - float(distance)
        hits.append(item)
    return hits


def _query_frame_index(
    cache_dir: Path,
    question: str,
    time_ranges: list[tuple[float, float]],
    top_k: int,
) -> list[dict[str, Any]]:
    try:
        collection = chromadb.PersistentClient(path=str(cache_dir / "frame_index")).get_collection(
            "frames"
        )
    except chromadb.errors.NotFoundError:
        return []
    query_embedding = get_siglip().encode_text([question])[0].tolist()
    # Always blend a slice of un-gated global SigLIP top-K so a wrong BGE
    # caption→scene routing can be recovered. Quota stays under top_k so total
    # frame count (and VLM token cost) is unchanged.
    scene_quota = _scene_quota(top_k, time_ranges)
    hits = _query_frames(collection, query_embedding, scene_quota, _build_where(time_ranges))
    if len(hits) < top_k:
        fallback = _query_frames(collection, query_embedding, top_k * 3, None)
        existing = {float(item["timestamp"]) for item in hits}
        for item in fallback:
            if float(item["timestamp"]) not in existing:
                hits.append(item)
                existing.add(float(item["timestamp"]))
            if len(hits) >= top_k:
                break
    return hits[:top_k]


def _query_frames(
    collection,
    query_embedding: list[float],
    n_results: int,
    where: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if n_results <= 0:
        return []
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances"],
        )
    except Exception:
        if where is None:
            raise
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(n_results * 5, n_results),
            include=["metadatas", "distances"],
        )
        return _client_side_filter(results, where, n_results)

    hits: list[dict[str, Any]] = []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for metadata, distance in zip(metadatas, distances, strict=False):
        item = dict(metadata or {})
        item["score"] = 1.0 - float(distance)
        hits.append(item)
    return hits


def _scene_quota(top_k: int, time_ranges: list[tuple[float, float]]) -> int:
    """How many of the top_k frames should come from scene-gated search.

    When no scene ranges were found, use the full quota for un-gated retrieval.
    Otherwise reserve ~1/3 of the budget for un-gated frames so a wrong scene
    routing can be recovered by the global SigLIP top-K.
    """
    if not time_ranges:
        return top_k
    return max(1, (top_k * 2) // 3)


def _build_where(time_ranges: list[tuple[float, float]]) -> dict[str, Any] | None:
    if not time_ranges:
        return None
    return {
        "$or": [
            {
                "$and": [
                    {"timestamp": {"$gte": float(start)}},
                    {"timestamp": {"$lte": float(end)}},
                ]
            }
            for start, end in time_ranges
        ]
    }


def _client_side_filter(
    results: dict[str, Any],
    where: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    ranges = []
    for group in where.get("$or", []):
        terms = group.get("$and", [])
        start = terms[0]["timestamp"]["$gte"]
        end = terms[1]["timestamp"]["$lte"]
        ranges.append((start, end))
    hits: list[dict[str, Any]] = []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for metadata, distance in zip(metadatas, distances, strict=False):
        timestamp = float(metadata["timestamp"])
        if any(start <= timestamp <= end for start, end in ranges):
            item = dict(metadata)
            item["score"] = 1.0 - float(distance)
            hits.append(item)
        if len(hits) >= limit:
            break
    return hits
