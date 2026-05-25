import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import chromadb
import decord
import numpy as np
from PIL import Image
from scenedetect import ContentDetector, detect

from app.cache import ensure_cache_dirs, save_meta, set_video_status
from app.config import settings
from app.models import get_bge, get_siglip, release_bge, release_siglip
from app.progress import stage_label
from app.vqa import generate_caption

logger = logging.getLogger(__name__)


def dense_frame_filename(timestamp: float) -> str:
    return f"t{timestamp:06.1f}.jpg"


ProgressCallback = Callable[[str, str | None, float | None], None]


async def preprocess_video(
    video_id: str,
    video_path: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run all preprocessing stages and return the saved metadata."""
    cache_dir = ensure_cache_dirs(video_id)
    _clear_previous_artifacts(cache_dir)

    _emit(progress_callback, "probe", 0.02)
    meta = _timed("probe", lambda: _probe(video_path))
    meta["video_id"] = video_id
    meta["source_path"] = str(video_path)
    save_meta(video_id, meta)

    _emit(progress_callback, "scenes", 0.12)
    scenes = _timed(
        "scene detection",
        lambda: _detect_scenes(video_path, float(meta["fps"]), float(meta["duration"])),
    )
    meta["scene_count"] = len(scenes)
    save_meta(video_id, meta)

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    scene_items: list[dict[str, Any]] = []
    frames_scene_dir = cache_dir / "frames_scene"
    for scene_id, (start_sec, end_sec) in enumerate(scenes):
        t_mid = (start_sec + end_sec) / 2
        image = _extract_scene_frame(vr, float(meta["fps"]), t_mid)
        filename = f"scene_{scene_id:04d}.jpg"
        image.save(frames_scene_dir / filename, format="JPEG", quality=85, optimize=True)
        scene_items.append(
            {
                "scene_id": scene_id,
                "start": start_sec,
                "end": end_sec,
                "t_mid": t_mid,
                "filename": filename,
                "frame": image,
            }
        )
        if scenes:
            _emit(progress_callback, "scenes", 0.12 + 0.18 * ((scene_id + 1) / len(scenes)))

    captions = await _timed_async(
        "caption scenes",
        lambda: _caption_scenes(scene_items, progress_callback),
    )
    with (cache_dir / "captions.jsonl").open("w", encoding="utf-8") as handle:
        for item in captions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    _emit(progress_callback, "indexing", 0.68)
    _timed("caption index", lambda: _build_caption_index(cache_dir, captions))
    _emit(progress_callback, "embed", 0.76)
    dense_count = _timed(
        "dense frame index",
        lambda: _extract_and_index_dense_frames(
            vr,
            float(meta["fps"]),
            float(meta["duration"]),
            cache_dir,
            progress_callback,
        ),
    )
    meta["dense_frame_count"] = dense_count
    save_meta(video_id, meta)
    set_video_status(video_id, "done")
    _emit(progress_callback, "done", 1.0)
    return meta


def _probe(video_path: Path) -> dict[str, Any]:
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    duration = total_frames / fps if fps > 0 else 0.0
    frame = vr[0].asnumpy() if total_frames else np.zeros((0, 0, 3), dtype=np.uint8)
    height = int(frame.shape[0]) if frame.size else 0
    width = int(frame.shape[1]) if frame.size else 0
    return {
        "fps": fps,
        "duration": duration,
        "total_frames": total_frames,
        "width": width,
        "height": height,
    }


def _detect_scenes(video_path: Path, fps: float, duration: float) -> list[tuple[float, float]]:
    try:
        scene_list = detect(
            str(video_path),
            ContentDetector(threshold=settings.scene_detect_threshold),
        )
        scenes = [
            (
                max(0.0, start.get_seconds()),
                min(duration, end.get_seconds()),
            )
            for start, end in scene_list
            if end.get_seconds() > start.get_seconds()
        ]
    except Exception:
        logger.exception("Scene detection failed; falling back to uniform chunks")
        scenes = []

    if len(scenes) >= 3:
        return scenes
    return _uniform_scenes(duration, chunk_sec=10.0 if fps > 0 else duration)


def _extract_scene_frame(vr: decord.VideoReader, fps: float, t_mid: float) -> Image.Image:
    idx = _time_to_index(t_mid, fps, len(vr))
    return Image.fromarray(vr[idx].asnumpy()).convert("RGB")


async def _caption_scenes(
    scenes_with_frames: list[dict[str, Any]],
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(scenes_with_frames)
    for start in range(0, len(scenes_with_frames), 8):
        batch = scenes_with_frames[start : start + 8]
        _emit(progress_callback, "captions", 0.30 + 0.36 * (start / max(total, 1)))
        captions = await asyncio.gather(
            *(generate_caption(item["frame"]) for item in batch)
        )
        for item, caption in zip(batch, captions, strict=False):
            results.append(
                {
                    "scene_id": item["scene_id"],
                    "start": item["start"],
                    "end": item["end"],
                    "t_mid": item["t_mid"],
                    "filename": item["filename"],
                    "caption": caption,
                }
            )
        _emit(
            progress_callback,
            "captions",
            0.30 + 0.36 * (min(start + len(batch), total) / max(total, 1)),
        )
    return results


def _build_caption_index(cache_dir: Path, captions: list[dict[str, Any]]) -> None:
    client = chromadb.PersistentClient(path=str(cache_dir / "caption_index"))
    _recreate_collection(client, "captions")
    collection = client.get_or_create_collection(
        name="captions",
        metadata={"hnsw:space": "cosine"},
    )
    if not captions:
        return
    documents = [item["caption"] for item in captions]
    try:
        embeddings = get_bge().encode_text(documents)
    finally:
        if settings.unload_models_after_use:
            release_bge()
    collection.add(
        ids=[str(item["scene_id"]) for item in captions],
        documents=documents,
        metadatas=[
            {
                "scene_id": int(item["scene_id"]),
                "start": float(item["start"]),
                "end": float(item["end"]),
                "t_mid": float(item["t_mid"]),
                "caption": item["caption"],
                "filename": item["filename"],
            }
            for item in captions
        ],
        embeddings=embeddings.tolist(),
    )


def _extract_and_index_dense_frames(
    vr: decord.VideoReader,
    fps: float,
    duration: float,
    cache_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> int:
    client = chromadb.PersistentClient(path=str(cache_dir / "frame_index"))
    _recreate_collection(client, "frames")
    collection = client.get_or_create_collection(
        name="frames",
        metadata={"hnsw:space": "cosine"},
    )
    if duration <= 0 or settings.dense_fps <= 0:
        return 0

    frames_dir = cache_dir / "frames_dense"
    interval = 1.0 / settings.dense_fps
    timestamps = [round(float(t), 1) for t in np.arange(0.0, duration, interval)]
    count = 0
    total = len(timestamps)
    try:
        for start in range(0, len(timestamps), 32):
            batch_times = timestamps[start : start + 32]
            _emit(progress_callback, "embed", 0.76 + 0.20 * (start / max(total, 1)))
            images: list[Image.Image] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []
            for timestamp in batch_times:
                idx = _time_to_index(timestamp, fps, len(vr))
                image = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
                filename = dense_frame_filename(timestamp)
                image.save(frames_dir / filename, format="JPEG", quality=85, optimize=True)
                images.append(image)
                ids.append(f"{timestamp:.1f}")
                metadatas.append({"timestamp": float(timestamp), "filename": filename})
            embeddings = get_siglip().encode_image(images)
            collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
            count += len(images)
            _emit(progress_callback, "embed", 0.76 + 0.20 * (count / max(total, 1)))
    finally:
        if settings.unload_models_after_use:
            release_siglip()
    return count


def _uniform_scenes(duration: float, chunk_sec: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    scenes = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk_sec)
        scenes.append((start, end))
        start = end
    return scenes


def _time_to_index(timestamp: float, fps: float, total_frames: int) -> int:
    if total_frames <= 0:
        raise ValueError("Cannot extract frames from an empty video")
    return min(total_frames - 1, max(0, int(round(timestamp * fps))))


def _recreate_collection(client: chromadb.PersistentClient, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass


def _clear_previous_artifacts(cache_dir: Path) -> None:
    for child in ("frames_scene", "frames_dense", "caption_index", "frame_index"):
        path = cache_dir / child
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for file_name in ("captions.jsonl", ".done"):
        path = cache_dir / file_name
        if path.exists():
            path.unlink()


def _timed(label: str, fn):
    start = time.perf_counter()
    logger.info("Starting %s", label)
    result = fn()
    logger.info("Finished %s in %.2fs", label, time.perf_counter() - start)
    return result


async def _timed_async(label: str, fn):
    start = time.perf_counter()
    logger.info("Starting %s", label)
    result = await fn()
    logger.info("Finished %s in %.2fs", label, time.perf_counter() - start)
    return result


def _emit(
    progress_callback: ProgressCallback | None,
    stage: str,
    progress: float | None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, stage_label(stage), progress)
