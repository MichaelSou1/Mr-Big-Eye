import asyncio
import json
import logging
import shutil
import subprocess
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
from app.text_assets import build_text_indexes, write_slides, write_transcripts
from app.vqa import generate_caption

logger = logging.getLogger(__name__)


def dense_frame_filename(timestamp: float) -> str:
    return f"t{timestamp:06.1f}.jpg"


ProgressCallback = Callable[[str, str | None, float | None], None]


async def preprocess_video(
    video_id: str,
    video_path: Path,
    progress_callback: ProgressCallback | None = None,
    hotwords: list[str] | None = None,
) -> dict[str, Any]:
    """Run all preprocessing stages and return the saved metadata."""
    cache_dir = ensure_cache_dirs(video_id)
    _clear_previous_artifacts(cache_dir)

    _emit(progress_callback, "probe", 0.02)
    video_path, meta = _timed(
        "probe",
        lambda: _probe_with_remux(video_id, video_path, cache_dir, progress_callback),
    )
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

    captions_task = asyncio.create_task(
        _timed_async("caption scenes", lambda: _caption_scenes(scene_items, progress_callback))
    )
    asr_task = asyncio.create_task(
        asyncio.to_thread(_run_asr_stage, video_id, video_path, hotwords, progress_callback)
    )
    captions = await captions_task
    asr_meta = await asr_task
    meta.update(asr_meta)
    save_meta(video_id, meta)

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
    slide_meta = _timed(
        "slide OCR",
        lambda: _run_slide_stage(video_id, cache_dir, progress_callback),
    )
    meta.update(slide_meta)
    save_meta(video_id, meta)
    set_video_status(video_id, "done")
    _emit(progress_callback, "done", 1.0)
    return meta


def probe_video_lenient(video_path: Path) -> dict[str, Any]:
    """Probe for upload validation; fall back to ffprobe when decord cannot open."""
    try:
        return _probe(video_path)
    except Exception as exc:
        logger.warning("decord probe failed during upload preflight: %s", exc)
        duration = _ffprobe_duration(video_path)
        return {
            "fps": 0.0,
            "duration": duration,
            "total_frames": 0,
            "width": 0,
            "height": 0,
            "probe_fallback": "ffprobe",
        }


def _probe_with_remux(
    video_id: str,
    video_path: Path,
    cache_dir: Path,
    progress_callback: ProgressCallback | None,
) -> tuple[Path, dict[str, Any]]:
    try:
        meta = _probe(video_path)
        meta["remuxed"] = False
        return video_path, meta
    except Exception as exc:
        if not _should_try_remux(exc):
            raise
        _emit(progress_callback, "remux", 0.05)
        remuxed = cache_dir / "remuxed.mp4"
        logger.warning("Probe failed for %s; trying lossless remux: %s", video_id, exc)
        _remux_video(video_path, remuxed)
        meta = _probe(remuxed)
        meta["remuxed"] = True
        meta["source_remux_reason"] = _remux_reason(exc)
        meta["original_source_path"] = str(video_path)
        return remuxed, meta


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


def _should_try_remux(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("eof", "moov", "invalid", "error", "decord"))


def _remux_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "eof" in text:
        return "decord_eof"
    if "moov" in text:
        return "decord_moov"
    return "decord_probe_failed"


def _remux_video(source: Path, dest: Path) -> None:
    import imageio_ffmpeg

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-loglevel",
        "error",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


def _ffprobe_duration(video_path: Path) -> float:
    import imageio_ffmpeg

    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffprobe = ffmpeg.with_name("ffprobe")
    if not ffprobe.exists():
        ffprobe = Path("ffprobe")
    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float((result.stdout or "0").strip() or 0.0)


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


def _run_asr_stage(
    video_id: str,
    video_path: Path,
    hotwords: list[str] | None,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    if not settings.enable_asr:
        return {"has_transcript": False, "asr_skipped": "disabled"}
    _emit(progress_callback, "asr", 0.30)
    start = time.perf_counter()
    try:
        from app.asr import ASR_MODEL_NAME, transcribe

        segments = transcribe(
            video_path,
            language=settings.asr_language,
            hotwords=hotwords,
        )
        segment_count = write_transcripts(video_id, segments)
        chunk_count = build_text_indexes(video_id, kind="transcript") if segment_count else 0
        _emit(progress_callback, "asr", 0.66)
        return {
            "has_transcript": segment_count > 0,
            "asr_model": ASR_MODEL_NAME,
            "asr_duration_sec": round(time.perf_counter() - start, 3),
            "transcript_segment_count": segment_count,
            "transcript_chunk_count": chunk_count,
        }
    except Exception as exc:
        logger.exception("ASR stage failed for %s", video_id)
        return {
            "has_transcript": False,
            "asr_duration_sec": round(time.perf_counter() - start, 3),
            "asr_error": str(exc),
        }


def _run_slide_stage(
    video_id: str,
    cache_dir: Path,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    if not settings.enable_slide_ocr:
        return {"has_slides": False, "slide_ocr_skipped": "disabled"}
    _emit(progress_callback, "slides", 0.88)
    start = time.perf_counter()
    try:
        candidates = _detect_slide_candidates(cache_dir)
        slides = _ocr_slide_candidates(cache_dir, candidates)
        slide_count = write_slides(video_id, slides)
        chunk_count = build_text_indexes(video_id, kind="slide") if slide_count else 0
        _emit(progress_callback, "slides", 0.96)
        return {
            "has_slides": slide_count > 0,
            "slide_count": slide_count,
            "slide_chunk_count": chunk_count,
            "slide_ocr_model": "RapidOCR-PP-OCRv4",
            "slide_ocr_duration_sec": round(time.perf_counter() - start, 3),
        }
    except Exception as exc:
        logger.exception("Slide OCR stage failed for %s", video_id)
        return {
            "has_slides": False,
            "slide_ocr_duration_sec": round(time.perf_counter() - start, 3),
            "slide_ocr_error": str(exc),
        }


def _detect_slide_candidates(cache_dir: Path) -> list[tuple[float, Path]]:
    frames_dir = cache_dir / "frames_dense"
    paths = sorted(frames_dir.glob("t*.jpg"))
    if not paths:
        return []
    sample_step = max(1, int(round(max(settings.dense_fps, 0.1) / max(settings.slide_sample_fps, 0.1))))
    candidates: list[tuple[float, Path]] = []
    prev_hash: int | None = None
    prev_gray: np.ndarray | None = None
    for index, path in enumerate(paths):
        if index % sample_step != 0 and index != 0:
            continue
        timestamp = _timestamp_from_dense_path(path)
        image = Image.open(path).convert("RGB")
        gray = np.asarray(image.convert("L").resize((64, 36), Image.Resampling.BILINEAR))
        dhash = _dhash(gray)
        if prev_hash is None or prev_gray is None:
            candidates.append((timestamp, path))
        else:
            hash_delta = (dhash ^ prev_hash).bit_count()
            frame_delta = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            if (
                hash_delta >= int(settings.slide_change_hash_threshold)
                and frame_delta >= float(settings.slide_change_frame_threshold)
            ):
                candidates.append((timestamp, path))
        prev_hash = dhash
        prev_gray = gray
    return candidates


def _ocr_slide_candidates(cache_dir: Path, candidates: list[tuple[float, Path]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("rapidocr_onnxruntime is not installed") from exc

    ocr = RapidOCR()
    slides_dir = cache_dir / "frames_slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    slides: list[dict[str, Any]] = []
    for slide_id, (timestamp, path) in enumerate(candidates):
        filename = f"slide_{slide_id:04d}.jpg"
        shutil.copy2(path, slides_dir / filename)
        raw, _elapse = ocr(str(path))
        blocks = _rapid_blocks(raw)
        text = "\n".join(block["text"] for block in blocks if block.get("text")).strip()
        if not text:
            continue
        slides.append(
            {
                "slide_id": slide_id,
                "text": text,
                "t_start": float(timestamp),
                "t_end": float(timestamp),
                "filename": filename,
                "ocr_blocks": blocks,
                "source": "rapidocr",
            }
        )
    return slides


def _rapid_blocks(raw: Any) -> list[dict[str, Any]]:
    """RapidOCR returns [[box, text, score], ...] or None for blank images."""
    blocks: list[dict[str, Any]] = []
    if not raw:
        return blocks
    for item in raw:
        try:
            box = item[0]
            text = str(item[1])
            score = float(item[2])
        except (TypeError, IndexError, ValueError):
            continue
        blocks.append({"text": text, "score": score, "box": box})
    return blocks


def _dhash(gray: np.ndarray) -> int:
    resized = np.asarray(Image.fromarray(gray).resize((9, 8), Image.Resampling.BILINEAR))
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def _timestamp_from_dense_path(path: Path) -> float:
    stem = path.stem
    if stem.startswith("t"):
        try:
            return float(stem[1:])
        except ValueError:
            return 0.0
    return 0.0


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
    for child in (
        "frames_scene",
        "frames_dense",
        "caption_index",
        "frame_index",
        "transcript_index",
        "slide_index",
        "frames_slides",
    ):
        path = cache_dir / child
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for file_name in ("captions.jsonl", "transcripts.jsonl", "slides.jsonl", "subtitles.vtt", "remuxed.mp4", ".done"):
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
