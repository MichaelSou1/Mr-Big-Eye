from __future__ import annotations

import html
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import chromadb

from app.cache import video_cache_dir
from app.config import settings
from app.models import get_text_embed, release_text_embed

logger = logging.getLogger(__name__)

AssetKind = Literal["transcript", "slide"]


@dataclass(frozen=True)
class TextHit:
    kind: AssetKind
    text: str
    t_start: float
    t_end: float
    score: float
    source: str
    metadata: dict[str, Any]

    @property
    def marker(self) -> str:
        if self.kind == "slide":
            return f"[SLIDE:t={self.t_start:.1f}]"
        return f"[TRANSCRIPT:t={self.t_start:.1f}-{self.t_end:.1f}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "score": self.score,
            "source": self.source,
            "marker": self.marker,
            **self.metadata,
        }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL row at %s:%d", path, line_number)
    return rows


def transcript_path(video_id: str) -> Path:
    return video_cache_dir(video_id) / "transcripts.jsonl"


def slides_path(video_id: str) -> Path:
    return video_cache_dir(video_id) / "slides.jsonl"


def vtt_path(video_id: str) -> Path:
    return video_cache_dir(video_id) / "subtitles.vtt"


def load_transcripts(video_id: str) -> list[dict[str, Any]]:
    return read_jsonl(transcript_path(video_id))


def load_slides(video_id: str) -> list[dict[str, Any]]:
    return read_jsonl(slides_path(video_id))


def write_transcripts(video_id: str, segments: list[dict[str, Any]]) -> int:
    normalized = [_normalize_segment(item, "transcript", index) for index, item in enumerate(segments)]
    count = write_jsonl(transcript_path(video_id), normalized)
    write_vtt(video_id, normalized)
    return count


def write_slides(video_id: str, slides: list[dict[str, Any]]) -> int:
    normalized = [_normalize_segment(item, "slide", index) for index, item in enumerate(slides)]
    return write_jsonl(slides_path(video_id), normalized)


def write_vtt(video_id: str, segments: list[dict[str, Any]]) -> Path:
    path = vtt_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for index, segment in enumerate(segments, start=1):
        start = _format_vtt_time(float(segment.get("t_start", 0.0)))
        end = _format_vtt_time(float(segment.get("t_end", segment.get("t_start", 0.0))))
        text = html.escape(str(segment.get("text") or ""))
        lines.extend([str(index), f"{start} --> {end}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_text_indexes(video_id: str, *, kind: AssetKind) -> int:
    rows = load_transcripts(video_id) if kind == "transcript" else load_slides(video_id)
    chunks = chunk_transcripts(rows) if kind == "transcript" else chunk_slides(rows)
    _write_fts(video_id, kind, chunks)
    try:
        _write_dense_index(video_id, kind, chunks)
    except Exception:
        logger.exception("Dense %s index failed for %s; sparse search remains available", kind, video_id)
    return len(chunks)


def search_text(
    video_id: str,
    query: str,
    *,
    kind: AssetKind,
    top_k: int = 5,
) -> list[TextHit]:
    sparse = _search_fts(video_id, query, kind=kind, top_k=max(top_k * 2, 10))
    dense = _search_dense(video_id, query, kind=kind, top_k=max(top_k * 2, 10))
    return _rrf(sparse, dense)[:top_k]


def search_keyword(
    video_id: str,
    keyword: str,
    *,
    kind: AssetKind,
    top_k: int = 10,
) -> list[TextHit]:
    rows = load_transcripts(video_id) if kind == "transcript" else load_slides(video_id)
    needle = keyword.strip().lower()
    hits: list[TextHit] = []
    if not needle:
        return hits
    for row in rows:
        text = str(row.get("text") or "")
        if needle not in text.lower():
            continue
        hits.append(_row_to_hit(row, kind=kind, score=1.0, source="keyword"))
        if len(hits) >= top_k:
            break
    return hits


def nearby_text(video_id: str, timestamp: float, *, window_sec: float, kind: AssetKind) -> list[TextHit]:
    rows = load_transcripts(video_id) if kind == "transcript" else load_slides(video_id)
    start = max(0.0, float(timestamp) - max(0.0, window_sec))
    end = float(timestamp) + max(0.0, window_sec)
    hits = []
    for row in rows:
        t_start = float(row.get("t_start", 0.0))
        t_end = float(row.get("t_end", t_start))
        if t_start <= end and t_end >= start:
            hits.append(_row_to_hit(row, kind=kind, score=1.0, source="time_window"))
    return hits


def chunk_transcripts(segments: list[dict[str, Any]], *, window: int = 4, overlap: int = 1) -> list[dict[str, Any]]:
    if not segments:
        return []
    chunks: list[dict[str, Any]] = []
    step = max(1, window - overlap)
    for start in range(0, len(segments), step):
        group = segments[start : start + window]
        if not group:
            continue
        text = " ".join(str(item.get("text") or "").strip() for item in group).strip()
        if not text:
            continue
        chunks.append(
            {
                "chunk_id": f"transcript_{start:05d}",
                "text": text,
                "t_start": float(group[0].get("t_start", 0.0)),
                "t_end": float(group[-1].get("t_end", group[-1].get("t_start", 0.0))),
                "source": "asr",
                "segment_start": start,
                "segment_count": len(group),
            }
        )
    return chunks


def chunk_slides(slides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, slide in enumerate(slides):
        text = str(slide.get("text") or "").strip()
        if not text:
            continue
        chunks.append(
            {
                "chunk_id": f"slide_{index:05d}",
                "text": text,
                "t_start": float(slide.get("t_start", slide.get("timestamp", 0.0))),
                "t_end": float(slide.get("t_end", slide.get("timestamp", 0.0))),
                "source": str(slide.get("source") or "rapidocr"),
                "slide_id": int(slide.get("slide_id", index)),
                "filename": str(slide.get("filename") or ""),
            }
        )
    return chunks


def _normalize_segment(item: dict[str, Any], kind: AssetKind, index: int) -> dict[str, Any]:
    t_start = float(item.get("t_start", item.get("start", item.get("timestamp", 0.0))) or 0.0)
    t_end = float(item.get("t_end", item.get("end", t_start)) or t_start)
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    normalized = dict(item)
    normalized["text"] = str(item.get("text") or "").strip()
    normalized["t_start"] = round(t_start, 3)
    normalized["t_end"] = round(t_end, 3)
    normalized["source"] = str(item.get("source") or ("asr" if kind == "transcript" else "rapidocr"))
    if kind == "slide":
        normalized["slide_id"] = int(item.get("slide_id", index))
    return normalized


def _format_vtt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{millis:03d}"


def _fts_path() -> Path:
    path = settings.data_dir / "transcripts.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect_fts() -> sqlite3.Connection:
    conn = sqlite3.connect(_fts_path())
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS text_chunks USING fts5("
        "video_id UNINDEXED, kind UNINDEXED, chunk_id UNINDEXED, "
        "text, t_start UNINDEXED, t_end UNINDEXED, source UNINDEXED, metadata UNINDEXED"
        ")"
    )
    return conn


def _write_fts(video_id: str, kind: AssetKind, chunks: list[dict[str, Any]]) -> None:
    conn = _connect_fts()
    try:
        conn.execute("DELETE FROM text_chunks WHERE video_id = ? AND kind = ?", (video_id, kind))
        conn.executemany(
            "INSERT INTO text_chunks(video_id, kind, chunk_id, text, t_start, t_end, source, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    video_id,
                    kind,
                    chunk["chunk_id"],
                    chunk["text"],
                    float(chunk["t_start"]),
                    float(chunk["t_end"]),
                    chunk.get("source", kind),
                    json.dumps(
                        {key: value for key, value in chunk.items() if key not in {"text", "t_start", "t_end", "source"}},
                        ensure_ascii=False,
                    ),
                )
                for chunk in chunks
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _search_fts(video_id: str, query: str, *, kind: AssetKind, top_k: int) -> list[TextHit]:
    if not query.strip() or not _fts_path().exists():
        return []
    conn = _connect_fts()
    try:
        try:
            rows = conn.execute(
                "SELECT text, t_start, t_end, source, metadata, bm25(text_chunks) AS rank "
                "FROM text_chunks WHERE video_id = ? AND kind = ? AND text_chunks MATCH ? "
                "ORDER BY rank LIMIT ?",
                (video_id, kind, _fts_query(query), int(top_k)),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(
                "SELECT text, t_start, t_end, source, metadata, 0.0 AS rank "
                "FROM text_chunks WHERE video_id = ? AND kind = ? AND text LIKE ? LIMIT ?",
                (video_id, kind, f"%{query.strip()}%", int(top_k)),
            ).fetchall()
    finally:
        conn.close()
    hits: list[TextHit] = []
    for row in rows:
        metadata = _json_dict(row[4])
        hits.append(
            TextHit(
                kind=kind,
                text=str(row[0]),
                t_start=float(row[1]),
                t_end=float(row[2]),
                score=1.0 / (1.0 + abs(float(row[5] or 0.0))),
                source=str(row[3] or "fts"),
                metadata=metadata,
            )
        )
    return hits


def _write_dense_index(video_id: str, kind: AssetKind, chunks: list[dict[str, Any]]) -> None:
    cache_dir = video_cache_dir(video_id)
    path = cache_dir / ("transcript_index" if kind == "transcript" else "slide_index")
    client = chromadb.PersistentClient(path=str(path))
    name = f"{kind}_chunks"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    if not chunks:
        return
    documents = [chunk["text"] for chunk in chunks]
    try:
        embeddings = get_text_embed().encode_text(documents)
    finally:
        if settings.unload_models_after_use:
            release_text_embed()
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=documents,
        metadatas=[
            {
                "kind": kind,
                "t_start": float(chunk["t_start"]),
                "t_end": float(chunk["t_end"]),
                "source": str(chunk.get("source") or kind),
                "metadata": json.dumps(
                    {key: value for key, value in chunk.items() if key not in {"text", "t_start", "t_end", "source"}},
                    ensure_ascii=False,
                ),
            }
            for chunk in chunks
        ],
        embeddings=embeddings.tolist(),
    )


def _search_dense(video_id: str, query: str, *, kind: AssetKind, top_k: int) -> list[TextHit]:
    if not query.strip():
        return []
    cache_dir = video_cache_dir(video_id)
    path = cache_dir / ("transcript_index" if kind == "transcript" else "slide_index")
    try:
        collection = chromadb.PersistentClient(path=str(path)).get_collection(f"{kind}_chunks")
    except Exception:
        return []
    try:
        embedding = get_text_embed().encode_text([query])[0].tolist()
    finally:
        if settings.unload_models_after_use:
            release_text_embed()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=int(top_k),
        include=["documents", "metadatas", "distances"],
    )
    hits: list[TextHit] = []
    for document, metadata, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
        strict=False,
    ):
        meta = dict(metadata or {})
        extra = _json_dict(meta.get("metadata"))
        hits.append(
            TextHit(
                kind=kind,
                text=str(document or ""),
                t_start=float(meta.get("t_start", 0.0)),
                t_end=float(meta.get("t_end", meta.get("t_start", 0.0))),
                score=1.0 - float(distance),
                source=str(meta.get("source") or "dense"),
                metadata=extra,
            )
        )
    return hits


def _rrf(*ranked_lists: list[TextHit], k: int = 60) -> list[TextHit]:
    scores: dict[tuple[str, float, float], float] = {}
    by_key: dict[tuple[str, float, float], TextHit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            key = (hit.kind, round(hit.t_start, 2), round(hit.t_end, 2))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in by_key or hit.score > by_key[key].score:
                by_key[key] = hit
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        TextHit(
            kind=by_key[key].kind,
            text=by_key[key].text,
            t_start=by_key[key].t_start,
            t_end=by_key[key].t_end,
            score=scores[key],
            source=by_key[key].source,
            metadata=by_key[key].metadata,
        )
        for key in ordered
    ]


def _row_to_hit(row: dict[str, Any], *, kind: AssetKind, score: float, source: str) -> TextHit:
    return TextHit(
        kind=kind,
        text=str(row.get("text") or ""),
        t_start=float(row.get("t_start", 0.0)),
        t_end=float(row.get("t_end", row.get("t_start", 0.0))),
        score=float(score),
        source=source,
        metadata={key: value for key, value in row.items() if key not in {"text", "t_start", "t_end", "source"}},
    )


def _fts_query(query: str) -> str:
    terms = [term.strip('"') for term in query.replace("'", " ").replace('"', " ").split() if term.strip()]
    if not terms:
        return query
    return " OR ".join(f'"{term}"' for term in terms[:12])


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
