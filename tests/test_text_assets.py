from app.config import settings
from app import text_assets


def test_transcript_assets_write_vtt_chunk_and_sparse_search(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    video_id = "vid-text-assets"
    segments = [
        {"text": "REINFORCE estimates returns.", "t_start": 1.0, "t_end": 2.0},
        {"text": "A2C adds a learned baseline.", "t_start": 2.0, "t_end": 3.0},
    ]

    assert text_assets.write_transcripts(video_id, segments) == 2
    loaded = text_assets.load_transcripts(video_id)
    assert loaded[0]["source"] == "asr"
    assert text_assets.vtt_path(video_id).read_text(encoding="utf-8").startswith("WEBVTT")

    chunks = text_assets.chunk_transcripts(loaded, window=2, overlap=1)
    assert chunks[0]["t_start"] == 1.0
    assert "A2C" in chunks[0]["text"]

    text_assets._write_fts(video_id, "transcript", chunks)
    hits = text_assets._search_fts(video_id, "REINFORCE", kind="transcript", top_k=3)
    assert hits
    assert hits[0].kind == "transcript"
    assert "[TRANSCRIPT:t=" in hits[0].marker


def test_slide_assets_keyword_and_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    video_id = "vid-slide-assets"

    assert text_assets.write_slides(
        video_id,
        [{"text": "TD target versus return", "timestamp": 72.0, "source": "rapidocr"}],
    ) == 1

    hits = text_assets.search_keyword(video_id, "TD target", kind="slide")
    assert len(hits) == 1
    assert hits[0].marker == "[SLIDE:t=72.0]"
