import importlib
import re

from app import cache


def test_video_id_from_file_is_stable(tmp_path):
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"same bytes")

    first = cache.video_id_from_file(path)
    second = cache.video_id_from_file(path)

    assert first == second
    assert re.fullmatch(r"[0-9a-f]{16}", first)


def test_ensure_cache_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.settings, "data_dir", tmp_path)

    root = cache.ensure_cache_dirs("abc")

    assert root == tmp_path / "cache" / "abc"
    for name in ("frames_scene", "frames_dense", "caption_index", "frame_index"):
        assert (root / name).is_dir()


def test_done_status_survives_module_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.settings, "data_dir", tmp_path)
    cache.set_video_status("abc", "done")
    reloaded = importlib.reload(cache)
    monkeypatch.setattr(reloaded.settings, "data_dir", tmp_path)

    assert reloaded.get_video_status("abc") == "done"


def test_preprocess_video_marks_status_done(tmp_path, monkeypatch):
    """preprocess_video must call set_video_status('done') so CLI ingest skips
    the manual .done workaround the smoke run had to use."""
    import asyncio

    from app import preprocess

    monkeypatch.setattr(preprocess.settings, "data_dir", tmp_path)
    monkeypatch.setattr(cache.settings, "data_dir", tmp_path)

    monkeypatch.setattr(preprocess, "_clear_previous_artifacts", lambda cache_dir: None)
    monkeypatch.setattr(
        preprocess,
        "_probe",
        lambda path: {"fps": 1.0, "duration": 1.0, "total_frames": 1, "width": 8, "height": 8},
    )
    monkeypatch.setattr(preprocess, "_detect_scenes", lambda *a, **kw: [(0.0, 1.0)])

    class FakeVR(list):
        def get_avg_fps(self):
            return 1.0

    fake_vr = FakeVR([type("F", (), {"asnumpy": lambda self: __import__("numpy").zeros((8, 8, 3), dtype="uint8")})()])
    monkeypatch.setattr(preprocess.decord, "VideoReader", lambda *a, **kw: fake_vr)

    async def fake_caption(image):
        return "stub caption"

    monkeypatch.setattr(preprocess, "generate_caption", fake_caption)
    monkeypatch.setattr(preprocess, "_build_caption_index", lambda *a, **kw: None)
    monkeypatch.setattr(
        preprocess,
        "_extract_and_index_dense_frames",
        lambda *a, **kw: 0,
    )

    asyncio.run(preprocess.preprocess_video("abc123", tmp_path / "fake.mp4"))

    assert cache.get_video_status("abc123") == "done"
