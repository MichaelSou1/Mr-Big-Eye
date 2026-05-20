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
