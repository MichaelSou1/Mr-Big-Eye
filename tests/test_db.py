from app import db


def test_create_user_session_and_video(tmp_path, monkeypatch):
    monkeypatch.setattr(db.settings, "data_dir", tmp_path)
    monkeypatch.setattr(db.settings, "database_path", None)

    db.init_db()
    user = db.create_or_get_user("Witch")
    same = db.create_or_get_user("Witch")
    session_id = db.create_session(user["user_id"], title="First question")
    db.register_video("vid001", user["user_id"], "clip.mp4", 12.5)
    db.update_session(session_id, video_id="vid001")

    assert same["user_id"] == user["user_id"]
    assert db.get_user_by_id(user["user_id"])["username"] == "Witch"
    assert db.list_sessions(user["user_id"])[0]["video_id"] == "vid001"
    assert db.list_videos(user["user_id"])[0]["filename"] == "clip.mp4"
