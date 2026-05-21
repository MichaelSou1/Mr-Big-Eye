from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from app.config import settings


def _database_path() -> Path:
    path = settings.database_path or (settings.data_dir / "mr_big_eye.sqlite3")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_database_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def init_db() -> None:
    """Create SQLite tables if they do not already exist."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                video_id TEXT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                duration REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(video_id, user_id),
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_videos_user_created
                ON videos(user_id, created_at DESC);
            """
        )


def create_or_get_user(username: str) -> dict[str, Any]:
    """Insert a user if absent and return its row."""
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("username cannot be empty")

    with _connect() as conn:
        existing = conn.execute(
            "SELECT user_id, username, created_at FROM users WHERE username = ?",
            (clean_username,),
        ).fetchone()
        if existing is not None:
            return dict(existing)

        user_id = uuid4().hex
        try:
            conn.execute(
                "INSERT INTO users (user_id, username) VALUES (?, ?)",
                (user_id, clean_username),
            )
        except sqlite3.IntegrityError:
            existing = conn.execute(
                "SELECT user_id, username, created_at FROM users WHERE username = ?",
                (clean_username,),
            ).fetchone()
            if existing is None:
                raise
            return dict(existing)

        row = conn.execute(
            "SELECT user_id, username, created_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return dict(row)


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id, username, created_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return _row_to_dict(row)


def list_usernames() -> set[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT username FROM users").fetchall()
        return {str(row["username"]) for row in rows}


def create_session(
    user_id: str,
    video_id: str | None = None,
    title: str | None = None,
) -> str:
    """Create a session and return its uuid4 session_id."""
    session_id = uuid4().hex
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, video_id, title)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, user_id, video_id, title),
        )
    return session_id


def get_session(session_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT s.session_id, s.user_id, s.video_id, s.title,
                   s.created_at, s.updated_at, v.filename AS video_filename
            FROM sessions s
            LEFT JOIN videos v ON v.video_id = s.video_id AND v.user_id = s.user_id
            WHERE s.session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return _row_to_dict(row)


def list_sessions(user_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT s.session_id, s.user_id, s.video_id, s.title,
                   s.created_at, s.updated_at, v.filename AS video_filename
            FROM sessions s
            LEFT JOIN videos v ON v.video_id = s.video_id AND v.user_id = s.user_id
            WHERE s.user_id = ?
            ORDER BY s.updated_at DESC, s.created_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def update_session(session_id: str, **fields: Any) -> None:
    allowed = {"video_id", "title"}
    updates = {key: value for key, value in fields.items() if key in allowed}
    assignments = [f"{key} = ?" for key in updates]
    params = list(updates.values())
    assignments.append("updated_at = CURRENT_TIMESTAMP")
    params.append(session_id)

    with _connect() as conn:
        conn.execute(
            f"UPDATE sessions SET {', '.join(assignments)} WHERE session_id = ?",
            params,
        )


def register_video(video_id: str, user_id: str, filename: str, duration: float) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO videos (video_id, user_id, filename, duration)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(video_id, user_id) DO UPDATE SET
                filename = excluded.filename,
                duration = excluded.duration
            """,
            (video_id, user_id, filename, float(duration)),
        )


def list_videos(user_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT video_id, user_id, filename, duration, created_at
            FROM videos
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]

