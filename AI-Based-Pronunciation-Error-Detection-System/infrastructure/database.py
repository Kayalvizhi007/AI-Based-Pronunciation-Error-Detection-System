"""
Infrastructure Layer - SQLite Database
--------------------------------------
Handles persistence for users, sessions, and word-level results.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/pronunciation_tutor.db")).expanduser()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT UNIQUE NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    password    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    last_login  TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sentence        TEXT NOT NULL,
    score           INTEGER NOT NULL DEFAULT 0,
    feedback        TEXT,
    audio_file_path TEXT,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    summary         TEXT
);

CREATE TABLE IF NOT EXISTS phoneme_scores (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    phoneme    TEXT NOT NULL,
    score      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date            TEXT NOT NULL,
    total_sessions  INTEGER NOT NULL DEFAULT 0,
    avg_score       REAL NOT NULL DEFAULT 0.0,
    practice_minutes INTEGER NOT NULL DEFAULT 0,
    UNIQUE(user_id, date)
);

CREATE TABLE IF NOT EXISTS word_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER REFERENCES sessions(id),
    word          TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    passed        INTEGER NOT NULL DEFAULT 0,
    best_accuracy REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS phoneme_errors (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    word_result_id   INTEGER REFERENCES word_results(id),
    expected_phoneme TEXT,
    detected_phoneme TEXT,
    error_type       TEXT,
    severity         TEXT,
    confidence       REAL
);
"""


def init_db() -> None:
    """Create tables if they do not exist."""
    with _connect() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", DB_PATH)


def create_user(username: str, email: str, password_hash: str) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, _now()),
        )
        user_id = cursor.lastrowid
    logger.info("Created user id=%s username='%s'", user_id, username)
    return user_id


def get_user(user_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    found = row is not None
    logger.debug("Lookup user_id=%s found=%s", user_id, found)
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    found = row is not None
    logger.debug("Lookup username=%s found=%s", username, found)
    return dict(row) if row else None


def update_last_login(user_id: int) -> None:
    with _connect() as conn:
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (_now(), user_id))
    logger.info("Updated last_login for user_id=%s", user_id)


def verify_user(username: str, plain_password: str) -> Optional[dict]:
    import bcrypt

    user = get_user_by_username(username)
    if not user:
        return None
    if bcrypt.checkpw(plain_password.encode('utf-8'), user['password'].encode('utf-8')):
        update_last_login(user['id'])
        return user
    return None


def start_session(sentence: str, user_id: Optional[int] = None) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (user_id, sentence, started_at) VALUES (?, ?, ?)",
            (user_id, sentence, _now()),
        )
        session_id = cursor.lastrowid
    logger.info("Started session id=%s user_id=%s", session_id, user_id)
    return session_id


def end_session(session_id: int, summary: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
            (_now(), summary, session_id),
        )
    logger.info("Ended session id=%s", session_id)


def save_word_result(
    session_id: int,
    word: str,
    attempts: int,
    passed: bool,
    best_accuracy: float,
    errors: List[dict],
) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO word_results (session_id, word, attempts, passed, best_accuracy)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, word, attempts, int(passed), round(best_accuracy, 4)),
        )
        word_result_id = cursor.lastrowid

        for error in errors:
            conn.execute(
                """
                INSERT INTO phoneme_errors
                (word_result_id, expected_phoneme, detected_phoneme, error_type, severity, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    word_result_id,
                    error.get("expected_phoneme"),
                    error.get("detected_phoneme"),
                    error.get("error_type"),
                    error.get("severity"),
                    error.get("confidence"),
                ),
            )

    logger.info(
        "Saved word result id=%s session_id=%s word='%s' attempts=%d passed=%s errors=%d",
        word_result_id,
        session_id,
        word,
        attempts,
        passed,
        len(errors),
    )
    return word_result_id


def get_session_results(session_id: int) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM word_results WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    logger.info("Fetched %d word results for session_id=%s", len(rows), session_id)
    return [dict(row) for row in rows]


def record_full_session(
    user_id: int,
    sentence: str,
    score: int,
    feedback: str,
    audio_file_path: str | None,
    started_at: str | None = None,
    ended_at: str | None = None,
    summary: str | None = None,
) -> int:
    started_at = started_at or _now()
    ended_at = ended_at or _now()

    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (user_id, sentence, score, feedback, audio_file_path, started_at, ended_at, summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, sentence, score, feedback, audio_file_path, started_at, ended_at, summary),
        )
        session_id = cursor.lastrowid
    logger.info("Recorded full session id %s for user_id=%s", session_id, user_id)
    return session_id


def add_phoneme_score(session_id: int, phoneme: str, score: int) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO phoneme_scores (session_id, phoneme, score) VALUES (?, ?, ?)",
            (session_id, phoneme, score),
        )
        ps_id = cursor.lastrowid
    return ps_id


def upsert_daily_stats(user_id: int, date: str, session_score: int, minutes: int) -> None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT total_sessions, avg_score, practice_minutes FROM daily_stats WHERE user_id = ? AND date = ?",
            (user_id, date),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO daily_stats (user_id, date, total_sessions, avg_score, practice_minutes) VALUES (?, ?, ?, ?, ?)",
                (user_id, date, 1, float(session_score), minutes),
            )
        else:
            total = row["total_sessions"] + 1
            avg = (row["avg_score"] * row["total_sessions"] + session_score) / total
            conn.execute(
                "UPDATE daily_stats SET total_sessions = ?, avg_score = ?, practice_minutes = ? WHERE user_id = ? AND date = ?",
                (total, avg, row["practice_minutes"] + minutes, user_id, date),
            )


def get_user_sessions(user_id: int, limit: int = 50) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_session_counts(user_id: int, days: int = 30) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT date(started_at) AS d, COUNT(*) AS sessions, AVG(score) AS avg_score FROM sessions WHERE user_id = ? AND date(started_at) >= date('now', ? || ' days') GROUP BY d ORDER BY d",
            (user_id, f'-{days}'),
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_daily_stats(user_id: int, days: int = 30) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT date AS d, total_sessions, avg_score, practice_minutes FROM daily_stats WHERE user_id = ? AND date >= date('now', ? || ' days') ORDER BY date",
            (user_id, f'-{days}'),
        ).fetchall()
    return [dict(r) for r in rows]


def get_phoneme_scores(user_id: int) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ps.* FROM phoneme_scores ps JOIN sessions s ON ps.session_id = s.id WHERE s.user_id = ?",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_scores(user_id: int, limit: int = 20) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, started_at, score, sentence FROM sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
