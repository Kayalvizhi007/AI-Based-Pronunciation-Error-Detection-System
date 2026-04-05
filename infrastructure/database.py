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
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id),
    sentence   TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    summary    TEXT
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


def create_user(name: str) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO users (name, created_at) VALUES (?, ?)",
            (name, _now()),
        )
        user_id = cursor.lastrowid
    logger.info("Created user id=%s name='%s'", user_id, name)
    return user_id


def get_user(user_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    found = row is not None
    logger.debug("Lookup user_id=%s found=%s", user_id, found)
    return dict(row) if row else None


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


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
