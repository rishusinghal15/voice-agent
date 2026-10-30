"""Database management and helper functions for SQLite persistence.

Follows design doc §8:
- tables: sessions, turns, turn_metrics, errors
- helper functions for writes and reads
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.config import settings

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  started_at TEXT,
  ended_at TEXT,
  end_reason TEXT,
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS turns (
  session_id TEXT,
  turn_id INTEGER,
  role TEXT,
  text TEXT,
  interrupted INTEGER DEFAULT 0,
  tool_calls_json TEXT,
  PRIMARY KEY (session_id, turn_id, role)
);

CREATE TABLE IF NOT EXISTS turn_metrics (
  session_id TEXT,
  turn_id INTEGER,
  vad_end_ms REAL,
  stt_final_ms REAL,
  llm_first_token_ms REAL,
  tts_first_audio_ms REAL,
  total_ms REAL,
  PRIMARY KEY (session_id, turn_id)
);

CREATE TABLE IF NOT EXISTS errors (
  session_id TEXT,
  turn_id INTEGER,
  stage TEXT,
  code TEXT,
  message TEXT,
  ts TEXT
);
"""


def _get_db_path(db_path: Optional[str] = None) -> str:
    """Resolve database path, ensuring parent directories exist."""
    path = db_path or settings.DB_PATH
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _sync_init_db(target_path: str) -> None:
    with sqlite3.connect(target_path) as conn:
        conn.executescript(CREATE_TABLES_SQL)
        conn.commit()


async def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the SQLite database schema on startup."""
    target_path = _get_db_path(db_path)
    await asyncio.to_thread(_sync_init_db, target_path)


def _sync_create_session(
    target_path: str,
    session_id: str,
    now_iso: str,
    config_json: str,
) -> None:
    with sqlite3.connect(target_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, started_at, ended_at, end_reason, config_json)
            VALUES (?, ?, NULL, NULL, ?)
            """,
            (session_id, now_iso, config_json),
        )
        conn.commit()


async def create_session(
    session_id: str,
    started_at: Optional[str] = None,
    config_json: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new session record in the database."""
    target_path = _get_db_path(db_path)
    now_iso = started_at or datetime.now(timezone.utc).isoformat()
    if config_json is None:
        config_data = {
            "system_prompt": settings.SYSTEM_PROMPT,
            "prompt_version": settings.PROMPT_VERSION,
            "vad_silence_ms": settings.VAD_SILENCE_MS,
            "vad_min_speech_ms": settings.VAD_MIN_SPEECH_MS,
            "max_call_seconds": settings.MAX_CALL_SECONDS,
            "silence_timeout_seconds": settings.SILENCE_TIMEOUT_SECONDS,
            "stt_provider": settings.STT_PROVIDER,
            "llm_provider": settings.LLM_PROVIDER,
            "tts_provider": settings.TTS_PROVIDER,
        }
        config_json = json.dumps(config_data)

    await asyncio.to_thread(
        _sync_create_session,
        target_path,
        session_id,
        now_iso,
        config_json,
    )

    return {
        "id": session_id,
        "started_at": now_iso,
        "ended_at": None,
        "end_reason": None,
        "config_json": config_json,
    }


def _sync_end_session(target_path: str, session_id: str, now_iso: str, end_reason: Optional[str]) -> None:
    with sqlite3.connect(target_path) as conn:
        conn.execute(
            """
            UPDATE sessions
            SET ended_at = ?, end_reason = ?
            WHERE id = ?
            """,
            (now_iso, end_reason, session_id),
        )
        conn.commit()


async def end_session(
    session_id: str,
    ended_at: Optional[str] = None,
    end_reason: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update a session record when the call ends."""
    target_path = _get_db_path(db_path)
    now_iso = ended_at or datetime.now(timezone.utc).isoformat()
    await asyncio.to_thread(_sync_end_session, target_path, session_id, now_iso, end_reason)


def _sync_get_session(target_path: str, session_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT id, started_at, ended_at, end_reason, config_json FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)


async def get_session(session_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single session by id."""
    target_path = _get_db_path(db_path)
    return await asyncio.to_thread(_sync_get_session, target_path, session_id)


def _sync_write_turn(
    target_path: str,
    session_id: str,
    turn_id: int,
    role: str,
    text: str,
    interrupted: int,
    tool_calls_json: Optional[str],
) -> None:
    with sqlite3.connect(target_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO turns (session_id, turn_id, role, text, interrupted, tool_calls_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, turn_id, role, text, interrupted, tool_calls_json),
        )
        conn.commit()


async def write_turn(
    session_id: str,
    turn_id: int,
    role: str,
    text: str,
    interrupted: int = 0,
    tool_calls_json: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Write or update a turn record for a session."""
    target_path = _get_db_path(db_path)
    await asyncio.to_thread(
        _sync_write_turn,
        target_path,
        session_id,
        turn_id,
        role,
        text,
        interrupted,
        tool_calls_json,
    )


def _sync_write_turn_metrics(
    target_path: str,
    session_id: str,
    turn_id: int,
    vad_end_ms: float,
    stt_final_ms: float,
    llm_first_token_ms: float,
    tts_first_audio_ms: float,
    total_ms: float,
) -> None:
    with sqlite3.connect(target_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO turn_metrics (
                session_id, turn_id, vad_end_ms, stt_final_ms,
                llm_first_token_ms, tts_first_audio_ms, total_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_id,
                vad_end_ms,
                stt_final_ms,
                llm_first_token_ms,
                tts_first_audio_ms,
                total_ms,
            ),
        )
        conn.commit()


async def write_turn_metrics(
    session_id: str,
    turn_id: int,
    vad_end_ms: float,
    stt_final_ms: float,
    llm_first_token_ms: float,
    tts_first_audio_ms: float,
    total_ms: float,
    db_path: Optional[str] = None,
) -> None:
    """Write or update turn metrics record."""
    target_path = _get_db_path(db_path)
    await asyncio.to_thread(
        _sync_write_turn_metrics,
        target_path,
        session_id,
        turn_id,
        vad_end_ms,
        stt_final_ms,
        llm_first_token_ms,
        tts_first_audio_ms,
        total_ms,
    )


def _sync_write_error(
    target_path: str,
    session_id: str,
    turn_id: Optional[int],
    stage: str,
    code: str,
    message: str,
    now_iso: str,
) -> None:
    with sqlite3.connect(target_path) as conn:
        conn.execute(
            """
            INSERT INTO errors (session_id, turn_id, stage, code, message, ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, turn_id, stage, code, message, now_iso),
        )
        conn.commit()


async def write_error(
    session_id: str,
    turn_id: Optional[int],
    stage: str,
    code: str,
    message: str,
    ts: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Record an error event into the database."""
    target_path = _get_db_path(db_path)
    now_iso = ts or datetime.now(timezone.utc).isoformat()
    await asyncio.to_thread(
        _sync_write_error,
        target_path,
        session_id,
        turn_id,
        stage,
        code,
        message,
        now_iso,
    )


def _sync_get_turns(target_path: str, session_id: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT session_id, turn_id, role, text, interrupted, tool_calls_json
            FROM turns
            WHERE session_id = ?
            ORDER BY turn_id ASC, role DESC
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


async def get_turns(session_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all turns for a session ordered by turn_id, role."""
    target_path = _get_db_path(db_path)
    return await asyncio.to_thread(_sync_get_turns, target_path, session_id)


def _sync_get_turn_metrics(target_path: str, session_id: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT session_id, turn_id, vad_end_ms, stt_final_ms,
                   llm_first_token_ms, tts_first_audio_ms, total_ms
            FROM turn_metrics
            WHERE session_id = ?
            ORDER BY turn_id ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


async def get_turn_metrics(session_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all metrics for a session ordered by turn_id."""
    target_path = _get_db_path(db_path)
    return await asyncio.to_thread(_sync_get_turn_metrics, target_path, session_id)


def _sync_get_errors(target_path: str, session_id: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT session_id, turn_id, stage, code, message, ts
            FROM errors
            WHERE session_id = ?
            ORDER BY ts ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


async def get_errors(session_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all error entries for a session."""
    target_path = _get_db_path(db_path)
    return await asyncio.to_thread(_sync_get_errors, target_path, session_id)


async def get_session_details(session_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve complete session package: session metadata, turns, metrics, errors."""
    session = await get_session(session_id, db_path)
    if session is None:
        return None
    turns = await get_turns(session_id, db_path)
    metrics = await get_turn_metrics(session_id, db_path)
    errors = await get_errors(session_id, db_path)
    return {
        "session": session,
        "turns": turns,
        "metrics": metrics,
        "errors": errors,
    }
