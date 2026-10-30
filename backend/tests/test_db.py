"""Unit tests for SQLite persistence layer.

Follows design doc §8 schema and db.py helper functions.
"""

import asyncio
from pathlib import Path

from backend.app import db


def test_db_lifecycle(tmp_path: Path) -> None:
    """Test full database lifecycle including table creation, writes and reads."""
    test_db = str(tmp_path / "test_voice.db")

    async def _run() -> None:
        # 1. Initialize DB schema
        await db.init_db(test_db)

        # 2. Create session
        session_id = "test-session-123"
        created = await db.create_session(session_id=session_id, db_path=test_db)
        assert created["id"] == session_id
        assert created["ended_at"] is None
        assert created["end_reason"] is None

        # Retrieve session
        fetched = await db.get_session(session_id, db_path=test_db)
        assert fetched is not None
        assert fetched["id"] == session_id

        # 3. Write turns
        await db.write_turn(
            session_id=session_id,
            turn_id=1,
            role="user",
            text="What time is it in Tokyo?",
            db_path=test_db,
        )
        await db.write_turn(
            session_id=session_id,
            turn_id=1,
            role="agent",
            text="It is 4:10 PM in Tokyo.",
            tool_calls_json='[{"name": "get_current_time"}]',
            db_path=test_db,
        )

        turns = await db.get_turns(session_id, db_path=test_db)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["text"] == "What time is it in Tokyo?"
        assert turns[1]["role"] == "agent"
        assert turns[1]["text"] == "It is 4:10 PM in Tokyo."

        # 4. Write turn metrics
        await db.write_turn_metrics(
            session_id=session_id,
            turn_id=1,
            vad_end_ms=600.0,
            stt_final_ms=190.5,
            llm_first_token_ms=380.0,
            tts_first_audio_ms=240.0,
            total_ms=1410.5,
            db_path=test_db,
        )

        metrics = await db.get_turn_metrics(session_id, db_path=test_db)
        assert len(metrics) == 1
        assert metrics[0]["turn_id"] == 1
        assert metrics[0]["vad_end_ms"] == 600.0
        assert metrics[0]["total_ms"] == 1410.5

        # 5. Write error
        await db.write_error(
            session_id=session_id,
            turn_id=1,
            stage="stt",
            code="STT_TIMEOUT",
            message="STT final transcript exceeded 3s timeout",
            db_path=test_db,
        )

        errors = await db.get_errors(session_id, db_path=test_db)
        assert len(errors) == 1
        assert errors[0]["code"] == "STT_TIMEOUT"
        assert errors[0]["stage"] == "stt"

        # 6. End session
        await db.end_session(session_id=session_id, end_reason="user_end", db_path=test_db)
        updated = await db.get_session(session_id, db_path=test_db)
        assert updated is not None
        assert updated["ended_at"] is not None
        assert updated["end_reason"] == "user_end"

        # 7. Get full session details
        details = await db.get_session_details(session_id, db_path=test_db)
        assert details is not None
        assert details["session"]["id"] == session_id
        assert len(details["turns"]) == 2
        assert len(details["metrics"]) == 1
        assert len(details["errors"]) == 1

    asyncio.run(_run())


def test_get_nonexistent_session(tmp_path: Path) -> None:
    """Verify None is returned when session does not exist."""
    test_db = str(tmp_path / "test_voice.db")

    async def _run() -> None:
        await db.init_db(test_db)
        assert await db.get_session("non-existent", db_path=test_db) is None
        assert await db.get_session_details("non-existent", db_path=test_db) is None

    asyncio.run(_run())
