"""Unit tests for silence timeout, max call duration, and ASGI RuntimeError handling."""

import asyncio
from unittest.mock import AsyncMock
import pytest

from backend.app import db
from backend.app.config import settings
from backend.app.pipeline import VoicePipeline


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Ensure a fresh clean database and mock providers for timeout tests."""
    test_db = str(tmp_path / "test_timeouts.db")
    settings.DB_PATH = test_db
    orig_tts = settings.TTS_PROVIDER
    settings.TTS_PROVIDER = "mock"
    asyncio.run(db.init_db(test_db))
    yield
    settings.DB_PATH = "voice_agent.db"
    settings.TTS_PROVIDER = orig_tts


@pytest.mark.asyncio
async def test_silence_timeout_triggers_goodbye_and_ends_session():
    """Verify that silence exceeding SILENCE_TIMEOUT_SECONDS speaks goodbye and ends session."""
    orig_silence = settings.SILENCE_TIMEOUT_SECONDS
    settings.SILENCE_TIMEOUT_SECONDS = 0.1

    session = await db.create_session("test-silence-timeout")
    session_id = session["id"]

    emitted_json = []
    async def mock_emit_json(payload):
        emitted_json.append(payload)

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=mock_emit_json,
        emit_audio=AsyncMock(),
    )

    try:
        await pipeline.start()
        # Wait for timeout to trigger and pipeline to close
        for _ in range(50):
            if pipeline.closed:
                break
            await asyncio.sleep(0.05)

        # Verify goodbye was spoken and call_ended was emitted
        types = [m.get("type") for m in emitted_json]
        assert "agent_text" in types
        assert "call_ended" in types

        call_ended_msg = next(m for m in emitted_json if m.get("type") == "call_ended")
        assert call_ended_msg["reason"] == "silence_timeout"

        # Verify session is ended in DB
        record = await db.get_session(session_id)
        assert record is not None
        assert record["end_reason"] == "silence_timeout"
        assert record["ended_at"] is not None
    finally:
        await pipeline.close(reason="user_end")
        settings.SILENCE_TIMEOUT_SECONDS = orig_silence


@pytest.mark.asyncio
async def test_max_duration_timeout_triggers_goodbye_and_ends_session():
    """Verify that call duration exceeding MAX_CALL_SECONDS speaks goodbye and ends session."""
    orig_max = settings.MAX_CALL_SECONDS
    orig_silence = settings.SILENCE_TIMEOUT_SECONDS
    settings.MAX_CALL_SECONDS = 0.1
    settings.SILENCE_TIMEOUT_SECONDS = 10.0  # Make silence timeout much longer

    session = await db.create_session("test-max-duration")
    session_id = session["id"]

    emitted_json = []
    async def mock_emit_json(payload):
        emitted_json.append(payload)

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=mock_emit_json,
        emit_audio=AsyncMock(),
    )

    try:
        await pipeline.start()
        # Wait for timeout to trigger and pipeline to close
        for _ in range(50):
            if pipeline.closed:
                break
            await asyncio.sleep(0.05)

        types = [m.get("type") for m in emitted_json]
        assert "agent_text" in types
        assert "call_ended" in types

        call_ended_msg = next(m for m in emitted_json if m.get("type") == "call_ended")
        assert call_ended_msg["reason"] == "max_duration"

        record = await db.get_session(session_id)
        assert record is not None
        assert record["end_reason"] == "max_duration"
        assert record["ended_at"] is not None
    finally:
        await pipeline.close(reason="user_end")
        settings.MAX_CALL_SECONDS = orig_max
        settings.SILENCE_TIMEOUT_SECONDS = orig_silence


@pytest.mark.asyncio
async def test_emit_json_swallows_unexpected_asgi_runtime_error():
    """Verify emit_json swallows RuntimeError when ASGI socket is closed."""
    session = await db.create_session("test-asgi-error")
    session_id = session["id"]

    async def mock_emit_json_fail(payload):
        raise RuntimeError("Unexpected ASGI message 'websocket.send', after sending 'websocket.close'")

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=mock_emit_json_fail,
        emit_audio=AsyncMock(),
    )

    # Calling emit_json should swallow the RuntimeError cleanly
    await pipeline.emit_json({"type": "ping"})

    # And if pipeline is closed, emit_json is a no-op
    pipeline.closed = True
    await pipeline.emit_json({"type": "ping"})
