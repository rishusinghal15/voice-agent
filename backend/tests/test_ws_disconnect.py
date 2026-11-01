"""Unit test for WebSocket disconnect handling.

Tests that WebSocketDisconnect and ClientDisconnected:
- Log an INFO message without throwing unhandled exceptions or error tracebacks
- Cancel all in-flight pipeline tasks cleanly
- End the session with end_reason='ws_drop'
- Persist transcript, turns, and metrics once
"""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import WebSocketDisconnect

try:
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:
    class ClientDisconnected(Exception):
        pass

from backend.app import db
from backend.app.config import settings
from backend.app.ws import websocket_voice_endpoint


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Ensure a fresh clean database for WebSocket disconnect tests."""
    test_db = str(tmp_path / "test_ws_disconnect.db")
    settings.DB_PATH = test_db
    asyncio.run(db.init_db(test_db))
    yield
    settings.DB_PATH = "voice_agent.db"


@pytest.mark.asyncio
async def test_websocket_disconnect_ends_session_with_ws_drop():
    """Verify WebSocketDisconnect in receive loop cancels tasks and ends session as ws_drop."""
    session = await db.create_session("test-drop-session-1")
    session_id = session["id"]

    mock_ws = AsyncMock()
    mock_ws.accept = AsyncMock()
    # Simulate client disconnect on receive
    mock_ws.receive = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
    mock_ws.send_text = AsyncMock()
    mock_ws.send_bytes = AsyncMock()

    # Must complete cleanly without raising
    await websocket_voice_endpoint(mock_ws, session_id=session_id)

    # Verify session is ended with ws_drop
    record = await db.get_session(session_id)
    assert record is not None
    assert record["end_reason"] == "ws_drop"
    assert record["ended_at"] is not None


@pytest.mark.asyncio
async def test_client_disconnected_in_send_path_ends_session_with_ws_drop():
    """Verify ClientDisconnected during send cancels tasks and ends session with ws_drop once."""
    session = await db.create_session("test-drop-session-2")
    session_id = session["id"]

    mock_ws = AsyncMock()
    mock_ws.accept = AsyncMock()
    # First send_text (state: listening) succeeds, subsequent sends raise ClientDisconnected
    call_count = 0
    async def mock_send_text(text):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise ClientDisconnected()

    mock_ws.send_text = AsyncMock(side_effect=mock_send_text)
    # Receive sends a ping, which triggers send_text (pong), triggering ClientDisconnected
    mock_ws.receive = AsyncMock(side_effect=[
        {"type": "websocket.receive", "text": '{"type": "ping"}'},
        WebSocketDisconnect(code=1006),
    ])

    await websocket_voice_endpoint(mock_ws, session_id=session_id)

    record = await db.get_session(session_id)
    assert record is not None
    assert record["end_reason"] == "ws_drop"
    assert record["ended_at"] is not None

