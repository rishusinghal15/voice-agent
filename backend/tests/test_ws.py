"""Unit and integration tests for WebSocket /ws/voice endpoint.

Follows design doc §6 and assignment M1 requirements:
- Rejects unknown session IDs
- Accepts binary PCM frames and JSON text frames
- Handles ping -> pong
- ECHO mode: on end_call, streams back agent_audio_start, binary PCM chunks,
  agent_audio_end, and call_ended {reason: "user_end"}
- Every server JSON message includes ts (epoch ms)
"""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app import db
from backend.app.config import settings
from backend.app.main import app


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path):
    """Ensure a fresh clean database for WebSocket tests."""
    test_db = str(tmp_path / "test_ws.db")
    settings.DB_PATH = test_db
    asyncio.run(db.init_db(test_db))
    yield
    settings.DB_PATH = "voice_agent.db"


def test_reject_missing_session_id():
    """Verify connection is rejected when session_id is missing."""
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/voice"):
            pass


def test_reject_unknown_session_id():
    """Verify connection is rejected when session_id does not exist in DB."""
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/voice?session_id=invalid-uuid-1234"):
            pass


def test_ws_connect_and_ping_pong():
    """Verify valid connection receives state: listening, and ping receives pong."""
    client = TestClient(app)

    # 1. Create a session first via REST
    res = client.post("/sessions")
    assert res.status_code == 201
    session_id = res.json()["session_id"]

    # 2. Connect to WebSocket
    with client.websocket_connect(f"/ws/voice?session_id={session_id}") as ws:
        # Initial state message must be 'listening' with 'ts'
        init_msg = ws.receive_json()
        assert init_msg["type"] == "state"
        assert init_msg["state"] == "listening"
        assert isinstance(init_msg["ts"], int)

        # Send ping
        ws.send_text(json.dumps({"type": "ping"}))
        pong_msg = ws.receive_json()
        assert pong_msg["type"] == "pong"
        assert isinstance(pong_msg["ts"], int)


def test_ws_echo_mode_on_end_call():
    """Verify PCM frames are buffered and echoed back on end_call with correct schema."""
    client = TestClient(app)

    # 1. Create session
    res = client.post("/sessions")
    session_id = res.json()["session_id"]

    # 2. Connect
    with client.websocket_connect(f"/ws/voice?session_id={session_id}") as ws:
        # Receive initial state
        state_msg = ws.receive_json()
        assert state_msg["type"] == "state"
        assert state_msg["state"] == "listening"

        # Send 2 chunks of binary PCM (e.g. 640 bytes each = 20ms at 16kHz 16-bit mono)
        sample_chunk_1 = b"\x01\x02" * 320
        sample_chunk_2 = b"\x03\x04" * 320
        ws.send_bytes(sample_chunk_1)
        ws.send_bytes(sample_chunk_2)

        # Send end_call
        ws.send_text(json.dumps({"type": "end_call"}))

        # 1. State speaking
        speaking_msg = ws.receive_json()
        assert speaking_msg["type"] == "state"
        assert speaking_msg["state"] == "speaking"
        assert isinstance(speaking_msg["ts"], int)

        # 2. agent_audio_start
        start_msg = ws.receive_json()
        assert start_msg["type"] == "agent_audio_start"
        assert start_msg["sample_rate"] == 16000
        assert start_msg["encoding"] == "pcm_s16le"
        assert start_msg["channels"] == 1
        assert start_msg["turn_id"] == 1
        assert start_msg["sentence_index"] == 0
        assert isinstance(start_msg["ts"], int)

        # 3. Binary chunks (total 1280 bytes = 2 x 640 bytes)
        received_bytes = bytearray()
        chunk1 = ws.receive_bytes()
        received_bytes.extend(chunk1)
        chunk2 = ws.receive_bytes()
        received_bytes.extend(chunk2)

        assert bytes(received_bytes) == sample_chunk_1 + sample_chunk_2

        # 4. agent_audio_end
        end_audio_msg = ws.receive_json()
        assert end_audio_msg["type"] == "agent_audio_end"
        assert end_audio_msg["turn_id"] == 1
        assert end_audio_msg["sentence_index"] == 0
        assert isinstance(end_audio_msg["ts"], int)

        # 5. call_ended
        call_ended_msg = ws.receive_json()
        assert call_ended_msg["type"] == "call_ended"
        assert call_ended_msg["reason"] == "user_end"
        assert isinstance(call_ended_msg["ts"], int)

    # Verify session persisted in DB with end_reason user_end
    detail_res = client.get(f"/sessions/{session_id}")
    assert detail_res.status_code == 200
    details = detail_res.json()
    assert details["session"]["end_reason"] == "user_end"
    assert details["session"]["ended_at"] is not None


def test_ws_disconnect_records_ws_drop():
    """Verify sudden client disconnect records end_reason ws_drop in DB."""
    client = TestClient(app)

    # 1. Create session
    res = client.post("/sessions")
    session_id = res.json()["session_id"]

    # 2. Connect and immediately close socket
    with client.websocket_connect(f"/ws/voice?session_id={session_id}") as ws:
        _ = ws.receive_json()  # state: listening
        # Exit context manager closes socket directly

    # Verify DB recorded ws_drop
    detail_res = client.get(f"/sessions/{session_id}")
    assert detail_res.status_code == 200
    details = detail_res.json()
    assert details["session"]["end_reason"] == "ws_drop"
    assert details["session"]["ended_at"] is not None
