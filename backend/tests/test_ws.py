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
import time
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

    # 2. Connect (explicitly testing M1 echo mode)
    with client.websocket_connect(f"/ws/voice?session_id={session_id}&echo=true") as ws:
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


def test_ws_pipeline_end_call():
    """Verify client sending end_call cleanly closes pipeline and session."""
    client = TestClient(app)

    res = client.post("/sessions")
    session_id = res.json()["session_id"]

    with client.websocket_connect(f"/ws/voice?session_id={session_id}") as ws:
        state_msg = ws.receive_json()
        assert state_msg["type"] == "state"
        assert state_msg["state"] == "listening"

        # Send end_call
        ws.send_text(json.dumps({"type": "end_call"}))

        ended_msg = ws.receive_json()
        assert ended_msg["type"] == "call_ended"
        assert ended_msg["reason"] == "user_end"

    # DB verified
    detail_res = client.get(f"/sessions/{session_id}")
    assert detail_res.json()["session"]["end_reason"] == "user_end"


def test_ws_pipeline_speech_turn():
    """Verify full pipeline interaction through real WebSocket endpoint."""
    client = TestClient(app)

    res = client.post("/sessions")
    session_id = res.json()["session_id"]

    # Use 0 min speech duration for test so instantaneous frames pass VAD
    settings.VAD_MIN_SPEECH_MS = 0

    with client.websocket_connect(f"/ws/voice?session_id={session_id}") as ws:
        # 1. State: listening
        init_state = ws.receive_json()
        assert init_state["type"] == "state"
        assert init_state["state"] == "listening"

        # 2. Send loud audio frames (RMS > 500)
        loud_chunk = (b"\xff\x7f" * 160) + (b"\x00\x80" * 160)
        for _ in range(5):
            ws.send_bytes(loud_chunk)
            time.sleep(0.01)

        # 3. Send silent audio frame (RMS = 0)
        silent_chunk = b"\x00\x00" * 320
        ws.send_bytes(silent_chunk)

        # 4. Receive events
        events_received = []
        audio_chunks_received = 0

        # Collect events until turn_done
        while True:
            raw = ws.receive()
            if "bytes" in raw and raw["bytes"] is not None:
                audio_chunks_received += 1
            elif "text" in raw and raw["text"] is not None:
                msg = json.loads(raw["text"])
                events_received.append(msg)
                if msg.get("type") == "turn_done":
                    break

        event_types = [e["type"] for e in events_received]
        assert "transcript_final" in event_types
        assert "agent_text" in event_types
        assert "turn_done" in event_types
        assert audio_chunks_received > 0

        # 5. Send playback_done
        ws.send_text(json.dumps({"type": "playback_done", "turn_id": 1}))

        # State should transition back to listening
        listening_msg = ws.receive_json()
        assert listening_msg["type"] == "state"
        assert listening_msg["state"] == "listening"

        # 6. End call
        ws.send_text(json.dumps({"type": "end_call"}))
        ended_msg = ws.receive_json()
        assert ended_msg["type"] == "call_ended"

    settings.VAD_MIN_SPEECH_MS = 250

