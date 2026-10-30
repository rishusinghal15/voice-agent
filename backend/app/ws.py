"""WebSocket endpoint for real-time bidirectional voice communication.

Follows design doc §6:
- Route: /ws/voice?session_id=<uuid>
- Rejects unknown session IDs
- Accepts binary PCM frames and JSON text frames
- Handles ping -> pong
- Runs VoicePipeline for real-time VAD, STT, LLM, TTS, tools, metrics, and persistence
- Supports echo mode (?echo=true) for M1 audio loopback testing
- Every server JSON message includes ts (epoch ms)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from backend.app import db
from backend.app.pipeline import VoicePipeline
from backend.app.session import Session, SessionEvent, SessionState

router = APIRouter()


def current_epoch_ms() -> int:
    """Return current server time as epoch milliseconds."""
    return int(time.time() * 1000)


async def send_json_message(websocket: WebSocket, payload: Dict[str, Any]) -> None:
    """Send a JSON text frame ensuring 'ts' (epoch ms) is included."""
    if "ts" not in payload:
        payload["ts"] = current_epoch_ms()
    await websocket.send_text(json.dumps(payload))


@router.websocket("/ws/voice")
async def websocket_voice_endpoint(
    websocket: WebSocket,
    session_id: Optional[str] = None,
    mode: Optional[str] = None,
    echo: bool = False,
) -> None:
    """Handle real-time voice streaming WebSocket connection."""
    # Reject connection if session_id is missing
    if not session_id:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Missing session_id query parameter",
        )
        return

    # Reject connection if session_id is not found in database
    session_record = await db.get_session(session_id)
    if not session_record:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=f"Unknown session_id: {session_id}",
        )
        return

    # Accept WebSocket connection
    await websocket.accept()

    # M1 ECHO mode fallback if requested
    if echo or mode == "echo":
        session = Session(session_id=session_id)
        session.transition(SessionEvent.SESSION_START)

        await send_json_message(
            websocket,
            {
                "type": "state",
                "state": session.state.value,
            },
        )

        pcm_buffer = bytearray()
        turn_id = 1
        sentence_index = 0
        sample_rate = 16000

        try:
            while True:
                message = await websocket.receive()
                msg_type = message.get("type")

                if msg_type == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"] is not None:
                    pcm_chunk = message["bytes"]
                    if pcm_chunk:
                        pcm_buffer.extend(pcm_chunk)
                    continue

                if "text" in message and message["text"] is not None:
                    raw_text = message["text"]
                    try:
                        data = json.loads(raw_text)
                    except json.JSONDecodeError:
                        await send_json_message(
                            websocket,
                            {
                                "type": "error",
                                "code": "BAD_JSON",
                                "message": "Malformed JSON frame received",
                                "recoverable": True,
                            },
                        )
                        continue

                    client_type = data.get("type")

                    if client_type == "ping":
                        await send_json_message(websocket, {"type": "pong"})
                        continue

                    if client_type == "end_call":
                        session.state = SessionState.SPEAKING
                        await send_json_message(
                            websocket,
                            {
                                "type": "state",
                                "state": session.state.value,
                            },
                        )
                        await send_json_message(
                            websocket,
                            {
                                "type": "agent_audio_start",
                                "turn_id": turn_id,
                                "sentence_index": sentence_index,
                                "sample_rate": sample_rate,
                                "encoding": "pcm_s16le",
                                "channels": 1,
                            },
                        )

                        chunk_size = 640
                        if len(pcm_buffer) > 0:
                            for i in range(0, len(pcm_buffer), chunk_size):
                                chunk = bytes(pcm_buffer[i : i + chunk_size])
                                await websocket.send_bytes(chunk)

                        await send_json_message(
                            websocket,
                            {
                                "type": "agent_audio_end",
                                "turn_id": turn_id,
                                "sentence_index": sentence_index,
                            },
                        )

                        try:
                            session.transition(SessionEvent.END_CALL)
                        except Exception:
                            session.state = SessionState.ENDED

                        await send_json_message(
                            websocket,
                            {
                                "type": "call_ended",
                                "reason": "user_end",
                            },
                        )

                        await db.end_session(session_id=session_id, end_reason="user_end")
                        await websocket.close()
                        return
        except WebSocketDisconnect:
            pass
        finally:
            if session.state != SessionState.ENDED:
                try:
                    session.transition(SessionEvent.WS_DROP)
                except Exception:
                    session.state = SessionState.ENDED
                await db.end_session(session_id=session_id, end_reason="ws_drop")
        return

    # Real-time Voice Pipeline mode (default)
    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=lambda payload: send_json_message(websocket, payload),
        emit_audio=lambda chunk: websocket.send_bytes(chunk),
    )
    await pipeline.start()

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            # Handle binary PCM audio frames
            if "bytes" in message and message["bytes"] is not None:
                pcm_chunk = message["bytes"]
                if pcm_chunk:
                    await pipeline.handle_audio(pcm_chunk)
                continue

            # Handle JSON text control frames
            if "text" in message and message["text"] is not None:
                raw_text = message["text"]
                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    await send_json_message(
                        websocket,
                        {
                            "type": "error",
                            "code": "BAD_JSON",
                            "message": "Malformed JSON frame received",
                            "recoverable": True,
                        },
                    )
                    continue

                client_type = data.get("type")

                if client_type == "ping":
                    await pipeline.handle_ping()
                    continue

                if client_type == "interrupt":
                    await pipeline.handle_interrupt(
                        turn_id=data.get("turn_id"),
                        played_ms=data.get("played_ms", 0.0),
                    )
                    continue

                if client_type == "playback_done":
                    await pipeline.handle_playback_done(
                        turn_id=data.get("turn_id"),
                    )
                    continue

                if client_type == "end_call":
                    await pipeline.handle_end_call(reason="user_end")
                    await websocket.close()
                    return

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await pipeline.close(reason="ws_drop")
