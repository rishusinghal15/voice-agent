"""WebSocket endpoint for real-time bidirectional voice communication.

Follows design doc §6:
- Route: /ws/voice?session_id=<uuid>
- Rejects unknown session IDs
- Accepts binary PCM frames and JSON text frames
- Handles ping -> pong
- ECHO mode: buffers received PCM, sends back agent_audio_start, binary chunks,
  agent_audio_end, and call_ended {reason: "user_end"} on end_call.
- Every server JSON message includes ts (epoch ms).
"""

import json
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from backend.app import db
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

    # Initialize session state machine
    session = Session(session_id=session_id)
    session.transition(SessionEvent.SESSION_START)

    # Send initial state event to client
    await send_json_message(
        websocket,
        {
            "type": "state",
            "state": session.state.value,
        },
    )

    # Buffer for PCM audio frames in ECHO mode
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

            # Handle binary PCM frames
            if "bytes" in message and message["bytes"] is not None:
                pcm_chunk = message["bytes"]
                if pcm_chunk:
                    pcm_buffer.extend(pcm_chunk)
                continue

            # Handle JSON text frames
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

                # Handle ping -> pong keepalive
                if client_type == "ping":
                    await send_json_message(websocket, {"type": "pong"})
                    continue

                # Handle end_call -> Trigger ECHO playback and close
                if client_type == "end_call":
                    # Step 1: Transition to SPEAKING and announce agent audio
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

                    # Step 2: Stream back buffered binary PCM chunks (640 bytes = 20ms frames)
                    chunk_size = 640
                    if len(pcm_buffer) > 0:
                        for i in range(0, len(pcm_buffer), chunk_size):
                            chunk = bytes(pcm_buffer[i : i + chunk_size])
                            await websocket.send_bytes(chunk)

                    # Step 3: Announce audio end
                    await send_json_message(
                        websocket,
                        {
                            "type": "agent_audio_end",
                            "turn_id": turn_id,
                            "sentence_index": sentence_index,
                        },
                    )

                    # Step 4: Announce call ended with reason "user_end"
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

                    # Persist session end in database
                    await db.end_session(session_id=session_id, end_reason="user_end")

                    # Gracefully close socket
                    await websocket.close()
                    return

    except WebSocketDisconnect:
        pass
    finally:
        # If connection dropped without end_call, persist ws_drop
        if session.state != SessionState.ENDED:
            try:
                session.transition(SessionEvent.WS_DROP)
            except Exception:
                session.state = SessionState.ENDED
            await db.end_session(session_id=session_id, end_reason="ws_drop")

