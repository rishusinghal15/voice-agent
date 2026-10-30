"""Integration test running recorded/synthesized audio file through the voice pipeline.

Follows assignment Acceptance Criterion 9:
- One integration test running a recorded audio file through the full pipeline
- Asserts state transitions, emitted events, audio chunks, and database persistence
"""

import asyncio
import io
from pathlib import Path
import wave
import pytest

from backend.app import db
from backend.app.config import settings
from backend.app.pipeline import VoicePipeline
from backend.app.session import SessionState
from scripts.benchmark import generate_silence_pcm, generate_speech_pcm


def create_wav_file(path: Path, duration_s: float = 0.5) -> Path:
    """Create a standard 16kHz 16-bit mono WAV file with speech followed by silence."""
    speech = generate_speech_pcm(duration_s=duration_s, sample_rate=16000)
    silence = generate_silence_pcm(duration_s=0.2, sample_rate=16000)
    pcm_data = speech + silence

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_data)

    return path


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path):
    """Ensure a fresh test database for integration testing."""
    test_db = str(tmp_path / "test_integration.db")
    settings.DB_PATH = test_db
    settings.VAD_MIN_SPEECH_MS = 20
    asyncio.run(db.init_db(test_db))
    yield
    settings.DB_PATH = "voice_agent.db"
    settings.VAD_MIN_SPEECH_MS = 250


@pytest.mark.asyncio
async def test_full_pipeline_audio_file_integration(tmp_path: Path):
    """Feed a recorded WAV audio file through the full voice pipeline."""
    session_id = "test-integration-wav-session"
    await db.create_session(session_id)

    # 1. Create recorded WAV audio file
    wav_file = tmp_path / "user_input.wav"
    create_wav_file(wav_file, duration_s=0.4)

    # 2. Setup pipeline event and audio collectors
    emitted_json = []
    emitted_audio = bytearray()

    async def emit_json(payload):
        emitted_json.append(payload)

    async def emit_audio(chunk):
        emitted_audio.extend(chunk)

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=emit_json,
        emit_audio=emit_audio,
    )
    await pipeline.start()

    # Verify initial state event
    assert pipeline.session.state == SessionState.LISTENING
    assert emitted_json[0]["type"] == "state"
    assert emitted_json[0]["state"] == "listening"

    # 3. Stream frames from the WAV file in 20ms chunks (640 bytes)
    with wave.open(str(wav_file), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2

        chunk_size = 640
        while True:
            frames = wf.readframes(320)  # 320 frames * 2 bytes = 640 bytes
            if not frames:
                break
            await pipeline.handle_audio(frames)
            await asyncio.sleep(0.005)

    # 4. Wait for turn processing task to finish
    if pipeline.active_turn_task:
        await pipeline.active_turn_task

    # 5. Verify emitted events from pipeline
    event_types = [e["type"] for e in emitted_json]
    assert "transcript_final" in event_types
    assert "agent_text" in event_types
    assert "agent_audio_start" in event_types
    assert "agent_audio_end" in event_types
    assert "turn_done" in event_types

    # Verify audio chunks were received
    assert len(emitted_audio) > 0
    assert pipeline.session.state == SessionState.SPEAKING

    # 6. Complete playback
    await pipeline.handle_playback_done(turn_id=1)
    assert pipeline.session.state == SessionState.LISTENING

    # 7. End call
    await pipeline.handle_end_call(reason="user_end")
    assert pipeline.session.state == SessionState.ENDED

    # 8. Assert SQLite persistence
    session_row = await db.get_session(session_id)
    assert session_row is not None
    assert session_row["end_reason"] == "user_end"
    assert session_row["ended_at"] is not None

    turns = await db.get_turns(session_id)
    assert len(turns) >= 2  # user and agent
    roles = [t["role"] for t in turns]
    assert "user" in roles
    assert "agent" in roles

    metrics = await db.get_turn_metrics(session_id)
    assert len(metrics) == 1
    assert metrics[0]["vad_end_ms"] > 0
    assert metrics[0]["total_ms"] > 0

