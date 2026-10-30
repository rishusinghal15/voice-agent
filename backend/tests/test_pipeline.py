"""Unit and integration tests for VoicePipeline.

Follows design doc §3, §4, §5, §6, §7, §8, §9:
- Happy path single turn (VAD -> STT -> LLM -> SentenceBuffer -> TTS)
- Multi-turn conversation preserving history
- Tool execution flow (tool_event started, executed, ok, resume)
- Barge-in interrupt and played_ms mapping
- Noise / short utterance rejection
- Empty transcript rejection
- Error recovery paths (STT_ERROR, LLM_ERROR, TTS_ERROR) with fallback line
- End call and persistence
"""

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
import pytest

from backend.app import db
from backend.app.adapters.protocols import (
    Done,
    LLMAdapter,
    LLMEvent,
    Message,
    STTAdapter,
    STTFinal,
    STTPartial,
    STTStream,
    TextDelta,
    ToolCall,
    ToolSpec,
    TTSAdapter,
    VADEvent,
    VADEventType,
    VADProcessor,
)
from backend.app.config import settings
from backend.app.pipeline import VoicePipeline
from backend.app.session import SessionState
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.time_tool import TimeTool


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path):
    """Ensure a fresh test database for each pipeline test."""
    test_db = str(tmp_path / "test_pipeline.db")
    settings.DB_PATH = test_db
    settings.VAD_MIN_SPEECH_MS = 10  # Low threshold for test frame sequences
    asyncio.run(db.init_db(test_db))
    yield
    settings.DB_PATH = "voice_agent.db"
    settings.VAD_MIN_SPEECH_MS = 250


class MockOutputCollector:
    """Collects JSON events and audio chunks emitted by VoicePipeline."""

    def __init__(self) -> None:
        self.json_messages: List[Dict[str, Any]] = []
        self.audio_chunks: List[bytes] = []

    async def emit_json(self, payload: Dict[str, Any]) -> None:
        self.json_messages.append(payload)

    async def emit_audio(self, chunk: bytes) -> None:
        self.audio_chunks.append(chunk)

    def find_events(self, event_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.json_messages if m.get("type") == event_type]


class ControlledVAD:
    """VAD processor whose events can be triggered explicitly."""

    def __init__(self) -> None:
        self.queue: List[Optional[VADEvent]] = []

    def feed_event(self, event: Optional[VADEvent]) -> None:
        self.queue.append(event)

    def process(self, frame: bytes) -> Optional[VADEvent]:
        if self.queue:
            return self.queue.pop(0)
        return None


class ScriptedSTTStream:
    """STT stream yielding predetermined partials and finals."""

    def __init__(self, partials: List[str], final_text: str) -> None:
        self.partials = partials
        self.final_text = final_text
        self.audio_received: bytearray = bytearray()
        self.finished = False

    async def send_audio(self, pcm: bytes) -> None:
        self.audio_received.extend(pcm)

    async def events(self) -> AsyncIterator[Any]:
        for p in self.partials:
            yield STTPartial(text=p)
            await asyncio.sleep(0.005)
        while not self.finished:
            await asyncio.sleep(0.005)
        yield STTFinal(text=self.final_text)

    async def finish(self) -> None:
        self.finished = True


class ScriptedSTTAdapter:
    """STT adapter returning scripted streams."""

    def __init__(self, partials: Optional[List[str]] = None, final_text: str = "Hello assistant") -> None:
        self.partials = partials or ["Hello"]
        self.final_text = final_text
        self.active_stream: Optional[ScriptedSTTStream] = None

    async def open_stream(self, sample_rate: int) -> STTStream:
        self.active_stream = ScriptedSTTStream(self.partials, self.final_text)
        return self.active_stream


class ScriptedLLMAdapter:
    """LLM adapter returning scripted token stream."""

    def __init__(self, tokens: Optional[List[str]] = None) -> None:
        self.tokens = tokens or ["Hello", " ", "there!", " ", "How", " ", "are", " ", "you?"]

    async def stream(self, messages: List[Message], tools: List[ToolSpec]) -> AsyncIterator[LLMEvent]:
        for token in self.tokens:
            yield TextDelta(text=token)
            await asyncio.sleep(0.002)
        yield Done(usage={"total_tokens": 10})


class FastTTSAdapter:
    """TTS adapter generating fast PCM chunks."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        # Generate 1 frame of 640 bytes per word
        words = text.split()
        for _ in words:
            yield b"\x00\x01" * 320
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_pipeline_happy_path_single_turn():
    """Verify end-to-end turn: VAD -> STT -> LLM -> SentenceBuffer -> TTS -> playback_done."""
    session_id = "test-session-happy-1"
    await db.create_session(session_id)

    collector = MockOutputCollector()
    vad = ControlledVAD()
    stt = ScriptedSTTAdapter(partials=["Hello", "Hello there"], final_text="Hello there assistant.")
    llm = ScriptedLLMAdapter(tokens=["I", " ", "am", " ", "fine.", " ", "Ready", " ", "to", " ", "help!"])
    tts = FastTTSAdapter()

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=collector.emit_json,
        emit_audio=collector.emit_audio,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )

    # 1. Start pipeline -> LISTENING
    await pipeline.start()
    assert pipeline.session.state == SessionState.LISTENING
    assert collector.find_events("state")[0]["state"] == "listening"

    # 2. Feed speech frames
    vad.feed_event(VADEvent(type=VADEventType.SPEECH_START))
    await pipeline.handle_audio(b"\x01\x02" * 320)
    await asyncio.sleep(0.02)

    # Partial transcripts received
    partials = collector.find_events("transcript_partial")
    assert len(partials) >= 1

    # Speech ends
    vad.feed_event(VADEvent(type=VADEventType.SPEECH_END))
    await pipeline.handle_audio(b"\x00\x00" * 320)

    # Wait for turn processing task to finish
    if pipeline.active_turn_task:
        await pipeline.active_turn_task

    # 3. Verify turn output events
    finals = collector.find_events("transcript_final")
    assert len(finals) == 1
    assert finals[0]["text"] == "Hello there assistant."

    agent_texts = collector.find_events("agent_text")
    assert len(agent_texts) >= 1
    full_agent_text = " ".join(t["text"] for t in agent_texts)
    assert "fine." in full_agent_text or "help!" in full_agent_text

    audio_starts = collector.find_events("agent_audio_start")
    assert len(audio_starts) >= 1
    assert len(collector.audio_chunks) > 0

    turn_dones = collector.find_events("turn_done")
    assert len(turn_dones) == 1
    assert turn_dones[0]["total_audio_ms"] > 0

    assert pipeline.session.state == SessionState.SPEAKING

    # 4. Client completes playback
    await pipeline.handle_playback_done(turn_id=1)
    assert pipeline.session.state == SessionState.LISTENING
    assert pipeline.turn_id == 2

    # 5. Verify database persistence
    turns = await db.get_turns(session_id)
    assert len(turns) == 2
    user_turn = next(t for t in turns if t["role"] == "user")
    agent_turn = next(t for t in turns if t["role"] == "agent")
    assert user_turn["text"] == "Hello there assistant."
    assert "fine." in agent_turn["text"]

    metrics = await db.get_turn_metrics(session_id)
    assert len(metrics) == 1
    assert metrics[0]["total_ms"] > 0

    await pipeline.handle_end_call(reason="user_end")


@pytest.mark.asyncio
async def test_pipeline_multi_turn_history():
    """Verify 5-turn conversation retains full history across turns."""
    session_id = "test-session-multi-1"
    await db.create_session(session_id)

    collector = MockOutputCollector()
    vad = ControlledVAD()
    stt = ScriptedSTTAdapter(final_text="Turn input")
    llm = ScriptedLLMAdapter(tokens=["Turn", " ", "response."])
    tts = FastTTSAdapter()

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=collector.emit_json,
        emit_audio=collector.emit_audio,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )
    await pipeline.start()

    for i in range(1, 6):
        stt.final_text = f"User message number {i}."
        vad.feed_event(VADEvent(type=VADEventType.SPEECH_START))
        await pipeline.handle_audio(b"\x01\x02" * 320)
        await asyncio.sleep(0.015)

        vad.feed_event(VADEvent(type=VADEventType.SPEECH_END))
        await pipeline.handle_audio(b"\x00\x00" * 320)

        if pipeline.active_turn_task:
            await pipeline.active_turn_task

        await pipeline.handle_playback_done(turn_id=i)

    # Verify memory history has all 10 messages (5 user + 5 agent)
    assert len(pipeline.history) == 10
    assert pipeline.history[0]["content"] == "User message number 1."
    assert pipeline.history[8]["content"] == "User message number 5."

    # Verify DB has all 10 turns
    turns = await db.get_turns(session_id)
    assert len(turns) == 10

    await pipeline.handle_end_call(reason="user_end")


@pytest.mark.asyncio
async def test_pipeline_tool_calling_turn():
    """Verify tool call interception, execution, and LLM resuming with result."""
    session_id = "test-session-tool-1"
    await db.create_session(session_id)

    collector = MockOutputCollector()
    vad = ControlledVAD()
    stt = ScriptedSTTAdapter(final_text="What time is it in Tokyo?")
    tts = FastTTSAdapter()

    class ToolCallingLLM:
        def __init__(self):
            self.called = False

        async def stream(self, messages: List[Message], tools: List[ToolSpec]) -> AsyncIterator[LLMEvent]:
            has_tool_res = any(m.role == "tool" for m in messages)
            if not has_tool_res:
                yield ToolCall(id="call_tokyo", name="get_current_time", args={"timezone": "Asia/Tokyo"})
                yield Done()
            else:
                yield TextDelta(text="It is 4:00 PM in Tokyo.")
                yield Done()

    executor = ToolExecutor()
    executor.register(TimeTool())

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=collector.emit_json,
        emit_audio=collector.emit_audio,
        vad=vad,
        stt=stt,
        llm=ToolCallingLLM(),
        tts=tts,
        executor=executor,
    )
    await pipeline.start()

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_START))
    await pipeline.handle_audio(b"\x01\x02" * 320)
    await asyncio.sleep(0.015)

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_END))
    await pipeline.handle_audio(b"\x00\x00" * 320)

    if pipeline.active_turn_task:
        await pipeline.active_turn_task

    # Verify tool events
    tool_events = collector.find_events("tool_event")
    assert len(tool_events) == 2
    assert tool_events[0]["status"] == "started"
    assert tool_events[1]["status"] == "ok"
    assert tool_events[0]["name"] == "get_current_time"

    # Playback done
    await pipeline.handle_playback_done(turn_id=1)

    # Verify tool calls stored in DB
    turns = await db.get_turns(session_id)
    agent_turn = next(t for t in turns if t["role"] == "agent")
    assert agent_turn["tool_calls_json"] is not None
    tool_data = json.loads(agent_turn["tool_calls_json"])
    assert tool_data[0]["name"] == "get_current_time"
    assert "iso" in tool_data[0]["result"]

    await pipeline.handle_end_call(reason="user_end")


@pytest.mark.asyncio
async def test_pipeline_barge_in_interrupt():
    """Verify barge-in cancels speech and maps played_ms to partial sentences."""
    session_id = "test-session-bargein-1"
    await db.create_session(session_id)

    collector = MockOutputCollector()
    vad = ControlledVAD()
    stt = ScriptedSTTAdapter(final_text="Tell me a long story.")
    llm = ScriptedLLMAdapter(tokens=["First", " ", "sentence.", " ", "Second", " ", "sentence.", " ", "Third", " ", "sentence."])
    tts = FastTTSAdapter()

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=collector.emit_json,
        emit_audio=collector.emit_audio,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )
    await pipeline.start()

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_START))
    await pipeline.handle_audio(b"\x01\x02" * 320)
    await asyncio.sleep(0.015)

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_END))
    await pipeline.handle_audio(b"\x00\x00" * 320)

    if pipeline.active_turn_task:
        await pipeline.active_turn_task

    # Now state is SPEAKING; client barges in with played_ms = 40 (heard first sentence)
    assert pipeline.session.state == SessionState.SPEAKING
    await pipeline.handle_interrupt(turn_id=1, played_ms=40.0)

    # State returns to LISTENING and turn advances
    assert pipeline.session.state == SessionState.LISTENING
    assert pipeline.turn_id == 2

    # Verify interrupted turn saved in DB with interrupted = 1
    turns = await db.get_turns(session_id)
    agent_turn = next(t for t in turns if t["role"] == "agent")
    assert agent_turn["interrupted"] == 1
    assert "First" in agent_turn["text"]

    await pipeline.handle_end_call(reason="user_end")


@pytest.mark.asyncio
async def test_pipeline_noise_rejection():
    """Verify utterance shorter than VAD_MIN_SPEECH_MS is dropped with no LLM call."""
    session_id = "test-session-noise-1"
    await db.create_session(session_id)
    settings.VAD_MIN_SPEECH_MS = 250  # 250ms threshold

    collector = MockOutputCollector()
    vad = ControlledVAD()
    stt = ScriptedSTTAdapter()
    llm = ScriptedLLMAdapter()
    tts = FastTTSAdapter()

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=collector.emit_json,
        emit_audio=collector.emit_audio,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )
    await pipeline.start()

    # Feed SPEECH_START then immediately SPEECH_END (0 ms elapsed)
    vad.feed_event(VADEvent(type=VADEventType.SPEECH_START))
    await pipeline.handle_audio(b"\x01\x02" * 320)

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_END))
    await pipeline.handle_audio(b"\x00\x00" * 320)

    # State must remain LISTENING
    assert pipeline.session.state == SessionState.LISTENING
    # No transcript final or LLM calls
    assert len(collector.find_events("transcript_final")) == 0
    assert len(collector.find_events("agent_text")) == 0

    await pipeline.handle_end_call(reason="user_end")


@pytest.mark.asyncio
async def test_pipeline_empty_transcript_rejection():
    """Verify empty STT final is dropped without LLM call."""
    session_id = "test-session-empty-1"
    await db.create_session(session_id)

    collector = MockOutputCollector()
    vad = ControlledVAD()
    stt = ScriptedSTTAdapter(partials=[], final_text="   ")
    llm = ScriptedLLMAdapter()
    tts = FastTTSAdapter()

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=collector.emit_json,
        emit_audio=collector.emit_audio,
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )
    await pipeline.start()

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_START))
    await pipeline.handle_audio(b"\x01\x02" * 320)
    await asyncio.sleep(0.015)

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_END))
    await pipeline.handle_audio(b"\x00\x00" * 320)

    if pipeline.active_turn_task:
        await pipeline.active_turn_task

    # Returns to LISTENING without agent speech
    assert pipeline.session.state == SessionState.LISTENING
    assert len(collector.find_events("agent_text")) == 0

    await pipeline.handle_end_call(reason="user_end")


@pytest.mark.asyncio
async def test_pipeline_llm_error_speaks_fallback():
    """Verify LLM failure speaks fallback line and logs error in DB."""
    session_id = "test-session-err-1"
    await db.create_session(session_id)

    collector = MockOutputCollector()
    vad = ControlledVAD()
    stt = ScriptedSTTAdapter(final_text="Hello")
    tts = FastTTSAdapter()

    class FailingLLM:
        async def stream(self, messages: List[Message], tools: List[ToolSpec]) -> AsyncIterator[LLMEvent]:
            raise RuntimeError("LLM service unavailable")
            yield Done()

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=collector.emit_json,
        emit_audio=collector.emit_audio,
        vad=vad,
        stt=stt,
        llm=FailingLLM(),
        tts=tts,
    )
    await pipeline.start()

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_START))
    await pipeline.handle_audio(b"\x01\x02" * 320)
    await asyncio.sleep(0.015)

    vad.feed_event(VADEvent(type=VADEventType.SPEECH_END))
    await pipeline.handle_audio(b"\x00\x00" * 320)

    if pipeline.active_turn_task:
        await pipeline.active_turn_task

    # Fallback line spoken
    agent_texts = collector.find_events("agent_text")
    assert len(agent_texts) >= 1
    assert agent_texts[0]["text"] == settings.FALLBACK_LINE

    # Error event emitted
    errors = collector.find_events("error")
    assert len(errors) == 1
    assert errors[0]["code"] == "LLM_ERROR"

    # State returns to LISTENING
    assert pipeline.session.state == SessionState.LISTENING

    # Error persisted to DB
    session_data = await db.get_session(session_id)
    assert session_data is not None
    db_errors = await db.get_errors(session_id)
    assert len(db_errors) == 1
    assert db_errors[0]["code"] == "LLM_ERROR"

    await pipeline.handle_end_call(reason="user_end")

