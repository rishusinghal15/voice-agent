"""Unit tests for mock adapters (VAD, STT, LLM, TTS) including cancellation tests."""

import asyncio
import struct

import pytest

from backend.app.adapters.llm_mock import MockLLMAdapter
from backend.app.adapters.protocols import (
    Done,
    Message,
    STTFinal,
    STTPartial,
    TextDelta,
    ToolCall,
    VADEventType,
)
from backend.app.adapters.stt_mock import MockSTTAdapter
from backend.app.adapters.tts_mock import MockTTSAdapter
from backend.app.adapters.vad_mock import MockVADProcessor


# ============================================================================
# VAD Adapter Tests
# ============================================================================

def test_mock_vad_silence_and_speech():
    """Verify MockVAD emits SPEECH_START on high energy and SPEECH_END on silence."""
    vad = MockVADProcessor(energy_threshold=500.0)

    # 1. Low energy silence frame (zeros)
    silence_frame = b"\x00\x00" * 320
    assert vad.process(silence_frame) is None
    assert not vad.is_speech

    # 2. High energy frame (amplitude 5000)
    speech_sample = struct.pack("<h", 5000)
    speech_frame = speech_sample * 320
    event1 = vad.process(speech_frame)
    assert event1 is not None
    assert event1.type == VADEventType.SPEECH_START
    assert vad.is_speech

    # 3. Continued speech frame emits None
    assert vad.process(speech_frame) is None
    assert vad.is_speech

    # 4. Return to silence emits SPEECH_END
    event2 = vad.process(silence_frame)
    assert event2 is not None
    assert event2.type == VADEventType.SPEECH_END
    assert not vad.is_speech

    # 5. Continued silence emits None
    assert vad.process(silence_frame) is None

    # 6. Reset
    vad.is_speech = True
    vad.reset()
    assert not vad.is_speech


# ============================================================================
# STT Adapter Tests
# ============================================================================

def test_mock_stt_events():
    """Verify MockSTT emits Partial then Final when finish() is called."""
    async def _run():
        adapter = MockSTTAdapter(
            canned_partial="test partial",
            canned_final="test final",
        )
        stream = await adapter.open_stream(sample_rate=16000)

        # Feed audio
        await stream.send_audio(b"\x01\x02" * 160)
        assert len(stream.audio_chunks) == 1

        # Trigger finish
        await stream.finish()

        # Collect events
        events = []
        async for ev in stream.events():
            events.append(ev)

        assert len(events) == 2
        assert isinstance(events[0], STTPartial)
        assert events[0].text == "test partial"
        assert isinstance(events[1], STTFinal)
        assert events[1].text == "test final"

    asyncio.run(_run())


def test_mock_stt_cancellation():
    """Verify MockSTT handles cancellation gracefully and drains internal queue."""
    async def _run():
        adapter = MockSTTAdapter()
        stream = await adapter.open_stream(sample_rate=16000)

        # Background consumer waiting on events
        async def consumer():
            async for _ in stream.events():
                pass

        task = asyncio.create_task(consumer())
        # Let task run and wait on queue
        await asyncio.sleep(0.01)

        # Cancel the consumer task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())


# ============================================================================
# LLM Adapter Tests
# ============================================================================

def test_mock_llm_canned_reply():
    """Verify MockLLM streams reply word-by-word ending with Done."""
    async def _run():
        adapter = MockLLMAdapter(delay_per_word=0.001)
        messages = [Message(role="user", content="How is the weather today?")]

        events = []
        async for ev in adapter.stream(messages):
            events.append(ev)

        assert len(events) > 1
        # All initial events are TextDelta
        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) > 0
        full_text = "".join(e.text for e in text_events)
        assert "voice assistant" in full_text

        # Last event is Done
        assert isinstance(events[-1], Done)
        assert events[-1].usage is not None

    asyncio.run(_run())


def test_mock_llm_time_tool_call():
    """Verify MockLLM emits ToolCall when user text contains 'time'."""
    async def _run():
        adapter = MockLLMAdapter()
        messages = [Message(role="user", content="What time is it in Tokyo?")]

        events = []
        async for ev in adapter.stream(messages):
            events.append(ev)

        assert len(events) == 2
        tool_call = events[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.name == "get_current_time"
        assert tool_call.args == {"timezone": "Asia/Tokyo"}
        assert isinstance(events[1], Done)

        # Next turn with tool result returns final speech
        tool_messages = [
            Message(role="user", content="What time is it in Tokyo?"),
            Message(role="assistant", content="", tool_calls=[{"name": "get_current_time"}]),
            Message(role="tool", content='{"iso": "2026-10-31T01:00:00", "human": "1:00 AM"}'),
        ]
        followup_events = []
        async for ev in adapter.stream(tool_messages):
            followup_events.append(ev)

        deltas = [e for e in followup_events if isinstance(e, TextDelta)]
        result_text = "".join(d.text for d in deltas)
        assert "current time" in result_text
        assert isinstance(followup_events[-1], Done)

    asyncio.run(_run())


def test_mock_llm_cancellation():
    """Verify MockLLM streaming is cancel-safe."""
    async def _run():
        adapter = MockLLMAdapter(delay_per_word=0.1)
        messages = [Message(role="user", content="Tell me a long story.")]

        async def run_stream():
            async for _ in adapter.stream(messages):
                pass

        task = asyncio.create_task(run_stream())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())


# ============================================================================
# TTS Adapter Tests
# ============================================================================

def test_mock_tts_synthesis():
    """Verify MockTTS yields sine-wave PCM chunks at 24000 Hz proportional to text."""
    async def _run():
        adapter = MockTTSAdapter(sample_rate=24000)
        assert adapter.sample_rate == 24000

        short_chunks = []
        async for chunk in adapter.synthesize("Hi."):
            short_chunks.append(chunk)

        long_chunks = []
        async for chunk in adapter.synthesize("This is a much longer sentence that should take more audio chunks."):
            long_chunks.append(chunk)

        # Longer text yields more chunks
        assert len(long_chunks) > len(short_chunks)

        # Verify chunks unpack as valid 16-bit PCM samples
        chunk = short_chunks[0]
        assert len(chunk) > 0
        assert len(chunk) % 2 == 0
        sample_count = len(chunk) // 2
        samples = struct.unpack(f"<{sample_count}h", chunk)
        assert any(s != 0 for s in samples)  # Not completely silent

    asyncio.run(_run())


def test_mock_tts_cancellation():
    """Verify MockTTS synthesis can be cancelled mid-stream without error."""
    async def _run():
        adapter = MockTTSAdapter(sec_per_char=0.1)  # slow synthesis

        async def run_synth():
            async for _ in adapter.synthesize("A very long sentence for cancellation testing."):
                pass

        task = asyncio.create_task(run_synth())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())

