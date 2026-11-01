"""Unit tests for keyword-based forced tool choice.

Tests:
1. Transcript matching /\\b(time|date|day|today|clock)\\b/i forces get_current_time
   on first LLM call with Asia/Kolkata default, then auto on second call.
2. Transcript matching /order/i forces get_order_status on first LLM call,
   then auto on second call.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import pytest

from backend.app.adapters.protocols import (
    Done,
    LLMAdapter,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from backend.app.pipeline import VoicePipeline
from backend.app.session import Session, SessionState
from backend.app.tools.executor import create_default_tool_executor


class SpyingLLMAdapter:
    """Mock LLM adapter that records tool_choice arguments passed across invocations."""

    def __init__(self, responses: List[List[LLMEvent]]) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._responses: List[List[LLMEvent]] = list(responses)

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append({
            "messages": list(messages),
            "tools": tools,
            "tool_choice": tool_choice,
        })
        if self._responses:
            current_resp = self._responses.pop(0)
            for ev in current_resp:
                yield ev


class DummyVAD:
    def process(self, frame: bytes):
        return None


class DummySTTStream:
    async def send_audio(self, pcm: bytes) -> None:
        pass

    async def events(self):
        if False:
            yield

    async def finish(self) -> None:
        pass


class DummySTT:
    async def open_stream(self, sample_rate: int):
        return DummySTTStream()


class DummyTTS:
    sample_rate = 16000

    async def synthesize(self, text: str):
        yield b"\x00" * 320


@pytest.mark.asyncio
async def test_forced_tool_choice_time_keyword():
    """Verify time keywords force get_current_time on first LLM call and auto on subsequent calls."""
    spy_llm = SpyingLLMAdapter([
        # Call 1 response: ToolCall without timezone
        [
            ToolCall(id="call_time_1", name="get_current_time", args={}),
            Done(),
        ],
        # Call 2 response: final reply text
        [
            TextDelta(text="It is currently 5:00 PM in Kolkata."),
            Done(),
        ],
    ])

    emitted_json = []

    async def mock_emit_json(data: Dict[str, Any]) -> None:
        emitted_json.append(data)

    async def mock_emit_audio(chunk: bytes) -> None:
        pass

    executor = create_default_tool_executor()
    pipeline = VoicePipeline(
        session_id="test_time_forced",
        emit_json=mock_emit_json,
        emit_audio=mock_emit_audio,
        vad=DummyVAD(),
        stt=DummySTT(),
        llm=spy_llm,
        tts=DummyTTS(),
        executor=executor,
    )

    # Start pipeline and set state to THINKING (as after speech_end & transcript_final)
    await pipeline.start()
    pipeline.session.state = SessionState.THINKING

    # Feed user transcript matching /\b(time|date|day|today|clock)\b/i
    pipeline.history.append({"role": "user", "content": "What day is today?"})
    await pipeline._stream_llm_and_tts()

    # Verify 2 LLM calls occurred
    assert len(spy_llm.calls) == 2, f"Expected 2 LLM calls, got {len(spy_llm.calls)}"

    # First call must force get_current_time
    call_1 = spy_llm.calls[0]
    assert call_1["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_current_time"},
    }

    # Second call must be normal (auto)
    call_2 = spy_llm.calls[1]
    assert call_2["tool_choice"] == "auto"

    # Verify tool call was executed and defaulted to Asia/Kolkata
    tool_events = [e for e in emitted_json if e.get("type") == "tool_event"]
    assert len(tool_events) == 2
    assert tool_events[0]["name"] == "get_current_time"
    assert tool_events[0]["status"] == "started"
    assert tool_events[0]["args"]["timezone"] == "Asia/Kolkata"

    assert tool_events[1]["name"] == "get_current_time"
    assert tool_events[1]["status"] == "ok"
    assert "IST" in tool_events[1]["result"]["human"] or "+05:30" in tool_events[1]["result"]["iso"]

    await pipeline.close()


@pytest.mark.asyncio
async def test_forced_tool_choice_order_keyword():
    """Verify order keyword forces get_order_status on first LLM call and auto on subsequent calls."""
    spy_llm = SpyingLLMAdapter([
        # Call 1 response: ToolCall for order status
        [
            ToolCall(id="call_ord_1", name="get_order_status", args={"order_id": "ord_101"}),
            Done(),
        ],
        # Call 2 response: final reply text
        [
            TextDelta(text="Your order has been shipped and is on the way."),
            Done(),
        ],
    ])

    emitted_json = []

    async def mock_emit_json(data: Dict[str, Any]) -> None:
        emitted_json.append(data)

    async def mock_emit_audio(chunk: bytes) -> None:
        pass

    executor = create_default_tool_executor()
    pipeline = VoicePipeline(
        session_id="test_order_forced",
        emit_json=mock_emit_json,
        emit_audio=mock_emit_audio,
        vad=DummyVAD(),
        stt=DummySTT(),
        llm=spy_llm,
        tts=DummyTTS(),
        executor=executor,
    )

    # Start pipeline and set state to THINKING (as after speech_end & transcript_final)
    await pipeline.start()
    pipeline.session.state = SessionState.THINKING

    # Feed user transcript matching /order/i
    pipeline.history.append({"role": "user", "content": "Please check the status of my order."})
    await pipeline._stream_llm_and_tts()

    # Verify 2 LLM calls occurred
    assert len(spy_llm.calls) == 2, f"Expected 2 LLM calls, got {len(spy_llm.calls)}"

    # First call must force get_order_status
    call_1 = spy_llm.calls[0]
    assert call_1["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_order_status"},
    }

    # Second call must be normal (auto)
    call_2 = spy_llm.calls[1]
    assert call_2["tool_choice"] == "auto"

    # Verify tool events
    tool_events = [e for e in emitted_json if e.get("type") == "tool_event"]
    assert len(tool_events) == 2
    assert tool_events[0]["name"] == "get_order_status"
    assert tool_events[0]["status"] == "started"

    assert tool_events[1]["name"] == "get_order_status"
    assert tool_events[1]["status"] == "ok"
    assert tool_events[1]["result"]["order_id"] == "ord_101"

    await pipeline.close()
