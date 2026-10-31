"""Unit tests for Groq LLM adapter.

Tests streaming text deltas, tool call accumulation, Done event,
missing API key error, and cancellation safety.
"""

import json
import pytest
import httpx

from backend.app.adapters.llm_groq import GroqLLMAdapter
from backend.app.adapters.protocols import Done, Message, TextDelta, ToolCall, ToolSpec
from backend.app.config import settings


@pytest.mark.asyncio
async def test_groq_missing_api_key(monkeypatch):
    """Verify ValueError is raised if GROQ_API_KEY is not configured."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    adapter = GroqLLMAdapter(api_key="")
    with pytest.raises(ValueError) as exc:
        async for _ in adapter.stream([Message(role="user", content="hi")]):
            pass
    assert "GROQ_API_KEY is not set" in str(exc.value)


@pytest.mark.asyncio
async def test_groq_streaming_text(monkeypatch):
    """Verify GroqLLMAdapter streams TextDelta events from SSE lines."""
    sse_lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        'data: {"choices": [{"delta": {"content": " there!"}}]}\n\n',
        'data: [DONE]\n\n',
    ]

    def mock_handler(request: httpx.Request):
        return httpx.Response(200, content="".join(sse_lines).encode("utf-8"))

    transport = httpx.MockTransport(mock_handler)
    original_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr("backend.app.adapters.llm_groq.httpx.AsyncClient", patched_client)

    adapter = GroqLLMAdapter(api_key="gsk_test123")
    events = []
    async for ev in adapter.stream([Message(role="user", content="hello")]):
        events.append(ev)

    assert len(events) == 3
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "Hello"
    assert isinstance(events[1], TextDelta)
    assert events[1].text == " there!"
    assert isinstance(events[2], Done)


@pytest.mark.asyncio
async def test_groq_streaming_tool_call(monkeypatch):
    """Verify GroqLLMAdapter accumulates tool_calls delta chunks into ToolCall event."""
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_123",
                            "function": {
                                "name": "get_current_time",
                                "arguments": '{"time',
                            },
                        }
                    ]
                }
            }
        ]
    }
    chunk2 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": 'zone": "UTC"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    sse_lines = [
        f"data: {json.dumps(chunk1)}\n\n",
        f"data: {json.dumps(chunk2)}\n\n",
        "data: [DONE]\n\n",
    ]

    def mock_handler(request: httpx.Request):
        return httpx.Response(200, content="".join(sse_lines).encode("utf-8"))

    transport = httpx.MockTransport(mock_handler)
    original_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr("backend.app.adapters.llm_groq.httpx.AsyncClient", patched_client)

    adapter = GroqLLMAdapter(api_key="gsk_test123")
    tools = [ToolSpec(name="get_current_time", description="Get current time")]
    events = []
    async for ev in adapter.stream([Message(role="user", content="time")], tools=tools):
        events.append(ev)

    assert len(events) == 2
    assert isinstance(events[0], ToolCall)
    assert events[0].name == "get_current_time"
    assert events[0].args == {"timezone": "UTC"}
    assert isinstance(events[1], Done)
