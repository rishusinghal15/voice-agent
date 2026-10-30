"""Groq streaming Large Language Model (LLM) adapter.

Implements LLMAdapter protocol using OpenAI-compatible streaming API via httpx.
Streams tokens as TextDelta, tool calls as ToolCall, and finishes with Done.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from backend.app.adapters.protocols import (
    Done,
    LLMAdapter,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from backend.app.config import settings


class GroqLLMAdapter:
    """Streaming LLM adapter communicating with Groq's high-speed inference API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        self.api_key: str = api_key or settings.GROQ_API_KEY or ""
        self.model: str = model
        self.base_url: str = base_url.rstrip("/")

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream TextDelta, ToolCall, or Done events from Groq API."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set GROQ_API_KEY in environment or .env.")

        # Convert protocols.Message to OpenAI format
        api_messages: List[Dict[str, Any]] = []
        for m in messages:
            msg_dict: Dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg_dict["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            api_messages.append(msg_dict)

        # Convert protocols.ToolSpec to OpenAI function format
        api_tools: Optional[List[Dict[str, Any]]] = None
        if tools:
            api_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
            "temperature": 0.6,
        }
        if api_tools:
            payload["tools"] = api_tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        active_tool_calls: Dict[int, Dict[str, Any]] = {}

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_S) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # Stream text content
                        content = delta.get("content")
                        if content:
                            yield TextDelta(text=content)

                        # Accumulate tool calls
                        tool_chunk_list = delta.get("tool_calls", [])
                        for tc in tool_chunk_list:
                            idx = tc.get("index", 0)
                            if idx not in active_tool_calls:
                                active_tool_calls[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "args_str": "",
                                }
                            if tc.get("id"):
                                active_tool_calls[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                active_tool_calls[idx]["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                active_tool_calls[idx]["args_str"] += tc["function"]["arguments"]

            # Emit any accumulated tool calls
            for idx in sorted(active_tool_calls.keys()):
                tc_info = active_tool_calls[idx]
                try:
                    args_parsed = json.loads(tc_info["args_str"])
                except Exception:
                    args_parsed = {}
                yield ToolCall(
                    id=tc_info["id"] or f"call_{idx}",
                    name=tc_info["name"],
                    args=args_parsed,
                )

            yield Done()

        except asyncio.CancelledError:
            raise

