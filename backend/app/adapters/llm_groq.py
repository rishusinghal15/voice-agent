"""Groq streaming Large Language Model (LLM) adapter.

Implements LLMAdapter protocol using OpenAI-compatible streaming API via httpx.
Streams tokens as TextDelta, tool calls as ToolCall, and finishes with Done.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union
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

logger = logging.getLogger("voice_agent.llm_groq")

# Persistent shared httpx.AsyncClient with IPv4 forced and keep-alive
_shared_client: Optional[httpx.AsyncClient] = None


def get_groq_client() -> httpx.AsyncClient:
    """Return the persistent httpx.AsyncClient for Groq API calls."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        transport = httpx.AsyncHTTPTransport(
            local_address="0.0.0.0",  # Force IPv4 to prevent IPv6 stalls
            http2=False,
            retries=0,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=60.0,
            ),
        )
        http_timeout = httpx.Timeout(
            timeout=15.0,
            connect=5.0,
            read=settings.LLM_TIMEOUT_S,
            write=5.0,
            pool=5.0,
        )
        _shared_client = httpx.AsyncClient(
            transport=transport,
            timeout=http_timeout,
        )
    return _shared_client


async def close_groq_client() -> None:
    """Close the persistent httpx.AsyncClient on shutdown."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


def reset_groq_client(new_client: Optional[httpx.AsyncClient] = None) -> None:
    """Reset or replace the shared persistent Groq client (e.g. for testing)."""
    global _shared_client
    _shared_client = new_client


class GroqLLMAdapter:
    """Streaming LLM adapter communicating with Groq's high-speed inference API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key: str = (
            api_key if api_key is not None else (settings.GROQ_API_KEY or "")
        )
        self.model: str = model or settings.GROQ_MODEL or "qwen/qwen3.8-27b"
        self.base_url: str = base_url.rstrip("/")
        self._custom_client: Optional[httpx.AsyncClient] = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._custom_client is not None and not self._custom_client.is_closed:
            return self._custom_client
        return get_groq_client()

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream TextDelta, ToolCall, or Done events from Groq API."""
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set GROQ_API_KEY in environment or .env."
            )

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
            "max_tokens": 120,
        }
        if api_tools:
            payload["tools"] = api_tools
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice in ("auto", "none", "required"):
                        payload["tool_choice"] = tool_choice
                    else:
                        payload["tool_choice"] = {
                            "type": "function",
                            "function": {"name": tool_choice},
                        }
                elif isinstance(tool_choice, dict):
                    payload["tool_choice"] = tool_choice
                else:
                    payload["tool_choice"] = "auto"
            else:
                payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        client = self._get_client()

        # Connect and timing hooks
        trace_data: Dict[str, float] = {}

        async def trace(event_name: str, info: dict) -> None:
            if event_name == "connection.connect_tcp.started":
                trace_data["t_tcp_start"] = time.monotonic()
            elif event_name in ("connection.start_tls.complete", "connection.connect_tcp.complete"):
                trace_data["t_connect"] = time.monotonic()

        active_tool_calls: Dict[int, Dict[str, Any]] = {}
        first_token_logged = False
        in_think = False

        max_retries = 1
        for attempt in range(max_retries + 1):
            t_start = time.monotonic()
            trace_data.clear()

            req = client.build_request(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                extensions={"trace": trace},
            )

            try:
                response = await client.send(req, stream=True)
                t_headers = time.monotonic()

                # Connection latency: if connection was reused from pool, t_tcp_start won't be set
                if "t_tcp_start" in trace_data and "t_connect" in trace_data:
                    connect_ms = (trace_data["t_connect"] - trace_data["t_tcp_start"]) * 1000.0
                else:
                    connect_ms = 0.0  # Reused keep-alive connection

                headers_ms = (t_headers - t_start) * 1000.0
                status_code = response.status_code
                retry_after_hdr = response.headers.get("retry-after") or response.headers.get("Retry-After")

                if status_code == 429:
                    try:
                        retry_s = min(float(retry_after_hdr), 1.0) if retry_after_hdr else 1.0
                    except (ValueError, TypeError):
                        retry_s = 1.0

                    logger.warning(
                        f"[LLM Groq] HTTP 429 received (Retry-After: {retry_after_hdr}). "
                        f"Retrying in {retry_s:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await response.aclose()
                    if attempt < max_retries:
                        await asyncio.sleep(retry_s)
                        continue
                    response.raise_for_status()

                response.raise_for_status()

                # Stream response lines
                try:
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

                        # Record first token latency and detailed request log
                        if not first_token_logged:
                            has_content = bool(delta.get("content"))
                            has_tools = bool(delta.get("tool_calls"))
                            if has_content or has_tools:
                                first_token_ms = (time.monotonic() - t_start) * 1000.0
                                logger.info(
                                    f"[LLM Groq] Request completed: status={status_code}, "
                                    f"connect_ms={connect_ms:.1f}, headers_ms={headers_ms:.1f}, "
                                    f"first_token_ms={first_token_ms:.1f}, "
                                    f"retry_after={retry_after_hdr or 'none'}, model={self.model}"
                                )
                                first_token_logged = True

                        # Handle reasoning tokens: ignore reasoning_content and strip <think>...</think>
                        content = delta.get("content")
                        if content:
                            if "<think>" in content:
                                parts = content.split("<think>")
                                before = parts[0]
                                after = parts[1]
                                in_think = True
                                if "</think>" in after:
                                    in_think = False
                                    content = before + after.split("</think>")[-1]
                                else:
                                    content = before
                            elif in_think:
                                if "</think>" in content:
                                    in_think = False
                                    content = content.split("</think>")[-1]
                                else:
                                    content = ""
                            elif "</think>" in content:
                                in_think = False
                                content = content.split("</think>")[-1]

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

                finally:
                    await response.aclose()

                # If first token was not logged during streaming (e.g. empty reply)
                if not first_token_logged:
                    total_ms = (time.monotonic() - t_start) * 1000.0
                    logger.info(
                        f"[LLM Groq] Request completed: status={status_code}, "
                        f"connect_ms={connect_ms:.1f}, headers_ms={headers_ms:.1f}, "
                        f"first_token_ms={total_ms:.1f}, "
                        f"retry_after={retry_after_hdr or 'none'}, model={self.model}"
                    )

                # Emit accumulated tool calls
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
                return

            except httpx.ConnectError as err:
                logger.warning(
                    f"[LLM Groq] Connect error: {err}. Retrying in 1.0s (attempt {attempt + 1}/{max_retries + 1})"
                )
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
                    continue
                raise
            except asyncio.CancelledError:
                raise
