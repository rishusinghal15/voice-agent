"""Mock Large Language Model (LLM) adapter implementation.

Streams canned multi-sentence replies word-by-word, and triggers get_current_time
tool calls when user text contains 'time'.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from backend.app.adapters.protocols import (
    Done,
    LLMAdapter,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolSpec,
)


class MockLLMAdapter:
    """Mock LLM streaming adapter with tool calling support."""

    def __init__(
        self,
        delay_per_word: float = 0.01,
        canned_reply: str = "Hello! I am your voice assistant. I am ready to help you today. What would you like to do next?",
    ) -> None:
        self.delay_per_word: float = delay_per_word
        self.canned_reply: str = canned_reply

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream words or tool calls based on user message and history."""
        try:
            # Find last user text and check for tool responses
            has_tool_result = any(m.role == "tool" for m in messages)
            user_messages = [m for m in messages if m.role == "user"]
            last_user_text = user_messages[-1].content if user_messages else ""

            forced_tool: Optional[str] = None
            if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                forced_tool = tool_choice
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                forced_tool = tool_choice.get("function", {}).get("name")

            # Check if tool should be called before tool execution
            if not has_tool_result:
                if forced_tool == "get_current_time" or (
                    forced_tool is None and re.search(r"\b(time|date|day|today|clock)\b", last_user_text, re.IGNORECASE)
                ):
                    tz = "Asia/Kolkata"
                    lower_text = last_user_text.lower()
                    if "tokyo" in lower_text:
                        tz = "Asia/Tokyo"
                    elif "new york" in lower_text or "nyc" in lower_text:
                        tz = "America/New_York"
                    elif "london" in lower_text:
                        tz = "Europe/London"
                    elif "utc" in lower_text:
                        tz = "UTC"

                    yield ToolCall(
                        id="call_mock_time_1",
                        name="get_current_time",
                        args={"timezone": tz},
                    )
                    yield Done(usage={"prompt_tokens": 15, "completion_tokens": 10})
                    return

                if forced_tool == "get_order_status" or (
                    forced_tool is None and re.search(r"order", last_user_text, re.IGNORECASE)
                ):
                    match = re.search(r"ord_\d+", last_user_text, re.IGNORECASE)
                    order_id = match.group(0).lower() if match else "ord_101"

                    yield ToolCall(
                        id="call_mock_order_1",
                        name="get_order_status",
                        args={"order_id": order_id},
                    )
                    yield Done(usage={"prompt_tokens": 15, "completion_tokens": 10})
                    return

            # Determine response text
            if has_tool_result:
                tool_msg = next((m for m in reversed(messages) if m.role == "tool"), None)
                if tool_msg and ("ord_" in str(tool_msg.content) or "status" in str(tool_msg.content)):
                    reply_text = "Your order status is confirmed and on schedule."
                elif tool_msg and ("iso" in str(tool_msg.content) or "human" in str(tool_msg.content)):
                    reply_text = "The current time is 4:10 PM. I am happy to assist you with anything else you need."
                else:
                    reply_text = "I have checked that for you. Is there anything else you need?"
            else:
                reply_text = self.canned_reply

            # Stream response word-by-word
            words = reply_text.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield TextDelta(text=token)
                if self.delay_per_word > 0:
                    await asyncio.sleep(self.delay_per_word)

            yield Done(usage={"prompt_tokens": 20, "completion_tokens": len(words)})

        except asyncio.CancelledError:
            # Perform any cleanup before re-raising
            raise

