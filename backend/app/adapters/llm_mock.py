"""Mock Large Language Model (LLM) adapter implementation.

Streams canned multi-sentence replies word-by-word, and triggers get_current_time
tool calls when user text contains 'time'.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List, Optional

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
    ) -> AsyncIterator[LLMEvent]:
        """Stream words or tool calls based on user message and history."""
        try:
            # Find last user text and check for tool responses
            has_tool_result = any(m.role == "tool" for m in messages)
            user_messages = [m for m in messages if m.role == "user"]
            last_user_text = user_messages[-1].content if user_messages else ""

            # Check if user asked for time and tool has not been called yet
            if "time" in last_user_text.lower() and not has_tool_result:
                # Detect target timezone from text if present, default to UTC
                tz = "UTC"
                lower_text = last_user_text.lower()
                if "tokyo" in lower_text:
                    tz = "Asia/Tokyo"
                elif "new york" in lower_text or "nyc" in lower_text:
                    tz = "America/New_York"
                elif "london" in lower_text:
                    tz = "Europe/London"

                yield ToolCall(
                    id="call_mock_time_1",
                    name="get_current_time",
                    args={"timezone": tz},
                )
                yield Done(usage={"prompt_tokens": 15, "completion_tokens": 10})
                return

            # Determine response text
            if has_tool_result:
                # After tool execution, craft response referencing time
                reply_text = "The current time is 4:10 PM. I am happy to assist you with anything else you need."
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

