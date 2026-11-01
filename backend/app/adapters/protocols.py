"""Protocol definitions and event types for voice pipeline adapters.

Follows design doc §3:
- VADProcessor
- STTStream & STTAdapter
- LLMAdapter
- TTSAdapter
- Tool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable


class VADEventType(str, Enum):
    """VAD detection event types."""
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VADEvent:
    """Event emitted by VAD processing."""
    type: VADEventType


@dataclass
class STTPartial:
    """Streaming STT partial transcript event."""
    text: str


@dataclass
class STTFinal:
    """Streaming STT final committed transcript event."""
    text: str


STTEvent = STTPartial | STTFinal


@dataclass
class Message:
    """Conversation history message."""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ToolSpec:
    """Specification of an executable tool."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextDelta:
    """Streamed text token from LLM."""
    text: str


@dataclass
class ToolCall:
    """Streamed tool call invocation from LLM."""
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class Done:
    """LLM completion event with usage info."""
    usage: Optional[Dict[str, Any]] = None


LLMEvent = TextDelta | ToolCall | Done


@runtime_checkable
class VADProcessor(Protocol):
    """Voice Activity Detector protocol."""

    def process(self, frame: bytes) -> Optional[VADEvent]:
        """Process a single PCM frame and return a VADEvent or None."""
        ...


@runtime_checkable
class STTStream(Protocol):
    """Streaming speech-to-text session stream protocol."""

    async def send_audio(self, pcm: bytes) -> None:
        """Feed audio chunk into STT stream."""
        ...

    def events(self) -> AsyncIterator[STTEvent]:
        """Iterate over partial and final STT events."""
        ...

    async def finish(self) -> None:
        """Finalize the stream and wait for trailing events."""
        ...


@runtime_checkable
class STTAdapter(Protocol):
    """Speech-to-text adapter factory protocol."""

    async def open_stream(self, sample_rate: int) -> STTStream:
        """Open a streaming STT session."""
        ...


@runtime_checkable
class LLMAdapter(Protocol):
    """Large language model adapter protocol."""

    def stream(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream tokens and tool calls from LLM given message history and tool specs."""
        ...


@runtime_checkable
class TTSAdapter(Protocol):
    """Text-to-speech adapter protocol."""

    sample_rate: int

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into raw PCM chunks."""
        ...


@runtime_checkable
class Tool(Protocol):
    """Tool execution protocol."""

    spec: ToolSpec

    async def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments and return JSON-serializable dictionary."""
        ...

