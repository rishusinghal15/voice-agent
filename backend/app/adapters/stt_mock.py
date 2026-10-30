"""Mock Speech-To-Text (STT) adapter implementation.

Emits canned partial and final transcripts when finish() is called.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List, Optional

from backend.app.adapters.protocols import (
    STTAdapter,
    STTEvent,
    STTFinal,
    STTPartial,
    STTStream,
)


class MockSTTStream:
    """Mock STT streaming session emitting canned events upon finish()."""

    def __init__(
        self,
        sample_rate: int = 16000,
        canned_partial: str = "what time is it",
        canned_final: str = "What time is it in Tokyo?",
    ) -> None:
        self.sample_rate: int = sample_rate
        self.canned_partial: str = canned_partial
        self.canned_final: str = canned_final
        self.audio_chunks: List[bytes] = []
        self._queue: asyncio.Queue[Optional[STTEvent]] = asyncio.Queue()
        self._finished: bool = False

    async def send_audio(self, pcm: bytes) -> None:
        """Buffer incoming PCM frames."""
        if not self._finished:
            self.audio_chunks.append(pcm)

    async def finish(self) -> None:
        """Trigger emission of partial and final transcripts."""
        if self._finished:
            return
        self._finished = True
        await self._queue.put(STTPartial(text=self.canned_partial))
        await self._queue.put(STTFinal(text=self.canned_final))
        await self._queue.put(None)  # Sentinel to terminate events iterator

    async def events(self) -> AsyncIterator[STTEvent]:
        """Asynchronously yield partial and final STT events."""
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                yield item
        except asyncio.CancelledError:
            # Drain queue and cleanup on cancellation
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            raise


class MockSTTAdapter:
    """Factory for MockSTTStream instances."""

    def __init__(
        self,
        canned_partial: str = "what time is it",
        canned_final: str = "What time is it in Tokyo?",
    ) -> None:
        self.canned_partial: str = canned_partial
        self.canned_final: str = canned_final

    async def open_stream(self, sample_rate: int = 16000) -> STTStream:
        """Open a new mock STT stream."""
        return MockSTTStream(
            sample_rate=sample_rate,
            canned_partial=self.canned_partial,
            canned_final=self.canned_final,
        )

