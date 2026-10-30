"""Deepgram live streaming Speech-To-Text (STT) adapter.

Implements STTAdapter and STTStream protocols using Deepgram's live WebSocket API.
Emits STTPartial for interim results and STTFinal for committed utterances.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Optional
import websockets

from backend.app.adapters.protocols import (
    STTAdapter,
    STTEvent,
    STTFinal,
    STTPartial,
    STTStream,
)
from backend.app.config import settings


class DeepgramSTTStream:
    """Live streaming session communicating with Deepgram STT WebSocket."""

    def __init__(self, api_key: str, sample_rate: int = 16000) -> None:
        self.api_key: str = api_key
        self.sample_rate: int = sample_rate
        self.ws: Optional[Any] = None
        self._events_queue: asyncio.Queue[Optional[STTEvent]] = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None
        self._connected: bool = False

    async def connect(self) -> None:
        """Establish WebSocket connection with Deepgram."""
        url = (
            f"wss://api.deepgram.com/v1/listen?"
            f"encoding=linear16&sample_rate={self.sample_rate}&channels=1"
            f"&interim_results=true&punctuate=true&smart_format=true"
        )
        headers = {"Authorization": f"Token {self.api_key}"}
        self.ws = await websockets.connect(url, extra_headers=headers)
        self._connected = True
        self._receive_task = asyncio.create_task(self._receiver())

    async def _receiver(self) -> None:
        """Receive transcripts from Deepgram WebSocket and push onto queue."""
        try:
            if not self.ws:
                return
            async for raw_message in self.ws:
                if isinstance(raw_message, str):
                    data = json.loads(raw_message)
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")
                        if transcript:
                            is_final = data.get("is_final", False)
                            if is_final:
                                await self._events_queue.put(STTFinal(text=transcript))
                            else:
                                await self._events_queue.put(STTPartial(text=transcript))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            await self._events_queue.put(None)

    async def send_audio(self, pcm: bytes) -> None:
        """Send raw PCM binary audio chunk to Deepgram."""
        if self.ws and self._connected:
            try:
                await self.ws.send(pcm)
            except Exception:
                pass

    async def finish(self) -> None:
        """Signal Deepgram that speech stream has ended."""
        if self.ws and self._connected:
            try:
                # Deepgram closes audio stream when empty bytes or CloseStream JSON is sent
                await self.ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass

    async def events(self) -> AsyncIterator[STTEvent]:
        """Asynchronously yield partial and final STT events."""
        try:
            while True:
                event = await self._events_queue.get()
                if event is None:
                    break
                yield event
        except asyncio.CancelledError:
            if self.ws:
                await self.ws.close()
            raise


class DeepgramSTTAdapter:
    """Factory for Deepgram streaming STT streams."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str = api_key or settings.DEEPGRAM_API_KEY or ""

    async def open_stream(self, sample_rate: int = 16000) -> STTStream:
        """Open a new streaming STT session."""
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY is not set. Please configure DEEPGRAM_API_KEY in environment or .env.")

        stream = DeepgramSTTStream(api_key=self.api_key, sample_rate=sample_rate)
        await stream.connect()
        return stream
