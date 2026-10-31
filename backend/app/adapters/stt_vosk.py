"""Vosk streaming Speech-To-Text (STT) adapter.

Implements STTAdapter and STTStream protocols using offline Kaldi-based Vosk models.
Emits STTPartial for interim results and STTFinal for committed utterances.
Executes blocking model inference in background threads via asyncio.to_thread.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Optional
import vosk

from backend.app.adapters.protocols import (
    STTAdapter,
    STTEvent,
    STTFinal,
    STTPartial,
    STTStream,
)
from backend.app.config import settings

# Global cached Vosk model instance to prevent reloading weights on each turn
_CACHED_VOSK_MODEL: Optional[vosk.Model] = None
_CACHED_VOSK_PATH: Optional[str] = None


def get_cached_vosk_model(model_path: str) -> vosk.Model:
    """Retrieve or initialize the cached Vosk model instance."""
    global _CACHED_VOSK_MODEL, _CACHED_VOSK_PATH
    if _CACHED_VOSK_MODEL is None or _CACHED_VOSK_PATH != model_path:
        p = Path(model_path)
        if not p.is_dir():
            raise FileNotFoundError(
                f"Vosk model directory not found at '{model_path}'. "
                f"Please ensure models/vosk-model-small-en-us-0.15 exists."
            )
        # Suppress verbose Vosk log output
        vosk.SetLogLevel(-1)
        _CACHED_VOSK_MODEL = vosk.Model(str(p))
        _CACHED_VOSK_PATH = model_path
    return _CACHED_VOSK_MODEL


class VoskSTTStream:
    """Streaming speech-to-text session powered by a Vosk recognizer."""

    def __init__(self, model: vosk.Model, sample_rate: int = 16000) -> None:
        self.sample_rate: int = sample_rate
        self.recognizer: vosk.KaldiRecognizer = vosk.KaldiRecognizer(model, sample_rate)
        self._events_queue: asyncio.Queue[Optional[STTEvent]] = asyncio.Queue()
        self._last_partial: str = ""
        self._finished: bool = False

    def _sync_accept_waveform(self, pcm: bytes) -> tuple[str, str]:
        """Process PCM chunk synchronously in worker thread."""
        if self.recognizer.AcceptWaveform(pcm):
            res = json.loads(self.recognizer.Result())
            return "final_phrase", res.get("text", "")
        else:
            partial_res = json.loads(self.recognizer.PartialResult())
            text = partial_res.get("partial", "") or partial_res.get("text", "")
            return "partial", text

    async def send_audio(self, pcm: bytes) -> None:
        """Feed audio chunk into Vosk recognizer without blocking the event loop."""
        if self._finished or not pcm:
            return

        res_type, text = await asyncio.to_thread(self._sync_accept_waveform, pcm)

        if text and text != self._last_partial:
            self._last_partial = text
            await self._events_queue.put(STTPartial(text=text))

    def _sync_final_result(self) -> str:
        """Extract remaining transcript from recognizer."""
        final_res = json.loads(self.recognizer.FinalResult())
        return final_res.get("text", "")

    async def finish(self) -> None:
        """Finalize the stream and emit STTFinal."""
        if self._finished:
            return
        self._finished = True

        final_text = await asyncio.to_thread(self._sync_final_result)
        # Prefer final result text; fallback to last seen partial if empty
        effective_final = final_text or self._last_partial
        await self._events_queue.put(STTFinal(text=effective_final))
        await self._events_queue.put(None)

    async def events(self) -> AsyncIterator[STTEvent]:
        """Asynchronously yield partial and final STT events."""
        try:
            while True:
                event = await self._events_queue.get()
                if event is None:
                    break
                yield event
        except asyncio.CancelledError:
            while not self._events_queue.empty():
                try:
                    self._events_queue.get_nowait()
                except Exception:
                    break
            raise


class VoskSTTAdapter:
    """Factory for Vosk streaming STT sessions."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path: str = model_path or settings.VOSK_MODEL_PATH

    async def open_stream(self, sample_rate: int = 16000) -> STTStream:
        """Open a new streaming Vosk STT session."""
        model = await asyncio.to_thread(get_cached_vosk_model, self.model_path)
        return VoskSTTStream(model=model, sample_rate=sample_rate)

