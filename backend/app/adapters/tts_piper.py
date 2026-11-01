"""Piper local Text-To-Speech (TTS) adapter.

Implements TTSAdapter protocol using local ONNX voice models via piper-tts.
sample_rate is read dynamically from the voice configuration.
synthesize(text) yields signed 16-bit PCM chunks (~100 ms each).
Executes blocking model synthesis in worker threads via asyncio.to_thread.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, List, Optional
import piper

from backend.app.adapters.protocols import TTSAdapter
from backend.app.config import settings

# Global cached Piper voice instance to avoid re-initializing ONNX session
_CACHED_PIPER_VOICE: Optional[piper.PiperVoice] = None
_CACHED_PIPER_PATH: Optional[str] = None


def get_cached_piper_voice(model_path: str) -> piper.PiperVoice:
    """Retrieve or initialize the cached PiperVoice instance."""
    global _CACHED_PIPER_VOICE, _CACHED_PIPER_PATH
    if _CACHED_PIPER_VOICE is None or _CACHED_PIPER_PATH != model_path:
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Piper voice model not found at '{model_path}'. "
                f"Please ensure models/en_US-lessac-medium.onnx exists."
            )
        _CACHED_PIPER_VOICE = piper.PiperVoice.load(str(p))
        _CACHED_PIPER_PATH = model_path
    return _CACHED_PIPER_VOICE


class PiperTTSAdapter:
    """Streaming TTS adapter using local Piper voice synthesis."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path: str = model_path or settings.PIPER_MODEL_PATH
        self.voice: piper.PiperVoice = get_cached_piper_voice(self.model_path)
        # Sample rate from voice config (e.g. 22050 Hz)
        self.sample_rate: int = self.voice.config.sample_rate

    def _sync_synthesize(self, text: str) -> List[bytes]:
        """Synthesize text into raw 16-bit PCM chunks synchronously."""
        return [chunk.audio_int16_bytes for chunk in self.voice.synthesize(text)]

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text and stream signed 16-bit PCM chunks (~100 ms each)."""
        if not text or not text.strip():
            return

        try:
            # Execute ONNX synthesis in worker thread without blocking event loop
            audio_segments = await asyncio.to_thread(self._sync_synthesize, text)

            # ~100 ms chunk size: sample_rate * 0.1 * 2 bytes/sample (s16le mono)
            chunk_size = int(self.sample_rate * 0.1) * 2

            for segment in audio_segments:
                if not segment:
                    continue
                for i in range(0, len(segment), chunk_size):
                    chunk = segment[i : i + chunk_size]
                    yield chunk
                    # Yield control to event loop to allow cancellation
                    await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            raise


def warmup_piper_model() -> None:
    """Warm up Piper TTS model by loading the voice and synthesizing a dummy word."""
    try:
        adapter = PiperTTSAdapter()
        list(adapter._sync_synthesize("Hello"))
    except Exception as exc:
        import logging
        logging.getLogger("voice_agent").warning(f"Piper TTS warmup failed: {exc}")


