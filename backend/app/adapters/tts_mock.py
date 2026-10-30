"""Mock Text-To-Speech (TTS) adapter implementation.

Synthesizes sine-wave PCM s16le chunks at 24000 Hz with duration proportional to text length.
Cancel-safe on asyncio.CancelledError.
"""

from __future__ import annotations

import asyncio
import math
import struct
from typing import AsyncIterator

from backend.app.adapters.protocols import TTSAdapter


class MockTTSAdapter:
    """Mock TTS adapter generating 24kHz signed 16-bit PCM sine-wave chunks."""

    def __init__(
        self,
        sample_rate: int = 24000,
        frequency_hz: float = 440.0,
        sec_per_char: float = 0.015,
        chunk_duration_ms: float = 20.0,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.frequency_hz: float = frequency_hz
        self.sec_per_char: float = sec_per_char
        self.chunk_duration_ms: float = chunk_duration_ms

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream sine-wave PCM s16le chunks for the synthesized text."""
        try:
            # Duration proportional to text length (minimum 100ms)
            total_duration_s = max(0.1, len(text) * self.sec_per_char)
            total_samples = int(self.sample_rate * total_duration_s)
            samples_per_chunk = int(self.sample_rate * (self.chunk_duration_ms / 1000.0))

            current_sample = 0
            while current_sample < total_samples:
                num_samples = min(samples_per_chunk, total_samples - current_sample)
                samples = []
                for i in range(num_samples):
                    n = current_sample + i
                    # 440Hz sine wave at 50% amplitude
                    amplitude = 32767 * 0.5
                    val = int(amplitude * math.sin(2 * math.pi * self.frequency_hz * (n / self.sample_rate)))
                    # Clamp to int16 range
                    val = max(-32768, min(32767, val))
                    samples.append(val)

                chunk = struct.pack(f"<{len(samples)}h", *samples)
                yield chunk
                current_sample += num_samples

                # Yield control to event loop to allow cancellation
                await asyncio.sleep(0.002)

        except asyncio.CancelledError:
            raise

