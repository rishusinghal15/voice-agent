"""Mock Voice Activity Detector (VAD) implementation.

Emits SPEECH_START and SPEECH_END events based on PCM frame energy.
"""

from __future__ import annotations

import math
import struct
from typing import Optional

from backend.app.adapters.protocols import VADEvent, VADEventType, VADProcessor


class MockVADProcessor:
    """Mock VAD detecting speech start and end transitions based on frame RMS energy."""

    def __init__(self, energy_threshold: float = 500.0) -> None:
        self.energy_threshold: float = energy_threshold
        self.is_speech: bool = False
        self.last_prob: float = 0.0
        self.max_prob: float = 0.0

    def reset(self) -> None:
        """Reset internal speech state."""
        self.is_speech = False
        self.last_prob = 0.0
        self.max_prob = 0.0

    def get_and_reset_max_prob(self) -> float:
        """Return the maximum speech probability since last call and reset it."""
        p = self.max_prob
        self.max_prob = self.last_prob
        return p

    def calculate_energy(self, frame: bytes) -> float:
        """Calculate Root Mean Square (RMS) energy for 16-bit mono PCM."""
        count = len(frame) // 2
        if count == 0:
            return 0.0
        samples = struct.unpack(f"<{count}h", frame[: count * 2])
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / count)

    def process(self, frame: bytes) -> Optional[VADEvent]:
        """Process a PCM frame and emit SPEECH_START or SPEECH_END on state changes."""
        rms = self.calculate_energy(frame)
        speech_detected = rms >= self.energy_threshold
        prob = 1.0 if speech_detected else 0.0
        self.last_prob = prob
        if prob > self.max_prob:
            self.max_prob = prob

        if speech_detected and not self.is_speech:
            self.is_speech = True
            return VADEvent(type=VADEventType.SPEECH_START)

        if not speech_detected and self.is_speech:
            self.is_speech = False
            return VADEvent(type=VADEventType.SPEECH_END)

        return None

