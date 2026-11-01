"""WebRTC VAD processor implementation using webrtcvad.

Processes 20 ms (640-byte) 16kHz s16le PCM frames and detects speech boundaries.
"""

from __future__ import annotations

from typing import Optional
import webrtcvad

from backend.app.adapters.protocols import VADEvent, VADEventType, VADProcessor
from backend.app.config import settings


class WebRTCVADProcessor:
    """Voice Activity Detector powered by WebRTC VAD."""

    def __init__(
        self,
        mode: int = 3,
        silence_threshold_ms: Optional[float] = None,
        min_speech_ms: Optional[float] = None,
    ) -> None:
        self.mode: int = mode
        self.vad = webrtcvad.Vad(mode)
        self.silence_threshold_ms: float = (
            silence_threshold_ms
            if silence_threshold_ms is not None
            else float(settings.VAD_SILENCE_MS)
        )
        self.min_speech_ms: float = (
            min_speech_ms
            if min_speech_ms is not None
            else float(settings.VAD_MIN_SPEECH_MS)
        )
        self.last_prob: float = 0.0
        self.max_prob: float = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset internal speech state and timers."""
        self._is_speaking = False
        self._consecutive_silence_ms = 0.0
        self.last_prob = 0.0
        self.max_prob = 0.0

    def get_and_reset_max_prob(self) -> float:
        """Return the maximum speech probability since last call and reset it."""
        p = self.max_prob
        self.max_prob = self.last_prob
        return p

    def process(self, frame: bytes) -> Optional[VADEvent]:
        """Process 20ms 16kHz mono 16-bit PCM frame (640 bytes)."""
        # WebRTC VAD requires exact frame length (e.g. 10ms, 20ms, or 30ms)
        # At 16kHz 16-bit mono, 20ms = 320 samples = 640 bytes.
        if len(frame) != 640:
            return None

        is_speech = self.vad.is_speech(frame, 16000)
        prob = 1.0 if is_speech else 0.0
        self.last_prob = prob
        if prob > self.max_prob:
            self.max_prob = prob

        if not self._is_speaking:
            if is_speech:
                self._is_speaking = True
                self._consecutive_silence_ms = 0.0
                return VADEvent(type=VADEventType.SPEECH_START)
        else:
            if not is_speech:
                self._consecutive_silence_ms += 20.0
                if self._consecutive_silence_ms >= self.silence_threshold_ms:
                    self._is_speaking = False
                    self._consecutive_silence_ms = 0.0
                    return VADEvent(type=VADEventType.SPEECH_END)
            else:
                self._consecutive_silence_ms = 0.0

        return None

