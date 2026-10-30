"""Per-turn latency metrics timer and statistical analysis helpers.

Follows design doc §7:
- Timestamps captured with time.monotonic() per turn:
  t_speech_start, t_vad_end, t_stt_final, t_llm_first_token,
  t_first_sentence_ready, t_tts_first_audio.
- Exact formulas from design §7:
  vad_end_ms = silence_threshold
  stt_final_ms = t_stt_final - t_vad_end
  llm_first_token_ms = t_llm_first_token - t_stt_final
  tts_first_audio_ms = t_tts_first_audio - t_first_sentence_ready
  total_ms = t_tts_first_audio - (t_vad_end - silence_threshold)
  total_ms_from_vad_end = t_tts_first_audio - t_vad_end
- Percentile helpers (median, p90) for benchmarks and latency reports.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from backend.app.config import settings


@dataclass
class TurnMetrics:
    """Calculated latency metrics for a single conversational turn in milliseconds."""

    vad_end_ms: float
    stt_final_ms: float
    llm_first_token_ms: float
    tts_first_audio_ms: float
    total_ms: float
    total_ms_from_vad_end: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "vad_end_ms": self.vad_end_ms,
            "stt_final_ms": self.stt_final_ms,
            "llm_first_token_ms": self.llm_first_token_ms,
            "tts_first_audio_ms": self.tts_first_audio_ms,
            "total_ms": self.total_ms,
            "total_ms_from_vad_end": self.total_ms_from_vad_end,
        }


class TurnTimer:
    """Captures timestamps with monotonic clock and computes turn latency metrics."""

    def __init__(self, silence_threshold_ms: Optional[float] = None) -> None:
        self.silence_threshold_ms: float = (
            silence_threshold_ms
            if silence_threshold_ms is not None
            else float(settings.VAD_SILENCE_MS)
        )
        self.t_speech_start: Optional[float] = None
        self.t_vad_end: Optional[float] = None
        self.t_stt_final: Optional[float] = None
        self.t_llm_first_token: Optional[float] = None
        self.t_first_sentence_ready: Optional[float] = None
        self.t_tts_first_audio: Optional[float] = None

    def mark_speech_start(self, t: Optional[float] = None) -> float:
        """Stamp time when speech was first detected."""
        self.t_speech_start = t if t is not None else time.monotonic()
        return self.t_speech_start

    def mark_vad_end(self, t: Optional[float] = None) -> float:
        """Stamp time when VAD declared end of utterance."""
        self.t_vad_end = t if t is not None else time.monotonic()
        return self.t_vad_end

    def mark_stt_final(self, t: Optional[float] = None) -> float:
        """Stamp time when STT committed final transcript."""
        self.t_stt_final = t if t is not None else time.monotonic()
        return self.t_stt_final

    def mark_llm_first_token(self, t: Optional[float] = None) -> float:
        """Stamp time when LLM generated its first token."""
        self.t_llm_first_token = t if t is not None else time.monotonic()
        return self.t_llm_first_token

    def mark_first_sentence_ready(self, t: Optional[float] = None) -> float:
        """Stamp time when the first complete sentence was buffered and ready for TTS."""
        self.t_first_sentence_ready = t if t is not None else time.monotonic()
        return self.t_first_sentence_ready

    def mark_tts_first_audio(self, t: Optional[float] = None) -> float:
        """Stamp time when the first byte of synthesized audio was sent."""
        self.t_tts_first_audio = t if t is not None else time.monotonic()
        return self.t_tts_first_audio

    def compute(self) -> TurnMetrics:
        """Compute latency metrics using exact formulas from design doc §7."""
        silence_threshold_s = self.silence_threshold_ms / 1000.0

        # vad_end_ms = silence_threshold (the silence wait)
        vad_end_ms = self.silence_threshold_ms

        # stt_final_ms = t_stt_final - t_vad_end
        if self.t_stt_final is not None and self.t_vad_end is not None:
            stt_final_ms = max(0.0, (self.t_stt_final - self.t_vad_end) * 1000.0)
        else:
            stt_final_ms = 0.0

        # llm_first_token_ms = t_llm_first_token - t_stt_final
        if self.t_llm_first_token is not None and self.t_stt_final is not None:
            llm_first_token_ms = max(0.0, (self.t_llm_first_token - self.t_stt_final) * 1000.0)
        else:
            llm_first_token_ms = 0.0

        # tts_first_audio_ms = t_tts_first_audio - t_first_sentence_ready
        if self.t_tts_first_audio is not None and self.t_first_sentence_ready is not None:
            tts_first_audio_ms = max(0.0, (self.t_tts_first_audio - self.t_first_sentence_ready) * 1000.0)
        else:
            tts_first_audio_ms = 0.0

        # total_ms = t_tts_first_audio - (t_vad_end - silence_threshold)
        if self.t_tts_first_audio is not None and self.t_vad_end is not None:
            total_ms = max(
                0.0,
                (self.t_tts_first_audio - (self.t_vad_end - silence_threshold_s)) * 1000.0,
            )
            # total_ms_from_vad_end = t_tts_first_audio - t_vad_end
            total_ms_from_vad_end = max(0.0, (self.t_tts_first_audio - self.t_vad_end) * 1000.0)
        else:
            total_ms = 0.0
            total_ms_from_vad_end = 0.0

        return TurnMetrics(
            vad_end_ms=round(vad_end_ms, 2),
            stt_final_ms=round(stt_final_ms, 2),
            llm_first_token_ms=round(llm_first_token_ms, 2),
            tts_first_audio_ms=round(tts_first_audio_ms, 2),
            total_ms=round(total_ms, 2),
            total_ms_from_vad_end=round(total_ms_from_vad_end, 2),
        )


def calculate_percentile(values: Sequence[float], p: float) -> float:
    """Calculate the p-th percentile (0 <= p <= 100) using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    # Percentage index
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    weight = idx - lower

    val = sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight
    return round(float(val), 2)


def median(values: Sequence[float]) -> float:
    """Return median (50th percentile) of values."""
    return calculate_percentile(values, 50.0)


def p90(values: Sequence[float]) -> float:
    """Return 90th percentile of values."""
    return calculate_percentile(values, 90.0)

