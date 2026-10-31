"""Silero VAD processor implementation using ONNX Runtime.

Runs local silero_vad.onnx without PyTorch dependencies.
Buffers 20 ms / 640-byte audio frames into 512-sample windows at 16 kHz with
64-sample context prepending. Emits SPEECH_START and SPEECH_END transitions
based on VAD_SILENCE_MS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime as ort

from backend.app.adapters.protocols import VADEvent, VADEventType, VADProcessor
from backend.app.config import settings


class SileroVADProcessor:
    """Voice Activity Detector powered by Silero VAD v5 ONNX model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        silence_threshold_ms: Optional[float] = None,
        min_speech_ms: Optional[float] = None,
    ) -> None:
        self.model_path: str = model_path or settings.SILERO_MODEL_PATH
        self.threshold: float = threshold
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

        p = Path(self.model_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Silero VAD ONNX model not found at '{self.model_path}'. "
                f"Please ensure models/silero_vad.onnx exists."
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(p), sess_options=opts)

        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Reset internal recurrent state, context buffer, and speech trackers."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)
        self._buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._consecutive_silence_samples = 0
        self._speech_samples = 0

    def process(self, frame: bytes) -> Optional[VADEvent]:
        """Process incoming 16kHz s16le PCM frame and emit SPEECH_START / SPEECH_END."""
        if not frame:
            return None

        # Convert int16 PCM to float32 in range [-1.0, 1.0]
        pcm_samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if len(self._buffer) == 0:
            self._buffer = pcm_samples
        else:
            self._buffer = np.concatenate([self._buffer, pcm_samples])

        event_to_emit: Optional[VADEvent] = None

        # Process in 512-sample (32 ms) windows
        while len(self._buffer) >= 512:
            window = self._buffer[:512]
            self._buffer = self._buffer[512:]

            # Prepend 64-sample context for Silero v5 (total 576 samples)
            model_input = np.concatenate([self._context, window])[np.newaxis, :]
            out, self._state = self._session.run(
                None,
                {
                    "input": model_input,
                    "state": self._state,
                    "sr": self._sr,
                },
            )
            # Context for next window is the last 64 samples of current window
            self._context = window[-64:]

            prob = float(out[0][0])
            is_speech_window = prob >= self.threshold

            if not self._is_speaking:
                if is_speech_window:
                    self._is_speaking = True
                    self._consecutive_silence_samples = 0
                    self._speech_samples = 512
                    event_to_emit = VADEvent(type=VADEventType.SPEECH_START)
            else:
                self._speech_samples += 512
                if not is_speech_window:
                    self._consecutive_silence_samples += 512
                    silence_duration_ms = (self._consecutive_silence_samples / 16000.0) * 1000.0
                    if silence_duration_ms >= self.silence_threshold_ms:
                        self._is_speaking = False
                        self._consecutive_silence_samples = 0
                        event_to_emit = VADEvent(type=VADEventType.SPEECH_END)
                else:
                    self._consecutive_silence_samples = 0

        return event_to_emit

