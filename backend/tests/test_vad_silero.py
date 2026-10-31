"""Unit tests for Silero VAD adapter.

Tests frame buffering, window inference, speech start/end transitions,
error handling for missing model, and reset behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from backend.app.adapters.protocols import VADEventType
from backend.app.adapters.vad_silero import SileroVADProcessor


def test_silero_vad_file_not_found():
    """Verify FileNotFoundError is raised if model path is invalid."""
    with pytest.raises(FileNotFoundError):
        SileroVADProcessor(model_path="non_existent_silero.onnx")


def test_silero_vad_mocked_session():
    """Verify Silero VAD windowing, context prepending, and event emission using mock session."""
    with patch("onnxruntime.InferenceSession") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess_cls.return_value = mock_sess

        # Mock inference return: probability 0.9 (speech) then 0.1 (silence)
        # Output shape: (1, 1), state shape: (2, 1, 128)
        speech_out = np.array([[0.9]], dtype=np.float32)
        silence_out = np.array([[0.1]], dtype=np.float32)
        dummy_state = np.zeros((2, 1, 128), dtype=np.float32)

        mock_sess.run.return_value = (speech_out, dummy_state)

        vad = SileroVADProcessor(
            model_path="models/silero_vad.onnx",
            threshold=0.5,
            silence_threshold_ms=64.0,  # 2 windows of 512 samples = 64 ms
            min_speech_ms=10.0,
        )

        # 640 bytes = 320 samples of int16
        frame_speech = b"\x00\x10" * 320

        # Feed first 320 samples: buffered (<512)
        event1 = vad.process(frame_speech)
        assert event1 is None

        # Feed second 320 samples: total 640 samples >= 512 samples -> triggers window
        event2 = vad.process(frame_speech)
        assert event2 is not None
        assert event2.type == VADEventType.SPEECH_START

        # Now simulate silence
        mock_sess.run.return_value = (silence_out, dummy_state)
        frame_silence = b"\x00\x00" * 320

        # Feed silence frames to exceed silence_threshold_ms (64ms)
        end_event = None
        for _ in range(4):
            ev = vad.process(frame_silence)
            if ev:
                end_event = ev
                break

        assert end_event is not None
        assert end_event.type == VADEventType.SPEECH_END


def test_silero_vad_real_model():
    """Verify real Silero VAD model if file exists on disk."""
    model_path = Path("models/silero_vad.onnx")
    if not model_path.is_file():
        pytest.skip("models/silero_vad.onnx not found")

    vad = SileroVADProcessor(model_path=str(model_path))
    assert vad._session is not None

    # Silence frame should not trigger speech
    silence = b"\x00\x00" * 320
    for _ in range(5):
        ev = vad.process(silence)
        assert ev is None

    vad.reset()
    assert vad._is_speaking is False
    assert len(vad._buffer) == 0

