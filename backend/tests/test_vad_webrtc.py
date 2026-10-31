"""Unit tests for WebRTC VAD adapter."""

from backend.app.adapters.protocols import VADEventType
from backend.app.adapters.vad_webrtc import WebRTCVADProcessor


def test_webrtc_vad_invalid_frame_length():
    """Verify non-640 byte frames are ignored."""
    vad = WebRTCVADProcessor()
    assert vad.process(b"\x00" * 100) is None
    assert vad.process(b"") is None


def test_webrtc_vad_silence_and_speech(monkeypatch):
    """Verify WebRTC VAD speech start and end transitions."""
    vad = WebRTCVADProcessor(mode=2, silence_threshold_ms=60.0)
    frame = b"\x00" * 640

    # Stream of classification results: silence, silence, speech, speech, silence x3
    speech_flags = [False, False, True, True, False, False, False]
    monkeypatch.setattr(vad.vad, "is_speech", lambda f, sr: speech_flags.pop(0))

    # Initial silence
    assert vad.process(frame) is None
    assert vad.process(frame) is None

    # Speech starts
    ev_start = vad.process(frame)
    assert ev_start is not None
    assert ev_start.type == VADEventType.SPEECH_START

    # Continuing speech returns None
    assert vad.process(frame) is None

    # Feed silence frames until silence_threshold_ms (60ms = 3 frames)
    assert vad.process(frame) is None  # 20ms
    assert vad.process(frame) is None  # 40ms
    ev_end = vad.process(frame)        # 60ms
    assert ev_end is not None
    assert ev_end.type == VADEventType.SPEECH_END


def test_webrtc_vad_real_native_call():
    """Verify real WebRTC VAD native library executes without errors."""
    vad = WebRTCVADProcessor(mode=2)
    silence = b"\x00\x00" * 320
    # Calling native library with valid frame should not throw
    result = vad.vad.is_speech(silence, 16000)
    assert isinstance(result, bool)
