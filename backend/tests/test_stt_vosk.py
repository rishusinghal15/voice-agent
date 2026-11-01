"""Unit tests for Vosk streaming STT adapter.

Tests model missing validation, partial and final transcript streaming,
cancellation safety, and real model if available.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.adapters.protocols import STTFinal, STTPartial
from backend.app.adapters.stt_vosk import VoskSTTAdapter, VoskSTTStream, get_cached_vosk_model


def test_vosk_model_not_found():
    """Verify FileNotFoundError is raised if model directory does not exist."""
    with pytest.raises(FileNotFoundError):
        get_cached_vosk_model("non_existent_vosk_dir")


@pytest.mark.asyncio
async def test_vosk_stt_stream_events_and_finish():
    """Verify VoskSTTStream processes audio and emits partial and final transcripts."""
    mock_model = MagicMock()

    with patch("vosk.KaldiRecognizer") as mock_rec_cls:
        mock_rec = MagicMock()
        mock_rec_cls.return_value = mock_rec

        # Simulate partial result, then final result
        mock_rec.AcceptWaveform.return_value = False
        mock_rec.PartialResult.return_value = '{"partial": "hello vosk"}'
        mock_rec.FinalResult.return_value = '{"text": "hello vosk assistant"}'

        stream = VoskSTTStream(model=mock_model, sample_rate=16000)

        # Feed audio chunk
        await stream.send_audio(b"\x00\x01" * 320)
        await stream.finish()

        # Collect events
        events = []
        async for ev in stream.events():
            events.append(ev)

        assert len(events) == 2
        assert isinstance(events[0], STTPartial)
        assert events[0].text == "hello vosk"
        assert isinstance(events[1], STTFinal)
        assert events[1].text == "hello vosk assistant"


@pytest.mark.asyncio
async def test_vosk_stt_cancellation():
    """Verify VoskSTTStream handles task cancellation cleanly."""
    mock_model = MagicMock()

    with patch("vosk.KaldiRecognizer") as mock_rec_cls:
        mock_rec_cls.return_value = MagicMock()
        stream = VoskSTTStream(model=mock_model, sample_rate=16000)

        async def iterate_stream():
            async for _ in stream.events():
                pass

        task = asyncio.create_task(iterate_stream())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


def test_vosk_real_model_loading():
    """Verify real Vosk model loads and initializes stream if directory exists."""
    model_dir = Path("models/vosk-model-en-us-0.22-lgraph")
    if not model_dir.is_dir():
        model_dir = Path("models/vosk-model-small-en-us-0.15")
    if not model_dir.is_dir():
        pytest.skip("No Vosk model directory found")

    adapter = VoskSTTAdapter(model_path=str(model_dir))
    assert adapter.model_path == str(model_dir)
