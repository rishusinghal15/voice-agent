"""Unit tests for Piper local TTS adapter.

Tests model missing validation, sample rate extraction, 100ms chunk streaming,
cancellation safety, and real model if available.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.adapters.tts_piper import PiperTTSAdapter, get_cached_piper_voice


def test_piper_model_not_found():
    """Verify FileNotFoundError is raised if voice model does not exist."""
    with pytest.raises(FileNotFoundError):
        get_cached_piper_voice("non_existent_piper.onnx")


@pytest.mark.asyncio
async def test_piper_synthesize_chunking():
    """Verify PiperTTSAdapter chunks audio into ~100 ms segments."""
    with patch("piper.PiperVoice.load") as mock_load:
        mock_voice = MagicMock()
        mock_load.return_value = mock_voice
        mock_voice.config.sample_rate = 22050

        # Create mock AudioChunk with 1 second of audio (22050 * 2 = 44100 bytes)
        mock_chunk = MagicMock()
        mock_chunk.audio_int16_bytes = b"\x00\x01" * 22050
        mock_voice.synthesize.return_value = [mock_chunk]

        adapter = PiperTTSAdapter(model_path="models/en_US-lessac-medium.onnx")
        assert adapter.sample_rate == 22050

        chunks = []
        async for pcm in adapter.synthesize("Hello world!"):
            chunks.append(pcm)

        assert len(chunks) > 1
        # Each chunk should be ~4410 bytes (100 ms at 22050Hz s16le mono)
        assert len(chunks[0]) == int(22050 * 0.1) * 2
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes == 44100


@pytest.mark.asyncio
async def test_piper_synthesize_cancellation():
    """Verify Piper synthesis handles cancellation safely."""
    with patch("piper.PiperVoice.load") as mock_load:
        mock_voice = MagicMock()
        mock_load.return_value = mock_voice
        mock_voice.config.sample_rate = 22050

        mock_chunk = MagicMock()
        mock_chunk.audio_int16_bytes = b"\x00\x01" * 44100
        mock_voice.synthesize.return_value = [mock_chunk]

        adapter = PiperTTSAdapter(model_path="models/en_US-lessac-medium.onnx")

        async def run_synth():
            async for _ in adapter.synthesize("Hello"):
                await asyncio.sleep(0.01)

        task = asyncio.create_task(run_synth())
        await asyncio.sleep(0.005)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


def test_piper_real_model():
    """Verify real Piper voice model if file exists on disk."""
    model_path = Path("models/en_US-lessac-medium.onnx")
    if not model_path.is_file():
        pytest.skip("models/en_US-lessac-medium.onnx not found")

    adapter = PiperTTSAdapter(model_path=str(model_path))
    assert adapter.sample_rate == 22050
    assert adapter.voice is not None

