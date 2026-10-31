"""Generate test audio file tests/data/what_time_tokyo.wav using Piper TTS.

Synthesizes "what time is it in Tokyo", resamples to 16000 Hz 16-bit mono PCM,
and saves to tests/data/what_time_tokyo.wav.
"""

from __future__ import annotations

import os
from pathlib import Path
import wave
import numpy as np
from piper import PiperVoice


def generate_audio(
    text: str = "what time is it in Tokyo",
    output_path: str = "tests/data/what_time_tokyo.wav",
    model_path: str = "models/en_US-lessac-medium.onnx",
    target_sr: int = 16000,
) -> None:
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Piper voice from {model_path}...")
    voice = PiperVoice.load(model_path)
    orig_sr = voice.config.sample_rate

    print(f"Synthesizing: '{text}' at {orig_sr} Hz...")
    raw_pcm_chunks = []
    for chunk in voice.synthesize(text):
        raw_pcm_chunks.append(chunk.audio_int16_bytes)

    full_raw = b"".join(raw_pcm_chunks)
    samples = np.frombuffer(full_raw, dtype=np.int16)

    # Resample to target_sr (16000 Hz) using linear interpolation
    orig_duration = len(samples) / orig_sr
    num_target_samples = int(orig_duration * target_sr)
    orig_times = np.linspace(0, orig_duration, len(samples), endpoint=False)
    target_times = np.linspace(0, orig_duration, num_target_samples, endpoint=False)
    resampled_samples = np.interp(target_times, orig_times, samples).astype(np.int16)

    # Add 400ms leading silence (6400 samples) and 1200ms trailing silence (19200 samples)
    # This allows VAD to cleanly detect IDLE -> SPEECH_START -> speech -> SPEECH_END
    leading_silence = np.zeros(int(0.4 * target_sr), dtype=np.int16)
    trailing_silence = np.zeros(int(1.2 * target_sr), dtype=np.int16)
    final_audio = np.concatenate([leading_silence, resampled_samples, trailing_silence])

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(final_audio.tobytes())

    total_duration = len(final_audio) / target_sr
    print(f"Generated {output_path} ({total_duration:.2f}s, {target_sr} Hz, mono, {len(final_audio)} samples).")


if __name__ == "__main__":
    generate_audio()

