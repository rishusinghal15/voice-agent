"""Benchmark script for measuring voice pipeline latencies across multiple turns.

Follows design doc §7 and assignment requirements:
- Runs 20 test turns through the voice pipeline
- Measures per-stage latency metrics:
  * vad_end_ms
  * stt_final_ms
  * llm_first_token_ms
  * tts_first_audio_ms
  * total_ms
  * total_ms_from_vad_end
- Calculates Median and P90 percentiles
- Prints formatted latency report
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import db
from backend.app.config import settings
from backend.app.metrics import calculate_percentile, median, p90
from backend.app.pipeline import VoicePipeline


def generate_speech_pcm(duration_s: float = 0.5, sample_rate: int = 16000, freq_hz: float = 440.0) -> bytes:
    """Generate 16kHz signed 16-bit PCM mono sine-wave audio above VAD threshold."""
    num_samples = int(sample_rate * duration_s)
    samples = []
    amplitude = 16000.0  # RMS will be ~11313, well above 500 threshold
    for i in range(num_samples):
        val = int(amplitude * math.sin(2 * math.pi * freq_hz * (i / sample_rate)))
        samples.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{len(samples)}h", *samples)


def generate_silence_pcm(duration_s: float = 0.2, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM frames (RMS = 0)."""
    num_samples = int(sample_rate * duration_s)
    return b"\x00\x00" * num_samples


async def run_benchmark(num_turns: int = 20) -> List[Dict[str, float]]:
    """Execute benchmark over specified number of turns and return collected metrics."""
    # Initialize DB
    test_db = "voice_benchmark.db"
    settings.DB_PATH = test_db
    settings.VAD_MIN_SPEECH_MS = 50  # 50ms min speech length for benchmark
    await db.init_db(test_db)

    session_id = f"bench_sess_{int(asyncio.get_event_loop().time())}"
    await db.create_session(session_id)

    collected_metrics: List[Dict[str, float]] = []

    async def emit_json(_: Dict[str, Any]) -> None:
        pass

    async def emit_audio(_: bytes) -> None:
        pass

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=emit_json,
        emit_audio=emit_audio,
    )
    await pipeline.start()

    speech_chunk = generate_speech_pcm(duration_s=0.3)  # 300ms speech
    silence_chunk = generate_silence_pcm(duration_s=0.1)  # 100ms silence

    frame_size = 640  # 20ms frames at 16kHz s16le

    print(f"Running benchmark with {num_turns} turns...")
    for turn in range(1, num_turns + 1):
        # Feed speech frames in 20ms chunks
        for i in range(0, len(speech_chunk), frame_size):
            await pipeline.handle_audio(speech_chunk[i : i + frame_size])
            await asyncio.sleep(0.01)

        # Feed silence frames to trigger VAD SPEECH_END
        for i in range(0, len(silence_chunk), frame_size):
            await pipeline.handle_audio(silence_chunk[i : i + frame_size])
            await asyncio.sleep(0.01)

        # Await turn execution
        if pipeline.active_turn_task:
            await pipeline.active_turn_task

        # Acknowledge playback done
        await pipeline.handle_playback_done(turn_id=turn)

        # Compute turn metrics
        turn_metric = pipeline.turn_timer.compute()
        collected_metrics.append(turn_metric.to_dict())

    await pipeline.handle_end_call(reason="user_end")
    return collected_metrics


def print_latency_report(metrics_list: List[Dict[str, float]]) -> None:
    """Print formatted latency report table showing Median and P90 per stage."""
    stages = [
        ("VAD end-of-speech (silence wait)", "vad_end_ms", 600.0),
        ("STT final transcript", "stt_final_ms", 200.0),
        ("LLM first token", "llm_first_token_ms", 400.0),
        ("TTS first audio", "tts_first_audio_ms", 250.0),
        ("Total (end-of-speech to first audio)", "total_ms", 1500.0),
        ("Total from VAD end", "total_ms_from_vad_end", 900.0),
    ]

    print("\n" + "=" * 78)
    print("VOICE AGENT LATENCY BENCHMARK REPORT (20 TURNS)")
    print("=" * 78)
    print(f"{'Pipeline Stage':<40} {'Budget':<10} {'Median':<12} {'P90':<12}")
    print("-" * 78)

    for stage_name, key, budget in stages:
        values = [m[key] for m in metrics_list if key in m]
        med = median(values) if values else 0.0
        p_90 = p90(values) if values else 0.0
        status = "OK" if med <= budget else "EXCEEDS"
        print(f"{stage_name:<40} {budget:>6.0f} ms  {med:>7.2f} ms   {p_90:>7.2f} ms ({status})")

    print("-" * 78)
    total_med = median([m["total_ms"] for m in metrics_list])
    print(f"Overall Target (Budget <= 1500 ms): Median = {total_med:.2f} ms -> "
          f"{'PASSED' if total_med <= 1500 else 'FAILED'}")
    print("=" * 78 + "\n")


async def main() -> None:
    turns = 20
    if len(sys.argv) > 1:
        try:
            turns = int(sys.argv[1])
        except ValueError:
            pass

    metrics = await run_benchmark(num_turns=turns)
    print_latency_report(metrics)


if __name__ == "__main__":
    asyncio.run(main())
