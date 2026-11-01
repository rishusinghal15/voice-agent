"""Browser Microphone Simulator for WebSocket Voice Agent.

Streams a WAV file (tests/data/what_time_tokyo.wav) re-encoded exactly the way
the browser does it:
1. Resampled to browser AudioContext rate (48 kHz / 44.1 kHz).
2. Processed in ScriptProcessor buffer chunks (2048 samples).
3. Filtered via 7.5 kHz anti-aliasing low-pass biquad filter.
4. Downsampled back to 16 kHz with continuous phase tracking and carry-over buffer.
5. Clipped and scaled (x * 32767) to signed 16-bit PCM.
6. Emitted as strictly 640-byte (20 ms) frames streamed in real time over WebSocket.

Reproduces the browser audio path end-to-end without a browser.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
import wave
import httpx
import numpy as np
import websockets

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings


def format_ts(epoch_ms: int | float | None) -> str:
    """Format epoch millisecond timestamp to readable HH:MM:SS.mmm."""
    if epoch_ms is None:
        return "N/A"
    sec = epoch_ms / 1000.0
    dt = datetime.fromtimestamp(sec)
    return dt.strftime("%H:%M:%S.%f")[:-3]


def is_server_ready(base_url: str) -> bool:
    """Check if FastAPI server is responsive on GET /healthz."""
    try:
        r = httpx.get(f"{base_url}/healthz", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def start_in_process_server(host: str = "127.0.0.1", port: int = 8000) -> threading.Thread:
    """Start an in-process Uvicorn server in a daemon thread."""
    import uvicorn
    from backend.app.main import app

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t


class BiquadLowpassFilter:
    """Matches Web Audio API createBiquadFilter() type='lowpass'."""

    def __init__(self, cutoff_hz: float = 7500.0, sample_rate: float = 48000.0, q: float = 0.707) -> None:
        w0 = 2.0 * math.pi * cutoff_hz / sample_rate
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)

        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def process(self, input_samples: np.ndarray) -> np.ndarray:
        output = np.empty_like(input_samples)
        for i in range(len(input_samples)):
            x0 = float(input_samples[i])
            y0 = (
                self.b0 * x0
                + self.b1 * self.x1
                + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2
            )
            self.x2 = self.x1
            self.x1 = x0
            self.y2 = self.y1
            self.y1 = y0
            output[i] = y0
        return output


class BrowserAudioSimulator:
    """Simulates browser mic capture at browser_rate, low-pass filtering, and 16kHz framing."""

    def __init__(
        self,
        browser_rate: int = 48000,
        target_rate: int = 16000,
        buffer_size: int = 2048,
    ) -> None:
        self.browser_rate = browser_rate
        self.target_rate = target_rate
        self.buffer_size = buffer_size
        self.filter = BiquadLowpassFilter(cutoff_hz=7500.0, sample_rate=float(browser_rate))
        self.input_carry = np.array([], dtype=np.float32)
        self.resample_pos = 0.0
        self.output_carry = np.array([], dtype=np.float32)
        self.ratio = browser_rate / target_rate

    def process_buffer(self, input_chunk: np.ndarray) -> list[bytes]:
        """Process one ScriptProcessor buffer of browser_rate float samples."""
        filtered = self.filter.process(input_chunk)

        if len(self.input_carry) > 0:
            total_input = np.concatenate([self.input_carry, filtered])
        else:
            total_input = filtered

        resampled_samples = []
        pos = self.resample_pos
        ratio = self.ratio
        tot_len = len(total_input)

        while pos + 1 < tot_len:
            idx = int(math.floor(pos))
            frac = pos - idx
            s0 = total_input[idx]
            s1 = total_input[idx + 1]
            resampled_samples.append(s0 + frac * (s1 - s0))
            pos += ratio

        last_idx = int(math.floor(pos))
        self.input_carry = total_input[last_idx:]
        self.resample_pos = pos - last_idx

        if resampled_samples:
            resampled_arr = np.array(resampled_samples, dtype=np.float32)
            if len(self.output_carry) > 0:
                self.output_carry = np.concatenate([self.output_carry, resampled_arr])
            else:
                self.output_carry = resampled_arr

        frames = []
        FRAME_SAMPLES = 320  # 20 ms at 16 kHz
        while len(self.output_carry) >= FRAME_SAMPLES:
            frame_slice = self.output_carry[:FRAME_SAMPLES]
            self.output_carry = self.output_carry[FRAME_SAMPLES:]

            clipped = np.clip(frame_slice, -1.0, 1.0)
            pcm16 = np.round(clipped * 32767).astype(np.int16)
            frames.append(pcm16.tobytes())

        return frames


def prepare_browser_simulated_frames(
    wav_path: str,
    browser_rate: int = 48000,
    buffer_size: int = 2048,
) -> tuple[list[bytes], float]:
    """Load 16kHz WAV, upsample to browser rate, run browser simulation, return 640-byte frames."""
    with wave.open(wav_path, "rb") as wf:
        orig_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw_frames = wf.readframes(wf.getnframes())

    orig_duration = len(raw_frames) / (orig_rate * sampwidth * n_channels)
    samples_16k = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

    # 1. Upsample 16 kHz to browser_rate (e.g. 48 kHz) to simulate mic physical capture
    up_ratio = browser_rate / orig_rate
    num_browser_samples = int(len(samples_16k) * up_ratio)
    indices = np.linspace(0, len(samples_16k) - 1, num_browser_samples)
    browser_mic_samples = np.interp(indices, np.arange(len(samples_16k)), samples_16k).astype(np.float32)

    # 2. Feed into BrowserAudioSimulator in chunks of buffer_size
    sim = BrowserAudioSimulator(browser_rate=browser_rate, target_rate=16000, buffer_size=buffer_size)
    encoded_frames: list[bytes] = []

    for i in range(0, len(browser_mic_samples), buffer_size):
        chunk = browser_mic_samples[i : i + buffer_size]
        frames = sim.process_buffer(chunk)
        encoded_frames.extend(frames)

    # Append 1.0s of silence (50 frames of 20ms zeros) to simulate user pausing/stopping speech
    # This ensures VAD silence threshold (600ms) triggers SPEECH_END
    silence_samples = np.zeros(int(browser_rate * 1.0), dtype=np.float32)
    for i in range(0, len(silence_samples), buffer_size):
        chunk = silence_samples[i : i + buffer_size]
        frames = sim.process_buffer(chunk)
        encoded_frames.extend(frames)

    return encoded_frames, orig_duration


async def run_mic_sim_direct(
    frames: list[bytes],
    orig_duration: float,
    audio_path: str,
) -> None:
    """Run simulated browser frames directly through VoicePipeline without TCP socket."""
    import uuid
    from backend.app import db
    from backend.app.pipeline import VoicePipeline

    await db.init_db(settings.DB_PATH)
    session_id = str(uuid.uuid4())
    await db.create_session(
        session_id=session_id,
        config={
            "vad": settings.VAD_PROVIDER,
            "stt": settings.STT_PROVIDER,
            "llm": settings.LLM_PROVIDER,
            "tts": settings.TTS_PROVIDER,
        },
    )
    print(f"[MicSim] Direct pipeline mode started for session: {session_id}")

    turn_done_event = asyncio.Event()
    received_audio_bytes = 0

    async def emit_json(payload: dict) -> None:
        ev_type = payload.get("type", "unknown")
        ts = payload.get("ts", int(time.time() * 1000))
        details = ""
        if ev_type == "state":
            details = f"state={payload.get('state')}"
        elif ev_type in ("transcript_partial", "transcript_final"):
            details = f"text='{payload.get('text', '')}'"
        elif ev_type == "agent_text":
            details = f"text='{payload.get('text', '')}'"
        elif ev_type == "tool_event":
            details = f"name={payload.get('name')}, status={payload.get('status')}"
        elif ev_type == "turn_done":
            details = f"turn_id={payload.get('turn_id')}, total_audio_ms={payload.get('total_audio_ms')}"
            turn_done_event.set()
        elif ev_type == "call_ended":
            details = f"reason={payload.get('reason')}"
        elif ev_type == "error":
            details = f"code={payload.get('code')}, message={payload.get('message')}"
        print(f"{format_ts(ts):<12} | {ev_type:<22} | {details}", flush=True)

    async def emit_audio(chunk: bytes) -> None:
        nonlocal received_audio_bytes
        received_audio_bytes += len(chunk)

    pipeline = VoicePipeline(
        session_id=session_id,
        emit_json=emit_json,
        emit_audio=emit_audio,
    )
    await pipeline.start()

    print("-" * 80)
    print(f"{'TIME':<12} | {'EVENT TYPE':<22} | {'DETAILS'}")
    print("-" * 80)

    for frame in frames:
        await pipeline.handle_audio(frame)
        await asyncio.sleep(0.02)

    print("[MicSim] Audio streaming complete. Waiting for agent reply...")
    try:
        await asyncio.wait_for(turn_done_event.wait(), timeout=25.0)
        print("[MicSim] Received turn_done event successfully!")
    except asyncio.TimeoutError:
        print("[MicSim] WARNING: Timed out waiting for turn_done!")

    await pipeline.handle_playback_done(pipeline.turn_id)
    await asyncio.sleep(0.1)
    await pipeline.handle_end_call(reason="user_end")

    print("-" * 80)
    print(f"[MicSim] Call finished. Total agent audio received: {received_audio_bytes} bytes.")

    turns = await db.get_turns(session_id)
    metrics = await db.get_turn_metrics(session_id)
    print("\n" + "=" * 60)
    print(f"       SESSION REPORT ({session_id})")
    print("=" * 60)
    for t in turns:
        role = t.get("role", "unknown").upper()
        text = t.get("text", "")
        tc = t.get("tool_calls_json")
        extra = f" [tool_calls: {tc}]" if tc else ""
        print(f"  [{role} Turn {t.get('turn_id')}]: {text}{extra}")
    if metrics:
        m = metrics[0]
        print("\n--- Latency Metrics (DB Record) ---")
        print(f"  VAD Speech End Latency:       {m.get('vad_end_ms'):.2f} ms")
        print(f"  STT Final Transcript Latency: {m.get('stt_final_ms'):.2f} ms")
        print(f"  LLM First Token Latency:      {m.get('llm_first_token_ms'):.2f} ms")
        print(f"  TTS First Audio Latency:      {m.get('tts_first_audio_ms'):.2f} ms")
        print(f"  Total Latency:                {m.get('total_ms'):.2f} ms")
    print("=" * 60 + "\n")


async def run_mic_sim(
    base_url: str = "http://127.0.0.1:8000",
    audio_path: str = "tests/data/what_time_tokyo.wav",
    browser_rate: int = 48000,
    vad_provider: str = "silero",
    stt_provider: str = "vosk",
    llm_provider: str = "groq",
    tts_provider: str = "piper",
    direct: bool = False,
) -> None:
    settings.VAD_PROVIDER = vad_provider
    settings.STT_PROVIDER = stt_provider
    settings.LLM_PROVIDER = llm_provider
    settings.TTS_PROVIDER = tts_provider

    audio_file = Path(audio_path)
    if not audio_file.is_file():
        print(f"Audio file '{audio_path}' not found. Generating now...")
        from scripts.generate_test_audio import generate_audio
        generate_audio(output_path=str(audio_file))

    frames, orig_duration = prepare_browser_simulated_frames(
        str(audio_file),
        browser_rate=browser_rate,
        buffer_size=2048,
    )

    print(f"\n[MicSim] Loaded audio: {audio_path} ({orig_duration:.2f}s)")
    print(f"[MicSim] Simulated Browser Mic: {browser_rate} Hz -> Biquad Lowpass -> 16 kHz carry-over framing")
    print(f"[MicSim] Total frames generated: {len(frames)} (each exactly {len(frames[0]) if frames else 0} bytes)")
    print(f"[MicSim] Active Providers: vad={settings.VAD_PROVIDER}, stt={settings.STT_PROVIDER}, llm={settings.LLM_PROVIDER}, tts={settings.TTS_PROVIDER}")

    if direct:
        await run_mic_sim_direct(frames, orig_duration, audio_path)
        return

    # Check or start server
    try:
        server_up = is_server_ready(base_url)
    except Exception:
        server_up = False

    if not server_up:
        print(f"[MicSim] Server not detected at {base_url}. Attempting to start in-process server or run direct...")
        try:
            start_in_process_server()
            for _ in range(30):
                if is_server_ready(base_url):
                    server_up = True
                    break
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[MicSim] In-process server start failed ({e}). Switching to direct pipeline mode...")
            await run_mic_sim_direct(frames, orig_duration, audio_path)
            return

        if not server_up:
            print("[MicSim] TCP socket communication restricted in this environment. Falling back to direct pipeline mode...")
            await run_mic_sim_direct(frames, orig_duration, audio_path)
            return

    # 1. Create session via POST /sessions
    try:
        async with httpx.AsyncClient(base_url=base_url) as client:
            resp = await client.post("/sessions")
            resp.raise_for_status()
            session_id = resp.json()["session_id"]
    except Exception as e:
        print(f"[MicSim] Failed to connect to server ({e}). Falling back to direct pipeline mode...")
        await run_mic_sim_direct(frames, orig_duration, audio_path)
        return

    print(f"[MicSim] Session created: {session_id}")

    # 2. Connect WebSocket /ws/voice?session_id=...
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/voice?session_id={session_id}"
    print(f"[MicSim] Connecting to {ws_url} ...")

    received_audio_bytes = 0
    turn_metrics = {}
    turn_done_event = asyncio.Event()
    call_ended_event = asyncio.Event()

    async with websockets.connect(ws_url) as ws:
        print(f"[MicSim] WebSocket connected. Streaming browser-simulated frames...")
        print("-" * 80)
        print(f"{'TIME':<12} | {'EVENT TYPE':<22} | {'DETAILS'}")
        print("-" * 80)

        async def receiver_loop():
            nonlocal received_audio_bytes, turn_metrics
            try:
                async for message in ws:
                    now_ms = int(time.time() * 1000)
                    if isinstance(message, bytes):
                        received_audio_bytes += len(message)
                    else:
                        data = json.loads(message)
                        ev_type = data.get("type", "unknown")
                        ts = data.get("ts", now_ms)

                        details = ""
                        if ev_type == "state":
                            details = f"state={data.get('state')}"
                        elif ev_type in ("transcript_partial", "transcript_final"):
                            details = f"text='{data.get('text', '')}'"
                        elif ev_type == "agent_text":
                            details = f"text='{data.get('text', '')}'"
                        elif ev_type == "tool_event":
                            details = f"name={data.get('name')}, status={data.get('status')}"
                        elif ev_type == "agent_audio_start":
                            details = f"sr={data.get('sample_rate')}, encoding={data.get('encoding')}"
                        elif ev_type == "agent_audio_end":
                            details = f"total_audio_bytes={received_audio_bytes}"
                        elif ev_type == "turn_done":
                            turn_metrics = data.get("metrics", {})
                            details = f"turn_id={data.get('turn_id')}, total_audio_ms={data.get('total_audio_ms')}"
                            turn_done_event.set()
                        elif ev_type == "call_ended":
                            details = f"reason={data.get('reason')}"
                            call_ended_event.set()
                        elif ev_type == "error":
                            details = f"code={data.get('code')}, message={data.get('message')}"

                        print(f"{format_ts(ts):<12} | {ev_type:<22} | {details}", flush=True)
            except (websockets.ConnectionClosed, asyncio.CancelledError):
                pass

        recv_task = asyncio.create_task(receiver_loop())

        # Stream browser-simulated 20 ms frames with real-time pacing
        for frame in frames:
            await ws.send(frame)
            await asyncio.sleep(0.02)

        print("[MicSim] Audio streaming complete. Waiting for agent reply...")

        try:
            await asyncio.wait_for(turn_done_event.wait(), timeout=25.0)
            print("[MicSim] Received turn_done event successfully!")
        except asyncio.TimeoutError:
            print("[MicSim] WARNING: Timed out waiting for turn_done!")

        # Signal playback done and end call
        await ws.send(json.dumps({"type": "playback_done", "ts": int(time.time() * 1000)}))
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "end_call", "ts": int(time.time() * 1000)}))

        try:
            await asyncio.wait_for(call_ended_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    print("-" * 80)
    print(f"[MicSim] Call finished. Total agent audio received: {received_audio_bytes} bytes.")

    # 3. Fetch session summary from GET /sessions/{session_id}
    async with httpx.AsyncClient(base_url=base_url) as client:
        sess_resp = await client.get(f"/sessions/{session_id}")
        sess_resp.raise_for_status()
        sess_data = sess_resp.json()

    print("\n" + "=" * 60)
    print(f"       SESSION REPORT ({session_id})")
    print("=" * 60)
    turns = sess_data.get("turns", [])
    for t in turns:
        role = t.get("role", "unknown").upper()
        text = t.get("text", "")
        tc = t.get("tool_calls_json")
        extra = f" [tool_calls: {tc}]" if tc else ""
        print(f"  [{role} Turn {t.get('turn_id')}]: {text}{extra}")

    metrics = sess_data.get("metrics", [])
    if metrics:
        m = metrics[0]
        print("\n--- Latency Metrics (DB Record) ---")
        print(f"  VAD Speech End Latency:       {m.get('vad_end_ms'):.2f} ms")
        print(f"  STT Final Transcript Latency: {m.get('stt_final_ms'):.2f} ms")
        print(f"  LLM First Token Latency:      {m.get('llm_first_token_ms'):.2f} ms")
        print(f"  TTS First Audio Latency:      {m.get('tts_first_audio_ms'):.2f} ms")
        print(f"  Total Latency:                {m.get('total_ms'):.2f} ms")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser Microphone Simulator test.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of server")
    parser.add_argument("--audio", default="tests/data/what_time_tokyo.wav", help="Path to 16kHz mono WAV")
    parser.add_argument("--browser-rate", type=int, default=48000, help="Simulated browser mic rate (Hz)")
    parser.add_argument("--vad", default="silero", help="VAD provider (silero, webrtc, mock)")
    parser.add_argument("--stt", default="vosk", help="STT provider (vosk, mock)")
    parser.add_argument("--llm", default="groq", help="LLM provider (groq, mock)")
    parser.add_argument("--tts", default="piper", help="TTS provider (piper, mock)")
    parser.add_argument("--direct", action="store_true", help="Run directly through pipeline without network socket")
    args = parser.parse_args()

    asyncio.run(
        run_mic_sim(
            base_url=args.url,
            audio_path=args.audio,
            browser_rate=args.browser_rate,
            vad_provider=args.vad,
            stt_provider=args.stt,
            llm_provider=args.llm,
            tts_provider=args.tts,
            direct=args.direct,
        )
    )


if __name__ == "__main__":
    main()
