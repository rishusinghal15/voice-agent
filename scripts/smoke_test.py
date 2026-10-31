"""End-to-end WebSocket voice smoke test using real providers.

Streams tests/data/what_time_tokyo.wav over WebSocket /ws/voice?session_id=...,
prints every incoming event with epoch timestamps, and displays per-stage
latency metrics.

Supports:
- Connecting to an existing live server (default http://127.0.0.1:8000)
- Or automatically spawning an in-process Uvicorn server if no server is running.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
import wave
import httpx
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


async def run_smoke_test(
    base_url: str = "http://127.0.0.1:8000",
    audio_path: str = "tests/data/what_time_tokyo.wav",
    vad_provider: str = "silero",
    stt_provider: str = "vosk",
    llm_provider: str = "groq",
    tts_provider: str = "piper",
) -> None:
    # Ensure real providers are set in settings
    settings.VAD_PROVIDER = vad_provider
    settings.STT_PROVIDER = stt_provider
    settings.LLM_PROVIDER = llm_provider
    settings.TTS_PROVIDER = tts_provider

    audio_file = Path(audio_path)
    if not audio_file.is_file():
        print(f"Audio file '{audio_path}' not found. Generating now...")
        from scripts.generate_test_audio import generate_audio
        generate_audio(output_path=str(audio_file))

    # Read WAV file
    with wave.open(str(audio_file), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    duration = len(frames) / (framerate * sampwidth * n_channels)
    print(f"\n[SmokeTest] Audio loaded: {audio_path}")
    print(f"            {framerate} Hz, {n_channels} ch, {sampwidth*8}-bit, duration: {duration:.2f}s, {len(frames)} bytes")
    print(f"[SmokeTest] Configured Providers: VAD={settings.VAD_PROVIDER}, STT={settings.STT_PROVIDER}, LLM={settings.LLM_PROVIDER}, TTS={settings.TTS_PROVIDER}")
    print(f"[SmokeTest] Groq Model: {settings.GROQ_MODEL}")

    # Check or start server
    if not is_server_ready(base_url):
        print(f"[SmokeTest] Server not detected at {base_url}. Starting in-process server...")
        host = "127.0.0.1"
        port = 8000
        start_in_process_server(host, port)
        # Wait up to 5s for server to become responsive
        for _ in range(50):
            if is_server_ready(base_url):
                break
            await asyncio.sleep(0.1)
        if not is_server_ready(base_url):
            raise RuntimeError(f"Failed to start server at {base_url}")
        print(f"[SmokeTest] In-process server is UP at {base_url}.")
    else:
        print(f"[SmokeTest] Server is live at {base_url}.")

    # Step 1: Create session via POST /sessions
    async with httpx.AsyncClient(base_url=base_url) as client:
        resp = await client.post("/sessions")
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
    print(f"[SmokeTest] Created session: {session_id}")

    # Step 2: Connect WebSocket /ws/voice?session_id=...
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/voice?session_id={session_id}"
    print(f"[SmokeTest] Connecting to {ws_url} ...")

    received_events = []
    received_audio_bytes = 0
    turn_metrics = {}
    turn_done_event = asyncio.Event()
    call_ended_event = asyncio.Event()

    async with websockets.connect(ws_url) as ws:
        print(f"[SmokeTest] WebSocket connected. Starting bidirectional turn...")
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
                        received_events.append(data)

                        details = ""
                        if ev_type == "state":
                            details = f"state={data.get('state')}"
                        elif ev_type in ("transcript_partial", "transcript_final", "stt_partial", "stt_final"):
                            details = f"text='{data.get('text', '')}'"
                        elif ev_type == "agent_text":
                            details = f"text='{data.get('text', '')}'"
                        elif ev_type == "llm_token":
                            details = f"token='{data.get('token', '')}'"
                        elif ev_type == "tool_event":
                            details = f"event={data.get('event')}, tool={data.get('tool')}, args={data.get('args')}, result={data.get('result')}"
                        elif ev_type == "agent_audio_start":
                            details = f"sr={data.get('sample_rate')}, format={data.get('format', 'pcm_s16le')}"
                        elif ev_type == "agent_audio_end":
                            details = f"total_audio_bytes={received_audio_bytes}"
                        elif ev_type == "turn_done":
                            turn_metrics = data.get("metrics", {})
                            details = f"turn_id={data.get('turn_id')}, metrics={turn_metrics}"
                            turn_done_event.set()
                        elif ev_type == "call_ended":
                            details = f"reason={data.get('reason')}"
                            call_ended_event.set()
                        elif ev_type == "error":
                            details = f"code={data.get('code')}, message={data.get('message')}"

                        print(f"{format_ts(ts):<12} | {ev_type:<22} | {details}")
            except (websockets.ConnectionClosed, asyncio.CancelledError):
                pass

        recv_task = asyncio.create_task(receiver_loop())

        # Stream audio in 20 ms chunks (640 bytes at 16kHz s16le mono)
        chunk_size = 640
        print(f"[SmokeTest] Streaming {len(frames)} bytes in 20ms ({chunk_size}B) chunks...")
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = chunk + b"\x00" * (chunk_size - len(chunk))
            await ws.send(chunk)
            await asyncio.sleep(0.02)  # Real-time pacing (20ms)

        print("[SmokeTest] Finished streaming audio. Waiting for agent reply & turn completion...")

        # Wait for turn_done with timeout
        try:
            await asyncio.wait_for(turn_done_event.wait(), timeout=25.0)
            print("[SmokeTest] Received turn_done event.")
        except asyncio.TimeoutError:
            print("[SmokeTest] WARNING: Timed out waiting for turn_done!")

        # Send playback_done and end_call
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
    print(f"[SmokeTest] Call completed. Total received agent audio: {received_audio_bytes} bytes.")

    # Step 3: Fetch turn metrics from GET /sessions/{id}
    async with httpx.AsyncClient(base_url=base_url) as client:
        sess_resp = await client.get(f"/sessions/{session_id}")
        sess_resp.raise_for_status()
        sess_data = sess_resp.json()

    print("\n" + "=" * 60)
    print(f"       SESSION REPORT ({session_id})")
    print("=" * 60)
    sess_info = sess_data.get("session", {})
    print(f"Started At: {sess_info.get('started_at')}")
    print(f"Ended At:   {sess_info.get('ended_at')}")
    print(f"End Reason: {sess_info.get('end_reason')}")

    turns = sess_data.get("turns", [])
    print(f"\n--- Conversation Turns ({len(turns)}) ---")
    for t in turns:
        role = t.get("role", "unknown").upper()
        text = t.get("text", "")
        tc = t.get("tool_calls_json")
        extra = f" [tool_calls: {tc}]" if tc else ""
        print(f"  [{role} Turn {t.get('turn_id')}]: {text}{extra}")

    metrics = sess_data.get("metrics", [])
    if metrics:
        m = metrics[0]
        print("\n--- Actual Per-Stage Latency Metrics (DB Record) ---")
        print(f"  VAD Speech End Latency:    {m.get('vad_end_ms'):.2f} ms")
        print(f"  STT Final Transcript Latency: {m.get('stt_final_ms'):.2f} ms")
        print(f"  LLM First Token Latency:      {m.get('llm_first_token_ms'):.2f} ms")
        print(f"  TTS First Audio Latency:      {m.get('tts_first_audio_ms'):.2f} ms")
        print(f"  Total Latency (from Speech Start): {m.get('total_ms'):.2f} ms")
        if m.get('total_ms_from_vad_end') is not None:
            print(f"  Total Latency (from VAD End):      {m.get('total_ms_from_vad_end'):.2f} ms")
    elif turn_metrics:
        print("\n--- Actual Per-Stage Latency Metrics (from Event) ---")
        for k, v in turn_metrics.items():
            print(f"  {k}: {v} ms")
    else:
        print("\n[SmokeTest] No metrics recorded.")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end voice smoke test.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of server")
    parser.add_argument("--audio", default="tests/data/what_time_tokyo.wav", help="Path to 16kHz mono WAV")
    parser.add_argument("--vad", default="silero", help="VAD provider (silero, webrtc, mock)")
    parser.add_argument("--stt", default="vosk", help="STT provider (vosk, mock)")
    parser.add_argument("--llm", default="groq", help="LLM provider (groq, mock)")
    parser.add_argument("--tts", default="piper", help="TTS provider (piper, mock)")
    args = parser.parse_args()

    asyncio.run(
        run_smoke_test(
            base_url=args.url,
            audio_path=args.audio,
            vad_provider=args.vad,
            stt_provider=args.stt,
            llm_provider=args.llm,
            tts_provider=args.tts,
        )
    )


if __name__ == "__main__":
    main()
