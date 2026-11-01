# Voice Agent MVP

A real-time voice agent: speak in the browser, hear a spoken reply.
Pipeline: mic → VAD → streaming STT → LLM (+ tool call) → streaming TTS → speaker,
with barge-in, per-stage latency metrics, error handling and SQLite persistence.

## Architecture

```
Browser (mic, 16 kHz resampling, playback, captions, barge-in)
   │  WebSocket: binary PCM s16le 16 kHz 20 ms frames + JSON events
   ▼
FastAPI backend ── Session Manager (state machine: idle/listening/thinking/speaking/ended)
   ├─ VAD        Silero (ONNX)               adapters/vad_silero.py
   ├─ STT        Vosk streaming              adapters/stt_vosk.py
   ├─ LLM        Groq (tool calling)         adapters/llm_groq.py
   ├─ Tools      get_current_time, get_order_status   tools/
   ├─ TTS        Piper, sentence-level       adapters/tts_piper.py
   └─ Metrics    per-turn timings + transcript → SQLite
```

Every provider sits behind a Protocol (`adapters/protocols.py`) and is chosen by config
(`*_PROVIDER`). Mock and alternative adapters (Deepgram STT, WebRTC VAD) are included,
so swapping a provider needs one adapter file plus a config change.
Event schema, state machine and error matrix: `docs/design.md`.

## Setup

Requirements: Python 3.11+ (tested on 3.13), a Groq API key (free tier).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then set GROQ_API_KEY
```

Models (not in git), put them in `models/`:
- Vosk: `vosk-model-en-us-0.22-lgraph` (alphacephei.com/vosk/models), unzipped
- Piper voice: `en_US-lessac-medium.onnx` and `.onnx.json` (rhasspy/piper-voices on Hugging Face)
- Silero VAD: `silero_vad.onnx` (snakers4/silero-vad)

## Run

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000, click **Start Call**, and speak (use headphones to avoid echo).
`GET /healthz` shows the active providers. `GET /sessions/<id>` returns the saved
transcript and per-stage metrics.

## Configuration (.env)

| Variable | Default / example |
|---|---|
| GROQ_API_KEY, GROQ_MODEL | your key, `qwen/qwen3.8-27b` |
| VAD_PROVIDER / STT_PROVIDER / LLM_PROVIDER / TTS_PROVIDER | silero / vosk / groq / piper (`mock` for offline tests) |
| VOSK_MODEL_PATH, PIPER_MODEL_PATH, SILERO_MODEL_PATH | paths under `models/` |
| VAD_SILENCE_MS, VAD_MIN_SPEECH_MS | 600, 250 |
| MAX_CALL_SECONDS, SILENCE_TIMEOUT_SECONDS | 600, 120 |
| STT_TIMEOUT_S, LLM_TIMEOUT_S, TTS_TIMEOUT_S, TOOL_TIMEOUT_S | 3, 5, 5, 5 |
| SYSTEM_PROMPT, DB_PATH | see `.env.example` |

## Tests and benchmark

```bash
pytest -q                          # unit + integration tests (mock providers, offline)
python scripts/smoke_test.py       # one real turn through real providers
python scripts/benchmark.py        # 20 turns, real providers → docs/latency_report.md
```

## Latency (20 turns, real providers, synthetic Piper audio)

| Stage | Median | p90 |
|---|---|---|
| VAD end-of-speech wait | 600 ms | 600 ms |
| STT final | 106 ms | 248 ms |
| LLM first token | 237 ms | 291 ms |
| TTS first audio | 150 ms | 316 ms |
| **Total, last spoken word → first agent audio** | **1180 ms** | **1330 ms** |

Target was ≤ 1.5 s median. Details and raw rows: `docs/latency_report.md`.

## Design decisions

- Server-side Silero VAD; client-side adaptive-threshold barge-in (grace period 600 ms,
  sustained 250 ms) for fast stop, with the server as source of truth (cancels LLM/TTS).
- Sentence-level TTS, with an early first chunk at a clause boundary to cut time to first audio.
- Persistent IPv4 HTTP client for Groq. Per-call connections caused multi-second stalls on Windows.
- First-token timeout (5 s) applies per LLM call, not the whole reply.
- History keeps only what the user actually heard after a barge-in.

## Known limitations (honest notes)

- Vosk can mishear names and accented speech; speaking clearly helps. Providers are swappable.
- Latency numbers use synthetic audio; real microphones and noise will be slower.
- Barge-in is energy-based on the client; headphones recommended to avoid echo triggers.
- The browser capture uses ScriptProcessorNode (design doc said AudioWorklet); it works but is deprecated.
- Free-tier Groq limits can cause occasional delays or fallback replies.
- Out of scope per assignment: telephony, auth, multi-language, scaling, dashboards.

## Repo layout

```
backend/app/   main, api, ws, session, pipeline, sentence_buffer, metrics, config, db
backend/app/adapters/  vad_*, stt_*, llm_*, tts_*, registry, protocols
backend/app/tools/     time_tool, order_tool, executor
backend/tests/ unit + integration tests
frontend/      index.html, app.js, mic-worklet.js
scripts/       smoke_test.py, benchmark.py, mic_sim.py
docs/          design.md, assignment.md, latency_report.md
```