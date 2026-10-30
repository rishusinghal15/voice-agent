"""End-to-end voice pipeline orchestration.

Connects:
- VADProcessor (frame processing and speech detection)
- STTAdapter (streaming audio frames and partial/final transcripts)
- LLMAdapter (prompt construction with system prompt + history + transcript, token streaming)
- ToolExecutor (intercepting ToolCall, running tools, feeding results back to LLM)
- SentenceBuffer (accumulating tokens into complete sentences)
- TTSAdapter (synthesizing sentences to PCM audio chunks)
- TurnTimer (recording timestamps and computing turn metrics)
- db (persisting turns, metrics, and errors)
- Session (state machine enforcing design doc §4 transitions)

Follows design doc §3, §4, §5, §6, §7, §8, §9, §10.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from backend.app import db
from backend.app.adapters.protocols import (
    Done,
    LLMAdapter,
    Message,
    STTAdapter,
    STTFinal,
    STTPartial,
    STTStream,
    TextDelta,
    ToolCall,
    TTSAdapter,
    VADEventType,
    VADProcessor,
)
from backend.app.adapters.registry import (
    get_llm_adapter,
    get_stt_adapter,
    get_tts_adapter,
    get_vad_adapter,
)
from backend.app.config import settings
from backend.app.metrics import TurnTimer
from backend.app.sentence_buffer import SentenceBuffer
from backend.app.session import Session, SessionEvent, SessionState
from backend.app.tools.executor import ToolExecutor, create_default_tool_executor


def current_epoch_ms() -> int:
    """Return current server time as epoch milliseconds."""
    return int(time.time() * 1000)


class VoicePipeline:
    """Orchestrates the real-time bidirectional voice conversation pipeline."""

    def __init__(
        self,
        session_id: str,
        emit_json: Callable[[Dict[str, Any]], Awaitable[None]],
        emit_audio: Callable[[bytes], Awaitable[None]],
        vad: Optional[VADProcessor] = None,
        stt: Optional[STTAdapter] = None,
        llm: Optional[LLMAdapter] = None,
        tts: Optional[TTSAdapter] = None,
        executor: Optional[ToolExecutor] = None,
        session: Optional[Session] = None,
    ) -> None:
        self.session_id: str = session_id
        self._emit_json_raw: Callable[[Dict[str, Any]], Awaitable[None]] = emit_json
        self._emit_audio_raw: Callable[[bytes], Awaitable[None]] = emit_audio

        # Adapter dependencies
        self.vad: VADProcessor = vad or get_vad_adapter()
        self.stt: STTAdapter = stt or get_stt_adapter()
        self.llm: LLMAdapter = llm or get_llm_adapter()
        self.tts: TTSAdapter = tts or get_tts_adapter()
        self.executor: ToolExecutor = executor or create_default_tool_executor()

        # Session state machine
        self.session: Session = session or Session(session_id=session_id)

        # Multi-turn history: list of {"role": "user"|"agent", "content": str}
        self.history: List[Dict[str, str]] = []

        # Current turn state
        self.turn_id: int = 1
        self.turn_timer: TurnTimer = TurnTimer()
        self.in_speech: bool = False
        self.speech_start_time: float = 0.0

        # Background tasks & queues
        self.stt_stream: Optional[STTStream] = None
        self.stt_events_task: Optional[asyncio.Task] = None
        self.final_transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self.active_turn_task: Optional[asyncio.Task] = None
        self.silence_timer_task: Optional[asyncio.Task] = None
        self.max_duration_task: Optional[asyncio.Task] = None

        # Turn accumulators
        self.sent_sentences: List[Tuple[str, float]] = []  # (sentence_text, duration_ms)
        self.turn_tool_calls: List[Dict[str, Any]] = []
        self.current_agent_text: List[str] = []

        self.closed: bool = False

    async def emit_json(self, payload: Dict[str, Any]) -> None:
        """Send JSON message to client, ensuring 'ts' (epoch ms) is included."""
        if "ts" not in payload:
            payload["ts"] = current_epoch_ms()
        await self._emit_json_raw(payload)

    async def emit_audio(self, chunk: bytes) -> None:
        """Send raw binary PCM chunk to client."""
        await self._emit_audio_raw(chunk)

    async def start(self) -> None:
        """Initialize pipeline, transition to LISTENING and start timeouts."""
        self.session.transition(SessionEvent.SESSION_START)
        await self.emit_json({
            "type": "state",
            "state": self.session.state.value,
        })
        self._reset_silence_timer()
        self._start_max_duration_timer()

    def _reset_silence_timer(self) -> None:
        """Reset the 120s silence timeout watchdog."""
        self._cancel_silence_timer()
        if not self.closed:
            self.silence_timer_task = asyncio.create_task(self._silence_watchdog())

    def _cancel_silence_timer(self) -> None:
        """Cancel active silence timeout watchdog."""
        if self.silence_timer_task and not self.silence_timer_task.done():
            self.silence_timer_task.cancel()
            self.silence_timer_task = None

    async def _silence_watchdog(self) -> None:
        """Watchdog firing when silence exceeds SILENCE_TIMEOUT_SECONDS."""
        try:
            await asyncio.sleep(settings.SILENCE_TIMEOUT_SECONDS)
            if not self.closed:
                await self.handle_end_call(reason="silence_timeout")
        except asyncio.CancelledError:
            pass

    def _start_max_duration_timer(self) -> None:
        """Start watchdog for MAX_CALL_SECONDS."""
        if not self.closed:
            self.max_duration_task = asyncio.create_task(self._max_duration_watchdog())

    async def _max_duration_watchdog(self) -> None:
        """Watchdog firing when call duration reaches MAX_CALL_SECONDS."""
        try:
            await asyncio.sleep(settings.MAX_CALL_SECONDS)
            if not self.closed:
                await self.handle_end_call(reason="max_duration")
        except asyncio.CancelledError:
            pass

    async def handle_ping(self) -> None:
        """Respond to client ping keepalive."""
        await self.emit_json({"type": "pong"})

    async def handle_audio(self, frame: bytes) -> None:
        """Process an incoming PCM frame from the client."""
        if self.closed:
            return

        # Only process audio input when in LISTENING state
        if self.session.state != SessionState.LISTENING:
            return

        event = self.vad.process(frame)

        if event and event.type == VADEventType.SPEECH_START:
            self._cancel_silence_timer()
            self.in_speech = True
            self.speech_start_time = time.monotonic()
            self.turn_timer = TurnTimer()
            self.turn_timer.mark_speech_start()
            self.session.transition(SessionEvent.SPEECH_START)

            # Reset transcript queue and open STT stream
            self.final_transcript_queue = asyncio.Queue()
            self.stt_stream = await self.stt.open_stream(sample_rate=16000)
            self.stt_events_task = asyncio.create_task(
                self._consume_stt_events(self.stt_stream)
            )
            await self.stt_stream.send_audio(frame)

        elif self.in_speech:
            if self.stt_stream:
                await self.stt_stream.send_audio(frame)

            if event and event.type == VADEventType.SPEECH_END:
                self.in_speech = False
                self.turn_timer.mark_vad_end()

                # Utterance duration check: drop if shorter than VAD_MIN_SPEECH_MS
                speech_dur_ms = (time.monotonic() - self.speech_start_time) * 1000.0
                if speech_dur_ms < settings.VAD_MIN_SPEECH_MS:
                    self.session.transition(SessionEvent.SPEECH_END_NOISE)
                    if self.stt_stream:
                        await self.stt_stream.finish()
                    if self.stt_events_task and not self.stt_events_task.done():
                        self.stt_events_task.cancel()
                    self._reset_silence_timer()
                    return

                # Valid utterance -> Transition to THINKING and process turn
                self.session.transition(SessionEvent.SPEECH_END_VALID)
                await self.emit_json({
                    "type": "state",
                    "state": self.session.state.value,
                })
                self.active_turn_task = asyncio.create_task(self._run_turn())

    async def _consume_stt_events(self, stream: STTStream) -> None:
        """Background worker consuming events from streaming STT."""
        try:
            async for event in stream.events():
                if isinstance(event, STTPartial):
                    await self.emit_json({
                        "type": "transcript_partial",
                        "text": event.text,
                    })
                elif isinstance(event, STTFinal):
                    await self.final_transcript_queue.put(event.text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await self.final_transcript_queue.put("")

    async def _run_turn(self) -> None:
        """Coordinate STT finalization, LLM streaming, tools, and TTS synthesis."""
        try:
            # 1. Finalize STT
            if self.stt_stream:
                await self.stt_stream.finish()

            try:
                final_text = await asyncio.wait_for(
                    self.final_transcript_queue.get(),
                    timeout=settings.STT_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                await self._handle_turn_error(
                    stage="stt",
                    code="STT_ERROR",
                    message=f"STT final transcript timed out after {settings.STT_TIMEOUT_S}s",
                )
                return
            except Exception as exc:
                await self._handle_turn_error(
                    stage="stt",
                    code="STT_ERROR",
                    message=f"STT error: {exc}",
                )
                return

            # Ignore empty or noise-only transcript without LLM call
            if not final_text or not final_text.strip():
                self.session.state = SessionState.LISTENING
                await self.emit_json({
                    "type": "state",
                    "state": self.session.state.value,
                })
                self._reset_silence_timer()
                return

            clean_text = final_text.strip()
            self.turn_timer.mark_stt_final()
            self.session.transition(SessionEvent.TRANSCRIPT_FINAL)

            await self.emit_json({
                "type": "transcript_final",
                "text": clean_text,
                "turn_id": self.turn_id,
            })

            # Record user turn in memory and SQLite
            self.history.append({"role": "user", "content": clean_text})
            await db.write_turn(
                session_id=self.session_id,
                turn_id=self.turn_id,
                role="user",
                text=clean_text,
            )

            # 2. LLM streaming and TTS synthesis
            await self._stream_llm_and_tts()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_turn_error(
                stage="llm",
                code="LLM_ERROR",
                message=f"Turn execution error: {exc}",
            )

    async def _stream_llm_and_tts(self) -> None:
        """Stream LLM response with tool interception and sentence-level TTS synthesis."""
        messages: List[Message] = [
            Message(role="system", content=settings.SYSTEM_PROMPT)
        ]
        for m in self.history:
            messages.append(Message(role=m["role"], content=m["content"]))

        tools = self.executor.get_specs()
        sb = SentenceBuffer()
        sentence_index = 0
        self.sent_sentences = []
        self.turn_tool_calls = []
        self.current_agent_text = []
        first_token_stamped = False
        first_sentence_stamped = False

        while True:
            tool_call_to_run: Optional[ToolCall] = None
            first_token_received = False

            async def _run_llm_stream() -> None:
                nonlocal tool_call_to_run, first_token_stamped, first_sentence_stamped, sentence_index, first_token_received
                async for event in self.llm.stream(messages, tools):
                    if isinstance(event, ToolCall):
                        tool_call_to_run = event
                        break
                    elif isinstance(event, TextDelta):
                        first_token_received = True
                        if not first_token_stamped:
                            self.turn_timer.mark_llm_first_token()
                            first_token_stamped = True
                        self.current_agent_text.append(event.text)
                        for sentence in sb.feed(event.text):
                            await self._synthesize_and_send_sentence(
                                sentence=sentence,
                                sentence_index=sentence_index,
                                first_sentence_stamped=first_sentence_stamped,
                            )
                            first_sentence_stamped = True
                            sentence_index += 1
                    elif isinstance(event, Done):
                        pass

            try:
                # Apply LLM timeout
                await asyncio.wait_for(
                    _run_llm_stream(),
                    timeout=settings.LLM_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                await self._handle_turn_error(
                    stage="llm",
                    code="LLM_ERROR",
                    message=f"LLM timed out after {settings.LLM_TIMEOUT_S}s",
                )
                return
            except Exception as exc:
                await self._handle_turn_error(
                    stage="llm",
                    code="LLM_ERROR",
                    message=f"LLM streaming error: {exc}",
                )
                return

            if tool_call_to_run is not None:
                # Intercept tool call
                tc = tool_call_to_run
                self.session.transition(SessionEvent.TOOL_CALL_REQUESTED)
                await self.emit_json({
                    "type": "tool_event",
                    "turn_id": self.turn_id,
                    "name": tc.name,
                    "status": "started",
                })

                tool_res = await self.executor.run(tc.name, tc.args)
                status_str = "ok" if "error" not in tool_res else "error"
                await self.emit_json({
                    "type": "tool_event",
                    "turn_id": self.turn_id,
                    "name": tc.name,
                    "status": status_str,
                })

                self.turn_tool_calls.append({
                    "name": tc.name,
                    "args": tc.args,
                    "result": tool_res,
                })

                # Feed tool call and result back into prompt for final LLM reply
                messages.append(
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=[{
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.args),
                            },
                        }],
                    )
                )
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(tool_res),
                        tool_call_id=tc.id,
                    )
                )
                continue
            else:
                # Flush remaining tokens in sentence buffer
                remaining = sb.flush()
                if remaining and remaining.strip():
                    await self._synthesize_and_send_sentence(
                        sentence=remaining.strip(),
                        sentence_index=sentence_index,
                        first_sentence_stamped=first_sentence_stamped,
                    )
                    first_sentence_stamped = True
                    sentence_index += 1

                total_audio_ms = sum(s[1] for s in self.sent_sentences)
                await self.emit_json({
                    "type": "turn_done",
                    "turn_id": self.turn_id,
                    "total_audio_ms": round(total_audio_ms, 2),
                })
                break

    async def _synthesize_and_send_sentence(
        self,
        sentence: str,
        sentence_index: int,
        first_sentence_stamped: bool,
    ) -> None:
        """Synthesize a complete sentence into PCM chunks and stream to client."""
        if not first_sentence_stamped:
            self.turn_timer.mark_first_sentence_ready()
            self.session.transition(SessionEvent.FIRST_SENTENCE_READY)
            await self.emit_json({
                "type": "state",
                "state": self.session.state.value,
            })

        await self.emit_json({
            "type": "agent_text",
            "text": sentence,
            "turn_id": self.turn_id,
            "sentence_index": sentence_index,
        })

        await self.emit_json({
            "type": "agent_audio_start",
            "turn_id": self.turn_id,
            "sentence_index": sentence_index,
            "sample_rate": self.tts.sample_rate,
            "encoding": "pcm_s16le",
            "channels": 1,
        })

        sentence_bytes = 0
        try:
            async for chunk in self.tts.synthesize(sentence):
                if self.turn_timer.t_tts_first_audio is None:
                    self.turn_timer.mark_tts_first_audio()
                sentence_bytes += len(chunk)
                await self.emit_audio(chunk)
        except Exception as exc:
            # TTS failure: log error and notify client
            await db.write_error(
                session_id=self.session_id,
                turn_id=self.turn_id,
                stage="tts",
                code="TTS_ERROR",
                message=f"TTS synthesis error: {exc}",
            )
            await self.emit_json({
                "type": "error",
                "code": "TTS_ERROR",
                "message": f"TTS synthesis error: {exc}",
                "recoverable": True,
            })
            return

        await self.emit_json({
            "type": "agent_audio_end",
            "turn_id": self.turn_id,
            "sentence_index": sentence_index,
        })

        # Calculate sentence audio duration: 16-bit mono = 2 bytes per sample
        duration_ms = (sentence_bytes / (self.tts.sample_rate * 2)) * 1000.0
        self.sent_sentences.append((sentence, duration_ms))

    async def handle_playback_done(self, turn_id: Optional[int] = None) -> None:
        """Handle client notification that agent speech finished playing."""
        if turn_id is not None and turn_id != self.turn_id:
            return

        if self.session.state == SessionState.SPEAKING:
            self.session.transition(SessionEvent.PLAYBACK_DONE)
            await self.emit_json({
                "type": "state",
                "state": self.session.state.value,
            })

            # Commit full agent response to history and DB
            full_reply = "".join(self.current_agent_text).strip()
            if full_reply:
                self.history.append({"role": "agent", "content": full_reply})
                tool_calls_json = (
                    json.dumps(self.turn_tool_calls) if self.turn_tool_calls else None
                )
                await db.write_turn(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    role="agent",
                    text=full_reply,
                    interrupted=0,
                    tool_calls_json=tool_calls_json,
                )

            # Compute and persist turn latency metrics
            metrics = self.turn_timer.compute()
            await db.write_turn_metrics(
                session_id=self.session_id,
                turn_id=self.turn_id,
                vad_end_ms=metrics.vad_end_ms,
                stt_final_ms=metrics.stt_final_ms,
                llm_first_token_ms=metrics.llm_first_token_ms,
                tts_first_audio_ms=metrics.tts_first_audio_ms,
                total_ms=metrics.total_ms,
            )

            # Advance to next turn
            self.turn_id += 1
            self.turn_timer = TurnTimer()
            self._reset_silence_timer()

    async def handle_interrupt(
        self,
        turn_id: Optional[int] = None,
        played_ms: float = 0.0,
    ) -> None:
        """Handle barge-in interrupt: cancel tasks, truncate history to played portion."""
        if turn_id is not None and turn_id != self.turn_id:
            return

        # Cancel in-flight turn task
        if self.active_turn_task and not self.active_turn_task.done():
            self.active_turn_task.cancel()
            try:
                await self.active_turn_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        # Map played_ms to sentence boundaries per design doc §5
        heard_sentences: List[str] = []
        accumulated_ms = 0.0
        for sentence, dur_ms in self.sent_sentences:
            if accumulated_ms + dur_ms <= played_ms:
                heard_sentences.append(sentence)
                accumulated_ms += dur_ms
            elif accumulated_ms < played_ms:
                remaining_ms = played_ms - accumulated_ms
                fraction = max(0.1, min(1.0, remaining_ms / dur_ms))
                words = sentence.split()
                kept_count = max(1, int(len(words) * fraction))
                heard_sentences.append(" ".join(words[:kept_count]) + "...")
                break
            else:
                break

        if not heard_sentences and self.sent_sentences:
            heard_sentences.append(self.sent_sentences[0][0])

        spoken_text = " ".join(heard_sentences).strip()
        if spoken_text:
            self.history.append({"role": "agent", "content": spoken_text})
            tool_calls_json = (
                json.dumps(self.turn_tool_calls) if self.turn_tool_calls else None
            )
            await db.write_turn(
                session_id=self.session_id,
                turn_id=self.turn_id,
                role="agent",
                text=spoken_text,
                interrupted=1,
                tool_calls_json=tool_calls_json,
            )

        # Transition back to LISTENING
        self.session.transition(SessionEvent.INTERRUPT)
        await self.emit_json({
            "type": "state",
            "state": self.session.state.value,
        })

        self.turn_id += 1
        self.turn_timer = TurnTimer()
        self._reset_silence_timer()

    async def _handle_turn_error(self, stage: str, code: str, message: str) -> None:
        """Handle adapter errors: write error, notify client, speak fallback line."""
        await db.write_error(
            session_id=self.session_id,
            turn_id=self.turn_id,
            stage=stage,
            code=code,
            message=message,
        )
        await self.emit_json({
            "type": "error",
            "code": code,
            "message": message,
            "recoverable": True,
        })

        # Transition to SPEAKING to speak fallback line
        fallback = settings.FALLBACK_LINE
        if self.session.state == SessionState.THINKING:
            self.session.transition(SessionEvent.ERROR_OR_TIMEOUT)
            await self.emit_json({
                "type": "state",
                "state": self.session.state.value,
            })

        await self.emit_json({
            "type": "agent_text",
            "text": fallback,
            "turn_id": self.turn_id,
            "sentence_index": 0,
        })

        try:
            await self.emit_json({
                "type": "agent_audio_start",
                "turn_id": self.turn_id,
                "sentence_index": 0,
                "sample_rate": self.tts.sample_rate,
                "encoding": "pcm_s16le",
                "channels": 1,
            })
            async for chunk in self.tts.synthesize(fallback):
                await self.emit_audio(chunk)
            await self.emit_json({
                "type": "agent_audio_end",
                "turn_id": self.turn_id,
                "sentence_index": 0,
            })
        except Exception:
            # If TTS is down, fallback line is text only per design doc §9
            pass

        # Return to LISTENING
        self.session.transition(SessionEvent.PLAYBACK_DONE)
        await self.emit_json({
            "type": "state",
            "state": self.session.state.value,
        })

        self.turn_id += 1
        self.turn_timer = TurnTimer()
        self._reset_silence_timer()

    async def handle_end_call(self, reason: str = "user_end") -> None:
        """Terminate call session gracefully."""
        if self.closed:
            return
        self.closed = True

        self._cancel_silence_timer()
        if self.max_duration_task and not self.max_duration_task.done():
            self.max_duration_task.cancel()

        if self.active_turn_task and not self.active_turn_task.done():
            self.active_turn_task.cancel()

        if self.stt_events_task and not self.stt_events_task.done():
            self.stt_events_task.cancel()

        try:
            self.session.transition(SessionEvent.END_CALL)
        except Exception:
            self.session.state = SessionState.ENDED

        try:
            await self.emit_json({
                "type": "call_ended",
                "reason": reason,
            })
        except Exception:
            # Client socket may already be closed (e.g. ws_drop)
            pass

        await db.end_session(
            session_id=self.session_id,
            end_reason=reason,
        )

    async def close(self, reason: str = "ws_drop") -> None:
        """Clean up on socket disconnection."""
        if not self.closed:
            await self.handle_end_call(reason=reason)

