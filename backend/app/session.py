"""Session state machine and lifecycle manager.

Follows design doc §4:
- States: IDLE, LISTENING, THINKING, SPEAKING, ENDED
- Full transition table implementing side effects and illegal transition rejection
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set


class SessionState(str, Enum):
    """Session states from design doc §4."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ENDED = "ended"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            clean = value.strip().lower()
            for member in cls:
                if member.value == clean or member.name.lower() == clean:
                    return member
        return super()._missing_(value)


class SessionEvent(str, Enum):
    """Session events from design doc §4."""

    SESSION_START = "session_start"
    SPEECH_START = "speech_start"
    SPEECH_END_VALID = "speech_end_valid"
    SPEECH_END_NOISE = "speech_end_noise"
    TRANSCRIPT_FINAL = "transcript_final"
    FIRST_SENTENCE_READY = "first_sentence_ready"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    ERROR_OR_TIMEOUT = "error_or_timeout"
    PLAYBACK_DONE = "playback_done"
    INTERRUPT = "interrupt"
    END_CALL = "end_call"
    SILENCE_TIMEOUT = "silence_timeout"
    MAX_DURATION = "max_duration"
    WS_DROP = "ws_drop"
    CALL_ENDED = "call_ended"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            clean = value.strip().lower()
            if clean in EVENT_ALIASES:
                return EVENT_ALIASES[clean]
            for member in cls:
                if member.value == clean or member.name.lower() == clean:
                    return member
        return super()._missing_(value)


# Canonical mappings for exact text variations in design doc §4
EVENT_ALIASES: Dict[str, SessionEvent] = {
    "post /sessions + ws connected": SessionEvent.SESSION_START,
    "ws_connected": SessionEvent.SESSION_START,
    "start_call": SessionEvent.SESSION_START,
    "vad speech_start": SessionEvent.SPEECH_START,
    "vad speech_end with valid utterance": SessionEvent.SPEECH_END_VALID,
    "speech_end": SessionEvent.SPEECH_END_VALID,
    "vad speech_end with noise-only or too-short utterance": SessionEvent.SPEECH_END_NOISE,
    "speech_end_short": SessionEvent.SPEECH_END_NOISE,
    "transcript_final received": SessionEvent.TRANSCRIPT_FINAL,
    "first sentence ready for tts": SessionEvent.FIRST_SENTENCE_READY,
    "tool call requested": SessionEvent.TOOL_CALL_REQUESTED,
    "stt / llm / tts error or timeout": SessionEvent.ERROR_OR_TIMEOUT,
    "reply finished and playback done": SessionEvent.PLAYBACK_DONE,
    "interrupt from client": SessionEvent.INTERRUPT,
    "end call / 2 min silence / max duration / ws drop": SessionEvent.END_CALL,
    "end call": SessionEvent.END_CALL,
    "2 min silence": SessionEvent.SILENCE_TIMEOUT,
    "max duration": SessionEvent.MAX_DURATION,
    "ws drop": SessionEvent.WS_DROP,
}

# The termination events that can transition any active state to ENDED
TERMINATION_EVENTS: Set[SessionEvent] = {
    SessionEvent.END_CALL,
    SessionEvent.SILENCE_TIMEOUT,
    SessionEvent.MAX_DURATION,
    SessionEvent.WS_DROP,
    SessionEvent.CALL_ENDED,
}

# Strict transition table: (CurrentState, Event) -> TargetState
TRANSITION_TABLE: Dict[tuple[SessionState, SessionEvent], SessionState] = {
    # Row 1: IDLE + POST /sessions + WS connected -> LISTENING
    (SessionState.IDLE, SessionEvent.SESSION_START): SessionState.LISTENING,
    # Row 2: LISTENING + VAD speech_start -> LISTENING
    (SessionState.LISTENING, SessionEvent.SPEECH_START): SessionState.LISTENING,
    # Row 3: LISTENING + VAD speech_end with valid utterance -> THINKING
    (SessionState.LISTENING, SessionEvent.SPEECH_END_VALID): SessionState.THINKING,
    # Row 4: LISTENING + VAD speech_end with noise-only or too-short utterance -> LISTENING
    (SessionState.LISTENING, SessionEvent.SPEECH_END_NOISE): SessionState.LISTENING,
    # Row 5: THINKING + transcript_final received -> THINKING
    (SessionState.THINKING, SessionEvent.TRANSCRIPT_FINAL): SessionState.THINKING,
    # Row 6: THINKING + First sentence ready for TTS -> SPEAKING
    (SessionState.THINKING, SessionEvent.FIRST_SENTENCE_READY): SessionState.SPEAKING,
    # Row 7: THINKING + Tool call requested -> THINKING
    (SessionState.THINKING, SessionEvent.TOOL_CALL_REQUESTED): SessionState.THINKING,
    # Row 8: THINKING + STT / LLM / TTS error or timeout -> SPEAKING
    (SessionState.THINKING, SessionEvent.ERROR_OR_TIMEOUT): SessionState.SPEAKING,
    # Row 9: SPEAKING + Reply finished and playback done -> LISTENING
    (SessionState.SPEAKING, SessionEvent.PLAYBACK_DONE): SessionState.LISTENING,
    # Row 10: SPEAKING + interrupt from client -> LISTENING
    (SessionState.SPEAKING, SessionEvent.INTERRUPT): SessionState.LISTENING,
}


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, current_state: SessionState, event: Any) -> None:
        super().__init__(
            f"Cannot transition from {current_state.value} using event {event}"
        )
        self.current_state = current_state
        self.event = event


class Session:
    """Manages a single call session's state, turn history, and lifecycle."""

    def __init__(
        self,
        session_id: str,
        config: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
    ) -> None:
        self.session_id: str = session_id
        self.state: SessionState = SessionState.IDLE
        self.config: Dict[str, Any] = config or {}
        self.started_at: Optional[str] = started_at
        self.ended_at: Optional[str] = None
        self.end_reason: Optional[str] = None
        self.current_turn_id: int = 0
        self.history: List[Dict[str, Any]] = []
        self.audio_buffer: bytearray = bytearray()

    def transition(self, event: SessionEvent | str) -> SessionState:
        """Apply a state transition according to design §4 table.

        Raises:
            InvalidStateTransitionError: If the transition is illegal.
        """
        if not isinstance(event, SessionEvent):
            event = SessionEvent(event)

        # Terminal state check: Once ENDED, no further transitions are valid
        if self.state == SessionState.ENDED:
            raise InvalidStateTransitionError(self.state, event)

        # Termination events can transition any non-ended state to ENDED
        if event in TERMINATION_EVENTS:
            self.state = SessionState.ENDED
            if event == SessionEvent.END_CALL:
                self.end_reason = "user_end"
            elif event == SessionEvent.SILENCE_TIMEOUT:
                self.end_reason = "silence_timeout"
            elif event == SessionEvent.MAX_DURATION:
                self.end_reason = "max_duration"
            elif event == SessionEvent.WS_DROP:
                self.end_reason = "ws_drop"
            else:
                self.end_reason = event.value
            return self.state

        # Check explicit state transition table
        key = (self.state, event)
        if key in TRANSITION_TABLE:
            next_state = TRANSITION_TABLE[key]
            self.state = next_state
            return self.state

        # Transition is illegal
        raise InvalidStateTransitionError(self.state, event)

