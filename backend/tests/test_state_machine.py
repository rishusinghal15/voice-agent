"""Unit tests covering every row of the design doc §4 state transition table.

Table in §4:
1.  IDLE      + POST /sessions + WS connected                                -> LISTENING
2.  LISTENING + VAD speech_start                                             -> LISTENING
3.  LISTENING + VAD speech_end with valid utterance                          -> THINKING
4.  LISTENING + VAD speech_end with noise-only or too-short utterance        -> LISTENING
5.  THINKING  + transcript_final received                                    -> THINKING
6.  THINKING  + First sentence ready for TTS                                 -> SPEAKING
7.  THINKING  + Tool call requested                                          -> THINKING
8.  THINKING  + STT / LLM / TTS error or timeout                             -> SPEAKING
9.  SPEAKING  + Reply finished and playback done                             -> LISTENING
10. SPEAKING  + interrupt from client                                        -> LISTENING
11. any       + End Call / 2 min silence / max duration / WS drop            -> ENDED
"""

import pytest

from backend.app.session import (
    InvalidStateTransitionError,
    Session,
    SessionEvent,
    SessionState,
)


def test_initial_state() -> None:
    """Verify session starts in IDLE state."""
    session = Session(session_id="test-session-1")
    assert session.state == SessionState.IDLE


def test_row_1_idle_to_listening() -> None:
    """Row 1: IDLE + POST /sessions + WS connected -> LISTENING."""
    # Test canonical enum
    session = Session(session_id="s1")
    assert session.state == SessionState.IDLE
    next_state = session.transition(SessionEvent.SESSION_START)
    assert next_state == SessionState.LISTENING
    assert session.state == SessionState.LISTENING

    # Test exact design text string
    session2 = Session(session_id="s2")
    session2.transition("POST /sessions + WS connected")
    assert session2.state == SessionState.LISTENING

    # Test alias
    session3 = Session(session_id="s3")
    session3.transition("ws_connected")
    assert session3.state == SessionState.LISTENING


def test_row_2_listening_speech_start() -> None:
    """Row 2: LISTENING + VAD speech_start -> LISTENING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)
    assert session.state == SessionState.LISTENING

    # Canonical
    next_state = session.transition(SessionEvent.SPEECH_START)
    assert next_state == SessionState.LISTENING
    assert session.state == SessionState.LISTENING

    # Exact design text
    session.transition("VAD speech_start")
    assert session.state == SessionState.LISTENING


def test_row_3_listening_speech_end_valid() -> None:
    """Row 3: LISTENING + VAD speech_end with valid utterance -> THINKING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)

    # Canonical
    next_state = session.transition(SessionEvent.SPEECH_END_VALID)
    assert next_state == SessionState.THINKING
    assert session.state == SessionState.THINKING

    # Test from listening with exact string
    session2 = Session(session_id="s2")
    session2.transition(SessionEvent.SESSION_START)
    session2.transition("VAD speech_end with valid utterance")
    assert session2.state == SessionState.THINKING


def test_row_4_listening_speech_end_noise() -> None:
    """Row 4: LISTENING + VAD speech_end with noise-only or too-short utterance -> LISTENING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)

    # Canonical
    next_state = session.transition(SessionEvent.SPEECH_END_NOISE)
    assert next_state == SessionState.LISTENING
    assert session.state == SessionState.LISTENING

    # Exact design text
    session.transition("VAD speech_end with noise-only or too-short utterance")
    assert session.state == SessionState.LISTENING


def test_row_5_thinking_transcript_final() -> None:
    """Row 5: THINKING + transcript_final received -> THINKING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)
    session.transition(SessionEvent.SPEECH_END_VALID)
    assert session.state == SessionState.THINKING

    # Canonical
    next_state = session.transition(SessionEvent.TRANSCRIPT_FINAL)
    assert next_state == SessionState.THINKING
    assert session.state == SessionState.THINKING

    # Exact design text
    session.transition("transcript_final received")
    assert session.state == SessionState.THINKING


def test_row_6_thinking_first_sentence_ready() -> None:
    """Row 6: THINKING + First sentence ready for TTS -> SPEAKING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)
    session.transition(SessionEvent.SPEECH_END_VALID)
    assert session.state == SessionState.THINKING

    # Canonical
    next_state = session.transition(SessionEvent.FIRST_SENTENCE_READY)
    assert next_state == SessionState.SPEAKING
    assert session.state == SessionState.SPEAKING

    # Exact design text
    session2 = Session(session_id="s2")
    session2.transition(SessionEvent.SESSION_START)
    session2.transition(SessionEvent.SPEECH_END_VALID)
    session2.transition("First sentence ready for TTS")
    assert session2.state == SessionState.SPEAKING


def test_row_7_thinking_tool_call_requested() -> None:
    """Row 7: THINKING + Tool call requested -> THINKING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)
    session.transition(SessionEvent.SPEECH_END_VALID)
    assert session.state == SessionState.THINKING

    # Canonical
    next_state = session.transition(SessionEvent.TOOL_CALL_REQUESTED)
    assert next_state == SessionState.THINKING
    assert session.state == SessionState.THINKING

    # Exact design text
    session.transition("Tool call requested")
    assert session.state == SessionState.THINKING


def test_row_8_thinking_error_or_timeout() -> None:
    """Row 8: THINKING + STT / LLM / TTS error or timeout -> SPEAKING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)
    session.transition(SessionEvent.SPEECH_END_VALID)
    assert session.state == SessionState.THINKING

    # Canonical
    next_state = session.transition(SessionEvent.ERROR_OR_TIMEOUT)
    assert next_state == SessionState.SPEAKING
    assert session.state == SessionState.SPEAKING

    # Exact design text
    session2 = Session(session_id="s2")
    session2.transition(SessionEvent.SESSION_START)
    session2.transition(SessionEvent.SPEECH_END_VALID)
    session2.transition("STT / LLM / TTS error or timeout")
    assert session2.state == SessionState.SPEAKING


def test_row_9_speaking_playback_done() -> None:
    """Row 9: SPEAKING + Reply finished and playback done -> LISTENING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)
    session.transition(SessionEvent.SPEECH_END_VALID)
    session.transition(SessionEvent.FIRST_SENTENCE_READY)
    assert session.state == SessionState.SPEAKING

    # Canonical
    next_state = session.transition(SessionEvent.PLAYBACK_DONE)
    assert next_state == SessionState.LISTENING
    assert session.state == SessionState.LISTENING

    # Exact design text
    session.transition(SessionEvent.SPEECH_END_VALID)
    session.transition(SessionEvent.FIRST_SENTENCE_READY)
    assert session.state == SessionState.SPEAKING
    session.transition("Reply finished and playback done")
    assert session.state == SessionState.LISTENING


def test_row_10_speaking_interrupt() -> None:
    """Row 10: SPEAKING + interrupt from client -> LISTENING."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.SESSION_START)
    session.transition(SessionEvent.SPEECH_END_VALID)
    session.transition(SessionEvent.FIRST_SENTENCE_READY)
    assert session.state == SessionState.SPEAKING

    # Canonical
    next_state = session.transition(SessionEvent.INTERRUPT)
    assert next_state == SessionState.LISTENING
    assert session.state == SessionState.LISTENING

    # Exact design text
    session.transition(SessionEvent.SPEECH_END_VALID)
    session.transition(SessionEvent.FIRST_SENTENCE_READY)
    assert session.state == SessionState.SPEAKING
    session.transition("interrupt from client")
    assert session.state == SessionState.LISTENING


@pytest.mark.parametrize("initial_state", [
    SessionState.IDLE,
    SessionState.LISTENING,
    SessionState.THINKING,
    SessionState.SPEAKING,
])
@pytest.mark.parametrize("termination_event,expected_reason", [
    (SessionEvent.END_CALL, "user_end"),
    (SessionEvent.SILENCE_TIMEOUT, "silence_timeout"),
    (SessionEvent.MAX_DURATION, "max_duration"),
    (SessionEvent.WS_DROP, "ws_drop"),
])
def test_row_11_any_to_ended(
    initial_state: SessionState,
    termination_event: SessionEvent,
    expected_reason: str,
) -> None:
    """Row 11: any + End Call / 2 min silence / max duration / WS drop -> ENDED."""
    session = Session(session_id="s_any")
    # Advance to initial_state
    if initial_state == SessionState.LISTENING:
        session.transition(SessionEvent.SESSION_START)
    elif initial_state == SessionState.THINKING:
        session.transition(SessionEvent.SESSION_START)
        session.transition(SessionEvent.SPEECH_END_VALID)
    elif initial_state == SessionState.SPEAKING:
        session.transition(SessionEvent.SESSION_START)
        session.transition(SessionEvent.SPEECH_END_VALID)
        session.transition(SessionEvent.FIRST_SENTENCE_READY)

    assert session.state == initial_state

    # Transition to ENDED
    next_state = session.transition(termination_event)
    assert next_state == SessionState.ENDED
    assert session.state == SessionState.ENDED
    assert session.end_reason == expected_reason


def test_row_11_design_text_strings() -> None:
    """Row 11 with exact string aliases."""
    for text_event, expected_reason in [
        ("End Call", "user_end"),
        ("2 min silence", "silence_timeout"),
        ("max duration", "max_duration"),
        ("WS drop", "ws_drop"),
        ("End Call / 2 min silence / max duration / WS drop", "user_end"),
    ]:
        session = Session(session_id="test")
        session.transition(text_event)
        assert session.state == SessionState.ENDED
        assert session.end_reason == expected_reason


def test_ended_state_is_terminal() -> None:
    """Verify that no transitions are permitted once in ENDED state."""
    session = Session(session_id="s1")
    session.transition(SessionEvent.END_CALL)
    assert session.state == SessionState.ENDED

    # All events must raise InvalidStateTransitionError
    for event in SessionEvent:
        with pytest.raises(InvalidStateTransitionError):
            session.transition(event)


def test_illegal_transitions_from_idle() -> None:
    """Verify illegal transitions from IDLE state."""
    illegal_events = [
        SessionEvent.SPEECH_START,
        SessionEvent.SPEECH_END_VALID,
        SessionEvent.SPEECH_END_NOISE,
        SessionEvent.TRANSCRIPT_FINAL,
        SessionEvent.FIRST_SENTENCE_READY,
        SessionEvent.TOOL_CALL_REQUESTED,
        SessionEvent.ERROR_OR_TIMEOUT,
        SessionEvent.PLAYBACK_DONE,
        SessionEvent.INTERRUPT,
    ]
    for event in illegal_events:
        session = Session(session_id="s_idle")
        with pytest.raises(InvalidStateTransitionError):
            session.transition(event)


def test_illegal_transitions_from_listening() -> None:
    """Verify illegal transitions from LISTENING state."""
    illegal_events = [
        SessionEvent.SESSION_START,
        SessionEvent.TRANSCRIPT_FINAL,
        SessionEvent.FIRST_SENTENCE_READY,
        SessionEvent.TOOL_CALL_REQUESTED,
        SessionEvent.ERROR_OR_TIMEOUT,
        SessionEvent.PLAYBACK_DONE,
        SessionEvent.INTERRUPT,
    ]
    for event in illegal_events:
        session = Session(session_id="s_list")
        session.transition(SessionEvent.SESSION_START)
        with pytest.raises(InvalidStateTransitionError):
            session.transition(event)


def test_illegal_transitions_from_thinking() -> None:
    """Verify illegal transitions from THINKING state."""
    illegal_events = [
        SessionEvent.SESSION_START,
        SessionEvent.SPEECH_START,
        SessionEvent.SPEECH_END_VALID,
        SessionEvent.SPEECH_END_NOISE,
        SessionEvent.PLAYBACK_DONE,
        SessionEvent.INTERRUPT,
    ]
    for event in illegal_events:
        session = Session(session_id="s_think")
        session.transition(SessionEvent.SESSION_START)
        session.transition(SessionEvent.SPEECH_END_VALID)
        with pytest.raises(InvalidStateTransitionError):
            session.transition(event)


def test_illegal_transitions_from_speaking() -> None:
    """Verify illegal transitions from SPEAKING state."""
    illegal_events = [
        SessionEvent.SESSION_START,
        SessionEvent.SPEECH_START,
        SessionEvent.SPEECH_END_VALID,
        SessionEvent.SPEECH_END_NOISE,
        SessionEvent.TRANSCRIPT_FINAL,
        SessionEvent.FIRST_SENTENCE_READY,
        SessionEvent.TOOL_CALL_REQUESTED,
        SessionEvent.ERROR_OR_TIMEOUT,
    ]
    for event in illegal_events:
        session = Session(session_id="s_speak")
        session.transition(SessionEvent.SESSION_START)
        session.transition(SessionEvent.SPEECH_END_VALID)
        session.transition(SessionEvent.FIRST_SENTENCE_READY)
        with pytest.raises(InvalidStateTransitionError):
            session.transition(event)

