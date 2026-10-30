"""Unit tests for TurnTimer latency calculations and percentile helpers."""

import pytest

from backend.app.metrics import TurnTimer, calculate_percentile, median, p90


def test_turn_timer_exact_formulas():
    """Verify TurnTimer applies exact formulas from design doc §7."""
    # Silence threshold: 600 ms (0.6 s)
    timer = TurnTimer(silence_threshold_ms=600.0)

    # Set explicit monotonic timestamps in seconds
    t0 = 100.0
    timer.mark_speech_start(t0)

    t_vad_end = t0 + 1.0  # 101.0
    timer.mark_vad_end(t_vad_end)

    t_stt_final = t_vad_end + 0.2  # 101.2 -> stt_final_ms = 200 ms
    timer.mark_stt_final(t_stt_final)

    t_llm_first_token = t_stt_final + 0.4  # 101.6 -> llm_first_token_ms = 400 ms
    timer.mark_llm_first_token(t_llm_first_token)

    t_first_sentence_ready = t_llm_first_token + 0.1  # 101.7
    timer.mark_first_sentence_ready(t_first_sentence_ready)

    t_tts_first_audio = t_first_sentence_ready + 0.25  # 101.95 -> tts_first_audio_ms = 250 ms
    timer.mark_tts_first_audio(t_tts_first_audio)

    metrics = timer.compute()

    # vad_end_ms = silence_threshold (600 ms)
    assert metrics.vad_end_ms == 600.0

    # stt_final_ms = (101.2 - 101.0) * 1000 = 200 ms
    assert metrics.stt_final_ms == 200.0

    # llm_first_token_ms = (101.6 - 101.2) * 1000 = 400 ms
    assert metrics.llm_first_token_ms == 400.0

    # tts_first_audio_ms = (101.95 - 101.7) * 1000 = 250 ms
    assert metrics.tts_first_audio_ms == 250.0

    # total_ms = (101.95 - (101.0 - 0.6)) * 1000 = (101.95 - 100.4) * 1000 = 1550 ms
    assert metrics.total_ms == 1550.0

    # total_ms_from_vad_end = (101.95 - 101.0) * 1000 = 950 ms
    assert metrics.total_ms_from_vad_end == 950.0

    # Verify to_dict
    m_dict = metrics.to_dict()
    assert m_dict["total_ms"] == 1550.0
    assert m_dict["vad_end_ms"] == 600.0


def test_turn_timer_missing_timestamps():
    """Verify TurnTimer handles missing stages gracefully without raising."""
    timer = TurnTimer(silence_threshold_ms=600.0)
    timer.mark_vad_end(100.0)
    # No other timestamps stamped
    metrics = timer.compute()
    assert metrics.vad_end_ms == 600.0
    assert metrics.stt_final_ms == 0.0
    assert metrics.llm_first_token_ms == 0.0
    assert metrics.total_ms == 0.0


def test_percentile_calculations():
    """Verify median and p90 percentile helper functions."""
    # Empty list
    assert calculate_percentile([], 50.0) == 0.0
    assert median([]) == 0.0
    assert p90([]) == 0.0

    # Single item
    assert median([100.0]) == 100.0
    assert p90([100.0]) == 100.0

    # Odd count: 10, 20, 30, 40, 50 -> median = 30
    odd_list = [50.0, 10.0, 30.0, 20.0, 40.0]
    assert median(odd_list) == 30.0

    # Even count: 10, 20, 30, 40 -> median = 25.0
    even_list = [10.0, 20.0, 30.0, 40.0]
    assert median(even_list) == 25.0

    # 10 items from 1 to 10: 90th percentile = 9.1
    ten_items = [float(i) for i in range(1, 11)]
    assert p90(ten_items) == 9.1

