"""Unit tests for adapter registry and provider selection."""

import pytest

from backend.app.adapters.llm_mock import MockLLMAdapter
from backend.app.adapters.protocols import (
    LLMAdapter,
    STTAdapter,
    TTSAdapter,
    VADProcessor,
)
from backend.app.adapters.registry import (
    UnknownProviderError,
    get_llm_adapter,
    get_stt_adapter,
    get_tts_adapter,
    get_vad_adapter,
    register_llm,
    register_stt,
    register_tts,
    register_vad,
    validate_providers,
)
from backend.app.adapters.stt_mock import MockSTTAdapter
from backend.app.adapters.tts_mock import MockTTSAdapter
from backend.app.adapters.vad_mock import MockVADProcessor


def test_default_mock_adapters_selection() -> None:
    """Verify that default provider settings return mock adapters."""
    vad = get_vad_adapter()
    assert isinstance(vad, MockVADProcessor)
    assert isinstance(vad, VADProcessor)

    stt = get_stt_adapter()
    assert isinstance(stt, MockSTTAdapter)
    assert isinstance(stt, STTAdapter)

    llm = get_llm_adapter()
    assert isinstance(llm, MockLLMAdapter)
    assert isinstance(llm, LLMAdapter)

    tts = get_tts_adapter()
    assert isinstance(tts, MockTTSAdapter)
    assert isinstance(tts, TTSAdapter)


def test_unknown_provider_errors() -> None:
    """Verify clear error message is raised when unknown provider is requested."""
    with pytest.raises(UnknownProviderError) as exc_vad:
        get_vad_adapter("non_existent_vad")
    assert "Unknown VAD provider 'non_existent_vad'" in str(exc_vad.value)
    assert "mock" in str(exc_vad.value)

    with pytest.raises(UnknownProviderError) as exc_stt:
        get_stt_adapter("non_existent_stt")
    assert "Unknown STT provider 'non_existent_stt'" in str(exc_stt.value)

    with pytest.raises(UnknownProviderError) as exc_llm:
        get_llm_adapter("non_existent_llm")
    assert "Unknown LLM provider 'non_existent_llm'" in str(exc_llm.value)

    with pytest.raises(UnknownProviderError) as exc_tts:
        get_tts_adapter("non_existent_tts")
    assert "Unknown TTS provider 'non_existent_tts'" in str(exc_tts.value)


def test_custom_provider_registration() -> None:
    """Verify adding a new provider requires only a new adapter + a registry entry."""
    # Register dummy custom adapters
    register_vad("custom_vad", lambda: MockVADProcessor(energy_threshold=999.0))
    vad = get_vad_adapter("custom_vad")
    assert isinstance(vad, MockVADProcessor)
    assert vad.energy_threshold == 999.0

    register_stt("custom_stt", lambda: MockSTTAdapter(canned_final="custom final"))
    stt = get_stt_adapter("custom_stt")
    assert isinstance(stt, MockSTTAdapter)
    assert stt.canned_final == "custom final"

    register_llm("custom_llm", lambda: MockLLMAdapter(canned_reply="custom reply"))
    llm = get_llm_adapter("custom_llm")
    assert isinstance(llm, MockLLMAdapter)
    assert llm.canned_reply == "custom reply"

    register_tts("custom_tts", lambda: MockTTSAdapter(frequency_hz=880.0))
    tts = get_tts_adapter("custom_tts")
    assert isinstance(tts, MockTTSAdapter)
    assert tts.frequency_hz == 880.0


def test_validate_providers() -> None:
    """Verify validate_providers validates configured providers without error."""
    validate_providers()

