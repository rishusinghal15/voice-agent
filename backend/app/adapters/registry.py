"""Adapter registry for VAD, STT, LLM, and TTS providers.

Follows design doc §3 & §10:
- Selects adapters based on VAD_PROVIDER, STT_PROVIDER, LLM_PROVIDER, TTS_PROVIDER.
- Unknown provider raises a clear UnknownProviderError at startup.
- Adding a new provider requires only a new adapter file and a registry entry.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from backend.app.adapters.llm_mock import MockLLMAdapter
from backend.app.adapters.protocols import (
    LLMAdapter,
    STTAdapter,
    TTSAdapter,
    VADProcessor,
)
from backend.app.adapters.stt_mock import MockSTTAdapter
from backend.app.adapters.tts_mock import MockTTSAdapter
from backend.app.adapters.vad_mock import MockVADProcessor
from backend.app.config import settings


class UnknownProviderError(ValueError):
    """Raised when an unconfigured or unknown adapter provider is requested."""

    def __init__(self, provider_type: str, provider_name: str, available: List[str]) -> None:
        super().__init__(
            f"Unknown {provider_type} provider '{provider_name}'. "
            f"Configured options: {available}"
        )
        self.provider_type = provider_type
        self.provider_name = provider_name
        self.available = available


# Internal registries mapping provider name -> factory callable
_VAD_REGISTRY: Dict[str, Callable[[], VADProcessor]] = {}
_STT_REGISTRY: Dict[str, Callable[[], STTAdapter]] = {}
_LLM_REGISTRY: Dict[str, Callable[[], LLMAdapter]] = {}
_TTS_REGISTRY: Dict[str, Callable[[], TTSAdapter]] = {}


def register_vad(name: str, factory: Callable[[], VADProcessor]) -> None:
    """Register a VAD processor provider."""
    _VAD_REGISTRY[name.strip().lower()] = factory


def register_stt(name: str, factory: Callable[[], STTAdapter]) -> None:
    """Register an STT adapter provider."""
    _STT_REGISTRY[name.strip().lower()] = factory


def register_llm(name: str, factory: Callable[[], LLMAdapter]) -> None:
    """Register an LLM adapter provider."""
    _LLM_REGISTRY[name.strip().lower()] = factory


def register_tts(name: str, factory: Callable[[], TTSAdapter]) -> None:
    """Register a TTS adapter provider."""
    _TTS_REGISTRY[name.strip().lower()] = factory


def get_vad_adapter(name: Optional[str] = None) -> VADProcessor:
    """Retrieve an initialized VAD processor by name or from settings."""
    provider = (name or settings.VAD_PROVIDER).strip().lower()
    if provider not in _VAD_REGISTRY:
        raise UnknownProviderError("VAD", provider, sorted(_VAD_REGISTRY.keys()))
    return _VAD_REGISTRY[provider]()


def get_stt_adapter(name: Optional[str] = None) -> STTAdapter:
    """Retrieve an initialized STT adapter by name or from settings."""
    provider = (name or settings.STT_PROVIDER).strip().lower()
    if provider not in _STT_REGISTRY:
        raise UnknownProviderError("STT", provider, sorted(_STT_REGISTRY.keys()))
    return _STT_REGISTRY[provider]()


def get_llm_adapter(name: Optional[str] = None) -> LLMAdapter:
    """Retrieve an initialized LLM adapter by name or from settings."""
    provider = (name or settings.LLM_PROVIDER).strip().lower()
    if provider not in _LLM_REGISTRY:
        raise UnknownProviderError("LLM", provider, sorted(_LLM_REGISTRY.keys()))
    return _LLM_REGISTRY[provider]()


def get_tts_adapter(name: Optional[str] = None) -> TTSAdapter:
    """Retrieve an initialized TTS adapter by name or from settings."""
    provider = (name or settings.TTS_PROVIDER).strip().lower()
    if provider not in _TTS_REGISTRY:
        raise UnknownProviderError("TTS", provider, sorted(_TTS_REGISTRY.keys()))
    return _TTS_REGISTRY[provider]()


def validate_providers() -> None:
    """Validate that all providers specified in current settings exist in registry."""
    get_vad_adapter()
    get_stt_adapter()
    get_llm_adapter()
    get_tts_adapter()


# Register default 'mock' implementations
register_vad("mock", lambda: MockVADProcessor())
register_stt("mock", lambda: MockSTTAdapter())
register_llm("mock", lambda: MockLLMAdapter())
register_tts("mock", lambda: MockTTSAdapter())

# Register cloud providers
from backend.app.adapters.llm_groq import GroqLLMAdapter
from backend.app.adapters.stt_deepgram import DeepgramSTTAdapter

register_llm("groq", lambda: GroqLLMAdapter())
register_stt("deepgram", lambda: DeepgramSTTAdapter())
