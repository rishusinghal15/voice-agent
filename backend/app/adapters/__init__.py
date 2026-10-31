"""Adapters package for voice pipeline components."""

from backend.app.adapters.llm_groq import GroqLLMAdapter
from backend.app.adapters.llm_mock import MockLLMAdapter
from backend.app.adapters.protocols import (
    Done,
    LLMAdapter,
    LLMEvent,
    Message,
    STTAdapter,
    STTEvent,
    STTFinal,
    STTPartial,
    STTStream,
    TextDelta,
    Tool,
    ToolCall,
    ToolSpec,
    TTSAdapter,
    VADEvent,
    VADEventType,
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
from backend.app.adapters.stt_deepgram import DeepgramSTTAdapter
from backend.app.adapters.stt_mock import MockSTTAdapter, MockSTTStream
from backend.app.adapters.stt_vosk import VoskSTTAdapter, VoskSTTStream
from backend.app.adapters.tts_mock import MockTTSAdapter
from backend.app.adapters.tts_piper import PiperTTSAdapter
from backend.app.adapters.vad_mock import MockVADProcessor
from backend.app.adapters.vad_silero import SileroVADProcessor
from backend.app.adapters.vad_webrtc import WebRTCVADProcessor

__all__ = [
    "VADEventType",
    "VADEvent",
    "STTPartial",
    "STTFinal",
    "STTEvent",
    "Message",
    "ToolSpec",
    "TextDelta",
    "ToolCall",
    "Done",
    "LLMEvent",
    "VADProcessor",
    "STTStream",
    "STTAdapter",
    "LLMAdapter",
    "TTSAdapter",
    "Tool",
    "MockVADProcessor",
    "MockSTTStream",
    "MockSTTAdapter",
    "MockLLMAdapter",
    "MockTTSAdapter",
    "PiperTTSAdapter",
    "GroqLLMAdapter",
    "DeepgramSTTAdapter",
    "VoskSTTAdapter",
    "VoskSTTStream",
    "SileroVADProcessor",
    "WebRTCVADProcessor",
    "UnknownProviderError",
    "register_vad",
    "register_stt",
    "register_llm",
    "register_tts",
    "get_vad_adapter",
    "get_stt_adapter",
    "get_llm_adapter",
    "get_tts_adapter",
    "validate_providers",
]
