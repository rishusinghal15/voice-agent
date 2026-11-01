"""Application configuration loaded from environment and .env file.

Follows design doc §10 configuration parameters and defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    """Application settings with defaults from design §10 and task specifications."""

    model_config = SettingsConfigDict(
        env_file=(str(ENV_FILE), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # System & Prompt Configuration
    SYSTEM_PROMPT: str = "You are a helpful voice assistant."
    PROMPT_VERSION: str = "v1.0"
    FALLBACK_LINE: str = "Sorry, I didn't catch that, could you repeat?"

    # VAD Configuration
    VAD_SILENCE_MS: int = 600
    VAD_MIN_SPEECH_MS: int = 250

    # Call Timeouts & Limits
    MAX_CALL_SECONDS: int = 600
    SILENCE_TIMEOUT_SECONDS: int = 120
    STT_TIMEOUT_S: float = 3.0
    LLM_TIMEOUT_S: float = 5.0
    TTS_TIMEOUT_S: float = 5.0
    TOOL_TIMEOUT_S: float = 5.0

    # Provider Selection (defaults to mock per Task P2)
    VAD_PROVIDER: str = "mock"
    STT_PROVIDER: str = "mock"
    LLM_PROVIDER: str = "mock"
    TTS_PROVIDER: str = "mock"

    # Per-provider API Keys & Models
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    DEEPGRAM_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    CARTESIA_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None

    # Model Paths
    SILERO_MODEL_PATH: str = "models/silero_vad.onnx"
    VOSK_MODEL_PATH: str = "models/vosk-model-small-en-us-0.15"
    PIPER_MODEL_PATH: str = "models/en_US-lessac-medium.onnx"

    # Persistence & Server Configuration
    DB_PATH: str = "voice_agent.db"
    HOST: str = "127.0.0.1"
    PORT: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()


settings = get_settings()


def get_providers_line() -> str:
    """Format single-line description of active providers."""
    return (
        f"Providers: vad={settings.VAD_PROVIDER} "
        f"stt={settings.STT_PROVIDER} "
        f"llm={settings.LLM_PROVIDER} "
        f"tts={settings.TTS_PROVIDER} "
        f"model={settings.GROQ_MODEL}"
    )
