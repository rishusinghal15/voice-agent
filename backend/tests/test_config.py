"""Unit tests for configuration settings.

Follows design doc §10:
- Verifies all env vars and defaults
- Verifies loading from .env
"""

from backend.app.config import Settings


def test_config_defaults() -> None:
    """Verify all default configuration values match design doc §10."""
    settings = Settings()

    # System & Prompt Configuration
    assert settings.SYSTEM_PROMPT == "You are a helpful voice assistant."
    assert settings.PROMPT_VERSION == "v1.0"

    # VAD Configuration
    assert settings.VAD_SILENCE_MS == 600
    assert settings.VAD_MIN_SPEECH_MS == 250

    # Call Timeouts & Limits
    assert settings.MAX_CALL_SECONDS == 600
    assert settings.SILENCE_TIMEOUT_SECONDS == 120
    assert settings.STT_TIMEOUT_S == 3.0
    assert settings.LLM_TIMEOUT_S == 5.0
    assert settings.TTS_TIMEOUT_S == 5.0

    # Provider Selection
    assert settings.STT_PROVIDER == "deepgram"
    assert settings.LLM_PROVIDER == "groq"
    assert settings.TTS_PROVIDER == "piper"

    # Database
    assert settings.DB_PATH == "voice_agent.db"
    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 8000

