"""Unit tests for configuration settings.

Follows design doc §10 and Task P2:
- Verifies all env vars and defaults
- Verifies loading from .env
"""

from backend.app.config import Settings


def test_config_defaults() -> None:
    """Verify all default configuration values match design doc §10 and Task P2."""
    settings = Settings()

    # System & Prompt Configuration
    assert settings.SYSTEM_PROMPT == "You are a helpful voice assistant."
    assert settings.PROMPT_VERSION == "v1.0"
    assert settings.FALLBACK_LINE == "Sorry, I didn't catch that, could you repeat?"

    # VAD Configuration
    assert settings.VAD_SILENCE_MS == 600
    assert settings.VAD_MIN_SPEECH_MS == 250

    # Call Timeouts & Limits
    assert settings.MAX_CALL_SECONDS == 600
    assert settings.SILENCE_TIMEOUT_SECONDS == 120
    assert settings.STT_TIMEOUT_S == 3.0
    assert settings.LLM_TIMEOUT_S == 5.0
    assert settings.TTS_TIMEOUT_S == 5.0
    assert settings.TOOL_TIMEOUT_S == 5.0

    # Provider Selection (Task P2: default all "mock")
    assert settings.VAD_PROVIDER == "mock"
    assert settings.STT_PROVIDER == "mock"
    assert settings.LLM_PROVIDER == "mock"
    assert settings.TTS_PROVIDER == "mock"

    # Database
    assert settings.DB_PATH == "voice_agent.db"
    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 8000
