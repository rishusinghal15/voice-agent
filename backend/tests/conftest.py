"""Pytest configuration and global fixtures.

Ensures the test suite runs with mock providers by default so tests
stay fast, deterministic, and offline per Task P4 instructions.
"""

import os
import pytest

# Set environment variables early before settings or modules load
os.environ["VAD_PROVIDER"] = "mock"
os.environ["STT_PROVIDER"] = "mock"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["TTS_PROVIDER"] = "mock"

from backend.app.config import settings

settings.VAD_PROVIDER = "mock"
settings.STT_PROVIDER = "mock"
settings.LLM_PROVIDER = "mock"
settings.TTS_PROVIDER = "mock"


@pytest.fixture(autouse=True)
def reset_mock_providers(monkeypatch: pytest.MonkeyPatch):
    """Ensure every test executes with mock providers unless explicitly overridden."""
    monkeypatch.setenv("VAD_PROVIDER", "mock")
    monkeypatch.setenv("STT_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("TTS_PROVIDER", "mock")
    monkeypatch.setattr(settings, "VAD_PROVIDER", "mock")
    monkeypatch.setattr(settings, "STT_PROVIDER", "mock")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "TTS_PROVIDER", "mock")

