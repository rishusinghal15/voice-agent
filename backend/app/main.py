"""Main application entry point for Voice Agent MVP.

Configures FastAPI, lifespan hooks for database initialization, CORS,
API routers, WebSocket endpoint, and static file serving for the frontend.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.adapters.registry import validate_providers
from backend.app.api import router as api_router
from backend.app.config import get_providers_line, settings
from backend.app.db import init_db
from backend.app.ws import router as ws_router

logger = logging.getLogger("voice_agent")
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


def log_startup_providers() -> None:
    """Log active providers and print a clear warning if any provider is mock."""
    line = get_providers_line()
    logger.info(line)
    print(line, flush=True)

    mock_providers = [
        p
        for p, val in [
            ("vad", settings.VAD_PROVIDER),
            ("stt", settings.STT_PROVIDER),
            ("llm", settings.LLM_PROVIDER),
            ("tts", settings.TTS_PROVIDER),
        ]
        if val == "mock"
    ]
    if mock_providers:
        warning_msg = (
            f"WARNING: Active provider(s) using 'mock': {', '.join(mock_providers)}. "
            "Offline/simulated responses will be used!"
        )
        logger.warning(warning_msg)
        print(warning_msg, flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager for startup and shutdown hooks."""
    # Validate adapter providers configured in settings
    validate_providers()
    # Log active providers and mock warnings
    log_startup_providers()
    # Initialize SQLite database tables on startup
    await init_db(settings.DB_PATH)
    # Warm up Vosk STT model in worker thread if active
    if settings.STT_PROVIDER.lower() == "vosk":
        import asyncio
        from backend.app.adapters.stt_vosk import warmup_vosk_model
        await asyncio.to_thread(warmup_vosk_model)
    # Warm up persistent Groq client if active
    if settings.LLM_PROVIDER.lower() == "groq":
        from backend.app.adapters.llm_groq import get_groq_client
        get_groq_client()
    yield
    # Shutdown hooks
    from backend.app.adapters.llm_groq import close_groq_client
    await close_groq_client()


app = FastAPI(
    title="Voice Agent MVP",
    description="Real-time voice agent pipeline with WebSockets and FastAPI",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for localhost and browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST and WebSocket routers first
app.include_router(api_router)
app.include_router(ws_router)

# Mount frontend directory for static serving
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

