"""Main application entry point for Voice Agent MVP.

Configures FastAPI, lifespan hooks for database initialization, CORS,
API routers, WebSocket endpoint, and static file serving for the frontend.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.adapters.registry import validate_providers
from backend.app.api import router as api_router
from backend.app.config import settings
from backend.app.db import init_db
from backend.app.ws import router as ws_router

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager for startup and shutdown hooks."""
    # Validate adapter providers configured in settings
    validate_providers()
    # Initialize SQLite database tables on startup
    await init_db(settings.DB_PATH)
    yield


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

