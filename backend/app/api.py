"""REST API endpoints for session management and health checks.

Follows design doc §6:
- POST /sessions
- GET /sessions/{id}
- GET /healthz
"""

import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app import db
from backend.app.config import get_providers_line

router = APIRouter()


class CreateSessionResponse(BaseModel):
    """Response payload for session creation."""
    session_id: str


class HealthzResponse(BaseModel):
    """Response payload for health check."""
    ok: bool = True
    providers: str


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new call session",
)
async def create_session_endpoint() -> CreateSessionResponse:
    """Create a new session record and return its UUID."""
    session_id = str(uuid.uuid4())
    await db.create_session(session_id=session_id)
    return CreateSessionResponse(session_id=session_id)


@router.get(
    "/sessions/{session_id}",
    response_model=Dict[str, Any],
    summary="Fetch saved session, turns, metrics, and errors",
)
async def get_session_endpoint(session_id: str) -> Dict[str, Any]:
    """Retrieve full session record including transcript turns, metrics, and errors."""
    session_data = await db.get_session_details(session_id)
    if session_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    return session_data


@router.get(
    "/healthz",
    response_model=HealthzResponse,
    summary="Health check endpoint",
)
async def healthz_endpoint() -> HealthzResponse:
    """Return health status and active providers."""
    return HealthzResponse(ok=True, providers=get_providers_line())

