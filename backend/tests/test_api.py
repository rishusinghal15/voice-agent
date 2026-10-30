"""Unit and integration tests for REST API endpoints.

Follows design doc §6:
- POST /sessions
- GET /sessions/{id}
- GET /healthz
"""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import db
from backend.app.config import settings
from backend.app.main import app


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path):
    """Ensure a fresh clean database for API tests."""
    test_db = str(tmp_path / "test_api.db")
    settings.DB_PATH = test_db
    asyncio.run(db.init_db(test_db))
    yield
    settings.DB_PATH = "voice_agent.db"


def test_healthz() -> None:
    """Verify GET /healthz returns ok: true."""
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_create_and_get_session() -> None:
    """Verify POST /sessions creates a session and GET /sessions/{id} retrieves it."""
    client = TestClient(app)
    # Create session
    post_res = client.post("/sessions")
    assert post_res.status_code == 201
    data = post_res.json()
    assert "session_id" in data
    session_id = data["session_id"]
    assert len(session_id) > 0

    # Retrieve session
    get_res = client.get(f"/sessions/{session_id}")
    assert get_res.status_code == 200
    details = get_res.json()
    assert details["session"]["id"] == session_id
    assert details["turns"] == []
    assert details["metrics"] == []
    assert details["errors"] == []


def test_get_session_not_found() -> None:
    """Verify GET /sessions/{id} returns 404 for missing session."""
    client = TestClient(app)
    response = client.get("/sessions/unknown-session-uuid")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
