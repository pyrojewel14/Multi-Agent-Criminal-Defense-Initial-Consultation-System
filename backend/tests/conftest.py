"""Shared pytest fixtures for the backend test suite."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.security.jwt import create_access_token

# ``main`` is imported eagerly at module load time so that the FastAPI app
# (and the heavy import chain it pulls in, including chromadb) is
# initialised exactly once for the test session. Doing it lazily inside the
# ``test_app`` fixture caused a macOS-specific
# ``ImportError: cannot load module more than once per process`` error from
# numpy when ``consultation_service`` was also imported by other test files.
from main import app as _fastapi_app  # noqa: F401


# ---------------------------------------------------------------------------
# LLM Gateway mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_gateway():
    """Patch the module-level ``llm_gateway`` singleton so no real LLM calls are made."""
    with patch("app.utils.llm_gateway.llm_gateway") as mock:
        mock.generate = AsyncMock(return_value="mocked LLM response")
        mock.generate_with_tools = AsyncMock(
            return_value={
                "content": "mocked LLM response",
                "tool_calls": [],
                "has_tool_call": False,
            }
        )
        yield mock


# ---------------------------------------------------------------------------
# Sample consultation state
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    """Return a standard ``ConsultationState`` dict with sensible defaults."""
    return {
        "consultation_id": "consult-001",
        "user_id": "user-001",
        "session_id": "session-001",
        "user_type": "suspect",
        "consent_given": False,
        "facts_raw": [""],
        "facts_structured": {},
        "applied_laws": [],
        "current_agent": "Receptionist",
        "pending_questions": [],
        "alert_triggered": False,
        "risk_assessment": None,
        "lawyer_review_needed": False,
        "final_output": "",
        "conversation_history": [],
        "report_draft": None,
        "service_plan": None,
        "lawyer_id": None,
        "current_input": None,
        "facts_coverage_rate": None,
        "element_to_law_mapping": None,
        "identity_info": None,
        "user_role": None,
        "awaiting_lawyer_review": None,
        "lawyer_decision": None,
        "lawyer_feedback": None,
    }


# ---------------------------------------------------------------------------
# Database session mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_session():
    """AsyncMock standing in for ``sqlalchemy.ext.asyncio.AsyncSession``."""
    session = AsyncMock(spec=["commit", "rollback", "close", "execute", "scalars", "get", "add", "refresh", "delete"])
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Redis cache mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis(monkeypatch):
    """Dict-based mock for the Redis cache helpers.

    - ``get_redis_cache_json`` returns ``None`` by default (cache miss).
    - ``set_redis_cache`` is a no-op that records calls.
    """
    _store: dict = {}

    async def _get_json(key: str):
        return _store.get(key)

    async def _set_cache(key: str, value, expire: int = 3600):
        _store[key] = value
        return True

    async def _get_str(key: str):
        return _store.get(key)

    monkeypatch.setattr("app.db.redis_config.get_redis_cache_json", _get_json)
    monkeypatch.setattr("app.db.redis_config.set_redis_cache", _set_cache)
    monkeypatch.setattr("app.db.redis_config.get_redis_cache_str", _get_str)

    return _store


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_headers():
    """Return dict with a valid JWT ``Authorization`` header for a client user."""
    token = create_access_token(user_id="test-user-001", role="client")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# FastAPI test application
# ---------------------------------------------------------------------------

@pytest.fixture
async def test_app(mock_llm_gateway, mock_redis, mock_db_session):
    """Create a FastAPI app instance for integration testing.

    The lifespan (DB / Redis init) is bypassed and key dependencies are
    overridden so tests run without external services.
    """
    from app.db.db_config import get_db

    # Override the DB dependency so every request gets the mock session.
    async def _override_get_db():
        yield mock_db_session

    _fastapi_app.dependency_overrides[get_db] = _override_get_db

    yield _fastapi_app

    # Clean up overrides after the test.
    _fastapi_app.dependency_overrides.clear()
