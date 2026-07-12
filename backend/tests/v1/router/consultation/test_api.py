"""Consultation API endpoint tests.

Tests cover:
1. POST /api/v1/sessions - Create session
2. POST /api/v1/sessions/{id}/confirm-consent - Confirm consent
3. GET /api/v1/sessions/{id}/state - Get session state
4. GET /api/v1/sessions - List sessions
5. POST /api/v1/sessions/{id}/close - Close session
6. Unauthenticated requests return 401/403
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ASGITransport, AsyncClient

from app.security.jwt import create_access_token
from tests.factories import make_consultation_state


# The consultation router has prefix="/sessions" and is included
# in main.py with prefix="/api/v1", so the full path is:
#   /api/v1/sessions
SESSIONS_PREFIX = "/api/v1/sessions"


@pytest.fixture
def client_auth_headers():
    """Auth headers for a client user."""
    token = create_access_token(user_id="test-user-001", role="client")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def lawyer_auth_headers():
    """Auth headers for a lawyer user."""
    token = create_access_token(user_id="lawyer-001", role="lawyer")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers():
    """Auth headers for an admin user."""
    token = create_access_token(user_id="admin-001", role="admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_session_state():
    """A session state stored in the orchestrator."""
    return make_consultation_state(
        session_id="sess-001",
        user_id="test-user-001",
        consent_given=False,
        facts_raw=["用户陈述"],
        current_agent="Receptionist",
        conversation_history=[],
    )


# ---------------------------------------------------------------------------
# POST /sessions - Create session
# ---------------------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_session_success(self, test_app, client_auth_headers, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            with patch("app.v1.service.consultation_service.create_consultation_record", new_callable=AsyncMock) as mock_create, \
                 patch("app.v1.service.consultation_service.start_session", new_callable=AsyncMock) as mock_start, \
                 patch("app.v1.service.consultation_service.generate_welcome_message", new_callable=AsyncMock) as mock_welcome, \
                 patch("app.v1.service.consultation_service.persist_state", new_callable=AsyncMock) as mock_persist:
                mock_create.return_value = "consult-001"
                mock_welcome.return_value = "欢迎语"
                mock_start.return_value = make_consultation_state(
                    session_id="placeholder",
                    current_agent="Receptionist",
                )

                response = await client.post(
                    f"{SESSIONS_PREFIX}",
                    json={"client_id": "test-user-001", "user_type": "suspect"},
                    headers=client_auth_headers,
                )

                assert response.status_code == 200
                data = response.json()
                assert "session_id" in data
                assert data["current_agent"] == "Receptionist"

    @pytest.mark.asyncio
    async def test_create_session_unauthenticated(self, test_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{SESSIONS_PREFIX}",
                json={"client_id": "test-user-001", "user_type": "suspect"},
            )
            assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# POST /sessions/{id}/confirm-consent - Confirm consent
# ---------------------------------------------------------------------------


class TestConfirmConsent:
    @pytest.mark.asyncio
    async def test_confirm_consent_accepted(self, test_app, client_auth_headers, sample_session_state, mock_db_session):
        # Mock the DB execute for consultation lookup
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        ))

        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state, \
             patch("app.v1.service.consultation_service.process_consent", new_callable=AsyncMock) as mock_process, \
             patch("app.v1.service.consultation_service.persist_state", new_callable=AsyncMock):
            mock_get_state.return_value = sample_session_state
            mock_process.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/confirm-consent",
                    json={
                        "session_id": "sess-001",
                        "consent_given": True,
                        "consent_timestamp": "2026-01-01T00:00:00Z",
                        "consent_version": "1.0",
                    },
                    headers=client_auth_headers,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    @pytest.mark.asyncio
    async def test_confirm_consent_session_not_found(self, test_app, client_auth_headers):
        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = None

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/nonexistent/confirm-consent",
                    json={
                        "session_id": "nonexistent",
                        "consent_given": True,
                        "consent_timestamp": "2026-01-01T00:00:00Z",
                        "consent_version": "1.0",
                    },
                    headers=client_auth_headers,
                )

                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_confirm_consent_unauthenticated(self, test_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{SESSIONS_PREFIX}/sess-001/confirm-consent",
                json={
                    "session_id": "sess-001",
                    "consent_given": True,
                    "consent_timestamp": "2026-01-01T00:00:00Z",
                    "consent_version": "1.0",
                },
            )
            assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# GET /sessions/{id}/state - Get session state
# ---------------------------------------------------------------------------


class TestGetSessionState:
    @pytest.mark.asyncio
    async def test_get_session_state_success(self, test_app, client_auth_headers, sample_session_state):
        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}/sess-001/state",
                    headers=client_auth_headers,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["session_id"] == "sess-001"

    @pytest.mark.asyncio
    async def test_get_session_state_not_found(self, test_app, client_auth_headers):
        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = None

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}/nonexistent/state",
                    headers=client_auth_headers,
                )

                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_state_unauthenticated(self, test_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get(
                f"{SESSIONS_PREFIX}/sess-001/state",
            )
            assert response.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_get_session_state_wrong_user_forbidden(self, test_app, sample_session_state):
        """A different client user cannot access another user's session."""
        other_token = create_access_token(user_id="other-user-999", role="client")
        other_headers = {"Authorization": f"Bearer {other_token}"}

        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}/sess-001/state",
                    headers=other_headers,
                )

                assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_session_state_lawyer_can_access(self, test_app, lawyer_auth_headers, sample_session_state):
        """A lawyer can access any session state."""
        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}/sess-001/state",
                    headers=lawyer_auth_headers,
                )

                assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /sessions - List sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_success(self, test_app, client_auth_headers, sample_session_state):
        with patch("app.v1.router.consultation.routes.orchestrator") as mock_orch:
            mock_orch.get_active_sessions.return_value = {
                "sess-001": sample_session_state,
            }

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}",
                    headers=client_auth_headers,
                )

                assert response.status_code == 200
                data = response.json()
                assert "sessions" in data
                assert data["total"] >= 0

    @pytest.mark.asyncio
    async def test_list_sessions_filters_by_client(self, test_app, client_auth_headers, sample_session_state):
        """Client users only see their own sessions."""
        other_session = make_consultation_state(
            session_id="sess-002",
            user_id="other-user-999",
            facts_raw=["陈述"],
        )

        with patch("app.v1.router.consultation.routes.orchestrator") as mock_orch:
            mock_orch.get_active_sessions.return_value = {
                "sess-001": sample_session_state,
                "sess-002": other_session,
            }

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}",
                    headers=client_auth_headers,
                )

                assert response.status_code == 200
                data = response.json()
                session_ids = [s["session_id"] for s in data["sessions"]]
                assert "sess-001" in session_ids
                assert "sess-002" not in session_ids

    @pytest.mark.asyncio
    async def test_list_sessions_unauthenticated(self, test_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get(f"{SESSIONS_PREFIX}")
            assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# POST /sessions/{id}/close - Close session
# ---------------------------------------------------------------------------


class TestCloseSession:
    @pytest.mark.asyncio
    async def test_close_session_success(self, test_app, client_auth_headers, sample_session_state):
        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state, \
             patch("app.v1.service.consultation_service.persist_state", new_callable=AsyncMock):
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/close",
                    json={"reason": "用户主动结束"},
                    headers=client_auth_headers,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    @pytest.mark.asyncio
    async def test_close_session_not_found(self, test_app, client_auth_headers):
        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = None

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/nonexistent/close",
                    json={"reason": "测试"},
                    headers=client_auth_headers,
                )

                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_close_session_wrong_user_forbidden(self, test_app, sample_session_state):
        other_token = create_access_token(user_id="other-user-999", role="client")
        other_headers = {"Authorization": f"Bearer {other_token}"}

        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/close",
                    json={"reason": "测试"},
                    headers=other_headers,
                )

                assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_close_session_lawyer_can_close(self, test_app, lawyer_auth_headers, sample_session_state):
        with patch("app.v1.service.consultation_service.get_session_state", new_callable=AsyncMock) as mock_get_state, \
             patch("app.v1.service.consultation_service.persist_state", new_callable=AsyncMock):
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/close",
                    json={"reason": "律师关闭"},
                    headers=lawyer_auth_headers,
                )

                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_close_session_unauthenticated(self, test_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{SESSIONS_PREFIX}/sess-001/close",
                json={"reason": "测试"},
            )
            assert response.status_code in (401, 403)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
