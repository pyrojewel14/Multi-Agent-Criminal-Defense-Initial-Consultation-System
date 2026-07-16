"""Consultation history router endpoint tests.

Tests cover:
1. GET /api/v1/consultations/list - List consultations
2. GET /api/v1/consultations/{id} - Get consultation
3. GET /api/v1/consultations/{id}/messages - Get consultation messages
4. POST /api/v1/consultations/assign - Assign lawyer
5. PUT /api/v1/consultations/{id}/status - Update consultation status
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


CONSULTATION_PREFIX = "/api/v1/consultations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_consultation(**overrides):
    """Build a SimpleNamespace standing in for a Consultation row."""
    c = SimpleNamespace(
        id="consult-001",
        client_id="client-001",
        assigned_lawyer_id="lawyer-001",
        user_type="suspect",
        consent_given=True,
        status=SimpleNamespace(value="in_progress"),
        facts_structured='{"key": "value"}',
        applied_laws='[{"article": "264"}]',
        final_output=None,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 2, 12, 0, 0),
        completed_at=None,
    )
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def _make_user(**overrides):
    user = SimpleNamespace(
        id="user-001",
        username="testuser",
        real_name="测试用户",
    )
    for k, v in overrides.items():
        setattr(user, k, v)
    return user


def _make_message(**overrides):
    msg = SimpleNamespace(
        id="msg-001",
        consultation_id="consult-001",
        sender_type="user",
        sender_id="client-001",
        content="hello",
        agent_name=None,
        message_type="text",
        created_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    for k, v in overrides.items():
        setattr(msg, k, v)
    return msg


def _setup_db(mock_db_session, *, scalars_returns=None, scalar_one=None, scalar_one_or_none=None, n_calls=10):
    if scalars_returns is not None:
        scalars_returns = list(scalars_returns) + [None] * (n_calls - len(scalars_returns))
    if scalar_one is not None and isinstance(scalar_one, list):
        scalar_one = list(scalar_one) + [None] * (n_calls - len(scalar_one))
    if scalar_one_or_none is not None and isinstance(scalar_one_or_none, list):
        scalar_one_or_none = list(scalar_one_or_none) + [None] * (n_calls - len(scalar_one_or_none))

    execute_call = {"i": 0}

    async def _execute(*args, **kwargs):
        idx = execute_call["i"]
        execute_call["i"] += 1
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        if scalars_returns is not None and idx < len(scalars_returns):
            data = scalars_returns[idx]
            if data is not None:
                result.scalars.return_value.all.return_value = data
        if scalar_one is not None and idx < len(scalar_one):
            val = scalar_one[idx]
            if val is not None:
                result.scalar_one.return_value = val
        if scalar_one_or_none is not None and idx < len(scalar_one_or_none):
            val = scalar_one_or_none[idx]
            if val is not None:
                result.scalar_one_or_none.return_value = val
        return result

    mock_db_session.execute = AsyncMock(side_effect=_execute)
    return execute_call


@pytest.fixture
async def admin_app(test_app, monkeypatch):
    """Test app with admin role override."""
    import app.security.rbac as rbac_module
    import app.v1.router.consultation_history as history_module

    async def _admin_override():
        return {"user_id": "admin-001", "role": "admin"}

    async def _lawyer_override():
        return {"user_id": "lawyer-001", "role": "lawyer"}

    async def _client_override():
        return {"user_id": "client-001", "role": "client"}

    monkeypatch.setattr(history_module, "require_admin", _admin_override, raising=False)
    monkeypatch.setattr(history_module, "require_lawyer", _lawyer_override, raising=False)
    monkeypatch.setattr(history_module, "get_current_user", _client_override, raising=False)

    test_app.dependency_overrides[rbac_module.require_admin] = _admin_override
    test_app.dependency_overrides[rbac_module.require_lawyer] = _lawyer_override
    test_app.dependency_overrides[rbac_module.get_current_user] = _client_override

    yield test_app


# ---------------------------------------------------------------------------
# GET /consultations/list
# ---------------------------------------------------------------------------


class TestListConsultations:
    @pytest.mark.asyncio
    async def test_list_as_admin(self, admin_app, mock_db_session):
        """Admin sees all consultations."""
        c1 = _make_consultation()
        # 0) count -> scalar_one, 1) main -> scalars, 2) client -> scalar_one_or_none
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [c1], None],
            scalar_one=[1, None, None],
            scalar_one_or_none=[None, None, _make_user()],
        )

        # Override get_current_user to admin
        import app.security.rbac as rbac_module
        async def _admin_user():
            return {"user_id": "admin-001", "role": "admin"}
        admin_app.dependency_overrides[rbac_module.get_current_user] = _admin_user

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{CONSULTATION_PREFIX}/list")

            assert response.status_code == 200
            data = response.json()
            assert data["data"]["total"] == 1
            assert len(data["data"]["consultations"]) == 1

    @pytest.mark.asyncio
    async def test_list_as_lawyer(self, admin_app, mock_db_session):
        """Lawyer sees only assigned consultations."""
        c1 = _make_consultation(assigned_lawyer_id="lawyer-001")
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [c1], None],
            scalar_one=[1, None, None],
            scalar_one_or_none=[None, None, _make_user()],
        )

        import app.security.rbac as rbac_module
        async def _lawyer_user():
            return {"user_id": "lawyer-001", "role": "lawyer"}
        admin_app.dependency_overrides[rbac_module.get_current_user] = _lawyer_user

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{CONSULTATION_PREFIX}/list")

            assert response.status_code == 200
            assert response.json()["data"]["total"] == 1

    @pytest.mark.asyncio
    async def test_list_as_client(self, admin_app, mock_db_session):
        """Client sees only their own consultations."""
        c1 = _make_consultation(client_id="client-001")
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [c1], None],
            scalar_one=[1, None, None],
            scalar_one_or_none=[None, None, _make_user()],
        )

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{CONSULTATION_PREFIX}/list")

            assert response.status_code == 200
            assert response.json()["data"]["total"] == 1

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, admin_app, mock_db_session):
        c1 = _make_consultation()
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [c1], None],
            scalar_one=[1, None, None],
            scalar_one_or_none=[None, None, _make_user()],
        )

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{CONSULTATION_PREFIX}/list?status=in_progress"
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, admin_app, mock_db_session):
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [], None],
            scalar_one=[0, None, None],
            scalar_one_or_none=[None, None, None],
        )

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{CONSULTATION_PREFIX}/list?page=2&page_size=10"
            )

            assert response.status_code == 200
            assert response.json()["data"]["page"] == 2


# ---------------------------------------------------------------------------
# GET /consultations/{id}
# ---------------------------------------------------------------------------


class TestGetConsultation:
    @pytest.mark.asyncio
    async def test_get_as_client_owner(self, admin_app, mock_db_session):
        c = _make_consultation(client_id="client-001")
        # 0) consultation, 1) client, 2) lawyer (if assigned)
        _setup_db(
            mock_db_session,
            scalar_one_or_none=[c, _make_user(), _make_user(id="lawyer-001")],
        )

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{CONSULTATION_PREFIX}/consult-001")

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["id"] == "consult-001"

    @pytest.mark.asyncio
    async def test_get_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{CONSULTATION_PREFIX}/missing")
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_as_client_not_owner(self, admin_app, mock_db_session):
        c = _make_consultation(client_id="other-client")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{CONSULTATION_PREFIX}/consult-001")
            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_as_lawyer_not_assigned(self, admin_app, mock_db_session):
        c = _make_consultation(assigned_lawyer_id="other-lawyer")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        import app.security.rbac as rbac_module
        async def _lawyer_user():
            return {"user_id": "lawyer-001", "role": "lawyer"}
        admin_app.dependency_overrides[rbac_module.get_current_user] = _lawyer_user

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{CONSULTATION_PREFIX}/consult-001")
            assert response.status_code == 403


# ---------------------------------------------------------------------------
# GET /consultations/{id}/messages
# ---------------------------------------------------------------------------


class TestGetConsultationMessages:
    @pytest.mark.asyncio
    async def test_get_messages_success(self, admin_app, mock_db_session):
        c = _make_consultation(client_id="client-001")
        m1 = _make_message()
        # 0) consultation, 1) messages query
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [m1]],
            scalar_one_or_none=[c, None],
        )

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{CONSULTATION_PREFIX}/consult-001/messages"
            )

            assert response.status_code == 200
            assert len(response.json()["data"]) == 1

    @pytest.mark.asyncio
    async def test_get_messages_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{CONSULTATION_PREFIX}/missing/messages"
            )
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_messages_forbidden(self, admin_app, mock_db_session):
        c = _make_consultation(client_id="other-client")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{CONSULTATION_PREFIX}/consult-001/messages"
            )
            assert response.status_code == 403


# ---------------------------------------------------------------------------
# POST /consultations/assign
# ---------------------------------------------------------------------------


class TestAssignLawyer:
    @pytest.mark.asyncio
    async def test_assign_lawyer_success(self, admin_app, mock_db_session):
        lawyer = _make_user(id="lawyer-001", real_name="王律师")
        c = _make_consultation()
        # 0) lawyer, 1) consultation
        _setup_db(mock_db_session, scalar_one_or_none=[lawyer, c])
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{CONSULTATION_PREFIX}/assign",
                json={"consultation_id": "consult-001", "lawyer_id": "lawyer-001"},
            )

            assert response.status_code == 200
            assert c.assigned_lawyer_id == "lawyer-001"
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_assign_lawyer_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{CONSULTATION_PREFIX}/assign",
                json={"consultation_id": "consult-001", "lawyer_id": "missing"},
            )
            assert response.status_code == 404
            assert "律师" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_assign_consultation_not_found(self, admin_app, mock_db_session):
        lawyer = _make_user(id="lawyer-001")
        # 0) lawyer found, 1) consultation not found
        _setup_db(mock_db_session, scalar_one_or_none=[lawyer, None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{CONSULTATION_PREFIX}/assign",
                json={"consultation_id": "missing", "lawyer_id": "lawyer-001"},
            )
            assert response.status_code == 404
            assert "咨询" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# PUT /consultations/{id}/status
# ---------------------------------------------------------------------------


class TestUpdateConsultationStatus:
    @pytest.mark.asyncio
    async def test_update_status_success(self, admin_app, mock_db_session):
        c = _make_consultation()
        _setup_db(mock_db_session, scalar_one_or_none=[c])
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{CONSULTATION_PREFIX}/consult-001/status",
                json={"status": "completed"},
            )

            assert response.status_code == 200
            assert c.status.value == "completed"
            assert c.completed_at is not None
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{CONSULTATION_PREFIX}/missing/status",
                json={"status": "completed"},
            )
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_status_invalid_value(self, admin_app, mock_db_session):
        c = _make_consultation()
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{CONSULTATION_PREFIX}/consult-001/status",
                json={"status": "unknown_status"},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_update_status_lawyer_not_assigned(self, admin_app, mock_db_session):
        c = _make_consultation(assigned_lawyer_id="other-lawyer")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        import app.security.rbac as rbac_module
        async def _lawyer_user():
            return {"user_id": "lawyer-001", "role": "lawyer"}
        admin_app.dependency_overrides[rbac_module.get_current_user] = _lawyer_user
        admin_app.dependency_overrides[rbac_module.require_lawyer] = _lawyer_user

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{CONSULTATION_PREFIX}/consult-001/status",
                json={"status": "completed"},
            )
            assert response.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
