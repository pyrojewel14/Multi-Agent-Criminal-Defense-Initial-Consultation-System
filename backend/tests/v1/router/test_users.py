"""User management router endpoint tests.

Tests cover:
1. GET /api/v1/users/ - List users
2. GET /api/v1/users/{id} - Get user
3. PUT /api/v1/users/{id} - Update user
4. PUT /api/v1/users/{id}/role - Update user role
5. DELETE /api/v1/users/{id} - Delete user
"""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


USERS_PREFIX = "/api/v1/users"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(**overrides):
    """Build a SimpleNamespace standing in for a User row."""
    user = SimpleNamespace(
        id=str(uuid.uuid4()),
        username="testuser",
        email="test@example.com",
        phone="13800138000",
        real_name="测试用户",
        role="client",
        is_active=True,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 2, 12, 0, 0),
        last_login_at=None,
        password_hash="hashed",
    )
    for k, v in overrides.items():
        setattr(user, k, v)
    return user


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
    """Test app with ``require_admin`` overridden to a fixed admin user."""
    import app.security.rbac as rbac_module
    import app.v1.router.users as users_module

    async def _admin_override():
        return {"user_id": "admin-001", "role": "admin"}

    monkeypatch.setattr(users_module, "require_admin", _admin_override, raising=False)
    test_app.dependency_overrides[rbac_module.require_admin] = _admin_override

    yield test_app


# ---------------------------------------------------------------------------
# GET /users/
# ---------------------------------------------------------------------------


class TestListUsers:
    @pytest.mark.asyncio
    async def test_list_users_success(self, admin_app, mock_db_session):
        u1 = _make_user(username="user1")
        u2 = _make_user(username="user2")
        # 0) count query -> scalars, 1) main query -> scalars
        _setup_db(mock_db_session, scalars_returns=[[u1, u2], [u1, u2]])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{USERS_PREFIX}/")

            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 200
            assert data["data"]["total"] == 2
            assert len(data["data"]["users"]) == 2

    @pytest.mark.asyncio
    async def test_list_users_with_role_filter(self, admin_app, mock_db_session):
        lawyer = _make_user(role="lawyer")
        _setup_db(mock_db_session, scalars_returns=[[lawyer], [lawyer]])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{USERS_PREFIX}/?role=lawyer")

            assert response.status_code == 200
            assert response.json()["data"]["total"] == 1

    @pytest.mark.asyncio
    async def test_list_users_pagination(self, admin_app, mock_db_session):
        u1 = _make_user(username="user1")
        _setup_db(mock_db_session, scalars_returns=[[u1], [u1]])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{USERS_PREFIX}/?skip=0&limit=10")

            assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /users/{id}
# ---------------------------------------------------------------------------


class TestGetUser:
    @pytest.mark.asyncio
    async def test_get_user_success(self, admin_app, mock_db_session):
        u = _make_user()
        _setup_db(mock_db_session, scalar_one_or_none=[u])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{USERS_PREFIX}/{u.id}")

            assert response.status_code == 200
            data = response.json()
            assert data["data"]["username"] == "testuser"

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{USERS_PREFIX}/missing")
            assert response.status_code == 404


# ---------------------------------------------------------------------------
# PUT /users/{id}
# ---------------------------------------------------------------------------


class TestUpdateUser:
    @pytest.mark.asyncio
    async def test_update_user_success(self, admin_app, mock_db_session):
        u = _make_user()
        # 0) user lookup, 1) email check (returns None)
        _setup_db(mock_db_session, scalar_one_or_none=[u, None])

        async def _refresh(obj):
            pass
        mock_db_session.refresh = AsyncMock(side_effect=_refresh)
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{USERS_PREFIX}/{u.id}",
                json={"real_name": "新名字", "phone": "13900000000"},
            )

            assert response.status_code == 200
            assert u.real_name == "新名字"
            assert u.phone == "13900000000"
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_user_with_email_conflict(self, admin_app, mock_db_session):
        u = _make_user()
        # 0) user lookup, 1) email lookup returns conflict user
        conflict = _make_user(email="conflict@example.com")
        _setup_db(mock_db_session, scalar_one_or_none=[u, conflict])

        async def _refresh(obj):
            pass
        mock_db_session.refresh = AsyncMock(side_effect=_refresh)

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{USERS_PREFIX}/{u.id}",
                json={"email": "conflict@example.com"},
            )
            assert response.status_code == 400
            assert "邮箱" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_update_user_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{USERS_PREFIX}/missing",
                json={"real_name": "X"},
            )
            assert response.status_code == 404


# ---------------------------------------------------------------------------
# PUT /users/{id}/role
# ---------------------------------------------------------------------------


class TestUpdateUserRole:
    @pytest.mark.asyncio
    async def test_update_user_role_success(self, admin_app, mock_db_session):
        u = _make_user(role="client")
        _setup_db(mock_db_session, scalar_one_or_none=[u])

        async def _refresh(obj):
            pass
        mock_db_session.refresh = AsyncMock(side_effect=_refresh)
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{USERS_PREFIX}/{u.id}/role",
                json={"role": "lawyer"},
            )

            assert response.status_code == 200
            assert u.role == "lawyer"
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_user_role_invalid(self, admin_app, mock_db_session):
        u = _make_user()
        _setup_db(mock_db_session, scalar_one_or_none=[u])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{USERS_PREFIX}/{u.id}/role",
                json={"role": "unknown_role"},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_update_user_role_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{USERS_PREFIX}/missing/role",
                json={"role": "lawyer"},
            )
            assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /users/{id}
# ---------------------------------------------------------------------------


class TestDeleteUser:
    @pytest.mark.asyncio
    async def test_delete_user_success(self, admin_app, mock_db_session):
        u = _make_user()
        _setup_db(mock_db_session, scalar_one_or_none=[u])
        mock_db_session.commit = AsyncMock()
        mock_db_session.delete = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(f"{USERS_PREFIX}/{u.id}")

            assert response.status_code == 200
            mock_db_session.delete.assert_awaited_once_with(u)
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(f"{USERS_PREFIX}/missing")
            assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
