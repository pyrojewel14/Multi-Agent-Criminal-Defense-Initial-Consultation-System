"""Lawyer management router endpoint tests.

Tests cover:
1. GET /api/v1/lawyers/ - List all lawyers
2. POST /api/v1/lawyers/ - Create new lawyer
3. PUT /api/v1/lawyers/{id} - Update lawyer
"""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


LAWYERS_PREFIX = "/api/v1/lawyers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(**overrides):
    """Build a SimpleNamespace standing in for a User row with role=lawyer."""
    user = SimpleNamespace(
        id=str(uuid.uuid4()),
        username="lawyer01",
        email="lawyer01@example.com",
        phone="13800138000",
        real_name="王律师",
        role="lawyer",
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
    import app.v1.router.lawyers as lawyers_module
    import app.v1.router.users as users_module

    async def _admin_override():
        return {"user_id": "admin-001", "role": "admin"}

    for mod in [lawyers_module, users_module]:
        monkeypatch.setattr(mod, "require_admin", _admin_override, raising=False)
    test_app.dependency_overrides[rbac_module.require_admin] = _admin_override

    yield test_app


# ---------------------------------------------------------------------------
# GET /lawyers/
# ---------------------------------------------------------------------------


class TestListLawyers:
    @pytest.mark.asyncio
    async def test_list_lawyers_success(self, admin_app, mock_db_session):
        l1 = _make_user(username="lawyer1")
        l2 = _make_user(username="lawyer2")
        _setup_db(
            mock_db_session,
            scalars_returns=[[l1, l2]],
        )

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{LAWYERS_PREFIX}/")

            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 200
            assert data["data"]["total"] == 2
            assert len(data["data"]["lawyers"]) == 2

    @pytest.mark.asyncio
    async def test_list_lawyers_empty(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalars_returns=[[]])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{LAWYERS_PREFIX}/")

            assert response.status_code == 200
            data = response.json()
            assert data["data"]["total"] == 0
            assert data["data"]["lawyers"] == []


# ---------------------------------------------------------------------------
# POST /lawyers/
# ---------------------------------------------------------------------------


class TestCreateLawyer:
    @pytest.mark.asyncio
    async def test_create_lawyer_success(self, admin_app, mock_db_session):
        _setup_db(
            mock_db_session,
            scalar_one_or_none=[None, None],  # no existing username, no existing email
        )

        async def _refresh(obj):
            obj.id = "new-lawyer-id"
            if not getattr(obj, "is_active", None):
                obj.is_active = True
            if not getattr(obj, "created_at", None):
                obj.created_at = datetime(2026, 1, 1, 12, 0, 0)
            if not getattr(obj, "updated_at", None):
                obj.updated_at = datetime(2026, 1, 2, 12, 0, 0)
        mock_db_session.refresh = AsyncMock(side_effect=_refresh)
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYERS_PREFIX}/",
                json={
                    "username": "newlawyer",
                    "password": "password123",
                    "email": "new@example.com",
                    "phone": "13800138000",
                    "real_name": "新律师",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 200
            assert data["data"]["username"] == "newlawyer"
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_create_lawyer_duplicate_username(self, admin_app, mock_db_session):
        existing = _make_user(username="existinglawyer")
        _setup_db(mock_db_session, scalar_one_or_none=[existing])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYERS_PREFIX}/",
                json={
                    "username": "existinglawyer",
                    "password": "password123",
                },
            )
            assert response.status_code == 400
            assert "用户名" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_create_lawyer_duplicate_email(self, admin_app, mock_db_session):
        existing = _make_user(email="existing@example.com")
        # First query (username) returns None, second query (email) returns existing
        _setup_db(mock_db_session, scalar_one_or_none=[None, existing])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYERS_PREFIX}/",
                json={
                    "username": "newlawyer",
                    "password": "password123",
                    "email": "existing@example.com",
                },
            )
            assert response.status_code == 400
            assert "邮箱" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# PUT /lawyers/{id}
# ---------------------------------------------------------------------------


class TestUpdateLawyer:
    @pytest.mark.asyncio
    async def test_update_lawyer_success(self, admin_app, mock_db_session):
        lawyer = _make_user()
        _setup_db(mock_db_session, scalar_one_or_none=[lawyer])

        async def _refresh(obj):
            pass
        mock_db_session.refresh = AsyncMock(side_effect=_refresh)
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYERS_PREFIX}/{lawyer.id}",
                json={"real_name": "新名字", "phone": "13900000000"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 200
            assert lawyer.real_name == "新名字"
            assert lawyer.phone == "13900000000"
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_lawyer_not_found(self, admin_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYERS_PREFIX}/missing-id",
                json={"real_name": "X"},
            )
            assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
