"""Auth API endpoint tests.

Tests cover:
1. POST /api/v1/auth/register - Register new user
2. POST /api/v1/auth/login - Login returns tokens
3. POST /api/v1/auth/refresh - Refresh token
4. GET /api/v1/auth/me - Get current user
5. Unauthenticated access returns 401
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.security.jwt import create_access_token, create_refresh_token, hash_password
from app.models.user import User, UserRole


AUTH_PREFIX = "/api/v1/auth"


def _make_mock_user(
    user_id="user-001",
    username="testuser",
    role=UserRole.CLIENT,
    is_active=True,
):
    """Create a mock User object with the expected attributes."""
    user = MagicMock(spec=User)
    user.id = user_id
    user.username = username
    user.password_hash = hash_password("password123")
    user.email = "test@example.com"
    user.phone = None
    user.real_name = None
    user.role = role
    user.is_active = is_active
    user.created_at = datetime.utcnow()
    user.updated_at = datetime.utcnow()
    user.last_login_at = None
    user.refresh_token = None
    user.refresh_token_expires_at = None
    return user


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------


class TestRegister:
    @pytest.mark.asyncio
    async def test_register_success(self, test_app, mock_db_session):
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        ))

        # After db.add + db.refresh, the user object should have an id.
        # We simulate this by making refresh populate the id on the added object.
        original_refresh = mock_db_session.refresh

        async def _refresh_side_effect(obj):
            obj.id = "generated-user-id"
            if not hasattr(obj, 'is_active') or obj.is_active is None:
                obj.is_active = True
            if not hasattr(obj, 'created_at') or obj.created_at is None:
                obj.created_at = datetime.utcnow()
            if not hasattr(obj, 'updated_at') or obj.updated_at is None:
                obj.updated_at = datetime.utcnow()

        mock_db_session.refresh = AsyncMock(side_effect=_refresh_side_effect)

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/register",
                json={
                    "username": "newuser",
                    "password": "password123",
                    "email": "new@example.com",
                },
            )

            assert response.status_code == 201
            data = response.json()
            assert data["code"] == 201
            assert data["data"]["username"] == "newuser"

    @pytest.mark.asyncio
    async def test_register_duplicate_username(self, test_app, mock_db_session):
        existing_user = _make_mock_user(username="existinguser")

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=existing_user)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/register",
                json={
                    "username": "existinguser",
                    "password": "password123",
                },
            )

            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, test_app, mock_db_session):
        """Username is free but email is already taken -> 400."""
        existing_user = _make_mock_user()
        # Set the email on the existing mock user
        existing_user.email = "taken@example.com"

        # First execute call (username) returns None (free).
        # Second execute call (email) returns existing user.
        call_count = {"n": 0}

        async def _execute(*args, **kwargs):
            call_count["n"] += 1
            return MagicMock(
                scalar_one_or_none=MagicMock(
                    return_value=existing_user if call_count["n"] == 2 else None
                )
            )

        mock_db_session.execute = AsyncMock(side_effect=_execute)

        async def _refresh(obj):
            obj.id = "generated-user-id"
            if not getattr(obj, "is_active", None):
                obj.is_active = True
            if not getattr(obj, "created_at", None):
                obj.created_at = datetime.utcnow()
            if not getattr(obj, "updated_at", None):
                obj.updated_at = datetime.utcnow()

        mock_db_session.refresh = AsyncMock(side_effect=_refresh)

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/register",
                json={
                    "username": "newuser",
                    "password": "password123",
                    "email": "taken@example.com",
                },
            )

            assert response.status_code == 400
            assert "邮箱" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------


class TestLogin:
    @pytest.mark.asyncio
    async def test_login_success(self, test_app, mock_db_session):
        user = _make_mock_user()

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/login",
                json={
                    "username": "testuser",
                    "password": "password123",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data["data"]
            assert "refresh_token" in data["data"]
            assert data["data"]["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, test_app, mock_db_session):
        user = _make_mock_user()

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/login",
                json={
                    "username": "testuser",
                    "password": "wrongpassword",
                },
            )

            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, test_app, mock_db_session):
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/login",
                json={
                    "username": "nonexistent",
                    "password": "password123",
                },
            )

            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_inactive_user(self, test_app, mock_db_session):
        user = _make_mock_user(is_active=False)

        # Need to return user for the first query (find user), but
        # verify_password will succeed since we use the real hash.
        # The is_active check happens after password verification.
        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/login",
                json={
                    "username": "testuser",
                    "password": "password123",
                },
            )

            assert response.status_code == 403


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------


class TestRefreshToken:
    @pytest.mark.asyncio
    async def test_refresh_success(self, test_app, mock_db_session):
        user = _make_mock_user()
        refresh_token = create_refresh_token(user.id)
        user.refresh_token = refresh_token
        user.refresh_token_expires_at = datetime.utcnow() + timedelta(days=7)

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/refresh",
                json={"refresh_token": refresh_token},
            )

            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data["data"]
            assert "refresh_token" in data["data"]

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, test_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/refresh",
                json={"refresh_token": "invalid-token"},
            )

            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_user_not_found(self, test_app, mock_db_session):
        refresh_token = create_refresh_token("nonexistent-user")

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/refresh",
                json={"refresh_token": refresh_token},
            )

            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token_mismatch(self, test_app, mock_db_session):
        """Stored refresh_token doesn't match the provided one -> 401."""
        user = _make_mock_user()
        user.refresh_token = "different-token-stored"
        user.refresh_token_expires_at = datetime.utcnow() + timedelta(days=7)

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))

        # Use a valid refresh token (different from the stored one)
        fresh_token = create_refresh_token(user.id)

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/refresh",
                json={"refresh_token": fresh_token},
            )

            assert response.status_code == 401
            assert "无效" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_refresh_token_expired(self, test_app, mock_db_session):
        """Refresh token expiry is in the past -> 401."""
        user = _make_mock_user()
        refresh_token = create_refresh_token(user.id)
        user.refresh_token = refresh_token
        user.refresh_token_expires_at = datetime.utcnow() - timedelta(days=1)

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/refresh",
                json={"refresh_token": refresh_token},
            )

            assert response.status_code == 401
            assert "过期" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_refresh_inactive_user(self, test_app, mock_db_session):
        user = _make_mock_user(is_active=False)
        refresh_token = create_refresh_token(user.id)
        user.refresh_token = refresh_token
        user.refresh_token_expires_at = datetime.utcnow() + timedelta(days=7)

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/refresh",
                json={"refresh_token": refresh_token},
            )

            assert response.status_code == 403


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------


class TestGetMe:
    @pytest.mark.asyncio
    async def test_get_me_success(self, test_app, mock_db_session):
        user = _make_mock_user()
        token = create_access_token(user_id=user.id, role="client")
        headers = {"Authorization": f"Bearer {token}"}

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get(
                f"{AUTH_PREFIX}/me",
                headers=headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["data"]["username"] == "testuser"

    @pytest.mark.asyncio
    async def test_get_me_unauthenticated(self, test_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get(f"{AUTH_PREFIX}/me")

            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_me_user_not_found(self, test_app, mock_db_session):
        token = create_access_token(user_id="nonexistent-user", role="client")
        headers = {"Authorization": f"Bearer {token}"}

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get(
                f"{AUTH_PREFIX}/me",
                headers=headers,
            )

            assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------


class TestLogout:
    @pytest.mark.asyncio
    async def test_logout_success(self, test_app, mock_db_session):
        user = _make_mock_user()
        user.refresh_token = "some-token"
        user.refresh_token_expires_at = datetime.utcnow() + timedelta(days=7)
        token = create_access_token(user_id=user.id, role="client")
        headers = {"Authorization": f"Bearer {token}"}

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=user)
        ))
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/logout",
                headers=headers,
            )

            assert response.status_code == 200
            assert user.refresh_token is None
            assert user.refresh_token_expires_at is None
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_logout_user_not_found(self, test_app, mock_db_session):
        """If user doesn't exist, logout still returns success but doesn't commit."""
        token = create_access_token(user_id="nonexistent", role="client")
        headers = {"Authorization": f"Bearer {token}"}

        mock_db_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None)
        ))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                f"{AUTH_PREFIX}/logout",
                headers=headers,
            )

            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
