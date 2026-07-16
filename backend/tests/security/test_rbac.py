"""RBAC (Role-Based Access Control) unit tests.

Tests cover:
1. RoleChecker.__call__: Allows correct role, raises for wrong role
2. get_current_user: Valid token returns user dict, invalid token raises 401
3. require_admin / require_lawyer: FastAPI 依赖执行真实角色检查
4. get_optional_user_from_header: Header parsing for middleware
5. attach_user_to_request: Middleware attaching user to request state
6. get_user_from_request: Reading user back from request state
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from app.security.jwt import create_access_token
from app.security.rbac import (
    RoleChecker,
    attach_user_to_request,
    get_current_user,
    get_optional_user,
    get_optional_user_from_header,
    get_user_from_request,
    require_admin,
    require_lawyer,
    require_roles,
)


# ---------------------------------------------------------------------------
# RoleChecker
# ---------------------------------------------------------------------------


class TestRoleChecker:
    @pytest.mark.asyncio
    async def test_allows_correct_role(self):
        checker = RoleChecker(["admin", "lawyer"])
        user = {"user_id": "u1", "role": "lawyer"}
        result = await checker(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_allows_admin_role(self):
        checker = RoleChecker(["admin"])
        user = {"user_id": "u1", "role": "admin"}
        result = await checker(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_raises_for_wrong_role(self):
        checker = RoleChecker(["admin", "lawyer"])
        user = {"user_id": "u1", "role": "client"}
        with pytest.raises(HTTPException) as exc_info:
            await checker(user)
        assert exc_info.value.status_code == 403
        assert "权限不足" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_raises_for_single_role_mismatch(self):
        checker = RoleChecker(["admin"])
        user = {"user_id": "u1", "role": "client"}
        with pytest.raises(HTTPException) as exc_info:
            await checker(user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_raises_for_empty_roles_list(self):
        checker = RoleChecker([])
        user = {"user_id": "u1", "role": "admin"}
        with pytest.raises(HTTPException) as exc_info:
            await checker(user)
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_valid_token_returns_user_dict(self):
        token = create_access_token(user_id="user-001", role="client")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=token
        )
        result = await get_current_user(credentials)
        assert result["user_id"] == "user-001"
        assert result["role"] == "client"

    @pytest.mark.asyncio
    async def test_no_credentials_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None)
        assert exc_info.value.status_code == 401
        assert "请先登录" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self):
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid-token"
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token_type_raises_401(self):
        """A refresh token should not be accepted for get_current_user."""
        from app.security.jwt import create_refresh_token

        refresh_token = create_refresh_token("user-001")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=refresh_token
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials)
        assert exc_info.value.status_code == 401
        assert "Token 声明" in exc_info.value.detail


# ---------------------------------------------------------------------------
# require_admin / require_lawyer / require_roles
# ---------------------------------------------------------------------------


class TestRoleDependencies:
    def test_require_roles_custom_roles(self):
        checker = require_roles(["client", "lawyer"])
        assert isinstance(checker, RoleChecker)
        assert checker.allowed_roles == ["client", "lawyer"]

    @pytest.mark.asyncio
    async def test_require_admin_allows_admin(self):
        user = {"user_id": "u1", "role": "admin"}
        result = await require_admin(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_require_admin_rejects_lawyer(self):
        user = {"user_id": "u1", "role": "lawyer"}
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_require_lawyer_allows_lawyer(self):
        user = {"user_id": "u1", "role": "lawyer"}
        result = await require_lawyer(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_require_lawyer_allows_admin(self):
        user = {"user_id": "u1", "role": "admin"}
        result = await require_lawyer(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_require_lawyer_rejects_client(self):
        user = {"user_id": "u1", "role": "client"}
        with pytest.raises(HTTPException) as exc_info:
            await require_lawyer(user)
        assert exc_info.value.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# get_optional_user_from_header
# ---------------------------------------------------------------------------


class TestGetOptionalUserFromHeader:
    @pytest.mark.asyncio
    async def test_no_authorization_header(self):
        """When Authorization header is missing, return None."""
        request = MagicMock()
        request.headers = {}
        request.url.path = "/api/test"
        result = await get_optional_user_from_header(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_wrong_authorization_scheme(self):
        """When Authorization scheme is not 'Bearer', return None."""
        request = MagicMock()
        request.headers = {"Authorization": "Basic abc123"}
        request.url.path = "/api/test"
        result = await get_optional_user_from_header(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        """When the token is invalid, return None."""
        request = MagicMock()
        request.headers = {"Authorization": "Bearer invalid.token.value"}
        request.url.path = "/api/test"
        result = await get_optional_user_from_header(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_refresh_token_rejected(self):
        """A refresh token is not an access token — return None."""
        from app.security.jwt import create_refresh_token

        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {create_refresh_token('user-001')}"}
        request.url.path = "/api/test"
        result = await get_optional_user_from_header(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_access_token(self):
        """A valid access token should yield a user dict."""
        token = create_access_token(user_id="u-007", role="lawyer")
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.url.path = "/api/test"
        result = await get_optional_user_from_header(request)
        assert result == {"user_id": "u-007", "role": "lawyer"}


# ---------------------------------------------------------------------------
# get_optional_user
# ---------------------------------------------------------------------------


class TestGetOptionalUser:
    @pytest.mark.asyncio
    async def test_no_credentials_returns_none(self):
        """When no credentials provided, return None without raising."""
        result = await get_optional_user(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_token_returns_none(self):
        """An invalid token returns None (does not raise)."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-token")
        result = await get_optional_user(credentials)
        assert result is None

    @pytest.mark.asyncio
    async def test_refresh_token_returns_none(self):
        """A refresh token returns None (only access tokens are accepted)."""
        from app.security.jwt import create_refresh_token

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=create_refresh_token("user-001")
        )
        result = await get_optional_user(credentials)
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_access_token_returns_user(self):
        """A valid access token returns the user dict."""
        token = create_access_token(user_id="u-001", role="admin")
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        result = await get_optional_user(credentials)
        assert result == {"user_id": "u-001", "role": "admin"}


# ---------------------------------------------------------------------------
# get_user_from_request
# ---------------------------------------------------------------------------


class TestGetUserFromRequest:
    @pytest.mark.asyncio
    async def test_no_user_attribute(self):
        """When request.state has no 'user' attribute, return None."""
        # Use a real object with no attribute 'user' to exercise the getattr default
        class _NoUserState:
            pass

        request = MagicMock(spec=Request)
        request.state = _NoUserState()

        result = await get_user_from_request(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_user_present(self):
        """When request.state.user is set, return it."""
        request = MagicMock()
        request.state.user = {"user_id": "u-001", "role": "client"}
        result = await get_user_from_request(request)
        assert result == {"user_id": "u-001", "role": "client"}


# ---------------------------------------------------------------------------
# attach_user_to_request – middleware
# ---------------------------------------------------------------------------


class TestAttachUserToRequest:
    @pytest.mark.asyncio
    async def test_attaches_user_when_valid_token(self):
        """When a valid token is present, request.state.user should be populated."""
        token = create_access_token(user_id="u-007", role="client")
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {token}"}
        request.url.path = "/api/test"

        response_mock = MagicMock()
        call_next = AsyncMock(return_value=response_mock)

        result = await attach_user_to_request(request, call_next)
        assert result is response_mock
        assert request.state.user == {"user_id": "u-007", "role": "client"}
        call_next.assert_awaited_once_with(request)

    @pytest.mark.asyncio
    async def test_no_user_when_token_missing(self):
        """When no token is present, request.state.user is not set, but call_next is still called."""
        request = MagicMock()
        request.headers = {}
        request.url.path = "/api/test"

        response_mock = MagicMock()
        call_next = AsyncMock(return_value=response_mock)

        result = await attach_user_to_request(request, call_next)
        assert result is response_mock
        call_next.assert_awaited_once_with(request)
