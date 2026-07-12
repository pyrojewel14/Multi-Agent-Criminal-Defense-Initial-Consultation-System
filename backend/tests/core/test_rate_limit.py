import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.core.rate_limit import InMemoryRateLimiter, _get_client_ip, rate_limit
from app.core.success_response import success_response


class TestInMemoryRateLimiter:
    """Tests for InMemoryRateLimiter.is_allowed."""

    def test_within_limit_returns_true(self):
        limiter = InMemoryRateLimiter(limit=3, window=60)
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True

    def test_over_limit_returns_false(self):
        limiter = InMemoryRateLimiter(limit=2, window=60)
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False

    def test_different_clients_independent(self):
        limiter = InMemoryRateLimiter(limit=1, window=60)
        assert limiter.is_allowed("client_a") is True
        assert limiter.is_allowed("client_b") is True
        assert limiter.is_allowed("client_a") is False
        assert limiter.is_allowed("client_b") is False

    def test_window_expiry_resets_counter(self):
        limiter = InMemoryRateLimiter(limit=2, window=1)  # 1-second window
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False

        # Wait for the window to expire
        time.sleep(1.1)
        assert limiter.is_allowed("client1") is True

    def test_default_parameters(self):
        limiter = InMemoryRateLimiter()
        # Default limit is 10, so 10 requests should be allowed
        for _ in range(10):
            assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False


class TestSuccessResponse:
    """Tests for success_response."""

    def test_returns_correct_structure(self):
        response = success_response(message="操作成功", data={"key": "value"})
        import json

        body = json.loads(response.body)
        assert body["code"] == 200
        assert body["message"] == "操作成功"
        assert body["data"] == {"key": "value"}

    def test_default_message(self):
        response = success_response()
        import json

        body = json.loads(response.body)
        assert body["message"] == "success"
        assert body["data"] is None

    def test_data_can_be_list(self):
        response = success_response(data=[1, 2, 3])
        import json

        body = json.loads(response.body)
        assert body["data"] == [1, 2, 3]

    def test_status_code_is_200(self):
        response = success_response()
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# _get_client_ip
# ---------------------------------------------------------------------------


def _make_request(client_host=None, x_forwarded_for=None):
    request = MagicMock()
    if client_host is None:
        request.client = None
    else:
        client = MagicMock()
        client.host = client_host
        request.client = client

    headers = {}
    if x_forwarded_for is not None:
        headers["X-Forwarded-For"] = x_forwarded_for
    request.headers = headers
    return request


class TestGetClientIp:
    def test_uses_client_host(self):
        req = _make_request(client_host="1.2.3.4")
        assert _get_client_ip(req) == "1.2.3.4"

    def test_no_client_uses_xff(self):
        req = _make_request(client_host=None, x_forwarded_for="5.6.7.8, 9.10.11.12")
        assert _get_client_ip(req) == "5.6.7.8"

    def test_no_client_no_xff_returns_unknown(self):
        req = _make_request(client_host=None)
        assert _get_client_ip(req) == "unknown"

    def test_empty_xff_returns_unknown(self):
        req = _make_request(client_host=None, x_forwarded_for="")
        assert _get_client_ip(req) == "unknown"


# ---------------------------------------------------------------------------
# rate_limit dependency
# ---------------------------------------------------------------------------


class FakeRequest:
    """Minimal FastAPI Request-like stand-in."""

    def __init__(self, client_host=None, x_forwarded_for=None):
        if client_host is None:
            self.client = None
        else:
            client = MagicMock()
            client.host = client_host
            self.client = client
        self.headers = {}
        if x_forwarded_for is not None:
            self.headers["X-Forwarded-For"] = x_forwarded_for


class TestRateLimitDependency:
    @pytest.mark.asyncio
    async def test_redis_path_under_limit(self):
        req = FakeRequest(client_host="1.1.1.1")
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock()
        with patch_redis(redis_mock):
            dep = rate_limit(limit=5, window=60)
            # Should not raise
            await dep(req)

    @pytest.mark.asyncio
    async def test_redis_path_at_limit_raises(self):
        req = FakeRequest(client_host="1.1.1.1")
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=b"5")
        with patch_redis(redis_mock):
            dep = rate_limit(limit=5, window=60)
            with pytest.raises(HTTPException) as exc_info:
                await dep(req)
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_redis_path_increments_existing(self):
        req = FakeRequest(client_host="1.1.1.1")
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=b"2")
        redis_mock.incr = AsyncMock()
        with patch_redis(redis_mock):
            dep = rate_limit(limit=5, window=60)
            await dep(req)
        redis_mock.incr.assert_awaited_once()
        redis_mock.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_memory_on_redis_error(self):
        req = FakeRequest(client_host="1.1.1.1")
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(side_effect=RuntimeError("redis down"))
        # Use a fresh in-memory limiter with the test's desired limit
        fresh_limiter = InMemoryRateLimiter(limit=2, window=60)
        with patch_redis(redis_mock), patch("app.core.rate_limit._limiter", fresh_limiter):
            dep = rate_limit(limit=2, window=60)
            # 1st and 2nd calls allowed
            await dep(req)
            await dep(req)
            # 3rd call should raise
            with pytest.raises(HTTPException) as exc_info:
                await dep(req)
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_redis_http_exception_re_propagated(self):
        """When the dependency raises HTTPException directly, it should propagate."""
        req = FakeRequest(client_host="1.1.1.1")
        redis_mock = MagicMock()
        redis_mock.get = AsyncMock(return_value=b"5")
        with patch_redis(redis_mock):
            dep = rate_limit(limit=5, window=60)
            with pytest.raises(HTTPException) as exc_info:
                await dep(req)
            assert exc_info.value.status_code == 429


def patch_redis(redis_mock):
    """Context manager that patches ``connect_redis`` to return redis_mock."""
    from contextlib import contextmanager

    @contextmanager
    def _patcher():
        async def _connect():
            return redis_mock

        with patch("app.db.redis_config.connect_redis", _connect):
            yield

    return _patcher()
