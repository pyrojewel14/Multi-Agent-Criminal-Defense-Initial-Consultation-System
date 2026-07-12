"""Unit tests for ``app.db.redis_config``.

The Redis client is mocked so the suite runs without a real Redis server.
Covers pool initialisation, singleton client caching, ``redis_available``,
``init_redis``, ``close_redis`` and the ``get_redis_cache_*`` /
``set_redis_cache`` helpers.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db import redis_config
from app.db.redis_config import (
    close_redis,
    connect_redis,
    get_redis_cache_json,
    get_redis_cache_str,
    init_redis,
    redis_available,
    set_redis_cache,
)


@pytest.fixture(autouse=True)
def reset_redis_globals():
    """Reset the module-level ``_pool`` / ``_redis_client`` between tests."""
    redis_config._pool = None
    redis_config._redis_client = None
    yield
    redis_config._pool = None
    redis_config._redis_client = None


def _make_client():
    """Return a ``MagicMock`` standing in for ``redis.asyncio.Redis``."""
    client = MagicMock()
    client.get = AsyncMock()
    client.set = AsyncMock()
    client.ping = AsyncMock()
    client.aclose = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# get_redis / connect_redis
# ---------------------------------------------------------------------------


class TestGetRedis:
    @pytest.mark.asyncio
    async def test_get_redis_creates_singleton_client(self):
        with patch.object(redis_config, "_get_pool", return_value=MagicMock()):
            with patch("app.db.redis_config.redis.Redis") as mock_redis_cls:
                mock_redis_cls.return_value = "fake-client-1"

                client1 = await redis_config.get_redis()
                client2 = await redis_config.get_redis()

        # Same singleton returned on subsequent calls
        assert client1 is client2
        assert client1 == "fake-client-1"
        mock_redis_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_redis_uses_get_redis(self):
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = "client"

            client = await connect_redis()

        assert client == "client"
        mock_get.assert_awaited_once()


# ---------------------------------------------------------------------------
# redis_available / init_redis
# ---------------------------------------------------------------------------


class TestRedisAvailable:
    @pytest.mark.asyncio
    async def test_returns_true_when_ping_succeeds(self):
        client = _make_client()
        with patch.object(redis_config, "connect_redis", new_callable=AsyncMock) as mock_conn:
            mock_conn.return_value = client

            result = await redis_available()

        assert result is True
        client.ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_ping_raises(self):
        client = _make_client()
        client.ping.side_effect = ConnectionError("refused")
        with patch.object(redis_config, "connect_redis", new_callable=AsyncMock) as mock_conn:
            mock_conn.return_value = client

            result = await redis_available()

        assert result is False


class TestInitRedis:
    @pytest.mark.asyncio
    async def test_init_redis_succeeds_when_ping_ok(self):
        client = _make_client()
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            await init_redis()

        client.ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_init_redis_raises_when_ping_fails(self):
        client = _make_client()
        client.ping.side_effect = ConnectionError("refused")
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            with pytest.raises(ConnectionError):
                await init_redis()


# ---------------------------------------------------------------------------
# close_redis
# ---------------------------------------------------------------------------


class TestCloseRedis:
    @pytest.mark.asyncio
    async def test_close_redis_with_no_globals_is_noop(self):
        # Both globals are None — should not raise
        await close_redis()

    @pytest.mark.asyncio
    async def test_close_redis_closes_client_and_pool(self):
        client = _make_client()
        pool = MagicMock()
        pool.aclose = AsyncMock()
        redis_config._redis_client = client
        redis_config._pool = pool

        await close_redis()

        client.aclose.assert_awaited_once()
        pool.aclose.assert_awaited_once()
        assert redis_config._redis_client is None
        assert redis_config._pool is None

    @pytest.mark.asyncio
    async def test_close_redis_supports_redis_py_5_close_methods(self):
        class LegacyRedisClient:
            def __init__(self):
                self.close = AsyncMock()

        class LegacyConnectionPool:
            def __init__(self):
                self.disconnect = AsyncMock()

        client = LegacyRedisClient()
        pool = LegacyConnectionPool()
        redis_config._redis_client = client
        redis_config._pool = pool

        await close_redis()

        client.close.assert_awaited_once()
        pool.disconnect.assert_awaited_once()
        assert redis_config._redis_client is None
        assert redis_config._pool is None

    @pytest.mark.asyncio
    async def test_close_redis_only_client(self):
        client = _make_client()
        redis_config._redis_client = client
        redis_config._pool = None

        await close_redis()

        client.aclose.assert_awaited_once()
        assert redis_config._redis_client is None

    @pytest.mark.asyncio
    async def test_close_redis_only_pool(self):
        pool = MagicMock()
        pool.aclose = AsyncMock()
        redis_config._redis_client = None
        redis_config._pool = pool

        await close_redis()

        pool.aclose.assert_awaited_once()
        assert redis_config._pool is None


# ---------------------------------------------------------------------------
# get_redis_cache_str
# ---------------------------------------------------------------------------


class TestGetRedisCacheStr:
    @pytest.mark.asyncio
    async def test_returns_value_when_present(self):
        client = _make_client()
        client.get.return_value = "cached-value"
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await get_redis_cache_str("k")

        assert result == "cached-value"
        client.get.assert_awaited_once_with("k")

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        client = _make_client()
        client.get.side_effect = ConnectionError("boom")
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await get_redis_cache_str("k")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(self):
        client = _make_client()
        client.get.return_value = None
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await get_redis_cache_str("missing-key")

        assert result is None


# ---------------------------------------------------------------------------
# get_redis_cache_json
# ---------------------------------------------------------------------------


class TestGetRedisCacheJson:
    @pytest.mark.asyncio
    async def test_deserializes_json_value(self):
        client = _make_client()
        client.get.return_value = json.dumps({"a": 1, "b": [1, 2]})
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await get_redis_cache_json("k")

        assert result == {"a": 1, "b": [1, 2]}

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(self):
        client = _make_client()
        client.get.return_value = None
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await get_redis_cache_json("k")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        client = _make_client()
        client.get.side_effect = ConnectionError("boom")
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await get_redis_cache_json("k")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self):
        client = _make_client()
        client.get.return_value = "not-json"
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await get_redis_cache_json("k")

        # json.loads raises — the function returns None on any exception
        assert result is None


# ---------------------------------------------------------------------------
# set_redis_cache
# ---------------------------------------------------------------------------


class TestSetRedisCache:
    @pytest.mark.asyncio
    async def test_set_string_value(self):
        client = _make_client()
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await set_redis_cache("k", "hello", expire=60)

        assert result is True
        client.set.assert_awaited_once_with("k", "hello", ex=60)

    @pytest.mark.asyncio
    async def test_set_dict_value_serialized_as_json(self):
        client = _make_client()
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await set_redis_cache("k", {"x": 1}, expire=120)

        assert result is True
        args = client.set.await_args
        assert args.args[0] == "k"
        assert json.loads(args.args[1]) == {"x": 1}
        assert args.kwargs.get("ex") == 120 or args.args[2] == 120

    @pytest.mark.asyncio
    async def test_set_list_value_serialized_as_json(self):
        client = _make_client()
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await set_redis_cache("k", [1, 2, 3], expire=10)

        assert result is True
        args = client.set.await_args
        assert json.loads(args.args[1]) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_set_other_types_coerced_to_str(self):
        client = _make_client()
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await set_redis_cache("k", 123, expire=10)

        assert result is True
        args = client.set.await_args
        assert args.args[1] == "123"

    @pytest.mark.asyncio
    async def test_set_returns_false_on_error(self):
        client = _make_client()
        client.set.side_effect = ConnectionError("boom")
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            result = await set_redis_cache("k", "value")

        assert result is False

    @pytest.mark.asyncio
    async def test_default_expire(self):
        client = _make_client()
        with patch.object(redis_config, "get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = client

            await set_redis_cache("k", "v")

        # Default expire is 3600 seconds
        args = client.set.await_args
        assert args.kwargs.get("ex") == 3600 or args.args[2] == 3600


# ---------------------------------------------------------------------------
# _get_pool internal
# ---------------------------------------------------------------------------


class TestGetPool:
    def test_get_pool_creates_pool_only_once(self):
        # The first call creates the pool.
        with patch("app.db.redis_config.redis.ConnectionPool") as mock_pool_cls:
            mock_pool_cls.return_value = "pool-1"
            pool1 = redis_config._get_pool()
            pool2 = redis_config._get_pool()

        assert pool1 is pool2
        mock_pool_cls.assert_called_once()
        # Verify the construction args.
        call = mock_pool_cls.call_args
        kwargs = call.kwargs
        assert kwargs["decode_responses"] is True
        assert kwargs["max_connections"] == redis_config.REDIS_MAX_CONNECTIONS
