import json
import os
from inspect import isawaitable
from typing import Any, Optional

import redis.asyncio as redis

from app.utils.logger import get_logger

_logger = get_logger("DB.Redis")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "3"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))

_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


async def _close_async_resource(resource: Any, method_names: tuple[str, ...]) -> bool:
    """兼容不同 redis-py 版本，关闭异步 Redis 资源。"""
    for method_name in method_names:
        close_method = getattr(resource, method_name, None)
        if close_method is None:
            continue

        result = close_method()
        if isawaitable(result):
            await result
        return True

    return False


def _get_pool() -> redis.ConnectionPool:
    """获取共享的 Redis 连接池，首次调用时创建。

    Returns:
        Redis 连接池实例。
    """
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            max_connections=REDIS_MAX_CONNECTIONS,
            decode_responses=True,
            retry_on_timeout=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        _logger.info(
            "【_get_pool】Redis 连接池已创建: %s:%d db=%d max_connections=%d",
            REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_MAX_CONNECTIONS,
        )
    return _pool


async def get_redis() -> redis.Redis:
    """获取单例 Redis 客户端，所有操作共享同一实例。

    Returns:
        Redis 客户端实例。
    """
    global _redis_client
    if _redis_client is None:
        pool = _get_pool()
        _redis_client = redis.Redis(connection_pool=pool)
    return _redis_client


async def connect_redis() -> redis.Redis:
    """返回使用共享连接池的 Redis 客户端。

    Deprecated: 请使用 get_redis() 获取单例客户端，避免反复创建实例导致连接泄漏。

    Returns:
        Redis 客户端实例。
    """
    return await get_redis()


async def redis_available() -> bool:
    """检查 Redis 是否可通过 ping 命令访问。

    Returns:
        如果 Redis 服务器响应 PING 返回 True，否则返回 False。
    """
    try:
        client = await connect_redis()
        await client.ping()
        return True
    except Exception:
        return False


async def close_redis() -> None:
    """关闭 Redis 客户端和连接池，释放所有连接。"""
    global _pool, _redis_client
    if _redis_client:
        await _close_async_resource(_redis_client, ("aclose", "close"))
        _redis_client = None
    if _pool:
        await _close_async_resource(_pool, ("aclose", "disconnect", "close"))
        _pool = None
        _logger.info("【close_redis】Redis 连接池已关闭")


async def init_redis() -> None:
    """初始化 Redis 连接池并验证连接。

    Raises:
        Exception: Redis 连接初始化失败时抛出。
    """
    client = await get_redis()
    try:
        await client.ping()
        _logger.info("【init_redis】Redis 连接初始化成功")
    except Exception as e:
        _logger.error("【init_redis】Redis 连接初始化失败: %s", e)
        raise


async def get_redis_cache_str(key: str) -> Optional[str]:
    """根据键从 Redis 检索字符串值。

    Args:
        key: Redis 键。

    Returns:
        缓存的字符串值，不存在或错误返回 None。
    """
    try:
        client = await get_redis()
        value = await client.get(key)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value
    except Exception as e:
        _logger.error("【get_redis_cache_str】Redis 获取失败 key=%s: %s", key, e)
        return None


async def get_redis_cache_json(key: str) -> Optional[dict]:
    """根据键从 Redis 检索并反序列化 JSON 值。

    Args:
        key: Redis 键。

    Returns:
        反序列化后的字典，不存在或错误返回 None。
    """
    try:
        client = await get_redis()
        data = await client.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        _logger.error("【get_redis_cache_json】Redis JSON 获取失败 key=%s: %s", key, e)
        return None


async def set_redis_cache(key: str, value: Any, expire: int = 3600) -> bool:
    """在 Redis 中存储值，设置过期时间（秒）。

    Args:
        key: Redis 键。
        value: 要存储的值（str、dict 或 list — 其他类型转换为 str）。
        expire: TTL 秒数（默认 3600）。

    Returns:
        成功返回 True，失败返回 False。
    """
    try:
        client = await get_redis()
        if isinstance(value, str):
            await client.set(key, value, ex=expire)
        elif isinstance(value, (dict, list)):
            await client.set(key, json.dumps(value, ensure_ascii=False), ex=expire)
        else:
            await client.set(key, str(value), ex=expire)
        return True
    except Exception as e:
        _logger.error("【set_redis_cache】Redis 存储失败 key=%s: %s", key, e)
        return False
