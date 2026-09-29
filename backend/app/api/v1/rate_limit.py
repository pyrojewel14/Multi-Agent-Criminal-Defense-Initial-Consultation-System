import time
from collections import defaultdict
from typing import Dict

from fastapi import HTTPException, Request

from app.infrastructure.logging import get_logger

_logger = get_logger("Core.RateLimit")


class InMemoryRateLimiter:
    """基于内存数据结构的滑动窗口限流器。

    当 Redis 不可用时作为后备。每个客户端（按 IP）独立跟踪，
    使用每窗口请求计数器。

    Attributes:
        limit: 窗口内允许的最大请求数。
        window: 时间窗口（秒）。
    """

    def __init__(self, limit: int = 10, window: int = 60):
        """初始化内存限流器。

        Args:
            limit: 窗口内允许的最大请求数。
            window: 时间窗口（秒）。
        """
        self._limit = limit
        self._window = window
        self._clients: Dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_key: str) -> bool:
        """检查客户端是否在限流范围内。

        Args:
            client_key: 客户端标识符（通常为 IP 地址）。

        Returns:
            如果请求应被允许返回 True。
        """
        now = time.time()
        cutoff = now - self._window
        timestamps = self._clients[client_key]

        timestamps[:] = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= self._limit:
            return False

        timestamps.append(now)
        return True


_limiter: InMemoryRateLimiter = InMemoryRateLimiter()


def _get_client_ip(request: Request) -> str:
    """从请求中提取真实的客户端 IP。

    Args:
        request: FastAPI 请求对象。

    Returns:
        客户端 IP 地址。
    """
    ip = request.client.host if request.client else None
    if not ip:
        forwarded = request.headers.get("X-Forwarded-For", "")
        ip = forwarded.split(",")[0].strip() if forwarded else "unknown"
    return ip


def rate_limit(limit: int = 10, window: int = 60):
    """FastAPI 依赖工厂，实现按端点的限流。

    优先使用基于 Redis 的限流。当 Redis 不可达时，
    回退到内存滑动窗口限流器。

    Args:
        limit: 每个窗口允许的最大请求数。
        window: 时间窗口大小（秒）。

    Returns:
        异步 FastAPI 依赖函数。
    """

    async def dependency(request: Request):
        client_ip = _get_client_ip(request)
        key = f"rate_limit:{client_ip}"

        try:
            from app.infrastructure.database.redis import connect_redis

            redis_client = await connect_redis()
            current = await redis_client.get(key)
            current = int(current) if current else 0

            if current >= limit:
                raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

            if current == 0:
                await redis_client.setex(key, window, 1)
            else:
                await redis_client.incr(key)
        except HTTPException:
            raise
        except Exception:
            _logger.debug("【dependency】Redis 限流不可用，使用内存后备")
            if not _limiter.is_allowed(client_ip):
                raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    return dependency
