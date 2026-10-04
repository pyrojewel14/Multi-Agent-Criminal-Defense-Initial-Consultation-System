"""已有 SQLite 运行资源的有界只读探测，不重连或读取会话。"""

import asyncio


class SQLiteReadiness:
    """由资源所有者标记初始化与关闭；探测只返回受控状态。"""

    def __init__(self):
        self._connection = None
        self._lock = None
        self._status = "not_initialized"

    def start(self, connection, lock=None):
        """绑定已初始化的 aiosqlite 连接，可复用 checkpoint 的写入锁。"""
        self._connection = connection
        self._lock = lock
        self._status = "initialized"

    def close(self):
        """撤销运行资源引用；实际资源仍由原所有者负责关闭。"""
        self._connection = None
        self._lock = None
        self._status = "closed"

    async def check(self, timeout=0.5):
        """超时覆盖锁等待和查询；取消向上传播，不遗留请求任务或游标。"""
        connection, lock = self._connection, self._lock
        if connection is None:
            return {"status": self._status, "initialized": False, "available": False}

        async def query():
            # execute_fetchall 在 SQLite 工作线程内完成取值与关闭游标。
            # 即使等待方取消，也不会留下由请求持有的游标或事务。
            if lock is None:
                await connection.execute_fetchall("SELECT 1")
            else:
                async with lock:
                    await connection.execute_fetchall("SELECT 1")

        try:
            await asyncio.wait_for(query(), timeout)
        except TimeoutError:
            return {"status": "error", "initialized": True, "available": False, "error_code": "timeout"}
        except Exception:
            return {"status": "error", "initialized": True, "available": False, "error_code": "query_failed"}
        return {"status": "ready", "initialized": True, "available": True}
