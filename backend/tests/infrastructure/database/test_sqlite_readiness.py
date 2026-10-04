"""探测已有隔离 SQLite 连接，禁止重连、写入和异常详情泄露。"""

import asyncio
import time

import aiosqlite
import pytest

from app.infrastructure.database.readiness import SQLiteReadiness


@pytest.mark.asyncio
async def test_probe_lifecycle_and_closed_connection(tmp_path):
    probe = SQLiteReadiness()
    assert (await probe.check())["status"] == "not_initialized"
    async with aiosqlite.connect(tmp_path / "probe.db") as connection:
        statements = []
        await connection.set_trace_callback(statements.append)
        probe.start(connection)
        assert await probe.check() == {"status": "ready", "initialized": True, "available": True}
        assert statements == ["SELECT 1"]
    failure = await probe.check()
    assert failure == {"status": "error", "initialized": True, "available": False, "error_code": "query_failed"}
    probe.close()
    assert (await probe.check())["status"] == "closed"
    assert (await probe.check())["initialized"] is False


@pytest.mark.asyncio
async def test_probe_timeout_and_cancel_keep_connection_usable(tmp_path):
    probe = SQLiteReadiness()
    async with aiosqlite.connect(tmp_path / "probe.db") as connection:
        # 阻塞真实 SQLite 工作线程；探测不获取、遗留游标。
        await connection.create_function("pause", 0, lambda: time.sleep(0.05))
        probe.start(connection)
        blocker = asyncio.create_task(connection.execute_fetchall("SELECT pause()"))
        await asyncio.sleep(0.005)
        assert (await probe.check(timeout=0.005))["error_code"] == "timeout"
        pending = asyncio.create_task(probe.check())
        await asyncio.sleep(0.005)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await blocker
        assert (await probe.check())["available"] is True
        assert await connection.execute_fetchall("SELECT count(*) FROM sqlite_master") == [(0,)]


@pytest.mark.asyncio
async def test_probe_timeout_includes_checkpoint_lock_wait(tmp_path):
    probe = SQLiteReadiness()
    lock = asyncio.Lock()
    async with aiosqlite.connect(tmp_path / "probe.db") as connection:
        probe.start(connection, lock)
        async with lock:
            assert (await probe.check(timeout=0.005))["error_code"] == "timeout"
        assert (await probe.check())["available"] is True
