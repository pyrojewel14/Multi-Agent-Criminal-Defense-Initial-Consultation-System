"""API 存活与重排序器 readiness 边界测试。"""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    import main
    from app.consultation.workflow import ConsultationOrchestrator
    from app.infrastructure.database import db
    from app.infrastructure.database.readiness import SQLiteReadiness

    raw = SQLiteReadiness()
    monkeypatch.setattr(db, "database_readiness", raw)
    monkeypatch.setattr(main, "database_readiness", raw)
    monkeypatch.setattr(main, "checkpoint_readiness", SQLiteReadiness())
    monkeypatch.setattr(main, "orchestrator", ConsultationOrchestrator())


@pytest.mark.asyncio
async def test_ready_real_sqlite_startup_shutdown_and_no_model_call(tmp_path, monkeypatch):
    import json
    from unittest.mock import AsyncMock

    from sqlalchemy.ext.asyncio import create_async_engine

    import main
    from app.infrastructure.database import db
    from app.infrastructure.llm.gateway import LLMGateway

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'transcript.db'}")
    monkeypatch.setattr(db, "async_engine", engine)
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB_PATH", str(tmp_path / "checkpoint.db"))
    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "6")
    monkeypatch.setenv("MEMORY_SUMMARY_INPUT_BUDGET", "1234")
    monkeypatch.setattr(main, "preflight_law_knowledge", lambda: None)
    monkeypatch.setattr(main, "init_redis", AsyncMock())
    monkeypatch.setattr(main, "close_redis", AsyncMock())
    monkeypatch.setattr(main.reorder_service, "readiness", lambda: {"available": True})
    async def forbidden(*args, **kwargs):
        pytest.fail("readiness must never call models")
    monkeypatch.setattr(LLMGateway, "generate", forbidden)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/ready")).json()["status"] == "not_ready"
        assert list(tmp_path.iterdir()) == []
        async with main.lifespan(app):
            response = await client.get("/ready")
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == payload["api"] == "ready"
            assert payload["dependencies"]["raw_transcript"]["initialized"] is True
            assert payload["dependencies"]["checkpoint"]["available"] is True
            assert payload["dependencies"]["checkpoint"]["restart_recovery"] is True
            assert payload["dependencies"]["checkpoint"]["recovery_verified_now"] is False
            assert payload["memory"]["summary"]["input_budget"] == 1234
            assert payload["memory"]["context"]["recent_messages"] == 6
            assert str(tmp_path) not in json.dumps(payload)
            transcript_statements, checkpoint_statements = [], []
            await db.database_readiness._connection.set_trace_callback(transcript_statements.append)
            await main.checkpoint_readiness._connection.set_trace_callback(checkpoint_statements.append)
            await client.get("/ready")
            assert transcript_statements == checkpoint_statements == ["SELECT 1"]
            async with main.orchestrator._checkpointer.lock:
                timed_out = (await client.get("/ready")).json()
                assert timed_out["status"] == "not_ready"
                assert timed_out["dependencies"]["checkpoint"]["error_code"] == "timeout"
                import asyncio
                cancelled = asyncio.create_task(client.get("/ready"))
                await asyncio.sleep(0.005)
                cancelled.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await cancelled
            assert (await client.get("/ready")).json()["status"] == "ready"
            with patch.object(main.reorder_service, "readiness", return_value={"available": False}):
                assert (await client.get("/ready")).json()["status"] == "degraded"
            # 当前驱动失效时不重连，也不以配置冒充可用。
            await db.database_readiness._connection.close()
            failure = (await client.get("/ready")).json()
            assert failure["status"] == failure["api"] == "not_ready"
            assert failure["dependencies"]["raw_transcript"]["error_code"] == "query_failed"
        closed = (await client.get("/ready")).json()
        assert closed["status"] == "not_ready"
        assert closed["dependencies"]["checkpoint"]["persistence"] == "closed"
        assert closed["dependencies"]["checkpoint"]["restart_recovery"] is False
        assert closed["dependencies"]["raw_transcript"]["status"] == "closed"


@pytest.mark.asyncio
async def test_current_process_saver_never_claims_restart_recovery(monkeypatch):
    import aiosqlite

    import main

    async with aiosqlite.connect(":memory:") as connection:
        main.database_readiness.start(connection)
        # 即使探测资源可用，持久化声明只取 orchestrator 的既有权威。
        main.checkpoint_readiness.start(connection)
        payload = await main.readiness_check()
        assert payload["dependencies"]["checkpoint"]["persistence"] == "process"
        assert payload["dependencies"]["checkpoint"]["restart_recovery"] is False


@pytest.mark.asyncio
async def test_readiness_distinguishes_api_liveness_from_reranker_error():
    reranker_state = {
        "status": "error",
        "available": False,
        "model": "test-model",
        "error": "RuntimeError: broken weights",
        "type": "cross_encoder",
    }
    with patch("main.reorder_service", create=True) as reorder_service:
        reorder_service.readiness.return_value = reranker_state
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            health_response = await client.get("/health")
            readiness_response = await client.get("/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy"}
    assert readiness_response.status_code == 200
    payload = readiness_response.json()
    assert payload["status"] == "not_ready"
    assert payload["api"] == "not_ready"
    assert payload["dependencies"]["reranker"] == reranker_state
    assert payload["dependencies"]["checkpoint"]["persistence"] == "process"
    assert payload["dependencies"]["checkpoint"]["restart_recovery"] is False
    assert payload["dependencies"]["raw_transcript"]["initialized"] is False
    assert payload["memory"]["summary"]["enabled"] is True
    assert payload["memory"]["structured_case"]["enabled"] is True
    assert payload["memory"]["context"]["recent_messages"] > 0
