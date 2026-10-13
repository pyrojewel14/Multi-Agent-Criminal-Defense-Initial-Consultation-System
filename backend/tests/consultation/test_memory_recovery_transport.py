"""挂载 HTTP 与真实 WS 连接验证人工核验提示，不透传内部错误。"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.consultation import service, workflow
from app.consultation.memory.transcript import append_record, stable_id
from app.infrastructure.database import db as db_module
from app.models import Base, ConsultationMessage
from app.security.jwt import create_access_token
from main import app

SID = "synthetic-recovery-transport"
OWNER = "synthetic-owner"
CONTENT = "合成待核验陈述"
MANUAL_MESSAGE = "消息执行状态待人工核验，请联系处理人员，勿重复发送"


@pytest.fixture
def recovery_transport(tmp_path, monkeypatch):
    async def reception(state):
        return dict(state, consent_given=True, current_agent="FactDigger", final_output="合成欢迎")

    async def forbidden_intake(state):
        raise AssertionError("未知执行状态不应再次摄取事实")

    monkeypatch.setattr(workflow, "receptionist_node", reception)
    monkeypatch.setattr(workflow, "_fact_intake_workflow_node", forbidden_intake)
    graph = workflow.ConsultationOrchestrator()
    monkeypatch.setattr(service, "orchestrator", graph)
    monkeypatch.setattr(service, "_session_commands", service._SessionCommandCoordinator())
    # 每个请求独立连接，避免同步 TestClient 的不同事件循环复用连接池。
    storage = TemporaryDirectory(prefix="memory-transport-", dir=tmp_path)
    engine = create_async_engine(f"sqlite+aiosqlite:///{Path(storage.name) / 'audit.db'}", poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db_module, "AsyncSessionLocal", factory)

    async def isolated_db():
        async with factory() as db:
            yield db

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with factory() as db:
            cid = await service.create_consultation_record(SID, OWNER, "suspect", db)
            await service.start_session({"session_id": SID, "consultation_id": cid, "user_id": OWNER,
                "consent_given": True, "facts_raw": [], "facts_structured": {},
                "conversation_history": [], "current_agent": "Receptionist"})
        return cid

    cid = asyncio.run(setup())
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[db_module.get_db] = isolated_db
    client = TestClient(app)  # 不进入 app lifespan，不连接本机案件库或模型。
    try:
        yield client, graph, factory, cid
    finally:
        client.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
        asyncio.run(engine.dispose())
        storage.cleanup()


async def seed_unknown(graph, factory, cid, kind, legacy):
    key = "message:target"
    command_id = stable_id(SID, key, "command")
    fingerprint = service._command_fingerprint({"content": CONTENT, "sender_id": OWNER})
    async with factory() as db:
        await append_record(db, cid, stable_id(SID, key, "user"), CONTENT, "user", command_id=command_id,
                            metadata={"fingerprint": fingerprint}, sender_id=OWNER)
        if legacy:
            await append_record(db, cid, stable_id(SID, key, "error_response"), "消息处理失败，请稍后重试",
                "system", command_id=command_id, metadata={"event": "error_response", "request_id": "legacy"})
        await db.commit()
    if kind == "running":
        await service.persist_state(SID, projection_updates={"message_audit": {
            "status": "running", "command_id": command_id, "input_id": stable_id(SID, key, "user"),
            "reply_id": stable_id(SID, key, "reply"), "fingerprint": fingerprint,
            "agent_name": "FactDigger", "request_id": "synthetic-request"}})
    return await audit_state(graph, factory)


async def audit_state(graph, factory):
    snapshot = await graph.get_snapshot(SID)
    async with factory() as db:
        rows = (await db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
        return snapshot, [(row.id, row.sequence, row.content, row.record_metadata) for row in rows]


def send_twice(client, transport):
    token = create_access_token(OWNER, "client")
    payload = {"session_id": SID, "content": CONTENT, "idempotency_key": "target"}
    if transport == "http":
        responses = [client.post(f"/api/v1/sessions/{SID}/message", json=payload,
                                 headers={"Authorization": f"Bearer {token}"}) for _ in range(2)]
        assert [response.status_code for response in responses] == [500, 500]
        return [response.json()["error"]["message"] for response in responses]
    with client.websocket_connect(f"/api/v1/sessions/{SID}/ws?token={token}") as websocket:
        assert websocket.receive_json()["type"] == "ack"
        contents = []
        for _ in range(2):
            websocket.send_json(dict(payload, type="message"))
            message = websocket.receive_json()
            assert message["type"] == "error"
            contents.append(message["content"])
        return contents


@pytest.mark.parametrize("transport", ["http", "websocket"])
@pytest.mark.parametrize("kind", ["missing_receipt", "running"])
@pytest.mark.parametrize("legacy", [False, True])
def test_mounted_unknown_execution_returns_fixed_manual_guidance(recovery_transport, transport, kind, legacy):
    client, graph, factory, cid = recovery_transport
    before, original_rows = asyncio.run(seed_unknown(graph, factory, cid, kind, legacy))
    assert send_twice(client, transport) == [MANUAL_MESSAGE, MANUAL_MESSAGE]
    after, rows = asyncio.run(audit_state(graph, factory))
    assert after.next == before.next == ("fact_intake",)
    assert after.values == before.values
    assert rows[:len(original_rows)] == original_rows
    assert len(rows) == (2 if legacy else 3)


@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_mounted_ordinary_error_keeps_generic_guidance(recovery_transport, monkeypatch, transport):
    client, graph, factory, _ = recovery_transport
    before = asyncio.run(audit_state(graph, factory))
    result = service.ProcessMessageResult(response_content="", next_agent="FactDigger", alert_triggered=False,
        result_state=None, error="内部敏感异常：synthetic-secret /checkpoint/path")
    monkeypatch.setattr(service, "process_message", AsyncMock(return_value=result))
    assert send_twice(client, transport) == ["消息处理失败，请稍后重试"] * 2
    assert asyncio.run(audit_state(graph, factory)) == before
