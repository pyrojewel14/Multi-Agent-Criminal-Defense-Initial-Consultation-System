"""真实业务 SQLite 与真实图的 memory 服务契约；模型节点使用确定性替身。"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.consultation import service, workflow
from app.models import Base, ConsultationMessage


@pytest.fixture
async def memory_db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db:
        yield db
    await engine.dispose()

@pytest.fixture
async def memory_graph(monkeypatch):
    async def receptionist(state):
        return dict(state, consent_given=True, current_agent="FactDigger", final_output="欢迎")
    async def intake(state):
        facts = list(state.get("facts_raw", [])) + [state.get("current_input")]
        return dict(state, facts_raw=facts[-8:], current_input=None, current_agent="FactDigger")
    async def law(state):
        return state
    async def coverage(state):
        return dict(state, facts_coverage_rate=0.0, final_output="继续补充")
    monkeypatch.setattr(workflow, "receptionist_node", receptionist)
    monkeypatch.setattr(workflow, "_fact_intake_workflow_node", intake)
    monkeypatch.setattr(workflow, "law_ref_node", law)
    monkeypatch.setattr(workflow, "_fact_digger_workflow_node", coverage)
    graph = workflow.ConsultationOrchestrator()
    monkeypatch.setattr(service, "orchestrator", graph)
    yield graph

async def start(graph, db, sid):
    cid = await service.create_consultation_record(sid, "synthetic-owner", "suspect", db)
    return await service.start_session({"session_id": sid, "consultation_id": cid,
        "user_id": "synthetic-owner", "current_agent": "Receptionist", "facts_raw": [],
        "facts_structured": {}, "conversation_history": [], "applied_laws": []})


@pytest.mark.asyncio
async def test_direct_exchange_without_consent_keeps_only_raw_audit(memory_db, memory_graph, monkeypatch):
    from app.infrastructure.llm.gateway import LLMGateway
    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "2")
    summary = AsyncMock(return_value="合成摘要")
    monkeypatch.setattr(LLMGateway, "generate", summary)
    sid = "memory-no-consent"
    await start(memory_graph, memory_db, sid)
    await service.persist_state(sid, projection_updates={"consent_given": False})
    before = await memory_graph.get_snapshot(sid)
    for index in range(3):
        await service.record_external_exchange(sid, db=memory_db, key=f"blocked-{index}",
            input_content=f"未同意陈述{index}", output="请先同意")
    rows = (await memory_db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
    assert [row.content for row in rows] == [text for index in range(3)
        for text in (f"未同意陈述{index}", "请先同意")]
    after = await memory_graph.get_snapshot(sid)
    assert after.values.get("memory") == before.values.get("memory")
    assert after.next == before.next
    summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_ws_refusal_does_not_summarize_until_consent(memory_db, monkeypatch):
    import json

    from fastapi import WebSocketDisconnect

    from app.api.v1.routers.consultation import websocket as ws_module
    from app.infrastructure.database import db as db_module
    from app.infrastructure.llm.gateway import LLMGateway
    from app.security.jwt import create_access_token

    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "2")
    summary = AsyncMock(return_value="同意后的增量摘要")
    monkeypatch.setattr(LLMGateway, "generate", summary)
    monkeypatch.setattr(db_module, "AsyncSessionLocal", async_sessionmaker(memory_db.bind, expire_on_commit=False))
    async def no_heartbeat(*args):
        return None
    monkeypatch.setattr(ws_module, "_heartbeat_loop", no_heartbeat)
    sid = "memory-ws-refusal"
    memory_graph = workflow.ConsultationOrchestrator()
    monkeypatch.setattr(service, "orchestrator", memory_graph)
    cid = await service.create_consultation_record(sid, "synthetic-owner", "suspect", memory_db)
    await service.start_session({"session_id": sid, "consultation_id": cid,
        "user_id": "synthetic-owner", "user_type": "suspect", "consent_given": False,
        "facts_raw": [], "facts_structured": {}, "conversation_history": []})
    before = await memory_graph.get_snapshot(sid)
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.query_params = {"token": create_access_token("synthetic-owner", "client")}
    websocket.headers = {}
    websocket.receive_text = AsyncMock(side_effect=[json.dumps({"type": "message",
        "content": f"未同意陈述{index}", "idempotency_key": f"refusal-{index}"}, ensure_ascii=False)
        for index in range(3)] + [WebSocketDisconnect()])
    await ws_module.websocket_endpoint(websocket, sid)
    summary.assert_not_awaited()
    rows = (await memory_db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
    assert len(rows) == 6
    assert [row.content for row in rows if row.sender_type == "user"] == [f"未同意陈述{index}" for index in range(3)]
    refusal = "请先回复'同意'确认您已阅读并理解权利义务告知。"
    assert [row.content for row in rows if row.sender_type == "agent"] == [refusal] * 3
    sent = [call.args[0] for call in websocket.send_json.await_args_list]
    assert [item["content"] for item in sent if item["type"] == "message"] == [refusal] * 3
    assert (await memory_graph.get_snapshot(sid)).next == before.next
    # 确认同意后，存档原文才可按相同预算进入摘要；不摄取先前被拒绝的陈述。
    await service.process_consent(sid, True, db=memory_db,
        raw_payload={"consent_given": True}, next_prompt="实际同意提示", idempotency_key="consent")
    summary.assert_awaited_once()
    payload = json.loads(summary.await_args.args[1])
    summarized = [row for row in payload["new_messages"] if row["sender_type"] == "user"]
    assert [row["id"] for row in summarized] == [row.id for row in rows if row.sender_type == "user"]
    assert [row["sequence"] for row in summarized] == [1, 3, 5]
    snapshot = await memory_graph.get_snapshot(sid)
    assert snapshot.next == ("fact_intake",)
    assert snapshot.values["memory"]["summary"]["through_sequence"] == 6
    assert snapshot.values["facts_raw"] == []


@pytest.mark.asyncio
async def test_direct_exchange_projection_failure_preserves_reply_and_pending(memory_db, memory_graph, monkeypatch, caplog):
    from app.consultation.memory import summary
    sid = "memory-direct-save-failure"
    await start(memory_graph, memory_db, sid)
    before = await memory_graph.get_snapshot(sid)
    original = service.persist_state
    async def fail_memory(session_id, state=None, *, projection_updates=None):
        if projection_updates and "memory" in projection_updates:
            raise RuntimeError("不可写入日志的合成正文")
        return await original(session_id, state, projection_updates=projection_updates)
    monkeypatch.setattr(service, "persist_state", fail_memory)
    summary._logger.addHandler(caplog.handler)
    try:
        for _ in range(2):
            await service.record_external_exchange(sid, db=memory_db, key="welcome", output="合成欢迎")
    finally:
        summary._logger.removeHandler(caplog.handler)
    rows = (await memory_db.scalars(select(ConsultationMessage))).all()
    assert [row.content for row in rows] == ["合成欢迎"]
    after = await memory_graph.get_snapshot(sid)
    assert after.next == before.next
    assert after.values.get("memory") == before.values.get("memory")
    assert caplog.text.count("summary_event=projection_failed") == 2
    assert "error_code=projection_error" in caplog.text
    assert "不可写入日志的合成正文" not in caplog.text
    assert "summary_event=saved" not in caplog.text
    monkeypatch.setattr(service, "persist_state", original)
    await service.record_external_exchange(sid, db=memory_db, key="welcome", output="合成欢迎")
    assert len((await memory_db.scalars(select(ConsultationMessage))).all()) == 1
    recovered = await memory_graph.get_snapshot(sid)
    assert recovered.next == before.next
    assert [row["content"] for row in recovered.values["memory"]["recent"]] == ["合成欢迎"]


@pytest.mark.asyncio
async def test_direct_exchange_raw_commit_failure_still_propagates(memory_db, memory_graph, monkeypatch):
    sid = "memory-direct-audit-failure"
    await start(memory_graph, memory_db, sid)
    before = await memory_graph.get_snapshot(sid)
    monkeypatch.setattr(memory_db, "commit", AsyncMock(side_effect=RuntimeError("audit unavailable")))
    with pytest.raises(RuntimeError, match="audit unavailable"):
        await service.record_external_exchange(sid, db=memory_db, key="welcome", output="合成欢迎")
    assert (await memory_db.scalars(select(ConsultationMessage))).all() == []
    after = await memory_graph.get_snapshot(sid)
    assert after.next == before.next
    assert after.values.get("memory") == before.values.get("memory")

@pytest.mark.asyncio
async def test_http_ws_share_raw_ids_and_no_duplicate_graph_on_replay(memory_db, memory_graph):
    state = await start(memory_graph, memory_db, "memory-http-ws")
    first = await service.process_message("memory-http-ws", "合成输入甲", state, "FactDigger",
        db=memory_db, transport="http", sender_id="synthetic-owner", idempotency_key="key1")
    service._session_commands = service._SessionCommandCoordinator()
    replay = await service.process_message("memory-http-ws", "合成输入甲", state, "FactDigger",
        db=memory_db, transport="websocket", sender_id="synthetic-owner", idempotency_key="key1")
    rows = (await memory_db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
    assert [row.content for row in rows] == ["合成输入甲", "继续补充"]
    assert [row.sequence for row in rows] == [1, 2]
    assert [row.record_kind for row in rows] == ["external", "external"]
    assert first.message_id == replay.message_id == rows[1].id
    assert (await memory_graph.get_snapshot("memory-http-ws")).values["facts_raw"] == ["合成输入甲"]
    assert (await memory_graph.get_snapshot("memory-http-ws")).next == ("fact_intake",)
    assert len(replay.result_state["memory"]["recent"]) == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("projection_failure", [False, True])
async def test_reply_audit_failure_repairs_without_rerun(memory_db, memory_graph, monkeypatch, projection_failure):
    state = await start(memory_graph, memory_db, "memory-repair")
    real_commit = memory_db.commit
    commits = 0
    async def fail_second():
        nonlocal commits
        commits += 1
        if commits == 2:
            raise RuntimeError("synthetic audit failure")
        await real_commit()
    monkeypatch.setattr(memory_db, "commit", fail_second)
    first = await service.process_message("memory-repair", "合成输入乙", state, "FactDigger",
        db=memory_db, transport="http", idempotency_key="repair", sender_id="synthetic-owner")
    assert first.error is not None
    snapshot = await memory_graph.get_snapshot("memory-repair")
    assert snapshot.values["message_audit"]["status"] == "applied"
    before_pending = snapshot.next
    monkeypatch.setattr(memory_db, "commit", real_commit)
    if projection_failure:
        original = service.persist_state
        async def fail_memory(session_id, state=None, *, projection_updates=None):
            if projection_updates and "memory" in projection_updates:
                raise RuntimeError("synthetic optional projection failure")
            return await original(session_id, state, projection_updates=projection_updates)
        monkeypatch.setattr(service, "persist_state", fail_memory)
    service._session_commands = service._SessionCommandCoordinator()
    repaired = await service.process_message("memory-repair", "合成输入乙", state, "FactDigger",
        db=memory_db, transport="http", idempotency_key="repair", sender_id="synthetic-owner")
    assert repaired.error is None
    snapshot = await memory_graph.get_snapshot("memory-repair")
    assert snapshot.next == before_pending
    assert snapshot.values["facts_raw"] == ["合成输入乙"]
    rows = (await memory_db.scalars(select(ConsultationMessage))).all()
    assert len(rows) == 4
    assert [row.content for row in rows if row.sender_type == "user"] == ["合成输入乙"]
    assert any(row.content == "消息处理失败，请稍后重试" and row.record_kind == "external" for row in rows)
    assert snapshot.values["message_audit"] is None

@pytest.mark.asyncio
async def test_raw_commit_failure_never_advances_graph(memory_db, memory_graph, monkeypatch):
    state = await start(memory_graph, memory_db, "memory-before")
    monkeypatch.setattr(memory_db, "commit", AsyncMock(side_effect=RuntimeError("audit unavailable")))
    first = await service.process_message("memory-before", "不得丢失输入", state, "FactDigger",
        db=memory_db, transport="http", idempotency_key="before")
    assert first.error is not None
    snapshot = await memory_graph.get_snapshot("memory-before")
    assert snapshot.values["facts_raw"] == []
    assert snapshot.next == ("fact_intake",)

@pytest.mark.asyncio
async def test_external_close_audit_failure_keeps_payload_and_repairs(memory_db, memory_graph, monkeypatch):
    await start(memory_graph, memory_db, "memory-close")
    real_commit = memory_db.commit
    count = 0
    async def fail_audit():
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("lifecycle audit failure")
        await real_commit()
    monkeypatch.setattr(memory_db, "commit", fail_audit)
    with pytest.raises(service.LifecycleConsistencyError):
        await service.execute_lifecycle_command("memory-close", "close", db=memory_db,
            actor_id="synthetic-owner", reason="合成关闭理由", idempotency_key="close-key", transport="http")
    snapshot = await memory_graph.get_snapshot("memory-close")
    assert snapshot.values["repair_required"] is True
    assert snapshot.values["lifecycle_audit"]["payload"]["reason"] == "合成关闭理由"
    monkeypatch.setattr(memory_db, "commit", real_commit)
    await service.execute_lifecycle_command("memory-close", "close", db=memory_db,
        actor_id="synthetic-owner", reason="合成关闭理由", idempotency_key="close-key", transport="http")
    rows = (await memory_db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
    assert len(rows) == 2
    assert rows[0].record_kind == "external"
    assert rows[1].record_kind == "internal"
    assert (await memory_graph.get_snapshot("memory-close")).values["repair_required"] is False

@pytest.mark.asyncio
async def test_real_receptionist_consent_boundary_and_recorded_actual_prompt(memory_db, monkeypatch):
    graph = workflow.ConsultationOrchestrator()
    monkeypatch.setattr(service, "orchestrator", graph)
    cid = await service.create_consultation_record("memory-consent", "owner", "suspect", memory_db)
    await service.start_session({"session_id": "memory-consent", "consultation_id": cid,
        "user_id": "owner", "user_type": "suspect", "consent_given": False, "facts_raw": [],
        "facts_structured": {}, "conversation_history": []})
    assert (await graph.get_snapshot("memory-consent")).next == ()
    # aupdate_state 基于实际 receptionist 输出重算同意边，无须重跑 Receptionist。
    result = await service.process_consent("memory-consent", True, db=memory_db,
        raw_payload={"consent_given": True}, next_prompt="实际返回的静态提示", idempotency_key="consent-key")
    assert result["consent_given"] is True
    assert (await graph.get_snapshot("memory-consent")).next == ("fact_intake",)
    rows = (await memory_db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
    assert [row.content for row in rows] == ['{"consent_given": true}', "实际返回的静态提示"]

@pytest.mark.asyncio
async def test_ws_consent_retry_never_becomes_a_fact_input(memory_db, monkeypatch):
    graph = workflow.ConsultationOrchestrator()
    monkeypatch.setattr(service, "orchestrator", graph)
    cid = await service.create_consultation_record("memory-ws-consent", "owner", "suspect", memory_db)
    state = await service.start_session({"session_id": "memory-ws-consent", "consultation_id": cid,
        "user_id": "owner", "user_type": "suspect", "consent_given": False, "facts_raw": [],
        "facts_structured": {}, "conversation_history": []})
    first = await service.process_message("memory-ws-consent", "同意", state, "Receptionist",
        db=memory_db, transport="websocket", sender_id="owner", idempotency_key="same-consent")
    service._session_commands = service._SessionCommandCoordinator()
    second = await service.process_message("memory-ws-consent", "同意", state, "Receptionist",
        db=memory_db, transport="websocket", sender_id="owner", idempotency_key="same-consent")
    assert first.message_id == second.message_id
    assert (await graph.get_snapshot("memory-ws-consent")).values["facts_raw"] == []


def test_memory_recovers_across_two_real_python_processes(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path
    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "4")
    monkeypatch.setenv("MEMORY_SUMMARY_INPUT_BUDGET", "4096")
    monkeypatch.setenv("MEMORY_SUMMARY_OUTPUT_BUDGET", "256")
    script = Path(__file__).with_name("memory_process_probe.py")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    for mode, expected in [("write", '"raw_rows": 25'), ("read", '"raw_rows": 27')]:
        result = subprocess.run([sys.executable, str(script), mode, str(tmp_path / "checkpoint.db"), str(tmp_path / "audit.db")],
            env=env, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert expected in result.stdout

@pytest.mark.asyncio
async def test_summary_save_failure_does_not_advance_cursor_or_rerun_graph(memory_db, memory_graph, monkeypatch):
    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "2")
    from app.infrastructure.llm.gateway import LLMGateway
    monkeypatch.setattr(LLMGateway, "generate", AsyncMock(return_value="增量合成摘要"))
    state = await start(memory_graph, memory_db, "memory-summary-save")
    first = await service.process_message("memory-summary-save", "第一轮", state, "FactDigger", db=memory_db,
        transport="http", idempotency_key="summary-1")
    real_persist = service.persist_state
    async def fail_memory(sid, state=None, *, projection_updates=None):
        if projection_updates and "memory" in projection_updates:
            raise RuntimeError("synthetic checkpoint write failure")
        return await real_persist(sid, state, projection_updates=projection_updates)
    monkeypatch.setattr(service, "persist_state", fail_memory)
    second = await service.process_message("memory-summary-save", "第二轮", first.result_state, "FactDigger", db=memory_db,
        transport="http", idempotency_key="summary-2")
    assert second.error is None
    snap = await memory_graph.get_snapshot("memory-summary-save")
    assert snap.values["memory"]["summary"]["through_sequence"] == 0
    monkeypatch.setattr(service, "persist_state", real_persist)
    third = await service.process_message("memory-summary-save", "第三轮", second.result_state, "FactDigger", db=memory_db,
        transport="http", idempotency_key="summary-3")
    assert third.error is None
    assert third.result_state["memory"]["summary"]["through_sequence"] == 4
    assert third.result_state["facts_raw"] == ["第二轮", "第三轮"]
    assert len((await memory_db.scalars(select(ConsultationMessage))).all()) == 6

@pytest.mark.asyncio
async def test_high_risk_raw_input_and_actual_reply_survive(memory_db, memory_graph, monkeypatch):
    from app.consultation.constants import HIGH_RISK_ALERT_MESSAGE
    async def real_intake(state):
        return await workflow.fact_intake_node(state)
    monkeypatch.setattr(workflow, "_fact_intake_workflow_node", real_intake)
    # 重新编译使用真实 FactIntake，高风险必须在模型调用前转人工警报。
    state = await start(memory_graph, memory_db, "memory-alert")
    result = await service.process_message("memory-alert", "我想销毁证据", state, "FactDigger", db=memory_db,
        transport="http", idempotency_key="alert-key")
    assert result.alert_triggered is True
    rows = (await memory_db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
    assert [row.content for row in rows] == ["我想销毁证据", HIGH_RISK_ALERT_MESSAGE]
    assert result.result_state["facts_raw"] == []


@pytest.mark.asyncio
async def test_typed_llm_failure_preserves_raw_and_retry_does_not_reingest(memory_db, memory_graph, monkeypatch):
    from app.errors.exceptions import LLMTimeoutException
    attempts = 0
    async def law_fail_once(state):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMTimeoutException(detail="synthetic model failure")
        return state
    monkeypatch.setattr(workflow, "law_ref_node", law_fail_once)
    state = await start(memory_graph, memory_db, "memory-typed-error")
    with pytest.raises(LLMTimeoutException):
        await service.process_message("memory-typed-error", "只摄取一次", state, "FactDigger", db=memory_db,
            transport="http", idempotency_key="typed-error-key")
    snapshot = await memory_graph.get_snapshot("memory-typed-error")
    assert snapshot.next == ("law_ref",)
    assert snapshot.values["facts_raw"] == ["只摄取一次"]
    result = await service.process_message("memory-typed-error", "只摄取一次", state, "FactDigger", db=memory_db,
        transport="http", idempotency_key="typed-error-key")
    assert result.error is None
    assert result.result_state["facts_raw"] == ["只摄取一次"]
    rows = (await memory_db.scalars(select(ConsultationMessage))).all()
    assert [row.content for row in rows if row.sender_type == "user"] == ["只摄取一次"]
    assert any(row.record_kind == "internal" for row in rows)

@pytest.mark.asyncio
async def test_reconfirming_consent_without_key_never_reextracts_old_facts(memory_db, memory_graph):
    state = await start(memory_graph, memory_db, "memory-reconsent")
    original = await service.process_message("memory-reconsent", "旧事实", state, "FactDigger", db=memory_db,
        transport="http", idempotency_key="original-fact")
    result = await service.process_consent("memory-reconsent", True, db=memory_db,
        raw_payload={"consent_given": True}, next_prompt="同意提示")
    assert result["facts_raw"] == ["旧事实"]
    assert (await memory_graph.get_snapshot("memory-reconsent")).next == ("fact_intake",)
    assert original.error is None

@pytest.mark.asyncio
async def test_oversized_single_input_is_archived_and_user_can_send_a_shorter_message(memory_db, memory_graph, monkeypatch):
    from app.infrastructure.llm.gateway import ContextBudgetException
    monkeypatch.setenv("MEMORY_CONTEXT_TOKEN_BUDGET", "300")
    async def reject_long(state):
        if len(state.get("current_input") or "") > 100:
            raise ContextBudgetException(detail="context_budget_exceeded")
        return dict(state, current_input=None, facts_raw=[state["current_input"]])
    monkeypatch.setattr(workflow, "_fact_intake_workflow_node", reject_long)
    state = await start(memory_graph, memory_db, "memory-big-input")
    for _ in range(2):
        with pytest.raises(ContextBudgetException):
            await service.process_message("memory-big-input", "字" * 1000, state, "FactDigger", db=memory_db,
                transport="http", idempotency_key="big-key")
    rows = (await memory_db.scalars(select(ConsultationMessage))).all()
    assert len(rows) == 3  # 原始输入、内部拒绝事件、实际错误回复；重放不再追加。
    assert any(row.content == "字" * 1000 for row in rows)
    result = await service.process_message("memory-big-input", "短输入", state, "FactDigger", db=memory_db,
        transport="http", idempotency_key="small-key")
    assert result.error is None
    assert result.result_state["facts_raw"] == ["短输入"]

@pytest.mark.asyncio
async def test_running_unknown_receipt_never_becomes_retryable_after_two_retries(memory_db, memory_graph):
    from app.consultation.memory.transcript import append_record, stable_id
    state = await start(memory_graph, memory_db, "memory-unknown")
    key = "message:unknown-key"
    command_id = stable_id("memory-unknown", key, "command")
    fingerprint = service._command_fingerprint({"content": "未知执行结果", "sender_id": None})
    await append_record(memory_db, state["consultation_id"], stable_id("memory-unknown", key, "user"),
        "未知执行结果", "user", metadata={"fingerprint": fingerprint})
    await memory_db.commit()
    await service.persist_state("memory-unknown", projection_updates={"message_audit": {
        "command_id": command_id, "fingerprint": fingerprint, "status": "running"}})
    for _ in range(2):
        result = await service.process_message("memory-unknown", "未知执行结果", state, "FactDigger",
            db=memory_db, transport="http", idempotency_key="unknown-key")
        assert result.error is not None
        snap = await memory_graph.get_snapshot("memory-unknown")
        assert snap.values["facts_raw"] == []
        assert snap.values["message_audit"]["status"] == "running"


@pytest.mark.asyncio
async def test_creation_input_is_archived_before_graph_start(memory_db):
    from app.models import Consultation
    cid = await service.create_consultation_record("memory-before-start", "owner", "suspect", memory_db,
        initial_message="创建时原始陈述", source="http")
    assert await memory_db.scalar(select(Consultation.id).where(Consultation.id == cid)) == cid
    rows = (await memory_db.scalars(select(ConsultationMessage))).all()
    assert len(rows) == 1
    assert rows[0].content == "创建时原始陈述"
    assert rows[0].record_kind == "external"
    assert rows[0].sequence == 1
