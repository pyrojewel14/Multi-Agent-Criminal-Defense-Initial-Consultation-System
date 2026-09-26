"""真实 SQLite checkpoint 的重建、恢复和会话隔离回归。"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.orchestrator.workflow import ConsultationOrchestrator
from tests.factories import make_consultation_state


@pytest.fixture
def deterministic_nodes():
    calls = []

    async def receptionist(state):
        calls.append((state["session_id"], "receptionist"))
        state["current_agent"] = "Receptionist"
        return state

    async def fact_intake(state):
        calls.append((state["session_id"], "fact_intake", state.get("current_input")))
        if state.get("current_input"):
            state["facts_raw"].append(state["current_input"])
            state["current_input"] = None
        state["current_agent"] = "FactDigger"
        return state

    async def law_ref(state):
        calls.append((state["session_id"], "law_ref"))
        state["current_agent"] = "LawRef"
        return state

    async def fact_digger(state):
        calls.append((state["session_id"], "fact_digger"))
        state["facts_coverage_rate"] = 0.2
        state["current_agent"] = "FactDigger"
        return state

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch("app.orchestrator.workflow.fact_intake_node", fact_intake),
        patch("app.orchestrator.workflow.law_ref_node", law_ref),
        patch("app.orchestrator.workflow.fact_coverage_node", fact_digger),
    ):
        yield calls


def _saver(conn):
    return AsyncSqliteSaver(
        conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None)
    )


def _state(session_id):
    return make_consultation_state(
        session_id=session_id,
        consultation_id=f"consultation-{session_id}",
        consent_given=False,
        facts_raw=[f"初始事实-{session_id}"],
        conversation_history=[{"source": session_id}],
        command_processed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resume_after_restart",
    [False, True],
    ids=["checkpoint_survives_recreation", "resume_pending_node_after_restart"],
)
async def test_checkpoint_recovery(tmp_path, deterministic_nodes, resume_after_restart):
    db_path = tmp_path / "checkpoints.db"
    session_id = "session-A"
    async with aiosqlite.connect(db_path) as conn:
        first = ConsultationOrchestrator()
        first.configure_checkpointer(_saver(conn), persistent=True)
        await first.start_workflow(_state(session_id))
        await first.resume_workflow(session_id, {"consent_given": True})
        before = await first.get_snapshot(session_id)
        history_before = [item async for item in first._compiled_graph().aget_state_history(first._config(session_id))]
        assert before.next == ("fact_intake",)
        assert before.values["current_agent"] == "FactDigger"
        assert before.values["facts_raw"] == ["初始事实-session-A"]
        assert before.values["command_processed_at"] == datetime(2026, 1, 1, tzinfo=timezone.utc)

    async with aiosqlite.connect(db_path) as conn:
        second = ConsultationOrchestrator()
        second.configure_checkpointer(_saver(conn), persistent=True)
        after = await second.get_snapshot(session_id)
        history_after = [item async for item in second._compiled_graph().aget_state_history(second._config(session_id))]
        assert after is not None
        assert after.next == before.next
        assert after.values["current_agent"] == before.values["current_agent"]
        assert after.values["facts_raw"] == before.values["facts_raw"]
        assert after.values["conversation_history"] == before.values["conversation_history"]
        assert after.values["command_processed_at"] == before.values["command_processed_at"]
        assert len(history_after) == len(history_before)

        if resume_after_restart:
            resumed = await second.resume_workflow(session_id, {"current_input": "补充事实"})
            assert resumed["facts_raw"] == ["初始事实-session-A", "补充事实"]
            assert resumed["current_input"] is None
            assert resumed["conversation_history"] == [{"source": session_id}]
            assert (await second.get_snapshot(session_id)).next == ("fact_intake",)
            assert deterministic_nodes.count((session_id, "receptionist")) == 1
            assert deterministic_nodes.count((session_id, "fact_intake", "补充事实")) == 1


@pytest.mark.asyncio
async def test_checkpoint_sessions_remain_isolated_after_recreation(tmp_path, deterministic_nodes):
    db_path = tmp_path / "checkpoints.db"
    async with aiosqlite.connect(db_path) as conn:
        first = ConsultationOrchestrator(checkpointer=_saver(conn), persistent=True)
        await first.start_workflow(_state("session-A"))
        await first.resume_workflow("session-A", {"consent_given": True})
        await first.start_workflow(_state("session-B"))

    async with aiosqlite.connect(db_path) as conn:
        second = ConsultationOrchestrator(checkpointer=_saver(conn), persistent=True)
        snapshot_a = await second.get_snapshot("session-A")
        snapshot_b = await second.get_snapshot("session-B")
        assert snapshot_a.next == ("fact_intake",)
        assert snapshot_b.next == ()
        assert snapshot_a.values["session_id"] == "session-A"
        assert snapshot_b.values["session_id"] == "session-B"
        assert snapshot_a.values["facts_raw"] == ["初始事实-session-A"]
        assert snapshot_b.values["facts_raw"] == ["初始事实-session-B"]
        history_a = [item async for item in second._compiled_graph().aget_state_history(second._config("session-A"))]
        history_b = [item async for item in second._compiled_graph().aget_state_history(second._config("session-B"))]
        assert all(item.values.get("session_id") in (None, "session-A") for item in history_a)
        assert all(item.values.get("session_id") in (None, "session-B") for item in history_b)


def test_configure_checkpointer_rejects_replacement_after_compile():
    orchestrator = ConsultationOrchestrator()
    orchestrator._compiled_graph()
    with pytest.raises(RuntimeError, match="已编译"):
        orchestrator.configure_checkpointer(object(), persistent=True)


@pytest.mark.asyncio
async def test_application_lifespan_installs_durable_saver(tmp_path, monkeypatch, deterministic_nodes):
    import main
    from app.v1.service import consultation_service

    runtime_orchestrator = ConsultationOrchestrator()
    monkeypatch.setattr(main, "orchestrator", runtime_orchestrator)
    monkeypatch.setattr(consultation_service, "orchestrator", runtime_orchestrator)
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB_PATH", str(tmp_path / "app-checkpoints.db"))
    with (
        patch.object(main, "preflight_law_knowledge"),
        patch.object(main, "init_db", new_callable=AsyncMock),
        patch.object(main, "init_redis", new_callable=AsyncMock),
        patch.object(main, "close_db", new_callable=AsyncMock),
        patch.object(main, "close_redis", new_callable=AsyncMock),
    ):
        async with main.lifespan(main.app):
            assert isinstance(runtime_orchestrator._checkpointer, AsyncSqliteSaver)
            assert runtime_orchestrator.can_resume_after_restart is True
            assert runtime_orchestrator.checkpoint_persistence == "persistent"
            assert consultation_service.orchestrator is runtime_orchestrator
            readiness = await main.readiness_check()
            assert readiness["dependencies"]["checkpoint"] == {
                "persistence": "persistent",
                "restart_recovery": True,
            }
            await runtime_orchestrator.start_workflow(_state("startup-session"))
            assert (tmp_path / "app-checkpoints.db").exists()


@pytest.mark.asyncio
async def test_application_startup_fails_when_checkpoint_database_cannot_open(tmp_path, monkeypatch):
    import main

    runtime_orchestrator = ConsultationOrchestrator()
    monkeypatch.setattr(main, "orchestrator", runtime_orchestrator)
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB_PATH", str(tmp_path))
    with patch.object(main, "init_db", new_callable=AsyncMock) as init_db:
        with pytest.raises(Exception):
            async with main.lifespan(main.app):
                pass
    init_db.assert_not_awaited()
    assert runtime_orchestrator.can_resume_after_restart is False


@pytest.mark.asyncio
async def test_lawyer_assignment_uses_persisted_workflow_session_id_after_restart(tmp_path, deterministic_nodes):
    from app.v1.service import consultation_service

    db_path = tmp_path / "checkpoints.db"
    async with aiosqlite.connect(db_path) as conn:
        first = ConsultationOrchestrator(checkpointer=_saver(conn), persistent=True)
        state = _state("assignment-session")
        state["consent_given"] = True
        await first.start_workflow(state)

    async with aiosqlite.connect(db_path) as conn:
        second = ConsultationOrchestrator(checkpointer=_saver(conn), persistent=True)
        assert second.get_active_sessions() == {}
        with patch.object(consultation_service, "orchestrator", second):
            session_id = await consultation_service.assign_lawyer_to_active_session(
                "consultation-assignment-session", "lawyer-1", workflow_session_id="assignment-session"
            )
        assert session_id == "assignment-session"
        assert (await second.get_snapshot(session_id)).values["lawyer_id"] == "lawyer-1"
