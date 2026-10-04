"""摘要有限日志与真实 checkpoint 保存结果保持一致。"""

import logging

import pytest

from app.consultation.memory.context import MemorySettings
from app.consultation.memory.summary import advance_summary
from tests.consultation.test_memory_persistence import memory_db as memory_db
from tests.consultation.test_memory_persistence import memory_graph as memory_graph
from tests.consultation.test_memory_persistence import start


@pytest.fixture
def memory_logs(caplog):
    loggers = [logging.getLogger(name) for name in ("Memory.Summary", "Memory.Coordinator", "Agent.FactDigger")]
    for logger in loggers:
        logger.addHandler(caplog.handler)
    yield caplog
    for logger in loggers:
        logger.removeHandler(caplog.handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,event", [("success", "generated"), ("failure", "failed"), ("invalid", "failed"),
                                       ("blocked", "blocked"), ("short", "skipped")])
async def test_summary_events_only_log_counts_cursors_and_controlled_errors(mode, event, memory_logs):
    rows = [{"sequence": i, "record_kind": "external", "content": "secret-transcript"} for i in range(1, 5)]
    if mode == "short":
        rows = rows[:1]
    if mode == "blocked":
        rows[0]["content"] *= 2000
    async def summarize(*args):
        if mode == "failure":
            raise RuntimeError("secret-exception-host-path-password")
        return "secret-summary" if mode == "success" else ""
    result = await advance_summary({}, rows, summarize, MemorySettings(recent_messages=2, summary_input_budget=1000))
    text = memory_logs.text
    assert f"summary_event={event}" in text
    assert "cursor_before=0" in text
    assert "duration_ms=" in text
    assert "batch_count=" in text
    assert "version=" in text
    assert "secret-" not in text
    assert "summary_event=saved" not in text
    assert result["summary"]["through_sequence"] == (2 if mode == "success" else 0)


@pytest.mark.asyncio
async def test_generated_summary_not_saved_on_projection_failure_then_retry_saves(memory_db, memory_graph, monkeypatch, memory_logs):
    from app.consultation import service
    from app.infrastructure.llm.gateway import LLMGateway

    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "2")
    async def summarize(*args, **kwargs):
        return "secret-summary"
    monkeypatch.setattr(LLMGateway, "generate", summarize)
    sid = "secret-session"
    state = await start(memory_graph, memory_db, sid)
    first = await service.process_message(sid, "secret-transcript-first", state, "FactDigger", db=memory_db,
                                          transport="http", idempotency_key="one")
    memory_logs.clear()
    real_persist = service.persist_state
    async def fail_projection(sid, state=None, *, projection_updates=None):
        if projection_updates and "memory" in projection_updates:
            raise RuntimeError("secret-projection-error")
        return await real_persist(sid, state, projection_updates=projection_updates)
    monkeypatch.setattr(service, "persist_state", fail_projection)
    second = await service.process_message(sid, "secret-transcript-second", first.result_state, "FactDigger", db=memory_db,
                                           transport="http", idempotency_key="two")
    assert second.command_applied is True
    assert second.error is None
    assert (await memory_graph.get_snapshot(sid)).values["memory"]["summary"]["through_sequence"] == 0
    assert "summary_event=generated" in memory_logs.text
    assert "summary_event=projection_failed" in memory_logs.text
    assert "summary_event=projection_failed, batch_count=unknown" in memory_logs.text
    assert "summary_event=saved" not in memory_logs.text
    assert "secret-" not in memory_logs.text
    monkeypatch.setattr(service, "persist_state", real_persist)
    memory_logs.clear()
    third = await service.process_message(sid, "secret-transcript-third", second.result_state, "FactDigger", db=memory_db,
                                          transport="http", idempotency_key="three")
    assert third.result_state["memory"]["summary"]["through_sequence"] == 4
    assert third.result_state["memory"]["summary"]["version"] == 1
    assert "summary_event=saved" in memory_logs.text
    assert "cursor_before=0, cursor_after=4" in memory_logs.text
    assert "batch_count=4" in memory_logs.text
    assert "secret-" not in memory_logs.text


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_failed_optional_summary_does_not_block_or_log_saved(memory_db, memory_graph, monkeypatch, memory_logs, invalid):
    from app.consultation import service
    from app.errors.exceptions import LLMServiceException
    from app.infrastructure.llm.gateway import LLMGateway

    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "2")
    async def summarize(*args, **kwargs):
        if invalid:
            return ""
        raise LLMServiceException(detail="secret-error")
    monkeypatch.setattr(LLMGateway, "generate", summarize)
    sid = "optional-summary"
    state = await start(memory_graph, memory_db, sid)
    first = await service.process_message(sid, "第一轮", state, "FactDigger", db=memory_db, transport="http", idempotency_key="first")
    memory_logs.clear()
    second = await service.process_message(sid, "第二轮", first.result_state, "FactDigger", db=memory_db, transport="http", idempotency_key="second")
    assert second.command_applied is True
    assert second.error is None
    assert second.result_state["memory"]["summary"]["status"] == "failed"
    assert second.result_state["memory"]["summary"]["through_sequence"] == 0
    assert "summary_event=failed" in memory_logs.text
    assert "summary_event=saved" not in memory_logs.text
    assert "secret-error" not in memory_logs.text


@pytest.mark.asyncio
async def test_consent_skip_reports_existing_cursor_without_read_or_generation(memory_logs):
    from app.consultation.memory.coordinator import refresh_memory

    await refresh_memory(None, None, "secret-session", {"consent_given": False,
        "memory": {"summary": {"through_sequence": 20, "version": 3, "text": "secret-summary"}}})
    assert "summary_event=skipped" in memory_logs.text
    assert "cursor_before=20, cursor_after=20, version=3" in memory_logs.text
    assert "secret-" not in memory_logs.text


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_invalid_case_candidate_logs_rejection_for_first_and_existing_memory(existing, monkeypatch, memory_logs):
    from copy import deepcopy

    from app.consultation.agents import fact_digger
    from app.consultation.memory.case import merge_case_memory, project_facts
    from app.consultation.schemas.artifacts import ArtifactSource

    async def extract(*args):
        return {"evidence_mentioned": ["secret-invalid-field"]}, ArtifactSource.CONTENT_JSON
    monkeypatch.setattr(fact_digger, "_extract_structured_facts", extract)
    case = merge_case_memory({}, {"incident_location": "secret-previous-field"}, "secret-source") if existing else {}
    memory = {"case": case, "summary": {"text": "secret-summary", "through_sequence": 42, "version": 3}}
    state = {"current_input": "合成正常输入", "current_message_id": "secret-source", "facts_raw": [],
             "facts_structured": project_facts(case), "memory": deepcopy(memory), "conversation_history": []}
    result = await fact_digger.fact_intake_node(state)
    assert result["memory"] == memory
    assert result["facts_structured"] == project_facts(case)
    assert result["artifact_results"]["fact"]["degraded_reason"] == "schema_validation_failed"
    assert result["current_input"] is None
    assert "case_candidate_event=rejected" in memory_logs.text
    assert "error_code=schema_validation_failed" in memory_logs.text
    assert "validation_error_count=" in memory_logs.text
    assert "secret-" not in memory_logs.text


@pytest.mark.asyncio
async def test_required_intake_typed_failure_is_propagated_without_candidate_event(monkeypatch, memory_logs):
    from app.consultation.agents import fact_digger
    from app.errors.exceptions import LLMServiceException

    async def extract(*args):
        raise LLMServiceException(detail="secret-llm-error")
    monkeypatch.setattr(fact_digger, "_extract_structured_facts", extract)
    state = {"current_input": "合成正常输入", "facts_raw": [], "facts_structured": {}, "conversation_history": []}
    with pytest.raises(LLMServiceException) as caught:
        await fact_digger.fact_intake_node(state)
    assert caught.value.detail == "secret-llm-error"
    assert state["current_input"] == "合成正常输入"
    assert state["facts_raw"] == []
    assert "case_candidate_event=" not in memory_logs.text
    assert "secret-" not in memory_logs.text
