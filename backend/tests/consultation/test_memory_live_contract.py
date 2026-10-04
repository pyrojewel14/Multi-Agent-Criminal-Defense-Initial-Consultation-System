"""真实模型试跑暴露的 memory 供应商契约，使用确定性响应验证边界。"""

import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from app.consultation.agents import fact_digger
from app.infrastructure.llm.factory import chat_model_factory
from app.infrastructure.llm.gateway import LLMGateway
from tests.consultation.test_memory_persistence import memory_db as memory_db
from tests.consultation.test_memory_persistence import memory_graph as memory_graph


@pytest.mark.asyncio
async def test_ollama_memory_candidate_uses_schema_before_case_merge(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "OLLAMA")
    facts = {"incident_time": "昨天", "incident_location": "甲城", "parties": [],
             "behavior_sequence": [], "consequence": None,
             "evidence_mentioned": [{"type": "监控", "description": "商店监控"}],
             "arrest_status": None, "surrender": None, "victim_forgiveness": None, "prior_record": None}
    seen = []

    async def provider(**kwargs):
        seen.append(kwargs)
        if kwargs.get("response_schema"):
            return {"content": json.dumps(facts, ensure_ascii=False), "has_tool_call": False, "tool_calls": []}
        invalid = dict(facts, evidence_mentioned=["商店监控"])
        return {"content": "", "has_tool_call": True,
                "tool_calls": [{"name": "extract_case_facts", "args": invalid}]}

    monkeypatch.setattr(fact_digger.llm_gateway, "generate_with_tools", provider)
    state = await fact_digger.fact_intake_node({"session_id": "synthetic-schema", "current_input": "昨天在甲城，有商店监控",
        "current_message_id": "source-1", "facts_raw": [], "facts_structured": {}, "conversation_history": []})
    assert state["artifact_results"]["fact"]["status"] == "success"
    assert state["artifact_results"]["fact"]["source"] == "content_json"
    assert state["facts_structured"]["evidence_mentioned"] == facts["evidence_mentioned"]
    assert seen[0]["tools"] == []
    evidence = seen[0]["response_schema"]["properties"]["evidence_mentioned"]
    assert evidence["items"]["type"] == "object"


@pytest.mark.asyncio
async def test_summary_can_disable_ollama_thinking_without_mutating_cached_model(monkeypatch):
    cached = ChatOllama(model="synthetic", reasoning=True)
    monkeypatch.setattr(chat_model_factory, "create_precise_model", lambda *_: cached)
    seen = []

    async def invoke(model, messages, **kwargs):
        seen.append(model)
        return AIMessage(content="案件陈述待核实", usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18})

    monkeypatch.setattr(ChatOllama, "ainvoke", invoke)
    result = await LLMGateway().generate("摘要", "合并新增消息", output_limit=333, reasoning=False)
    assert result == "案件陈述待核实"
    assert seen[0].reasoning is False
    assert seen[0].num_predict == 333
    assert cached.reasoning is True


@pytest.mark.asyncio
async def test_coordinator_summary_explicitly_disables_thinking(monkeypatch):
    from types import SimpleNamespace

    from app.consultation.memory import coordinator

    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "2")
    rows = [{"id": f"s{i}", "sequence": i, "command_id": f"c{(i-1)//2}",
             "record_kind": "external", "sender_type": "user" if i % 2 else "agent", "content": "合成内容"}
            for i in range(1, 7)]
    monkeypatch.setattr(coordinator, "memory_rows", AsyncMock(return_value=rows))
    generate = AsyncMock(return_value="合成案件尚待核实。")
    monkeypatch.setattr(LLMGateway, "generate", generate)
    persist = AsyncMock()
    await coordinator.refresh_memory(SimpleNamespace(persist_state=persist), object(), "synthetic",
        {"consent_given": True, "consultation_id": "case"})
    assert generate.call_args.kwargs["reasoning"] is False
    assert persist.call_args.kwargs["projection_updates"]["memory"]["summary"]["through_sequence"] == 4


@pytest.mark.asyncio
async def test_invalid_candidate_and_failed_summary_keep_raw_old_case_and_pending(memory_db, memory_graph, monkeypatch):
    from sqlalchemy import select

    from app.consultation import service, workflow
    from app.consultation.schemas.artifacts import ArtifactSource
    from app.models import ConsultationMessage
    from tests.consultation.test_memory_persistence import start

    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "2")
    monkeypatch.setattr(workflow, "_fact_intake_workflow_node", fact_digger.fact_intake_node)
    candidate = dict(incident_time="昨天", incident_location="甲城", parties=[], behavior_sequence=[],
                     consequence=None, evidence_mentioned=[], arrest_status=None,
                     surrender=None, victim_forgiveness=None, prior_record=False)
    calls = 0

    async def extract(rows):
        nonlocal calls
        calls += 1
        if calls == 2:
            return dict(candidate, incident_location="乙城", evidence_mentioned=["非法字符串"]), ArtifactSource.CONTENT_JSON
        return candidate, ArtifactSource.CONTENT_JSON

    monkeypatch.setattr(fact_digger, "_extract_structured_facts", extract)
    monkeypatch.setattr(LLMGateway, "generate", AsyncMock(side_effect=TimeoutError("synthetic summary timeout")))
    sid = "synthetic-candidate-summary-failure"
    state = await start(memory_graph, memory_db, sid)
    first = await service.process_message(sid, "昨天在甲城，没有前科", state, "FactDigger", db=memory_db,
                                          transport="http", idempotency_key="first")
    old_case = first.result_state["memory"]["case"]
    second = await service.process_message(sid, "补充地点乙城和监控", first.result_state, "FactDigger", db=memory_db,
                                           transport="websocket", idempotency_key="second")
    assert second.error is None
    assert second.result_state["memory"]["case"] == old_case
    assert second.result_state["artifact_results"]["fact"]["status"] == "degraded"
    assert second.result_state["facts_structured"]["incident_location"] == "甲城"
    assert second.result_state["facts_structured"]["prior_record"] is False
    assert second.result_state["memory"]["summary"] == {
        "text": "", "version": 0, "through_sequence": 0, "status": "failed"}
    assert (await memory_graph.get_snapshot(sid)).next == ("fact_intake",)
    rows = (await memory_db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
    assert [row.content for row in rows if row.sender_type == "user"] == ["昨天在甲城，没有前科", "补充地点乙城和监控"]
    assert len(rows) == 4
    monkeypatch.setattr(LLMGateway, "generate", AsyncMock(return_value="用户陈述尚待核实。"))
    third = await service.process_message(sid, "再次陈述甲城", second.result_state, "FactDigger", db=memory_db,
                                          transport="http", idempotency_key="third")
    assert third.error is None
    assert third.result_state["memory"]["summary"]["through_sequence"] == 4
    assert third.result_state["memory"]["summary"]["version"] == 1


def test_case_context_keeps_conflicts_and_corrections_without_audit_metadata():
    from app.consultation.memory.context import ContextBuilder

    memory = {"case": {"fields": {
        "incident_location": {"value": "丙城", "status": "user_corrected_unverified",
            "source_ids": ["raw-source-current"], "updated_at": "synthetic-timestamp", "alternatives": [
                {"value": "甲城", "status": "conflicted", "source_ids": ["raw-source-old"]}]},
        "prior_record": {"value": False, "status": "user_claim_unverified", "source_ids": ["raw-source-current"]},
        "parties": {"items": [{"value": {"name": "甲"}, "status": "conflicted", "source_ids": ["party-source"],
            "alternatives": [{"value": {"name": "乙"}, "status": "user_claim_unverified"}]}]}}}}
    before = json.dumps(memory, ensure_ascii=False)
    built = ContextBuilder().build("系统规则", "当前问题", {"memory": memory})
    context = "\n".join(message.content for message in built.messages)
    assert all(value in context for value in ("丙城", "甲城", "conflicted", "user_corrected_unverified", 'false', "乙"))
    assert all(value not in context for value in ("raw-source", "synthetic-timestamp", "party-source", "updated_at"))
    assert "未核实" in context
    assert json.dumps(memory, ensure_ascii=False) == before


def test_redaction_only_scalar_cannot_become_a_new_or_conflicting_fact():
    from app.consultation.memory.case import merge_case_memory

    old = merge_case_memory({}, {"incident_location": "甲城"}, "source-1")
    new = merge_case_memory(old, {"incident_location": "[NAME-MASKED]", "incident_time": "[ID-MASKED]"}, "source-2")
    assert new == old
    assert "incident_time" not in new["fields"]
    # 带有其他实际信息的脱敏陈述仍保留，不能删除有效描述或推断真实身份。
    description = merge_case_memory(old, {"arrest_status": "[NAME-MASKED]未被拘留"}, "source-3")
    assert description["fields"]["arrest_status"]["value"] == "[NAME-MASKED]未被拘留"
