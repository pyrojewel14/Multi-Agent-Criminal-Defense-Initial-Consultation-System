import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.consultation.agents import fact_digger
from app.consultation.memory.case import merge_case_memory, project_facts
from app.consultation.memory.context import ContextBuilder, ContextLimitError, MemorySettings, estimate_tokens
from app.consultation.memory.summary import advance_summary


def test_case_memory_preserves_empty_deduplicates_and_keeps_conflicts():
    first = merge_case_memory({}, {"incident_location": "甲城", "parties": [{"name": "甲"}]}, "msg-1")
    second = merge_case_memory(first, {"incident_location": None, "parties": None}, "msg-2")
    assert project_facts(second)["incident_location"] == "甲城"
    third = merge_case_memory(second, {"incident_location": "乙城", "parties": [{"name": "甲"}]}, "msg-3")
    assert len(project_facts(third)["parties"]) == 1
    assert project_facts(third)["incident_location"] == "甲城"
    assert third["fields"]["incident_location"]["status"] == "conflicted"
    assert third["fields"]["incident_location"]["alternatives"][0]["source_ids"] == ["msg-3"]
    assert third["fields"]["parties"]["items"][0]["source_ids"] == ["msg-1", "msg-3"]
    corrected = merge_case_memory(third, {"incident_location": "乙城"}, "msg-4", correction=True)
    assert project_facts(corrected)["incident_location"] == "乙城"
    assert corrected["fields"]["incident_location"]["status"] == "user_corrected_unverified"
    assert corrected["fields"]["incident_location"]["alternatives"][0]["value"] == "甲城"


def test_context_is_bounded_and_never_splits_tool_pairs():
    settings = MemorySettings(recent_messages=4, context_token_budget=1200, model_window=1800, output_reserve=300)
    history = [AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "t1"}]),
               ToolMessage(content="字" * 500, tool_call_id="t1"),
               AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "t2"}]),
               ToolMessage(content="证据来源乙", tool_call_id="t2")]
    context = ContextBuilder(settings).build("system", "current", {}, history=history)
    ids = [message.tool_call_id for message in context.messages if isinstance(message, ToolMessage)]
    calls = [call["id"] for message in context.messages if isinstance(message, AIMessage) for call in message.tool_calls]
    assert ids == calls == ["t2"]
    assert context.estimated_input_tokens <= 1200
    assert context.dropped_groups == 1


def test_long_single_input_is_rejected_without_truncating():
    with pytest.raises(ContextLimitError):
        ContextBuilder(MemorySettings(context_token_budget=600, model_window=1800, output_reserve=300)).build(
            "system", "字" * 1000, {})


@pytest.mark.asyncio
async def test_summary_incremental_success_failure_and_whole_message_budget():
    observed = []
    async def summarize(old, rows):
        observed.append((old, [row["sequence"] for row in rows]))
        return "增量摘要"
    settings = MemorySettings(recent_messages=2, summary_input_budget=1000, summary_output_budget=100)
    rows = [{"sequence": i, "content": f"内容{i}", "record_kind": "external", "sender_type": "user"} for i in range(1, 7)]
    memory = await advance_summary({}, rows, summarize, settings)
    assert memory["summary"]["through_sequence"] == 4
    assert memory["summary"]["version"] == 1
    assert len(memory["recent"]) == 2
    rows2 = rows + [{"sequence": 7, "content": "新轮", "record_kind": "external", "sender_type": "user"}]
    memory = await advance_summary(memory, rows2, summarize, settings)
    assert observed == [("", [1, 2, 3, 4]), ("增量摘要", [5])]
    async def fail(*args):
        raise TimeoutError()
    failed = await advance_summary(memory, rows2 + [{"sequence": 8, "content": "再补充", "record_kind": "external", "sender_type": "user"}], fail, settings)
    assert failed["summary"]["through_sequence"] == 5
    assert failed["summary"]["version"] == 2
    assert failed["summary"]["status"] == "failed"
    assert len(failed["recent"]) == 2
    assert "内容1" not in str(failed["recent"])


@pytest.mark.asyncio
async def test_short_conversation_does_not_call_summary():
    async def forbidden(*args):
        pytest.fail("short conversation must not summarize")
    result = await advance_summary({}, [{"sequence": 1, "content": "新输入", "record_kind": "external"}], forbidden, MemorySettings())
    assert result["summary"]["through_sequence"] == 0
    assert len(result["recent"]) == 1


@pytest.mark.asyncio
async def test_intake_only_extracts_current_input_and_merges_null_lists(monkeypatch):
    captured = []
    async def extract(rows):
        from app.consultation.schemas.artifacts import ArtifactSource
        captured.extend(rows)
        return {"incident_time": None, "incident_location": "乙城", "parties": None,
                "behavior_sequence": [], "consequence": None, "evidence_mentioned": None,
                "arrest_status": None, "surrender": None, "victim_forgiveness": None, "prior_record": None}, ArtifactSource.TOOL_CALL
    monkeypatch.setattr(fact_digger, "_extract_structured_facts", extract)
    state = {"facts_raw": ["旧轮内容"], "facts_structured": {"incident_location": "甲城", "parties": [{"name": "甲"}]},
             "current_input": "新轮补充", "current_message_id": "msg-new", "conversation_history": []}
    result = await fact_digger.fact_intake_node(state)
    assert captured == ["新轮补充"]
    assert result["facts_structured"]["incident_location"] == "甲城"
    assert result["facts_structured"]["parties"] == [{"name": "甲"}]
    assert result["memory"]["case"]["fields"]["incident_location"]["status"] == "conflicted"
    assert result["current_input"] is None


@pytest.mark.asyncio
async def test_gateway_rejects_over_budget_before_model_call(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from app.errors.exceptions import LLMServiceException
    from app.infrastructure.llm.factory import chat_model_factory
    from app.infrastructure.llm.gateway import LLMGateway
    monkeypatch.setenv("MEMORY_CONTEXT_TOKEN_BUDGET", "300")
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=AIMessage(content="unexpected model call"))
    monkeypatch.setattr(chat_model_factory, "create_precise_model", lambda *args: model)
    with pytest.raises(LLMServiceException) as error:
        await LLMGateway().generate("system", "字" * 1000)
    assert error.value.detail == "context_budget_exceeded"
    model.ainvoke.assert_not_awaited()

@pytest.mark.asyncio
async def test_recent_and_summary_never_split_external_command_pair():
    rows = [{"sequence": 1, "id": "u1", "command_id": "turn1", "sender_type": "user", "record_kind": "external", "content": "问一"},
            {"sequence": 2, "id": "a1", "command_id": "turn1", "sender_type": "agent", "record_kind": "external", "content": "答一"},
            {"sequence": 3, "id": "u2", "command_id": "turn2", "sender_type": "user", "record_kind": "external", "content": "问二"},
            {"sequence": 4, "id": "a2", "command_id": "turn2", "sender_type": "agent", "record_kind": "external", "content": "答二"}]
    batches = []
    async def summarize(old, batch):
        batches.append([r["sequence"] for r in batch])
        return "先前一轮"
    result = await advance_summary({}, rows, summarize, MemorySettings(recent_messages=3))
    assert [r["sequence"] for r in result["recent"]] == [3, 4]
    assert batches == [[1, 2]]


def test_new_behavior_is_first_in_compatibility_projection_for_retrieval():
    memory = merge_case_memory({}, {"behavior_sequence": [{"action": "旧动作"}]}, "old")
    memory = merge_case_memory(memory, {"behavior_sequence": [{"action": "本轮决定性行为"}]}, "new")
    assert project_facts(memory)["behavior_sequence"][0]["action"] == "本轮决定性行为"
    assert len(project_facts(memory)["behavior_sequence"]) == 2

@pytest.mark.asyncio
async def test_summary_budget_never_compresses_half_a_command():
    rows = [{"sequence": 1, "command_id": "old", "record_kind": "external", "content": "短问题"},
            {"sequence": 2, "command_id": "old", "record_kind": "external", "content": "字" * 1000},
            {"sequence": 3, "command_id": "new", "record_kind": "external", "content": "新问题"},
            {"sequence": 4, "command_id": "new", "record_kind": "external", "content": "新回复"}]
    async def forbidden(*args):
        pytest.fail("overlarge whole command must remain uncovered")
    result = await advance_summary({}, rows, forbidden, MemorySettings(recent_messages=2, summary_input_budget=700))
    assert result["summary"]["through_sequence"] == 0
    assert result["summary"]["status"] == "blocked_large_message"


def test_context_reserves_latest_tool_group_before_optional_memory():
    settings = MemorySettings(context_token_budget=1100, model_window=1800, output_reserve=300)
    state = {"memory": {"summary": {"text": "字" * 150}}}
    pair = [AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "t"}]),
            ToolMessage(content="字" * 180, tool_call_id="t")]
    built = ContextBuilder(settings).build("system", "query", state, history=pair)
    assert isinstance(built.messages[-1], ToolMessage)
    assert built.messages[-1].tool_call_id == "t"
    assert not any("滚动摘要" in str(message.content) for message in built.messages)


def test_named_party_updates_keep_conflicting_versions_without_duplicate_party():
    memory = merge_case_memory({}, {"parties": [{"name": "甲", "role": "嫌疑人"}]}, "p1")
    memory = merge_case_memory(memory, {"parties": [{"name": "甲", "role": "被害人"}]}, "p2")
    items = memory["fields"]["parties"]["items"]
    assert len(items) == 1
    assert items[0]["status"] == "conflicted"
    assert items[0]["alternatives"][0]["source_ids"] == ["p2"]
    memory = merge_case_memory(memory, {"parties": [{"name": "甲", "role": "被害人"}]}, "p3", correction=True)
    assert project_facts(memory)["parties"] == [{"name": "甲", "role": "被害人"}]
    assert memory["fields"]["parties"]["items"][0]["status"] == "user_corrected_unverified"

@pytest.mark.asyncio
async def test_no_new_input_uses_existing_case_memory_without_extraction(monkeypatch):
    async def forbidden(*args):
        pytest.fail("no new input must not reinterpret an old user's statement")
    monkeypatch.setattr(fact_digger, "_extract_structured_facts", forbidden)
    case = merge_case_memory({}, {"incident_location": "甲城"}, "source-1")
    state = {"facts_raw": ["原陈述"], "facts_structured": project_facts(case), "memory": {"case": case},
             "current_input": None, "current_message_id": "source-1", "conversation_history": []}
    result = await fact_digger.fact_intake_node(state)
    assert result["facts_structured"]["incident_location"] == "甲城"

@pytest.mark.asyncio
async def test_long_raw_archive_becomes_shorter_active_context():
    rows = [{"id": f"id-{i}", "sequence": i, "command_id": f"turn-{(i-1)//2}", "record_kind": "external",
             "sender_type": "user" if i % 2 else "agent", "content": f"statement{i}:" + "合成内容" * 100} for i in range(1, 25)]
    async def summarize(old, batch):
        return "此前用户陈述尚未核实，等待补充。"
    settings = MemorySettings(recent_messages=4, context_token_budget=8000, summary_input_budget=6000)
    memory = await advance_summary({}, rows, summarize, settings)
    context = ContextBuilder(settings).build("案件分析指令", "当前轮任务", {"memory": memory})
    text = "\n".join(str(message.content) for message in context.messages)
    assert "statement1:" not in text
    assert "statement24:" in text
    assert context.estimated_input_tokens < estimate_tokens(rows) // 2
    assert memory["summary"]["through_sequence"] > 0
