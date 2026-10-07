"""在实际供应商输入边界验证原始案情及 PII 透传，所有资料均为合成。"""

import json

import pytest
from langchain_core.messages import AIMessage

from app.consultation.agents import fact_digger, law_ref, legal_research
from app.consultation.memory.case import merge_case_memory
from app.consultation.memory.context import node_context
from app.infrastructure.llm.gateway import LLMGateway, chat_model_factory
from app.knowledge.law_knowledge import load_criminal_law_data

FACTS = [
    "昨晚我喝了酒，在城市道路上驾驶小汽车……",
    "没有提供任何案件发生经过",
    "我没有拿走任何物品",
    "行程记录显示我没有进入商店",
    "马路上没有发生碰撞。",
]
REVIEW_CASES = [
    "张三住在青禾市明月区长宁路18号。",
    "张三正在提供行程记录。",
    "我被张三打了，没有还手。",
    "我和李晓明一起到店里买东西。",
    "欧阳明月正在核对记录。",
]
INPUT_FACTS = [*FACTS, *REVIEW_CASES]
PII = "我叫张三，电话13800138000，身份证110101199003071234及110101900307123，车牌京A12345，住在青禾市明月区长宁路18号"


class SyntheticProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []
        self.model = "synthetic-provider"

    def bind_tools(self, _tools):
        return self

    def bind(self, **_kwargs):
        return self

    async def ainvoke(self, messages):
        self.messages.append(messages)
        return next(self.responses)


def assert_raw_messages(messages):
    serialized = "\n".join(message.content for message in messages)
    for fact in INPUT_FACTS:
        assert fact in serialized
    assert PII in serialized
    for marker in ("[NAME-MASKED]", "[PHONE-MASKED]", "[ID-MASKED]", "[ADDR-MASKED]", "[VEHICLE-MASKED]"):
        assert marker not in serialized


@pytest.mark.asyncio
async def test_lawref_tool_loop_and_extract_receive_original_facts(monkeypatch):
    monkeypatch.setenv("LAW_AGENT_FINAL_PROTOCOL", "legacy")
    provider = SyntheticProvider([
        AIMessage(content="", tool_calls=[{"name": "search_laws", "args": {"query": "盗窃"}, "id": "search-synthetic"}]),
        AIMessage(content="", tool_calls=[{"name": "get_article", "args": {"article_id": "第264条"}, "id": "read-synthetic"}]),
        AIMessage(content=json.dumps({"article_ids": ["第264条"], "matched_elements": {"第264条": []}, "confidence": "low"})),
        AIMessage(content='{"charges": []}'),
    ])
    monkeypatch.setattr(chat_model_factory, "create_precise_model", lambda *_: provider)
    facts = {"behavior_sequence": [*INPUT_FACTS, PII], "consequence": None}
    result = await legal_research.run_legal_research(facts, None, load_criminal_law_data())
    assert result.termination_reason == "final_answer"
    for messages in provider.messages:
        # legacy 首次事实是 JSON 字符串内嵌 JSON，需要还原外层转义再检查。
        human = next(message for message in messages if message.type == "human" and message.content.startswith('{"facts"'))
        content = json.loads(human.content)
        received = content["facts"]
        received = json.loads(received) if isinstance(received, str) else received
        assert received["behavior_sequence"] == [*INPUT_FACTS, PII]
    await law_ref.extract_structured_laws([{"article_number": "第264条", "content": "合成法条"}], facts)
    assert_raw_messages(provider.messages[-1])


@pytest.mark.asyncio
async def test_memory_intake_and_gateway_context_preserve_denial_and_evidence(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "ALIYUN")
    candidate = {"incident_time": None, "incident_location": None, "parties": [],
                 "behavior_sequence": [{"action": fact} for fact in INPUT_FACTS], "consequence": None,
                 "evidence_mentioned": [{"type": "记录", "description": FACTS[-1]}],
                 "arrest_status": None, "surrender": None, "victim_forgiveness": None, "prior_record": None}
    provider = SyntheticProvider([AIMessage(content=json.dumps(candidate, ensure_ascii=False)), AIMessage(content="合成回忆")])
    monkeypatch.setattr(chat_model_factory, "create_precise_model", lambda *_: provider)
    content = "；".join([*INPUT_FACTS, PII])
    state = await fact_digger.fact_intake_node({"session_id": "synthetic-pii", "current_input": content,
        "facts_raw": [], "facts_structured": {}, "conversation_history": []})
    assert state["facts_raw"] == ["；".join([*INPUT_FACTS, PII])]
    assert state["facts_structured"]["behavior_sequence"] == [{"action": fact} for fact in INPUT_FACTS]
    assert_raw_messages(provider.messages[0])
    memory = {"summary": {"text": content},
        "case": merge_case_memory({}, {"behavior_sequence": [*INPUT_FACTS, PII]}, "source-1"),
        "recent": [{"id": "source-1", "sequence": 1, "content": content, "sender_type": "user"}]}
    with node_context("law_ref", {"memory": memory}):
        assert await LLMGateway().generate("合成回忆任务", "请回忆") == "合成回忆"
    assert_raw_messages(provider.messages[1])


@pytest.mark.asyncio
async def test_incremental_summary_receives_original_content_from_raw_audit(memory_db, memory_graph, monkeypatch):
    from app.consultation import service
    from app.consultation.memory.coordinator import refresh_memory
    from tests.consultation.test_memory_persistence import start

    monkeypatch.setenv("MEMORY_RECENT_MESSAGES", "2")
    provider = SyntheticProvider([AIMessage(content="合成摘要")])
    monkeypatch.setattr(chat_model_factory, "create_precise_model", lambda *_: provider)
    sid = "synthetic-pii-summary"
    await start(memory_graph, memory_db, sid)
    content = "；".join([*INPUT_FACTS, PII])
    await service.record_external_exchange(sid, db=memory_db, key="first", input_content=content, output="合成回复")
    await service.record_external_exchange(sid, db=memory_db, key="second", input_content="新增陈述", output="合成回复")
    state = await service.get_session_state(sid)
    await refresh_memory(service, memory_db, sid, state)
    state = await service.get_session_state(sid)
    assert state["memory"]["summary"]["text"] == "合成摘要"
    assert len(provider.messages) == 1
    payload = json.loads(provider.messages[0][-1].content)
    assert payload["new_messages"][0]["content"] == "；".join([*INPUT_FACTS, PII])


# 复用真实 SQLite 和真实图的生命周期 fixture，不复用测试方法。
from tests.consultation.test_memory_persistence import memory_db as memory_db  # noqa: E402
from tests.consultation.test_memory_persistence import memory_graph as memory_graph  # noqa: E402
