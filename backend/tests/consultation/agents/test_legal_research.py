"""Bounded LawRef tool loop and downstream contract regressions."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.consultation.agents.fact_digger import fact_coverage_node
from app.consultation.agents.law_ref import law_ref_node
from app.consultation.agents.legal_research import run_legal_research
from app.consultation.workflow import check_facts_sufficient
from app.infrastructure.observability.tracing import trace_store
from app.knowledge.law_knowledge import load_criminal_law_data
from app.knowledge.law_retrieval import LawSearchResults
from tests.factories import make_consultation_state

FACTS = {"behavior_sequence": ["盗窃"], "consequence": "财物损失"}


@pytest.fixture(autouse=True)
def candidate_final_protocol(monkeypatch):
    monkeypatch.setenv("LAW_AGENT_FINAL_PROTOCOL", "native_candidate")


@pytest.mark.asyncio
async def test_application_default_keeps_candidate_protocol_disabled(monkeypatch):
    monkeypatch.delenv("LAW_AGENT_FINAL_PROTOCOL", raising=False)
    captured = []

    async def respond(system, prompt, tools, **kwargs):
        captured.append((tools, kwargs))
        return [decision("search_laws", {"query": "盗窃"}), decision("get_article", {"article_id": "第264条"}), final()][len(captured)-1]

    with (
        patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", side_effect=respond),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user", load_criminal_law_data())
    assert result.termination_reason == "final_answer"
    assert len(captured[2][0]) == 3
    assert captured[2][1]["response_schema"] is None
    assert len(captured[2][1]["message_history"]) == 4


@pytest.mark.asyncio
async def test_model_receives_assistant_calls_and_linked_tool_results():
    from langchain_core.messages import AIMessage, ToolMessage

    histories = []

    async def respond(system, prompt, tools, **kwargs):
        histories.append(list(kwargs.get("message_history", [])))
        return [decision("search_laws", {"query": "盗窃"}), decision("get_article", {"article_id": "第264条"}), final()][len(histories) - 1]

    with (
        patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", side_effect=respond),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user", load_criminal_law_data())
    assert result.termination_reason == "final_answer"
    assert len(histories[1]) == 2
    assert isinstance(histories[1][0], AIMessage)
    assert isinstance(histories[1][1], ToolMessage)
    assert histories[1][0].tool_calls[0]["id"] == histories[1][1].tool_call_id
    assert json.loads(histories[1][1].content)["result"]["candidates"][0]["article_id"] == "第264条"
    assert histories[2] == []


@pytest.mark.asyncio
async def test_final_generation_uses_only_read_articles_and_constrained_elements():
    captured = []

    async def respond(system, prompt, tools, **kwargs):
        captured.append((json.loads(prompt), tools, kwargs))
        return [decision("search_laws", {"query": "盗窃"}), decision("get_article", {"article_id": "第264条"}), final()][len(captured)-1]

    with (
        patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", side_effect=respond),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user", load_criminal_law_data())
    assert result.termination_reason == "final_answer"
    payload, tools, kwargs = captured[2]
    assert tools == []
    assert [a["article_id"] for a in payload["read_articles"]] == ["第264条"]
    schema = kwargs["response_schema"]
    assert schema["properties"]["article_ids"]["items"]["enum"] == ["第264条"]
    assert schema["properties"]["matched_elements"]["additionalProperties"] is False
    assert schema["properties"]["matched_elements"]["properties"]["第264条"]["items"]["enum"] == ["盗窃公私财物", "数额较大或者具备法定盗窃情形之一"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_content,reason", [
    ("", "empty_output"), ("not json", "invalid_json"), ("{}", "schema_error"),
    (json.dumps({"article_ids": ["第9999条"], "matched_elements": {}, "confidence": "low"}), "unknown_article"),
    (json.dumps({"article_ids": ["第264条"], "matched_elements": {"第264条": ["invented"]}, "confidence": "low"}), "unknown_element"),
])
async def test_rejected_final_has_specific_reason_and_one_contextual_retry(bad_content, reason):
    prompts = []

    async def respond(system, prompt, tools, **kwargs):
        prompts.append((prompt, list(kwargs.get("message_history", []))))
        return [decision("search_laws", {"query": "盗窃"}), decision("get_article", {"article_id": "第264条"}), {"content": bad_content, "tool_calls": []}, final()][len(prompts)-1]

    with (
        patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", side_effect=respond),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user", load_criminal_law_data())
    assert result.termination_reason == "final_answer"
    assert result.trajectory[2].tool_result_summary["rejection_reason"] == reason
    assert len(prompts[3][1]) == 2
    assert prompts[3][1][0].content == bad_content
    assert reason in prompts[3][1][1].content


@pytest.mark.asyncio
async def test_final_retry_is_bounded_and_never_accepts_unknown_elements():
    replies = [decision("search_laws", {"query": "盗窃"}), decision("get_article", {"article_id": "第264条"})]
    invalid = {"content": "{}", "tool_calls": []}
    with (
        patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=replies + [invalid]*6),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user", load_criminal_law_data(), max_steps=8)
    assert result.termination_reason == "invalid_final"
    assert result.step_count == 4
    assert result.candidate_laws == []


@pytest.mark.asyncio
async def test_final_rejects_searched_but_unread_article():
    rejected = {"content": json.dumps({"article_ids": ["第234条"], "matched_elements": {}, "confidence": "low"}), "tool_calls": []}
    with (
        patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=[decision("search_laws", {"query": "盗窃"}), decision("search_laws", {"query": "故意伤害"}), decision("get_article", {"article_id": "第264条"}), rejected]),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user", load_criminal_law_data(), max_steps=4)
    assert result.candidate_laws == []
    assert result.trajectory[-1].tool_result_summary["rejection_reason"] == "unread_article"


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected,reason", [
    ({"content": '{"article_ids":["第264条"],"matched_elements":{},"confidence":"low"}', "tool_calls": [], "response_metadata": {"done_reason": "length"}}, "output_truncated"),
    ({"content": "", "tool_calls": [{"name": "search_laws", "args": {"query": "盗窃"}}]}, "unexpected_tool_call"),
])
async def test_final_rejects_truncation_and_tool_calls_without_executing_them(rejected, reason):
    with (
        patch("app.consultation.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=[decision("search_laws", {"query": "盗窃"}), decision("get_article", {"article_id": "第264条"}), rejected]),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user", load_criminal_law_data(), max_steps=3)
    assert result.candidate_laws == []
    assert result.successful_tool_call_count == 2
    assert result.trajectory[-1].tool_result_summary["rejection_reason"] == reason


def decision(name, args):
    return {"content": "", "tool_calls": [{"name": name, "args": args}], "has_tool_call": True}


def final(article_id="第264条"):
    return {
        "content": json.dumps(
            {
                "article_ids": [article_id],
                "matched_elements": {article_id: ["盗窃公私财物"]},
                "confidence": "medium",
            },
            ensure_ascii=False,
        ),
        "tool_calls": [],
        "has_tool_call": False,
    }


@pytest.mark.asyncio
async def test_unreviewed_full_corpus_article_cannot_enter_applied_laws():
    article = {
        "article_number": "第一百三十三条之一",
        "title": "危险驾驶罪",
        "content": "在道路上驾驶机动车，追逐竞驶，情节恶劣的。",
        "elements": [{"key": "circumstances_vicious", "name": "情节恶劣"}],
        "base_sentence": "处拘役",
        "charge_tags": ["危险驾驶"],
        "common_keywords": ["醉驾"],
        "annotation_source": "manual-title-v1",
        "annotation_layer": "manual-title-v1",
    }
    law_data = {"chapters": [{"chapter": "第二章", "articles": [article]}]}
    rag_hit = {"article_number": "第133条之一", "content": article["content"], "data_source": "rag_unverified"}
    responses = [
        decision("search_laws", {"query": "危险驾驶"}),
        decision("get_article", {"article_id": "第133条之一"}),
        {
            "content": json.dumps(
                {"article_ids": ["第133条之一"], "matched_elements": {"第133条之一": ["情节恶劣"]}, "confidence": "high"},
                ensure_ascii=False,
            ),
            "tool_calls": [],
            "has_tool_call": False,
        },
    ]
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            side_effect=responses,
        ),
        patch("app.consultation.agents.law_ref.load_criminal_law_data", return_value=law_data),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[rag_hit]),
    ):
        state = make_consultation_state(facts_structured={"behavior_sequence": ["危险驾驶"]}, applied_laws=[])
        result = await law_ref_node(state)

    assert "applied_laws" in result and "law_search_status" in result
    assert result["applied_laws"] == []
    assert result["law_search_status"] != "success"


@pytest.mark.asyncio
async def test_research_search_article_final_produces_workflow_contract():
    responses = [
        decision("search_laws", {"query": "盗窃"}),
        decision("get_article", {"article_id": "第264条"}),
        final(),
    ]
    trace_before = len(trace_store.events())
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            side_effect=responses,
        ),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        state = make_consultation_state(facts_structured=FACTS, applied_laws=[])
        result = await law_ref_node(state)

    assert result["law_search_status"] == "success", json.dumps(result["law_research"], ensure_ascii=False)
    assert result["applied_laws"][0]["article_number"] == "第二百六十四条"
    assert result["applied_laws"][0]["data_source"] == "json_keyword"
    assert result["applied_laws"][0]["required_elements"]
    assert result["law_research"]["tool_call_count"] == 2
    assert [step["tool_name"] for step in result["law_research"]["trajectory"]] == ["search_laws", "get_article", None]
    assert "盗窃" not in json.dumps(result["law_research"], ensure_ascii=False)
    events = trace_store.events()[trace_before:]
    assert [event.name for event in events if event.event_type == "law_tool"] == ["search_laws", "get_article"]
    assert len([event for event in events if event.event_type == "law_agent_step"]) == 3
    result["facts_structured"] = FACTS | {"theft_property": "财物", "theft_threshold": "达到法定情形"}
    with patch(
        "app.consultation.agents.fact_digger._generate_fact_summary", new_callable=AsyncMock, return_value="事实摘要"
    ):
        covered = await fact_coverage_node(result)
    assert covered["facts_coverage_rate"] == 1.0
    assert check_facts_sufficient(covered) == "complete"


@pytest.mark.asyncio
async def test_research_search_elements_then_final():
    responses = [
        decision("search_laws", {"query": "盗窃"}),
        decision("search_elements", {"article_id": "第264条"}),
        final(),
    ]
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            side_effect=responses,
        ),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data())

    assert result.termination_reason == "final_answer"
    assert result.matched_elements["第264条"] == ["盗窃公私财物"]
    assert result.missing_elements["第264条"] == ["数额较大或者具备法定盗窃情形之一"]


@pytest.mark.asyncio
async def test_keyword_candidate_keeps_partial_rag_failure_visible():
    responses = [
        decision("search_laws", {"query": "盗窃"}),
        decision("get_article", {"article_id": "第264条"}),
        final(),
    ]
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            side_effect=responses,
        ),
        patch(
            "app.knowledge.law_retrieval.search_laws_by_rag",
            new_callable=AsyncMock,
            return_value=LawSearchResults(dependency_failed=True),
        ),
    ):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data())

    assert result.termination_reason == "final_answer"
    assert result.trajectory[0].tool_status == "partial_dependency_failure"
    assert result.trajectory[0].tool_result_summary["rag_dependency_failed"] is True
    assert result.failed_tool_call_count == 1


@pytest.mark.asyncio
async def test_research_stops_repeated_identical_call():
    repeated = decision("search_laws", {"query": " 盗窃 "})
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            side_effect=[decision("search_laws", {"query": "盗窃"}), repeated, repeated],
        ),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data())

    assert result.termination_reason == "duplicate_call"
    assert result.duplicate_tool_call_count == 1
    assert result.tool_call_count == 2
    assert result.step_count == 2


@pytest.mark.asyncio
async def test_research_max_steps_stops_without_false_success():
    responses = [decision("search_laws", {"query": "盗窃"}), decision("search_laws", {"query": "财物"}), final()]
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            side_effect=responses,
        ),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data(), max_steps=2)

    assert result.termination_reason == "max_steps"
    assert result.step_count == 2
    assert result.candidate_laws == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call,status",
    [
        (decision("shell", {"command": "echo hi"}), "unknown_tool"),
        (decision("search_laws", {"query": 42}), "invalid_arguments"),
    ],
)
async def test_research_rejects_invalid_tool_calls(call, status):
    with patch(
        "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
        new_callable=AsyncMock,
        return_value=call,
    ):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data(), max_steps=1)

    assert result.termination_reason == "max_steps"
    assert result.trajectory[0].tool_status == status
    assert result.successful_tool_call_count == 0
    assert result.failed_tool_call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,reason", [(TimeoutError(), "tool_timeout"), (RuntimeError("offline"), "dependency_failure")]
)
async def test_research_tool_failure_is_controlled(failure, reason):
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            return_value=decision("search_laws", {"query": "盗窃"}),
        ),
        patch("app.knowledge.law_retrieval.search_laws_by_keyword", new_callable=AsyncMock, side_effect=failure),
    ):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data(), max_steps=2)

    assert result.termination_reason == reason
    assert result.candidate_laws == []
    assert result.failed_tool_call_count == 1


@pytest.mark.asyncio
async def test_research_empty_search_is_observed_and_degraded():
    with (
        patch(
            "app.consultation.agents.legal_research.llm_gateway.generate_with_tools",
            new_callable=AsyncMock,
            return_value=decision("search_laws", {"query": "不存在的罪名"}),
        ),
        patch("app.knowledge.law_retrieval.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
    ):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data(), max_steps=1)

    assert result.trajectory[0].tool_status == "empty"
    assert result.termination_reason == "max_steps"
    assert result.candidate_laws == []
