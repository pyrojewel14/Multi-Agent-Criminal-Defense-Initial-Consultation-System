"""Bounded LawRef tool loop and downstream contract regressions."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.legal_research import run_legal_research
from app.agents.fact_digger import fact_coverage_node
from app.agents.law_ref import LawSearchResults, law_ref_node, load_criminal_law_data
from app.observability.tracing import trace_store
from app.orchestrator.workflow import check_facts_sufficient
from tests.factories import make_consultation_state


FACTS = {"behavior_sequence": ["盗窃"], "consequence": "财物损失"}


def decision(name, args):
    return {"content": "", "tool_calls": [{"name": name, "args": args}], "has_tool_call": True}


def final(article_id="第264条"):
    return {
        "content": json.dumps({
            "article_ids": [article_id],
            "matched_elements": {article_id: ["盗窃公私财物"]},
            "confidence": "medium",
        }, ensure_ascii=False),
        "tool_calls": [],
        "has_tool_call": False,
    }


@pytest.mark.asyncio
async def test_research_search_article_final_produces_workflow_contract():
    responses = [
        decision("search_laws", {"query": "盗窃"}),
        decision("get_article", {"article_id": "第264条"}),
        final(),
    ]
    trace_before = len(trace_store.events())
    with patch("app.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=responses), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]):
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
    with patch("app.agents.fact_digger._generate_fact_summary", new_callable=AsyncMock, return_value="事实摘要"):
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
    with patch("app.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=responses), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]):
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
    with patch("app.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=responses), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=LawSearchResults(dependency_failed=True)):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data())

    assert result.termination_reason == "final_answer"
    assert result.trajectory[0].tool_status == "partial_dependency_failure"
    assert result.trajectory[0].tool_result_summary["rag_dependency_failed"] is True
    assert result.failed_tool_call_count == 1


@pytest.mark.asyncio
async def test_research_stops_repeated_identical_call():
    repeated = decision("search_laws", {"query": " 盗窃 "})
    with patch("app.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=[decision("search_laws", {"query": "盗窃"}), repeated, repeated]), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data())

    assert result.termination_reason == "duplicate_call"
    assert result.duplicate_tool_call_count == 1
    assert result.tool_call_count == 2
    assert result.step_count == 2


@pytest.mark.asyncio
async def test_research_max_steps_stops_without_false_success():
    responses = [decision("search_laws", {"query": "盗窃"}), decision("search_laws", {"query": "财物"}), final()]
    with patch("app.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, side_effect=responses), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data(), max_steps=2)

    assert result.termination_reason == "max_steps"
    assert result.step_count == 2
    assert result.candidate_laws == []


@pytest.mark.asyncio
@pytest.mark.parametrize("call,status", [
    (decision("shell", {"command": "echo hi"}), "unknown_tool"),
    (decision("search_laws", {"query": 42}), "invalid_arguments"),
])
async def test_research_rejects_invalid_tool_calls(call, status):
    with patch("app.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, return_value=call):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data(), max_steps=1)

    assert result.termination_reason == "max_steps"
    assert result.trajectory[0].tool_status == status
    assert result.successful_tool_call_count == 0
    assert result.failed_tool_call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,reason", [(TimeoutError(), "tool_timeout"), (RuntimeError("offline"), "dependency_failure")])
async def test_research_tool_failure_is_controlled(failure, reason):
    with patch("app.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, return_value=decision("search_laws", {"query": "盗窃"})), \
         patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, side_effect=failure):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data(), max_steps=2)

    assert result.termination_reason == reason
    assert result.candidate_laws == []
    assert result.failed_tool_call_count == 1


@pytest.mark.asyncio
async def test_research_empty_search_is_observed_and_degraded():
    with patch("app.agents.legal_research.llm_gateway.generate_with_tools", new_callable=AsyncMock, return_value=decision("search_laws", {"query": "不存在的罪名"})), \
         patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]):
        result = await run_legal_research(FACTS, "user-1", load_criminal_law_data(), max_steps=1)

    assert result.trajectory[0].tool_status == "empty"
    assert result.termination_reason == "max_steps"
    assert result.candidate_laws == []
