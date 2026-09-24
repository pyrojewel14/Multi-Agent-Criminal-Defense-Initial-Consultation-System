"""四类 LLM 产物接入 Agent 状态机的回归测试。"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.fact_digger import fact_intake_node
from app.agents.law_ref import law_ref_node
from app.agents.risk_assessor import risk_assessor_node
from app.agents.service_planner import service_planner_node
from tests.factories import make_applied_law, make_consultation_state, make_risk_assessment


COMPLETE_FACT = {
    "incident_time": None,
    "incident_location": "北京市某区",
    "parties": [],
    "behavior_sequence": [],
    "consequence": "轻伤",
    "evidence_mentioned": [],
    "arrest_status": None,
    "surrender": None,
    "victim_forgiveness": None,
    "prior_record": None,
}

COMPLETE_RISK = {
    "predicted_sentence_range": "待律师核验",
    "mitigating_factors": [],
    "aggravating_factors": [],
    "compulsory_measure_risk": {
        "detention_status": "未知",
        "bail_possibility": "待评估",
        "measure_change_space": "待评估",
        "prolonged_detention_risk": "unknown",
    },
    "evidence_risk_points": [],
    "procedure_risks": [],
}

COMPLETE_SERVICE = {
    "service_plan": {
        "urgent_actions": {"immediate": [], "short_term": [], "follow_up": []},
        "defense_strategies": {"primary": "待律师制定", "alternatives": []},
        "service_phases": [],
        "fee_structure": {
            "recommended_plan": "面议",
            "total_fee_range": "以委托合同为准",
            "breakdown": {},
        },
    },
    "report_draft": "# 刑事辩护初期咨询报告\n待律师审核。",
}


@pytest.mark.asyncio
async def test_fact_tool_call_success_records_tool_call_source():
    state = make_consultation_state(facts_raw=["事实"], current_input="补充")
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": "",
                "tool_calls": [{"name": "extract_case_facts", "args": COMPLETE_FACT}],
                "has_tool_call": True,
            }
        )
        result = await fact_intake_node(state)

    assert result["artifact_results"]["fact"]["status"] == "success"
    assert result["artifact_results"]["fact"]["source"] == "tool_call"


@pytest.mark.asyncio
async def test_fact_content_json_success_records_content_json_source():
    state = make_consultation_state(facts_raw=["事实"], current_input="补充")
    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": json.dumps(COMPLETE_FACT, ensure_ascii=False),
                "tool_calls": [],
                "has_tool_call": False,
            }
        )
        result = await fact_intake_node(state)

    assert result["artifact_results"]["fact"]["status"] == "success"
    assert result["artifact_results"]["fact"]["source"] == "content_json"


@pytest.mark.asyncio
async def test_fact_missing_fields_preserves_previous_artifact_and_records_degraded():
    """Fact 缺字段时不得覆盖上一轮已验证事实。"""
    state = make_consultation_state(
        facts_raw=["既有事实"],
        facts_structured=COMPLETE_FACT,
        current_input="补充内容",
    )
    incomplete = {"incident_time": "2026-09-21", "consequence": "轻伤"}

    with patch("app.agents.fact_digger.llm_gateway") as mock_llm:
        mock_llm.generate_with_tools = AsyncMock(
            return_value={
                "content": "",
                "tool_calls": [{"name": "extract_case_facts", "args": incomplete}],
                "has_tool_call": True,
            }
        )
        result = await fact_intake_node(state)

    assert result["facts_structured"] == COMPLETE_FACT
    assert result["artifact_results"]["fact"]["status"] == "degraded"
    assert result["artifact_results"]["fact"]["source"] == "tool_call"
    assert result["degraded_reason"] == "schema_validation_failed"
    assert result["validation_errors"]


@pytest.mark.asyncio
async def test_law_valid_content_json_records_content_json_source():
    state = make_consultation_state(facts_structured=COMPLETE_FACT)
    candidate = make_applied_law(title="故意伤害罪", common_keywords=["伤害"])
    structured = [
        {
            "charge_name": "故意伤害罪",
            "article_number": candidate["article_number"],
            "elements_matched": [],
            "elements_missing": [],
            "base_sentence": candidate["base_sentence"],
            "probability": "medium",
        }
    ]
    with (
        patch("app.agents.law_ref.load_criminal_law_data", return_value={"chapters": [{}]}),
        patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
        patch(
            "app.agents.law_ref.search_laws_by_keyword",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch(
            "app.agents.law_ref.extract_structured_laws",
            new_callable=AsyncMock,
            return_value=structured,
        ),
    ):
        result = await law_ref_node(state)

    assert result["artifact_results"]["law"]["status"] == "success"
    assert result["artifact_results"]["law"]["source"] == "content_json"


@pytest.mark.asyncio
async def test_law_invalid_llm_artifact_uses_labeled_deterministic_fallback():
    """Law 输出缺字段时只能使用带来源标记的确定性候选回退。"""
    state = make_consultation_state(facts_structured=COMPLETE_FACT)
    candidate = make_applied_law(
        title="故意伤害罪",
        content="故意伤害他人身体的。",
        common_keywords=["伤害"],
    )

    with (
        patch("app.agents.law_ref.load_criminal_law_data", return_value={"chapters": [{}]}),
        patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
        patch(
            "app.agents.law_ref.search_laws_by_keyword",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch(
            "app.agents.law_ref.extract_structured_laws",
            new_callable=AsyncMock,
            return_value=[{"charge_name": "故意伤害罪"}],
        ),
    ):
        result = await law_ref_node(state)

    assert result["applied_laws"]
    assert result["artifact_results"]["law"]["status"] == "degraded"
    assert result["artifact_results"]["law"]["source"] == "deterministic_fallback"
    assert result["validation_errors"]


@pytest.mark.asyncio
async def test_risk_wrong_typed_output_routes_to_human_review_without_fake_assessment():
    """Risk 错类型输出不得生成默认成功评估或继续到 ServicePlanner。"""
    state = make_consultation_state(
        facts_structured=COMPLETE_FACT,
        applied_laws=[make_applied_law()],
    )

    with patch("app.agents.risk_assessor.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(
            return_value=json.dumps({"predicted_sentence_range": 3})
        )
        result = await risk_assessor_node(state)

    assert result["risk_assessment"] is None
    assert result["current_agent"] == "HumanReview"
    assert result["lawyer_review_needed"] is True
    assert result["artifact_results"]["risk"]["status"] == "human_review"
    assert result["artifact_results"]["risk"]["source"] == "content_json"
    assert result["validation_errors"]


@pytest.mark.asyncio
async def test_risk_valid_content_json_records_content_json_source():
    state = make_consultation_state(
        facts_structured=COMPLETE_FACT,
        applied_laws=[make_applied_law()],
    )
    with patch("app.agents.risk_assessor.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(
            return_value=json.dumps(COMPLETE_RISK, ensure_ascii=False)
        )
        result = await risk_assessor_node(state)

    assert result["artifact_results"]["risk"]["status"] == "success"
    assert result["artifact_results"]["risk"]["source"] == "content_json"


@pytest.mark.asyncio
async def test_service_plain_markdown_routes_to_human_without_accepting_report():
    """Service 的任意 Markdown 不能绕过 JSON schema 成为报告产物。"""
    state = make_consultation_state(
        facts_structured=COMPLETE_FACT,
        applied_laws=[make_applied_law()],
        risk_assessment=make_risk_assessment(
            compulsory_measure_risk={
                "detention_status": "未知",
                "bail_possibility": "待评估",
                "measure_change_space": "待评估",
                "prolonged_detention_risk": "unknown",
            },
            evidence_risk_points=[],
            procedure_risks=[],
        ),
    )

    with patch("app.agents.service_planner.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="# 刑事辩护初期咨询报告\n未结构化内容")
        result = await service_planner_node(state)

    assert result["service_plan"] is None
    assert result["report_draft"] is None
    assert result["current_agent"] == "HumanReview"
    assert result["artifact_results"]["service"]["status"] == "human_review"
    assert result["artifact_results"]["service"]["source"] == "content_json"
    assert result["validation_errors"]


@pytest.mark.asyncio
async def test_service_valid_content_json_records_content_json_source():
    state = make_consultation_state(
        facts_structured=COMPLETE_FACT,
        applied_laws=[make_applied_law()],
        risk_assessment=COMPLETE_RISK,
    )
    with patch("app.agents.service_planner.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(
            return_value=json.dumps(COMPLETE_SERVICE, ensure_ascii=False)
        )
        result = await service_planner_node(state)

    assert result["artifact_results"]["service"]["status"] == "success"
    assert result["artifact_results"]["service"]["source"] == "content_json"
