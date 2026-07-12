"""Integration tests for the ServicePlanner Agent node."""

import json
import re
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.agents import service_planner
from app.agents.service_planner import (
    _build_service_request_message,
    _extract_inline_text,
    _extract_list_item,
    _extract_phase_description,
    _extract_service_plan_structure,
    _get_current_timestamp,
    _get_default_service_planner_prompt,
    _load_service_planner_prompt,
    _parse_llm_response,
    service_planner_node,
)
from tests.factories import make_applied_law, make_consultation_state, make_risk_assessment


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------


def test_parse_llm_response_with_service_plan_and_report():
    """When response contains both service plan and report markers, both should be extracted."""
    response = """一些前言内容

【服务方案建议】
1. 立即行动：申请取保候审
2. 短期行动：收集证据

# 刑事辩护初期咨询报告

## 案件基本信息
这是一个测试报告。
"""
    result = _parse_llm_response(response)
    assert "service_plan" in result
    assert "report_draft" in result
    assert "# 刑事辩护初期咨询报告" in result["report_draft"]


def test_parse_llm_response_report_only():
    """When response only contains report marker, report_draft should be set."""
    response = """# 刑事辩护初期咨询报告

## 案件基本信息
这是一个测试报告。
"""
    result = _parse_llm_response(response)
    assert result["report_draft"] == response


def test_parse_llm_response_no_markers():
    """When response has no markers, entire content becomes report_draft."""
    response = "这是一段普通的文本回复"
    result = _parse_llm_response(response)
    assert result["report_draft"] == response
    assert result["service_plan"] == {}


# ---------------------------------------------------------------------------
# _extract_service_plan_structure
# ---------------------------------------------------------------------------


def test_extract_service_plan_structure_default():
    """With no recognizable sections, structure should have default empty values."""
    content = "一些不相关的内容"
    result = _extract_service_plan_structure(content)

    assert "urgent_actions" in result
    assert "defense_strategies" in result
    assert "service_phases" in result
    assert "fee_structure" in result
    # Default structure has empty lists/strings
    assert result["urgent_actions"]["immediate"] == []
    assert result["defense_strategies"]["primary"] == ""


def test_extract_service_plan_structure_with_actions():
    """Service plan should extract urgent actions from recognized sections."""
    content = """【服务方案】
立即行动
- 申请取保候审
- 会见当事人

短期行动
- 收集不在场证明

后续行动
- 制定辩护策略
"""
    result = _extract_service_plan_structure(content)
    assert len(result["urgent_actions"]["immediate"]) > 0
    assert len(result["urgent_actions"]["short_term"]) > 0
    assert len(result["urgent_actions"]["follow_up"]) > 0


# ---------------------------------------------------------------------------
# service_planner_node – valid response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_planner_node_valid_response():
    """With a valid LLM response, service_plan and report_draft should be populated."""
    llm_response = """# 刑事辩护初期咨询报告

## 案件基本信息
测试案件

## 紧急行动建议
- 立即会见当事人
- 申请取保候审
"""

    state = make_consultation_state(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[make_applied_law()],
        risk_assessment=make_risk_assessment(),
        conversation_history=[],
    )

    with patch("app.agents.service_planner.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value=llm_response)
        result = await service_planner_node(state)

    assert result["service_plan"] is not None
    assert result["report_draft"] is not None
    assert isinstance(result["service_plan"], dict)


# ---------------------------------------------------------------------------
# service_planner_node – state updates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_planner_node_state_updates():
    """service_planner_node should set lawyer_review_needed=True and current_agent=HumanReview."""
    llm_response = "# 刑事辩护初期咨询报告\n\n测试报告内容"

    state = make_consultation_state(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[make_applied_law()],
        risk_assessment=make_risk_assessment(),
        conversation_history=[],
    )

    with patch("app.agents.service_planner.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value=llm_response)
        result = await service_planner_node(state)

    assert result["lawyer_review_needed"] is True
    assert result["current_agent"] == "HumanReview"


# ---------------------------------------------------------------------------
# service_planner_node – report generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_planner_node_report_contains_markdown():
    """report_draft should contain markdown content from the LLM response."""
    llm_response = """# 刑事辩护初期咨询报告

## 案件基本信息
- 当事人：张某
- 案由：盗窃

## 风险评估
量刑预测：三年以下有期徒刑
"""

    state = make_consultation_state(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[make_applied_law()],
        risk_assessment=make_risk_assessment(),
        conversation_history=[],
    )

    with patch("app.agents.service_planner.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value=llm_response)
        result = await service_planner_node(state)

    assert result["report_draft"] is not None
    assert "# 刑事辩护初期咨询报告" in result["report_draft"]
    # report_draft is extracted from the parsed response (after disclaimer injection + re-extraction),
    # so it may or may not contain the disclaimer prefix depending on parse logic.
    # The key contract is that report_draft contains the report markdown content.
    assert "案件基本信息" in result["report_draft"]


# ---------------------------------------------------------------------------
# service_planner_node – conversation history update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_planner_node_conversation_history():
    """service_planner_node should append a ServicePlanner entry to conversation_history."""
    llm_response = "# 刑事辩护初期咨询报告\n\n测试"

    state = make_consultation_state(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[make_applied_law()],
        risk_assessment=make_risk_assessment(),
        conversation_history=[],
    )

    with patch("app.agents.service_planner.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value=llm_response)
        result = await service_planner_node(state)

    assert any(entry.get("agent") == "ServicePlanner" for entry in result["conversation_history"])


# ---------------------------------------------------------------------------
# _load_service_planner_prompt / _get_current_timestamp
# ---------------------------------------------------------------------------


def test_load_service_planner_prompt_uses_loader():
    """_load_service_planner_prompt should return the prompt when registered."""
    with patch.object(service_planner.prompt_loader, "load", return_value="loaded-prompt"):
        assert _load_service_planner_prompt() == "loaded-prompt"


def test_load_service_planner_prompt_falls_back_to_default():
    """_load_service_planner_prompt should return default when KeyError raised."""
    with patch.object(service_planner.prompt_loader, "load", side_effect=KeyError("nope")):
        text = _load_service_planner_prompt()
    # Default prompt mentions 服务方案 Agent 提示词
    assert "服务方案" in text
    # And it's the same as _get_default_service_planner_prompt()
    assert text == _get_default_service_planner_prompt()


def test_get_current_timestamp_returns_isoformat():
    """_get_current_timestamp should return an ISO 8601 string parseable by datetime.fromisoformat."""
    ts = _get_current_timestamp()
    # ISO 8601 includes date and time
    parsed = datetime.fromisoformat(ts)
    assert isinstance(parsed, datetime)


# ---------------------------------------------------------------------------
# _extract_list_item
# ---------------------------------------------------------------------------


def test_extract_list_item_bold():
    """Bold text wrapped in ** should be extracted."""
    assert _extract_list_item("**加粗文本**") == "加粗文本"


def test_extract_list_item_dash():
    """Dash-prefixed list items should be extracted."""
    assert _extract_list_item("- 项目内容") == "项目内容"
    assert _extract_list_item("• 项目内容") == "项目内容"


def test_extract_list_item_numbered():
    """Numbered list items (with various separators) should be extracted."""
    assert _extract_list_item("1. 第一项") == "第一项"
    assert _extract_list_item("2、第二项") == "第二项"
    assert _extract_list_item("3) 第三项") == "第三项"


def test_extract_list_item_not_a_list():
    """Non-list lines should return None."""
    assert _extract_list_item("普通文本") is None
    assert _extract_list_item("") is None


# ---------------------------------------------------------------------------
# _extract_phase_description
# ---------------------------------------------------------------------------


def test_extract_phase_description_full_width_colon():
    """Full-width colon separator should split title and description."""
    assert _extract_phase_description("侦查阶段：会见嫌疑人、了解案情") == "会见嫌疑人、了解案情"


def test_extract_phase_description_half_width_colon():
    """Half-width colon separator should also work."""
    assert _extract_phase_description("审查起诉阶段: 阅卷") == "阅卷"


def test_extract_phase_description_dash_separator():
    """Dash and em-dash separators should work."""
    assert _extract_phase_description("审判阶段 — 庭审") == "庭审"
    assert _extract_phase_description("审判阶段 – 庭审") == "庭审"
    assert _extract_phase_description("审判阶段 - 庭审") == "庭审"


def test_extract_phase_description_no_separator():
    """When no separator is present, return empty string."""
    assert _extract_phase_description("审判阶段") == ""


# ---------------------------------------------------------------------------
# _extract_inline_text
# ---------------------------------------------------------------------------


def test_extract_inline_text_finds_keyword():
    """When a keyword is in the line, the text after the keyword is returned."""
    result = _extract_inline_text("推荐方案：全程委托", ["推荐方案", "全程委托"])
    assert "全程委托" in result


def test_extract_inline_text_no_keyword():
    """When no keyword is in the line, return empty string."""
    assert _extract_inline_text("无关文本", ["推荐方案"]) == ""


def test_extract_inline_text_keyword_at_end():
    """When the keyword is at the end with no trailing text, return empty string."""
    assert _extract_inline_text("推荐方案", ["推荐方案"]) == ""


# ---------------------------------------------------------------------------
# _extract_service_plan_structure – deeper coverage
# ---------------------------------------------------------------------------


def test_extract_service_plan_structure_alternative_strategy():
    """The '备选辩护策略' section should populate alternatives."""
    content = """备选辩护策略
- 罪轻辩护
- 认罪认罚
"""
    result = _extract_service_plan_structure(content)
    assert len(result["defense_strategies"]["alternatives"]) >= 2


def test_extract_service_plan_structure_phase_with_description():
    """Phase lines with description (via the title-only line) should be parsed."""
    content = """侦查阶段：会见嫌疑人
审查起诉阶段：阅卷
审判阶段：出庭
"""
    result = _extract_service_plan_structure(content)
    # At least one phase should be added
    assert len(result["service_phases"]) >= 1


def test_extract_service_plan_structure_fee_recommended():
    """The '推荐方案' section should populate fee_structure.recommended_plan."""
    content = """推荐方案：全程委托
"""
    result = _extract_service_plan_structure(content)
    assert "全程委托" in result["fee_structure"]["recommended_plan"] or result["fee_structure"]["recommended_plan"] != ""


def test_extract_service_plan_structure_fee_breakdown():
    """A '阶段报价' / '费用明细' section should populate fee_structure.breakdown."""
    content = """费用明细
- 第一项费用：1万-3万元
- 第二项费用：2万-5万元
"""
    result = _extract_service_plan_structure(content)
    # The breakdown dict should have at least one entry
    assert result["fee_structure"]["breakdown"]


def test_extract_service_plan_structure_fee_range_in_text():
    """A fee range in the text should be extracted into total_fee_range."""
    content = """其他内容
全程费用：5万-10万元
"""
    result = _extract_service_plan_structure(content)
    assert "5万" in result["fee_structure"]["total_fee_range"]


# ---------------------------------------------------------------------------
# _parse_llm_response – exception path
# ---------------------------------------------------------------------------


def test_parse_llm_response_handles_unusual_inputs():
    """_parse_llm_response should never raise even with weird inputs."""
    # String with both markers, but weird positioning
    response = "【服务方案\n# 刑事辩护初期咨询报告"
    result = _parse_llm_response(response)
    assert "report_draft" in result


def test_parse_llm_response_full_text_extraction():
    """When only the report marker is present, report_draft is the slice from the marker."""
    response = "前置内容 # 刑事辩护初期咨询报告 后置内容"
    result = _parse_llm_response(response)
    assert "刑事辩护初期咨询报告" in result["report_draft"]


# ---------------------------------------------------------------------------
# _build_service_request_message
# ---------------------------------------------------------------------------


def test_build_service_request_message_full():
    """_build_service_request_message should produce a message with all sections."""
    msg = _build_service_request_message(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[{"article_number": "第264条"}],
        risk_assessment={"predicted_sentence_range": "三年以下"},
        conversation_history=[
            {"agent": "FactDigger", "summary": "test summary"},
        ],
    )
    assert "案件基本信息" in msg
    assert "适用法律法规" in msg
    assert "风险评估结果" in msg
    assert "对话历史摘要" in msg
    assert "服务方案请求" in msg


def test_build_service_request_message_no_history():
    """When conversation_history is empty, the history section is omitted."""
    msg = _build_service_request_message(
        facts_structured={},
        applied_laws=[],
        risk_assessment={},
        conversation_history=[],
    )
    assert "对话历史摘要" not in msg


# ---------------------------------------------------------------------------
# service_planner_node – exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_planner_node_raises_on_llm_error():
    """If the LLM call raises, service_planner_node re-raises the exception."""
    state = make_consultation_state(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[make_applied_law()],
        risk_assessment=make_risk_assessment(),
    )
    with patch("app.agents.service_planner.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM offline"))
        with pytest.raises(Exception, match="LLM offline"):
            await service_planner_node(state)
