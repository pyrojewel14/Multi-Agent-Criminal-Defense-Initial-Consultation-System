"""Integration tests for the RiskAssessor Agent node."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents import risk_assessor
from app.agents.risk_assessor import (
    _extract_key_risks,
    _load_prompt,
    _parse_fallback_assessment,
    format_risk_assessment_report,
    risk_assessor_node,
)
from tests.factories import make_applied_law, make_consultation_state


# ---------------------------------------------------------------------------
# _parse_fallback_assessment
# ---------------------------------------------------------------------------


def test_parse_fallback_assessment_structure():
    """Fallback assessment should have the correct nested structure."""
    result = _parse_fallback_assessment("some non-JSON text")

    assert "predicted_sentence_range" in result
    assert "mitigating_factors" in result
    assert "aggravating_factors" in result
    assert "compulsory_measure_risk" in result
    assert "evidence_risk_points" in result
    assert "procedure_risks" in result

    # compulsory_measure_risk is a nested dict, not a string
    assert isinstance(result["compulsory_measure_risk"], dict)
    assert "detention_status" in result["compulsory_measure_risk"]
    assert "bail_possibility" in result["compulsory_measure_risk"]
    assert "measure_change_space" in result["compulsory_measure_risk"]
    assert "prolonged_detention_risk" in result["compulsory_measure_risk"]


# ---------------------------------------------------------------------------
# risk_assessor_node – valid JSON LLM response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_assessor_node_valid_json():
    """With a valid JSON LLM response, risk_assessment should contain structured data."""
    valid_assessment = {
        "predicted_sentence_range": "三年以下有期徒刑",
        "mitigating_factors": ["初犯", "认罪认罚"],
        "aggravating_factors": ["涉案金额较大"],
        "compulsory_measure_risk": {
            "detention_status": "已羁押",
            "bail_possibility": "中等",
            "measure_change_space": "有空间",
            "prolonged_detention_risk": "low",
        },
        "evidence_risk_points": [
            {"gap": "证据链不完整", "exclusion_possibility": "低"},
        ],
        "procedure_risks": [
            {"type": "诉讼时效", "description": "可能接近时效", "severity": "medium"},
        ],
    }

    state = make_consultation_state(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[make_applied_law()],
    )

    with patch("app.agents.risk_assessor.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value=json.dumps(valid_assessment))
        result = await risk_assessor_node(state)

    assert result["risk_assessment"]["predicted_sentence_range"] == "三年以下有期徒刑"
    assert result["risk_assessment"]["compulsory_measure_risk"]["detention_status"] == "已羁押"
    assert len(result["risk_assessment"]["mitigating_factors"]) == 2
    assert len(result["risk_assessment"]["evidence_risk_points"]) == 1


# ---------------------------------------------------------------------------
# risk_assessor_node – non-JSON LLM response → fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_assessor_node_non_json_fallback():
    """When LLM returns non-JSON, fallback assessment should be used."""
    state = make_consultation_state(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[make_applied_law()],
    )

    with patch("app.agents.risk_assessor.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value="This is not JSON at all")
        result = await risk_assessor_node(state)

    # Should use fallback structure
    assert result["risk_assessment"]["predicted_sentence_range"] == "待评估"
    assert isinstance(result["risk_assessment"]["compulsory_measure_risk"], dict)
    assert result["risk_assessment"]["compulsory_measure_risk"]["detention_status"] == "待确认"


# ---------------------------------------------------------------------------
# risk_assessor_node – state updates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_assessor_node_state_updates():
    """risk_assessor_node should update current_agent, inject disclaimer, and update conversation_history."""
    assessment = {
        "predicted_sentence_range": "三年以下有期徒刑",
        "mitigating_factors": [],
        "aggravating_factors": [],
        "compulsory_measure_risk": {
            "detention_status": "未羁押",
            "bail_possibility": "高",
            "measure_change_space": "较大",
            "prolonged_detention_risk": "low",
        },
        "evidence_risk_points": [],
        "procedure_risks": [],
    }

    state = make_consultation_state(
        facts_structured={"consequence": "轻伤"},
        applied_laws=[make_applied_law()],
        conversation_history=[],
    )

    with patch("app.agents.risk_assessor.llm_gateway") as mock_llm:
        mock_llm.generate = AsyncMock(return_value=json.dumps(assessment))
        result = await risk_assessor_node(state)

    # current_agent should transition to ServicePlanner
    assert result["current_agent"] == "ServicePlanner"
    # conversation_history should have a RiskAssessor entry
    assert any(entry.get("agent") == "RiskAssessor" for entry in result["conversation_history"])
    # The history entry should contain assessment_summary
    risk_entry = [e for e in result["conversation_history"] if e.get("agent") == "RiskAssessor"][0]
    assert "assessment_summary" in risk_entry


# ---------------------------------------------------------------------------
# _load_prompt – fallback path
# ---------------------------------------------------------------------------


def test_load_prompt_uses_loader():
    """When the prompt is registered, it should be returned as-is."""
    with patch.object(risk_assessor.prompt_loader, "load", return_value="loaded-risks"):
        assert _load_prompt() == "loaded-risks"


def test_load_prompt_falls_back_to_default():
    """When KeyError is raised, return the default prompt."""
    with patch.object(risk_assessor.prompt_loader, "load", side_effect=KeyError("nope")):
        text = _load_prompt()
    assert "专业的刑事辩护律师" in text
    assert "predicted_sentence_range" in text


# ---------------------------------------------------------------------------
# _extract_key_risks
# ---------------------------------------------------------------------------


def test_extract_key_risks_prolonged_detention():
    """A 'high' prolonged_detention_risk should produce the 超期羁押风险 key risk."""
    assessment = {
        "compulsory_measure_risk": {"prolonged_detention_risk": "high"},
        "evidence_risk_points": [],
        "procedure_risks": [],
    }
    assert "超期羁押风险" in _extract_key_risks(assessment)


def test_extract_key_risks_evidence():
    """Evidence risk points should produce a count-based key risk."""
    assessment = {
        "compulsory_measure_risk": {},
        "evidence_risk_points": [{"gap": "g1"}, {"gap": "g2"}],
        "procedure_risks": [],
    }
    risks = _extract_key_risks(assessment)
    assert any("2个证据风险点" in r for r in risks)


def test_extract_key_risks_procedure_high():
    """High-severity procedure risks should produce a labeled key risk."""
    assessment = {
        "compulsory_measure_risk": {},
        "evidence_risk_points": [],
        "procedure_risks": [{"type": "诉讼时效", "severity": "high"}],
    }
    risks = _extract_key_risks(assessment)
    assert any("诉讼时效" in r for r in risks)


# ---------------------------------------------------------------------------
# format_risk_assessment_report
# ---------------------------------------------------------------------------


def test_format_risk_assessment_report_full():
    """format_risk_assessment_report should produce a non-empty structured report."""
    assessment = {
        "predicted_sentence_range": "三年以下",
        "mitigating_factors": ["自首"],
        "aggravating_factors": ["累犯"],
        "compulsory_measure_risk": {
            "detention_status": "已羁押",
            "bail_possibility": "中等",
            "measure_change_space": "有空间",
            "prolonged_detention_risk": "low",
        },
        "evidence_risk_points": [
            {"gap": "证据缺口", "exclusion_possibility": "中"},
        ],
        "procedure_risks": [
            {"type": "诉讼时效", "description": "接近时效", "severity": "medium"},
        ],
    }
    report = format_risk_assessment_report(assessment)
    assert "三年以下" in report
    assert "自首" in report
    assert "证据缺口" in report
    assert "MEDIUM" in report.upper() or "medium" in report


def test_format_risk_assessment_report_empty_lists():
    """Empty factor and risk lists should show fallback messages."""
    assessment = {
        "predicted_sentence_range": "待评估",
        "mitigating_factors": [],
        "aggravating_factors": [],
        "compulsory_measure_risk": {
            "detention_status": "待确认",
            "bail_possibility": "待评估",
            "measure_change_space": "待评估",
            "prolonged_detention_risk": "待评估",
        },
        "evidence_risk_points": [],
        "procedure_risks": [],
    }
    report = format_risk_assessment_report(assessment)
    assert "无" in report
    assert "未发现明显证据风险点" in report
    assert "未发现明显程序风险" in report
