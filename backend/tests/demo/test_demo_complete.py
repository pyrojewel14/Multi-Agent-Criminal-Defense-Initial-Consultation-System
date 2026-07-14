import json
from pathlib import Path

import pytest

from examples.demo_complete import REQUIRED_CASE_KEYS, run_case
from app.security.sensitive_filter import detect_high_risk


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = PROJECT_ROOT / "demos" / "consultation" / "cases"


@pytest.mark.parametrize(
    ("case_id", "expected_output_type"),
    [
        ("ordinary_assault", "lawyer_reviewed_report"),
        ("missing_facts", "follow_up_questions"),
        ("high_risk_collusion", "human_intervention_notice"),
    ],
)
def test_demo_case_contract(case_id, expected_output_type):
    case = json.loads((CASE_DIR / f"{case_id}.json").read_text(encoding="utf-8"))

    assert REQUIRED_CASE_KEYS <= case.keys()
    assert len(case["expected_fact_fields"]) == 10
    assert case["expected_law_keywords"]
    assert case["expected_final_output_type"] == expected_output_type
    assert isinstance(case["expected_workflow_finished"], bool)
    assert isinstance(case["expected_requires_human_intervention"], bool)


@pytest.mark.asyncio
async def test_ordinary_demo_reaches_explicit_lawyer_approval():
    result = await run_case("ordinary_assault")

    assert result["finished"] is True
    assert result["requires_human_intervention"] is False
    assert result["output_type"] == "lawyer_reviewed_report"
    assert result["law_keywords_present"] is True
    assert [item["stage"] for item in result["trace"]] == [
        "consent_gate",
        "fact_and_law_contract",
        "risk_assessment",
        "service_plan_and_draft",
        "lawyer_review",
        "approved",
    ]


@pytest.mark.asyncio
async def test_missing_facts_demo_stops_for_follow_up():
    result = await run_case("missing_facts")

    assert result["finished"] is False
    assert result["requires_human_intervention"] is False
    assert result["output_type"] == "follow_up_questions"
    assert result["next_node"] == "law_ref"
    assert result["trace"][-1]["pending_questions"]


@pytest.mark.asyncio
async def test_high_risk_demo_reaches_end_and_requires_human_intervention():
    result = await run_case("high_risk_collusion")

    assert result["finished"] is True
    assert result["requires_human_intervention"] is True
    assert result["output_type"] == "human_intervention_notice"
    assert result["alert_triggered"] is True
    assert result["current_agent"] == "HumanAlert"


def test_high_risk_case_matches_the_real_rule_detector():
    case = json.loads((CASE_DIR / "high_risk_collusion.json").read_text(encoding="utf-8"))

    triggered, risk_type = detect_high_risk(case["user_input"])

    assert triggered is True
    assert risk_type == "COLLUSION"
