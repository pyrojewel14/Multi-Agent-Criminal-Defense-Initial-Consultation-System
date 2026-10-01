"""Public corpus eval exercises real tool contract, without model calls offline."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


@pytest.mark.parametrize("demo_mode", ["on", "off"])
def test_offline_public_cases_and_distinct_evidence(tmp_path, monkeypatch, demo_mode):
    from evaluation.run_full_eval import run_evaluation

    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "full")
    monkeypatch.setenv("LAW_FULL_DEMO_ANNOTATIONS", demo_mode)
    result = asyncio.run(run_evaluation("offline", tmp_path / "result.json"))
    assert result["evidence_kind"] == "offline-deterministic-tools"
    assert result["llm_executed"] is False
    assert result["rag_executed"] is False
    assert result["passed"] == 8
    assert result["total"] == 8
    assert result["legal_review_completed"] is False
    assert result["annotation_mode"] == ("demo" if demo_mode == "on" else "text_only_with_six_regression")
    assert result == json.loads((tmp_path / "result.json").read_text())


def test_custom_public_fact_case_file_uses_same_real_tool_contract(tmp_path):
    from evaluation.run_full_eval import run_evaluation

    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "public-theft", "query": "盗窃", "article_id": "第264条", "contains": "盗窃公私财物", "coverage_eligible": True}) + "\n")
    result = asyncio.run(run_evaluation("offline", tmp_path / "result.json", cases_path=cases))
    assert result["passed"] == result["total"] == 1
    assert result["cases"][0]["id"] == "public-theft"


def test_guardrail_expectations_reject_model_claimed_matches_without_fact_support(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from evaluation.run_full_eval import run_evaluation

    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "no-facts", "query": "盗窃", "article_id": "第264条", "coverage_eligible": True, "expected_matched_elements": [], "expected_missing_count": 2}) + "\n")
    state = {
        "applied_laws": [{"article_number": "第264条", "required_elements": ["盗窃公私财物", "数额较大或者具备法定盗窃情形之一"], "elements_matched": ["盗窃公私财物"], "elements_missing": ["数额较大或者具备法定盗窃情形之一"], "data_source": "json_keyword"}],
        "law_research": {"termination_reason": "final_answer", "trajectory": []},
    }
    monkeypatch.setattr("app.consultation.agents.law_ref.law_ref_node", AsyncMock(return_value=state))
    result = asyncio.run(run_evaluation("live-lawref", tmp_path / "result.json", cases_path=cases))
    assert result["cases"][0]["passed"] is False
    assert result["cases"][0]["guardrail_passed"] is False
