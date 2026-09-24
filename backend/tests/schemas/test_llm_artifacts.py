"""LLM 结构化产物的共享契约测试。"""

import re

import pytest
from pydantic import ValidationError

from app.schemas.llm_artifacts import (
    ArtifactSource,
    ArtifactStatus,
    FactArtifact,
    LawArtifact,
    LLMArtifactResult,
    RiskArtifact,
    ServiceArtifact,
    parse_json_object,
    validate_artifact,
)
from app.utils.prompt_loader import prompt_loader
from app.tools.fact_tools import extract_case_facts


VALID_FACT = {
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

VALID_LAW = {
    "charges": [
        {
            "charge_name": "故意伤害罪",
            "article_number": "第二百三十四条",
            "elements_matched": ["伤害行为"],
            "elements_missing": ["伤情鉴定"],
            "base_sentence": "三年以下",
            "probability": "medium",
        }
    ],
    "procedural_notes": [],
}

VALID_RISK = {
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

VALID_SERVICE = {
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


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (FactArtifact, VALID_FACT),
        (LawArtifact, VALID_LAW),
        (RiskArtifact, VALID_RISK),
        (ServiceArtifact, VALID_SERVICE),
    ],
)
def test_each_artifact_accepts_its_complete_contract(schema, payload):
    """完整且类型正确的四类产物应使用同一验证入口成功。"""
    artifact, result = validate_artifact(schema, payload, source=ArtifactSource.CONTENT_JSON)

    assert artifact is not None
    assert result.status is ArtifactStatus.SUCCESS
    assert result.source is ArtifactSource.CONTENT_JSON
    assert result.degraded_reason is None
    assert result.validation_errors == []


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (FactArtifact, {key: value for key, value in VALID_FACT.items() if key != "consequence"}),
        (LawArtifact, {"charges": "not-a-list", "procedural_notes": []}),
        (RiskArtifact, {"predicted_sentence_range": "缺少其余字段"}),
        (ServiceArtifact, {"service_plan": {}, "report_draft": 42}),
    ],
)
def test_missing_or_wrong_typed_artifact_is_not_success(schema, payload):
    """缺字段或错类型不能被包装成成功产物。"""
    artifact, result = validate_artifact(schema, payload, source=ArtifactSource.CONTENT_JSON)

    assert artifact is None
    assert result.status is ArtifactStatus.DEGRADED
    assert result.degraded_reason == "schema_validation_failed"
    assert result.validation_errors
    assert all("input" not in issue for issue in result.validation_errors)


def test_artifact_result_rejects_success_with_validation_errors():
    """成功状态不得携带降级原因或验证错误。"""
    with pytest.raises(ValidationError):
        LLMArtifactResult(
            status=ArtifactStatus.SUCCESS,
            source=ArtifactSource.CONTENT_JSON,
            degraded_reason="schema_validation_failed",
            validation_errors=[{"loc": ["field"], "type": "missing", "msg": "Field required"}],
            retryable=False,
        )


def test_fact_tool_schema_requires_every_shared_contract_field():
    """Function Calling schema 不得允许模型省略 FactArtifact 字段。"""
    required = set(extract_case_facts.args_schema.model_json_schema().get("required", []))
    assert required == set(FactArtifact.model_fields)


@pytest.mark.parametrize(
    ("prompt_name", "schema"),
    [
        ("extract_case_facts_prompt", FactArtifact),
        ("lawref_prompt", LawArtifact),
        ("risk_assessor_prompt", RiskArtifact),
        ("service_planner_prompt", ServiceArtifact),
    ],
)
def test_external_prompt_json_example_matches_consumer_schema(prompt_name, schema):
    """外部 prompt 的 JSON 示例必须能被真实消费端 schema 接受。"""
    prompt = prompt_loader.load(prompt_name)
    examples = re.findall(r"```json\s*(.*?)```", prompt, flags=re.DOTALL)

    matching = []
    for example in examples:
        payload = parse_json_object(example)
        if payload is None:
            continue
        try:
            matching.append(schema.model_validate(payload))
        except ValidationError:
            continue

    assert len(matching) == 1
