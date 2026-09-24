"""四类 LLM 产物的统一结构化契约与验证结果。"""

import json
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ArtifactStatus(StrEnum):
    """结构化产物在工作流中的处理状态。"""

    SUCCESS = "success"
    DEGRADED = "degraded"
    HUMAN_REVIEW = "human_review"


class ArtifactSource(StrEnum):
    """结构化产物的来源。"""

    TOOL_CALL = "tool_call"
    CONTENT_JSON = "content_json"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class _StrictArtifact(BaseModel):
    """所有 LLM 产物共用的严格 Pydantic 基类。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class FactArtifact(_StrictArtifact):
    """FactDigger 的结构化事实契约。"""

    incident_time: str | None
    incident_location: str | None
    parties: list[dict[str, Any]]
    behavior_sequence: list[dict[str, Any]]
    consequence: str | None
    evidence_mentioned: list[dict[str, Any]]
    arrest_status: str | None
    surrender: bool | None
    victim_forgiveness: bool | None
    prior_record: bool | None


class LawChargeArtifact(_StrictArtifact):
    """LawRef 返回的单项罪名分析。"""

    charge_name: str
    article_number: str
    elements_matched: list[str]
    elements_missing: list[str]
    base_sentence: str
    probability: str


class LawArtifact(_StrictArtifact):
    """LawRef 的结构化法律分析契约。"""

    charges: list[LawChargeArtifact] = Field(min_length=1)
    procedural_notes: list[str]


class CompulsoryMeasureRisk(_StrictArtifact):
    """强制措施风险字段。"""

    detention_status: str
    bail_possibility: str
    measure_change_space: str
    prolonged_detention_risk: str


class EvidenceRiskPoint(_StrictArtifact):
    """单项证据风险。"""

    gap: str
    exclusion_possibility: str


class ProcedureRisk(_StrictArtifact):
    """单项程序风险。"""

    type: str
    description: str
    severity: str


class RiskArtifact(_StrictArtifact):
    """RiskAssessor 的扁平结构化契约。"""

    predicted_sentence_range: str
    mitigating_factors: list[str]
    aggravating_factors: list[str]
    compulsory_measure_risk: CompulsoryMeasureRisk
    evidence_risk_points: list[EvidenceRiskPoint]
    procedure_risks: list[ProcedureRisk]


class UrgentActions(_StrictArtifact):
    """按时效分组的行动建议。"""

    immediate: list[str]
    short_term: list[str]
    follow_up: list[str]


class DefenseStrategies(_StrictArtifact):
    """辩护策略概览。"""

    primary: str
    alternatives: list[str]


class ServicePhase(_StrictArtifact):
    """律师服务阶段。"""

    phase: str
    description: str


class FeeStructure(_StrictArtifact):
    """服务费用结构。"""

    recommended_plan: str
    total_fee_range: str
    breakdown: dict[str, str]


class ServicePlan(_StrictArtifact):
    """ServicePlanner 的服务方案。"""

    urgent_actions: UrgentActions
    defense_strategies: DefenseStrategies
    service_phases: list[ServicePhase]
    fee_structure: FeeStructure


class ServiceArtifact(_StrictArtifact):
    """ServicePlanner 的完整结构化产物。"""

    service_plan: ServicePlan
    report_draft: str = Field(min_length=1)


class LLMArtifactResult(BaseModel):
    """所有 Agent 共享的产物校验与降级元数据。"""

    model_config = ConfigDict(extra="forbid")

    status: ArtifactStatus
    source: ArtifactSource
    degraded_reason: str | None = None
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    retryable: bool = False

    @model_validator(mode="after")
    def _success_has_no_failure_metadata(self) -> "LLMArtifactResult":
        if self.status is ArtifactStatus.SUCCESS and (
            self.degraded_reason is not None or self.validation_errors or self.retryable
        ):
            raise ValueError("成功产物不能包含失败元数据")
        return self


ArtifactModel = TypeVar("ArtifactModel", bound=BaseModel)


def _safe_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """移除验证错误中的原始输入，只保留字段位置和错误类别。"""
    return [
        {
            "loc": list(error.get("loc", ())),
            "type": str(error.get("type", "validation_error")),
            "msg": str(error.get("msg", "validation failed")),
        }
        for error in exc.errors(include_input=False, include_url=False)
    ]


def validate_artifact(
    schema: type[ArtifactModel],
    payload: object,
    *,
    source: ArtifactSource,
    failure_status: ArtifactStatus = ArtifactStatus.DEGRADED,
) -> tuple[ArtifactModel | None, LLMArtifactResult]:
    """通过统一入口验证产物，并返回不含原始输入的结果元数据。"""
    try:
        artifact = schema.model_validate(payload)
    except ValidationError as exc:
        return None, LLMArtifactResult(
            status=failure_status,
            source=source,
            degraded_reason="schema_validation_failed",
            validation_errors=_safe_validation_errors(exc),
            retryable=False,
        )

    return artifact, LLMArtifactResult(
        status=ArtifactStatus.SUCCESS,
        source=source,
    )


def record_artifact_result(
    state: dict[str, Any], artifact_name: str, result: LLMArtifactResult
) -> None:
    """把最新结果和按产物分类的历史元数据写入工作流状态。"""
    dumped = result.model_dump(mode="json")
    artifact_results = dict(state.get("artifact_results") or {})
    artifact_results[artifact_name] = dumped
    state["artifact_results"] = artifact_results
    state["degraded_reason"] = result.degraded_reason
    state["source"] = result.source.value
    state["validation_errors"] = result.validation_errors


def parse_json_object(text: object) -> dict[str, Any] | None:
    """从纯 JSON 或带说明文字的响应中提取首个 JSON 对象。"""
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
