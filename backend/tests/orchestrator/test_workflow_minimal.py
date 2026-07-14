"""Phase 5 workflow integration contracts using deterministic Agent nodes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.errors.exceptions import LLMServiceException
from app.orchestrator.workflow import ConsultationOrchestrator, _fact_digger_workflow_node
from app.state.consultation_state import ConsultationState
from tests.factories import make_consultation_state


def _initial_state(session_id: str) -> ConsultationState:
    return make_consultation_state(
        session_id=session_id,
        consent_given=False,
        facts_raw=["确定性工作流测试输入"],
        facts_structured={},
        applied_laws=[],
        conversation_history=[],
        lawyer_decision=None,
    )


@asynccontextmanager
async def _workflow_fixture(
    session_id: str,
    *,
    fact_coverages: list[float],
    high_risk: bool = False,
) -> AsyncIterator[tuple[ConsultationOrchestrator, list[str]]]:
    trace: list[str] = []
    fact_calls = 0

    async def receptionist(state: ConsultationState) -> ConsultationState:
        trace.append("receptionist")
        state["current_agent"] = "Receptionist"
        return state

    async def fact_digger(state: ConsultationState) -> ConsultationState:
        nonlocal fact_calls
        trace.append("fact_digger")
        coverage = fact_coverages[min(fact_calls, len(fact_coverages) - 1)]
        fact_calls += 1
        state["current_agent"] = "FactDigger"
        state["facts_structured"] = {"behavior_sequence": ["示例行为"]}
        state["facts_coverage_rate"] = coverage
        state["fact_law_loop_count"] = fact_calls
        state["pending_questions"] = ["请补充关键事实"] if coverage < 0.8 else []
        state["alert_triggered"] = high_risk
        return state

    async def law_ref(state: ConsultationState) -> ConsultationState:
        trace.append("law_ref")
        state["current_agent"] = "LawRef"
        state["applied_laws"] = [{"article_number": "第234条", "elements": []}]
        return state

    async def risk_assessor(state: ConsultationState) -> ConsultationState:
        trace.append("risk_assessor")
        state["current_agent"] = "RiskAssessor"
        state["risk_assessment"] = {"risk_level": "medium"}
        return state

    async def service_planner(state: ConsultationState) -> ConsultationState:
        trace.append("service_planner")
        state["current_agent"] = "ServicePlanner"
        state["service_plan"] = {"next_step": "lawyer_review"}
        state["report_draft"] = "确定性报告草案"
        return state

    async def human_alert(state: ConsultationState) -> ConsultationState:
        trace.append("human_alert")
        state["current_agent"] = "HumanAlert"
        state["lawyer_review_needed"] = True
        return state

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch("app.orchestrator.workflow.fact_digger_node", fact_digger),
        patch("app.orchestrator.workflow.law_ref_node", law_ref),
        patch("app.orchestrator.workflow.risk_assessor_node", risk_assessor),
        patch("app.orchestrator.workflow.service_planner_node", service_planner),
        patch("app.orchestrator.workflow.human_alert_node", human_alert),
    ):
        orchestrator = ConsultationOrchestrator()
        await orchestrator.start_workflow(_initial_state(session_id))
        yield orchestrator, trace


@pytest.mark.asyncio
async def test_without_informed_consent_finishes_before_fact_collection():
    async with _workflow_fixture("phase5-no-consent", fact_coverages=[1.0]) as (orchestrator, trace):
        result = await orchestrator.resume_workflow("phase5-no-consent", {"consent_given": False})

        assert result["current_agent"] == "Receptionist"
        assert trace == ["receptionist"]
        assert await orchestrator.is_workflow_finished("phase5-no-consent") is True


@pytest.mark.asyncio
async def test_missing_facts_pause_then_resume_through_law_ref():
    async with _workflow_fixture("phase5-follow-up", fact_coverages=[0.4, 1.0]) as (orchestrator, trace):
        paused = await orchestrator.resume_workflow("phase5-follow-up", {"consent_given": True})

        assert paused["pending_questions"] == ["请补充关键事实"]
        assert await orchestrator.get_next_node("phase5-follow-up") == "law_ref"

        resumed = await orchestrator.resume_workflow("phase5-follow-up", {"current_input": "补充事实"})

        assert resumed["current_agent"] == "HumanReview"
        assert resumed["awaiting_lawyer_review"] is True
        assert trace == [
            "receptionist",
            "fact_digger",
            "law_ref",
            "fact_digger",
            "risk_assessor",
            "service_planner",
        ]


@pytest.mark.asyncio
async def test_high_risk_routes_to_human_alert():
    async with _workflow_fixture("phase5-alert", fact_coverages=[0.0], high_risk=True) as (orchestrator, trace):
        result = await orchestrator.resume_workflow("phase5-alert", {"consent_given": True})

        assert result["current_agent"] == "HumanAlert"
        assert result["lawyer_review_needed"] is True
        assert trace == ["receptionist", "fact_digger", "human_alert"]
        assert await orchestrator.is_workflow_finished("phase5-alert") is True


@pytest.mark.parametrize(
    ("decision", "rerun_agent"),
    [("revise_facts", "fact_digger"), ("revise_risk", "risk_assessor")],
)
@pytest.mark.asyncio
async def test_lawyer_rejection_routes_back_to_requested_stage(decision: str, rerun_agent: str):
    session_id = f"phase5-{decision}"
    async with _workflow_fixture(session_id, fact_coverages=[1.0]) as (orchestrator, trace):
        first_review = await orchestrator.resume_workflow(session_id, {"consent_given": True})
        assert first_review["current_agent"] == "HumanReview"

        before = trace.count(rerun_agent)
        second_review = await orchestrator.process_lawyer_feedback(session_id, decision, "请重新核查")

        assert trace.count(rerun_agent) == before + 1
        assert second_review["current_agent"] == "HumanReview"
        assert second_review["awaiting_lawyer_review"] is True
        assert second_review["lawyer_decision"] is None


@pytest.mark.asyncio
async def test_revise_facts_does_not_append_stale_current_input_again():
    state = make_consultation_state(
        facts_raw=["补充事实"],
        current_input="补充事实",
        lawyer_decision="revise_facts",
        lawyer_feedback="请重新核查事实",
        applied_laws=[],
    )

    with patch(
        "app.agents.fact_digger._extract_structured_facts",
        new_callable=AsyncMock,
        return_value={"behavior_sequence": ["补充事实"]},
    ) as extract_facts:
        result = await _fact_digger_workflow_node(state)

    assert result["facts_raw"] == ["补充事实"]
    assert result["current_input"] is None
    assert result["lawyer_decision"] is None
    assert result["lawyer_feedback"] == "请重新核查事实"
    extract_facts.assert_awaited_once_with(["补充事实"])


@pytest.mark.asyncio
async def test_orchestrator_failure_log_carries_session_id():
    async def failing_receptionist(state: ConsultationState) -> ConsultationState:
        raise LLMServiceException("模型不可用")

    with patch("app.orchestrator.workflow.receptionist_node", failing_receptionist):
        orchestrator = ConsultationOrchestrator()
        orchestrator._logger = MagicMock()

        with pytest.raises(LLMServiceException):
            await orchestrator.start_workflow(_initial_state("phase5-log-session"))

    error_args = orchestrator._logger.error.call_args.args
    assert "session_id=%s" in error_args[0]
    assert "phase5-log-session" in error_args
