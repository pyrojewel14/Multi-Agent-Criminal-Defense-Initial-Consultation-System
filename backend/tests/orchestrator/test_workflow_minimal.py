"""Phase 5 workflow integration contracts using deterministic Agent nodes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.errors.exceptions import LLMServiceException
from app.orchestrator.workflow import ConsultationOrchestrator, _fact_intake_workflow_node
from app.schemas.llm_artifacts import ArtifactSource
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

    async def fact_intake(state: ConsultationState) -> ConsultationState:
        trace.append("fact_intake")
        state["current_agent"] = "FactDigger"
        state["alert_triggered"] = high_risk
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
        patch("app.orchestrator.workflow.fact_intake_node", fact_intake),
        patch("app.orchestrator.workflow.fact_coverage_node", fact_digger),
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
        assert await orchestrator.get_next_node("phase5-follow-up") == "fact_intake"

        resumed = await orchestrator.resume_workflow("phase5-follow-up", {"current_input": "补充事实"})

        assert resumed["current_agent"] == "HumanReview"
        assert resumed["awaiting_lawyer_review"] is True
        assert trace == [
            "receptionist",
            "fact_intake",
            "law_ref",
            "fact_digger",
            "fact_intake",
            "law_ref",
            "fact_digger",
            "risk_assessor",
            "service_planner",
        ]


@pytest.mark.asyncio
async def test_invalid_risk_artifact_routes_directly_to_human_without_service_call():
    """风险产物降级时真实编译图不得继续调用 ServicePlanner。"""
    session_id = "risk-artifact-human-review"
    service_calls = 0

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        return state

    async def fact_intake(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "FactDigger"
        state["alert_triggered"] = False
        return state

    async def law_ref(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "LawRef"
        return state

    async def fact_coverage(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "FactDigger"
        state["facts_coverage_rate"] = 1.0
        return state

    async def risk_assessor(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "HumanReview"
        state["workflow_status"] = "degraded"
        state["artifact_results"] = {
            "risk": {
                "status": "human_review",
                "source": "content_json",
                "degraded_reason": "schema_validation_failed",
                "validation_errors": [{"loc": ["procedure_risks"], "type": "missing"}],
                "retryable": False,
            }
        }
        return state

    async def service_planner(state: ConsultationState) -> ConsultationState:
        nonlocal service_calls
        service_calls += 1
        return state

    initial = _initial_state(session_id)
    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch("app.orchestrator.workflow.fact_intake_node", fact_intake),
        patch("app.orchestrator.workflow.law_ref_node", law_ref),
        patch("app.orchestrator.workflow.fact_coverage_node", fact_coverage),
        patch("app.orchestrator.workflow.risk_assessor_node", risk_assessor),
        patch("app.orchestrator.workflow.service_planner_node", service_planner),
    ):
        orchestrator = ConsultationOrchestrator()
        await orchestrator.start_workflow(initial)
        result = await orchestrator.resume_workflow(session_id, {"consent_given": True})

    assert result["current_agent"] == "HumanReview"
    assert service_calls == 0
    assert await orchestrator.get_next_node(session_id) == "human_review"


@pytest.mark.asyncio
async def test_resumed_fact_refreshes_law_query_and_risk_laws():
    """本轮决定性事实必须先进入真实编译图的检索与风险输入。"""
    session_id = "p0-current-fact-before-law"
    rag_queries: list[str] = []
    risk_inputs: list[tuple[dict, list[dict]]] = []

    async def law_decision(_system: str, message: str, _tools: list, **_kwargs: object) -> dict:
        import json

        payload = json.loads(message)
        observations = payload["observations"]
        if not observations:
            facts = json.loads(payload["facts"])
            query = facts["behavior_sequence"][0]["action"]
            return {"content": "", "tool_calls": [{"name": "search_laws", "args": {"query": query}}], "has_tool_call": True}
        if len(observations) == 1:
            article_id = observations[0]["result"]["candidates"][0]["article_id"]
            return {"content": "", "tool_calls": [{"name": "get_article", "args": {"article_id": article_id}}], "has_tool_call": True}
        article = observations[-1]["result"]["article"]
        article_id = article["article_id"]
        element = article["required_elements"][0]
        return {"content": json.dumps({"article_ids": [article_id], "matched_elements": {article_id: [element]}, "confidence": "medium"}, ensure_ascii=False), "tool_calls": [], "has_tool_call": False}

    class FakeRagService:
        def __init__(self, **_: object) -> None:
            pass

        async def initialize_retriever(self, query: str) -> None:
            rag_queries.append(query)

        async def retrieve_documents(self, query: str) -> list[str]:
            article = "第二百六十三条" if "持刀" in query else "第二百九十三条"
            return [f"中华人民共和国刑法{article}"]

    law_data = {
        "chapters": [
            {
                "chapter": "侵犯财产罪",
                "articles": [
                    {
                        "article_number": "第二百六十三条",
                        "title": "抢劫罪",
                        "content": "持刀威胁并抢走手机",
                        "elements": ["暴力取财"],
                        "base_sentence": "三年以上十年以下有期徒刑",
                        "charge_tags": ["抢劫"],
                        "common_keywords": ["持刀威胁并抢走手机"],
                    },
                ],
            },
            {
                "chapter": "扰乱公共秩序罪",
                "articles": [
                    {
                        "article_number": "第二百九十三条",
                        "title": "寻衅滋事罪",
                        "content": "徒手推搡",
                        "elements": ["随意殴打"],
                        "base_sentence": "五年以下有期徒刑",
                        "charge_tags": ["寻衅滋事"],
                        "common_keywords": ["徒手推搡"],
                    },
                ],
            },
        ]
    }

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        return state

    async def extract_facts(facts_raw: list[str]) -> tuple[dict, ArtifactSource]:
        complete = {
            "incident_time": None,
            "incident_location": None,
            "parties": [],
            "evidence_mentioned": [],
            "arrest_status": None,
            "surrender": None,
            "victim_forgiveness": None,
            "prior_record": None,
        }
        if any("持刀" in fact for fact in facts_raw):
            payload = complete | {
                "behavior_sequence": [
                    {
                        "time": "案发时",
                        "actor": "当事人",
                        "action": "持刀威胁并抢走手机",
                        "method": "持刀威胁",
                        "target": "手机",
                    }
                ],
                "consequence": "手机被夺",
            }
            return payload, ArtifactSource.TOOL_CALL
        payload = complete | {
            "behavior_sequence": [
                {
                    "time": "案发时",
                    "actor": "当事人",
                    "action": "徒手推搡",
                    "method": "徒手",
                    "target": "他人",
                }
            ],
            "consequence": "轻微伤",
        }
        return payload, ArtifactSource.TOOL_CALL

    async def analyze_coverage(facts: dict, _: list[dict]) -> dict:
        changed_charge = "持刀" in str(facts.get("behavior_sequence", []))
        return {
            "total_elements": 1,
            "covered_elements": int(changed_charge),
            "coverage_rate": 1.0 if changed_charge else 0.0,
            "missing_elements": [] if changed_charge else ["决定性行为"],
            "weak_elements": [],
            "source": "json_knowledge",
        }

    async def risk_assessor(state: ConsultationState) -> ConsultationState:
        risk_inputs.append((state["facts_structured"], state["applied_laws"]))
        state["current_agent"] = "RiskAssessor"
        state["risk_assessment"] = {"risk_level": "high"}
        return state

    async def service_planner(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "ServicePlanner"
        state["service_plan"] = {"next_step": "lawyer_review"}
        state["report_draft"] = "确定性报告草案"
        return state

    initial_state = make_consultation_state(
        session_id=session_id,
        consent_given=False,
        facts_raw=["先前仅发生徒手推搡"],
        current_input=None,
        facts_structured={},
        applied_laws=[],
        conversation_history=[],
        lawyer_decision=None,
    )

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch("app.agents.fact_digger._extract_structured_facts", extract_facts),
        patch("app.agents.fact_digger._analyze_coverage", analyze_coverage),
        patch(
            "app.agents.fact_digger._generate_follow_up_questions",
            new_callable=AsyncMock,
            return_value=["请补充是否使用工具"],
        ),
        patch(
            "app.agents.fact_digger._generate_fact_summary",
            new_callable=AsyncMock,
            return_value="事实摘要",
        ),
        patch("app.rag.rag_service.RagService", FakeRagService),
        patch("app.agents.legal_research.llm_gateway.generate_with_tools", new=law_decision),
        patch("app.agents.law_ref.load_criminal_law_data", return_value=law_data),
        patch(
            "app.agents.law_ref.extract_structured_laws",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("app.orchestrator.workflow.risk_assessor_node", risk_assessor),
        patch("app.orchestrator.workflow.service_planner_node", service_planner),
    ):
        orchestrator = ConsultationOrchestrator()
        await orchestrator.start_workflow(initial_state)
        paused = await orchestrator.resume_workflow(session_id, {"consent_given": True})

        assert paused["facts_coverage_rate"] == 0.0
        assert await orchestrator.get_next_node(session_id) == "fact_intake"

        result = await orchestrator.resume_workflow(
            session_id,
            {"current_input": "本轮新增决定性事实：持刀威胁并抢走手机"},
        )

    assert "持刀威胁并抢走手机" in rag_queries[-1]
    assert [law["article_number"] for law in result["applied_laws"]] == ["第二百六十三条"]
    assert len(risk_inputs) == 1
    risk_facts, risk_laws = risk_inputs[0]
    assert "持刀威胁并抢走手机" in str(risk_facts)
    assert [law["article_number"] for law in risk_laws] == ["第二百六十三条"]
    assert all(law["article_number"] != "第二百九十三条" for law in risk_laws)


@pytest.mark.asyncio
async def test_compiled_workflow_retry_does_not_append_consumed_input_twice():
    """后续节点失败后从 checkpoint 重试不得再次摄取同一条输入。"""
    session_id = "p0-retry-current-input-once"
    law_calls = 0

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        return state

    async def extract_facts(facts_raw: list[str]) -> tuple[dict, ArtifactSource]:
        return {"behavior_sequence": list(facts_raw)}, ArtifactSource.TOOL_CALL

    async def law_ref(state: ConsultationState) -> ConsultationState:
        nonlocal law_calls
        law_calls += 1
        if law_calls == 2:
            raise RuntimeError("检索阶段暂时失败")
        state["current_agent"] = "LawRef"
        state["applied_laws"] = [{"article_number": "第二百六十三条", "elements": []}]
        return state

    async def fact_coverage(state: ConsultationState) -> ConsultationState:
        complete = "本轮补充事实" in state["facts_raw"]
        state["current_agent"] = "FactDigger"
        state["facts_coverage_rate"] = 1.0 if complete else 0.0
        state["pending_questions"] = [] if complete else ["请补充事实"]
        state["fact_law_loop_count"] = state.get("fact_law_loop_count", 0) + 1
        return state

    async def risk_assessor(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "RiskAssessor"
        state["risk_assessment"] = {"risk_level": "medium"}
        return state

    async def service_planner(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "ServicePlanner"
        state["service_plan"] = {"next_step": "lawyer_review"}
        state["report_draft"] = "确定性报告草案"
        return state

    initial_state = make_consultation_state(
        session_id=session_id,
        consent_given=False,
        facts_raw=["上一轮事实"],
        current_input=None,
        facts_structured={},
        applied_laws=[],
        conversation_history=[],
        lawyer_decision=None,
    )

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch("app.agents.fact_digger._extract_structured_facts", extract_facts),
        patch("app.orchestrator.workflow.law_ref_node", law_ref),
        patch("app.orchestrator.workflow.fact_coverage_node", fact_coverage),
        patch("app.orchestrator.workflow.risk_assessor_node", risk_assessor),
        patch("app.orchestrator.workflow.service_planner_node", service_planner),
    ):
        orchestrator = ConsultationOrchestrator()
        await orchestrator.start_workflow(initial_state)
        await orchestrator.resume_workflow(session_id, {"consent_given": True})

        with pytest.raises(RuntimeError, match="检索阶段暂时失败"):
            await orchestrator.resume_workflow(
                session_id,
                {"current_input": "本轮补充事实"},
            )

        failed_snapshot = await orchestrator.get_snapshot(session_id)
        assert failed_snapshot is not None
        assert failed_snapshot.values["facts_raw"] == ["上一轮事实", "本轮补充事实"]
        assert failed_snapshot.values["current_input"] is None
        assert failed_snapshot.next == ("law_ref",)

        result = await orchestrator.resume_workflow(
            session_id,
            {"current_input": "本轮补充事实"},
        )

    assert result["facts_raw"] == ["上一轮事实", "本轮补充事实"]
    assert result["current_input"] is None
    assert law_calls == 3


@pytest.mark.asyncio
async def test_high_risk_routes_to_human_alert():
    async with _workflow_fixture("phase5-alert", fact_coverages=[0.0], high_risk=True) as (orchestrator, trace):
        result = await orchestrator.resume_workflow("phase5-alert", {"consent_given": True})

        assert result["current_agent"] == "HumanAlert"
        assert result["lawyer_review_needed"] is True
        assert trace == ["receptionist", "fact_intake", "human_alert"]
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
        return_value=(
            {"behavior_sequence": ["补充事实"]},
            ArtifactSource.TOOL_CALL,
        ),
    ) as extract_facts:
        result = await _fact_intake_workflow_node(state)

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
