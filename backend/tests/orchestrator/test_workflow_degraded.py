"""FactDigger/LawRef 有限重试与降级路径回归。"""

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.fact_digger import fact_coverage_node, fact_digger_node
from app.agents.law_ref import LawSearchResults, law_ref_node
from app.orchestrator.workflow import (
    ConsultationOrchestrator,
    _fact_digger_workflow_node,
    _fact_intake_workflow_node,
    check_facts_sufficient,
    human_review_node,
    lawyer_decision,
)
from app.state.consultation_state import ConsultationState
from tests.factories import make_applied_law, make_consultation_state


async def _fixed_facts(_: list[str]) -> dict:
    return {"behavior_sequence": ["发生争执"], "consequence": "有人受伤"}


@pytest.mark.asyncio
async def test_law_ref_distinguishes_no_match_from_dependency_failure():
    no_match_state = make_consultation_state(
        session_id="no-match-session",
        facts_structured={"behavior_sequence": ["无法识别的行为"], "consequence": ""},
    )
    dependency_state = make_consultation_state(
        session_id="dependency-session",
        facts_structured={"behavior_sequence": ["发生争执"], "consequence": "有人受伤"},
    )
    rag_failure_state = make_consultation_state(
        session_id="rag-failure-session",
        facts_structured={"behavior_sequence": ["无法识别的行为"], "consequence": ""},
    )

    with (
        patch(
            "app.agents.law_ref.load_criminal_law_data",
            return_value={"chapters": [{"chapter": "测试", "articles": []}]},
        ),
        patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
        patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, return_value=[]),
    ):
        no_match_result = await law_ref_node(no_match_state)

    with (
        patch("app.agents.law_ref.load_criminal_law_data", return_value={"chapters": []}),
        patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
        patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, return_value=[]),
    ):
        dependency_result = await law_ref_node(dependency_state)

    with (
        patch(
            "app.agents.law_ref.load_criminal_law_data",
            return_value={"chapters": [{"chapter": "测试", "articles": []}]},
        ),
        patch(
            "app.agents.law_ref.search_laws_by_rag",
            new_callable=AsyncMock,
            return_value=LawSearchResults(dependency_failed=True),
        ),
        patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, return_value=[]),
    ):
        rag_failure_result = await law_ref_node(rag_failure_state)

    assert no_match_result["law_search_status"] == "no_law_match"
    assert dependency_result["law_search_status"] == "dependency_failure"
    assert rag_failure_result["law_search_status"] == "dependency_failure"
    assert no_match_result["conversation_history"][-1]["session_id"] == "no-match-session"
    assert dependency_result["conversation_history"][-1]["session_id"] == "dependency-session"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "message_fragment", "termination_reason"),
    [
        ("no_law_match", "未检索到匹配法条", "no_law_match_retry_exhausted"),
        ("dependency_failure", "知识或模型服务当前不可用", "dependency_failure_retry_exhausted"),
    ],
)
async def test_non_fact_failures_stop_after_three_retries(
    failure_kind: str,
    message_fragment: str,
    termination_reason: str,
):
    state = make_consultation_state(
        session_id=f"{failure_kind}-session",
        facts_raw=["已有案情"],
        facts_structured={"behavior_sequence": ["发生争执"]},
        applied_laws=[],
        law_search_status=failure_kind,
    )

    with patch("app.agents.fact_digger._extract_structured_facts", side_effect=_fixed_facts):
        for _ in range(3):
            state = await fact_digger_node(state)

    assert state["fact_law_loop_count"] == 3
    assert [item["attempt"] for item in state["fact_law_attempts"]] == [1, 2, 3]
    assert [item["outcome"] for item in state["fact_law_attempts"]] == [failure_kind] * 3
    assert {item["session_id"] for item in state["fact_law_attempts"]} == {
        f"{failure_kind}-session"
    }
    assert state["workflow_status"] == "degraded"
    assert state["fact_law_termination_reason"] == termination_reason
    assert state["lawyer_review_needed"] is True
    assert state["pending_questions"]
    assert message_fragment in state["final_output"]
    assert check_facts_sufficient(state) == "degraded"


@pytest.mark.asyncio
async def test_missing_facts_uses_fact_prompt_and_records_outcome():
    state = make_consultation_state(
        session_id="missing-facts-session",
        facts_raw=["只说明了时间"],
        facts_structured={"incident_time": "昨晚"},
        applied_laws=[
            make_applied_law(
                elements=[
                    {"name": "时间", "key": "time"},
                    {"name": "地点", "key": "location"},
                ]
            )
        ],
    )

    with (
        patch(
            "app.agents.fact_digger._extract_structured_facts",
            new_callable=AsyncMock,
            return_value={"incident_time": "昨晚"},
        ),
        patch(
            "app.agents.fact_digger._generate_follow_up_questions",
            new_callable=AsyncMock,
            return_value=["事发地点在哪里？"],
        ),
    ):
        result = await fact_digger_node(state)

    assert result["fact_law_loop_count"] == 1
    assert result["fact_law_attempts"][-1] == {
        "attempt": 1,
        "outcome": "missing_facts",
        "session_id": "missing-facts-session",
    }
    assert result["pending_questions"] == ["事发地点在哪里？"]
    assert "补充以下信息" in result["final_output"]
    assert "服务当前不可用" not in result["final_output"]


@pytest.mark.asyncio
async def test_eleven_resumes_exit_wait_for_user_with_auditable_degraded_state():
    session_id = "eleven-resumes-session"

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        return state

    initial_state = make_consultation_state(
        session_id=session_id,
        consent_given=False,
        facts_raw=["已有案情"],
        facts_structured={"behavior_sequence": ["发生争执"], "consequence": "有人受伤"},
        applied_laws=[],
        conversation_history=[],
    )

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch("app.agents.fact_digger._extract_structured_facts", side_effect=_fixed_facts),
        patch("app.agents.law_ref.load_criminal_law_data", return_value={"chapters": []}),
        patch("app.agents.law_ref.search_laws_by_rag", new_callable=AsyncMock, return_value=[]),
        patch("app.agents.law_ref.search_laws_by_keyword", new_callable=AsyncMock, return_value=[]),
    ):
        orchestrator = ConsultationOrchestrator()
        await orchestrator.start_workflow(initial_state)

        state = await orchestrator.resume_workflow(session_id, {"consent_given": True})
        for index in range(10):
            state = await orchestrator.resume_workflow(
                session_id,
                {"current_input": f"第 {index + 1} 次重试"},
            )

    assert state["current_agent"] == "HumanReview"
    assert state["workflow_status"] == "degraded"
    assert state["fact_law_loop_count"] == 3
    assert state["fact_law_termination_reason"] == "dependency_failure_retry_exhausted"
    assert len(state["fact_law_attempts"]) == 3
    assert state["fact_law_attempts"][-1]["session_id"] == session_id
    assert "知识或模型服务当前不可用" in state["final_output"]
    assert await orchestrator.get_next_node(session_id) == "human_review"


@pytest.mark.asyncio
async def test_lawyer_retry_clears_degraded_window_without_erasing_attempt_history():
    historical_attempts = [
        {"attempt": 1, "outcome": "dependency_failure", "session_id": "retry-session"},
        {"attempt": 2, "outcome": "dependency_failure", "session_id": "retry-session"},
        {"attempt": 3, "outcome": "dependency_failure", "session_id": "retry-session"},
    ]
    state = make_consultation_state(
        session_id="retry-session",
        current_agent="HumanReview",
        current_input="触发降级前的旧输入",
        facts_raw=["已确认案情"],
        facts_structured={"incident_time": "昨晚"},
        applied_laws=[make_applied_law(elements=[{"name": "时间", "key": "time"}])],
        fact_law_loop_count=3,
        fact_law_attempts=historical_attempts,
        fact_law_termination_reason="dependency_failure_retry_exhausted",
        law_search_status="dependency_failure",
        workflow_status="degraded",
        fact_law_failure_streak=3,
        fact_law_last_failure="dependency_failure",
        lawyer_review_needed=True,
        awaiting_lawyer_review=True,
        lawyer_decision="revise_facts",
    )

    reviewed = await human_review_node(state)
    assert lawyer_decision(reviewed) == "revise_facts"

    with (
        patch(
            "app.agents.fact_digger._extract_structured_facts",
            new_callable=AsyncMock,
            return_value={"incident_time": "昨晚"},
        ),
        patch(
            "app.agents.fact_digger._generate_fact_summary",
            new_callable=AsyncMock,
            return_value="已确认事实摘要",
        ),
    ):
        intaken = await _fact_intake_workflow_node(reviewed)
        retried = await _fact_digger_workflow_node(intaken)

    assert retried["workflow_status"] is None
    assert retried["fact_law_termination_reason"] is None
    assert retried["law_search_status"] is None
    assert retried["fact_law_failure_streak"] == 0
    assert retried["fact_law_last_failure"] is None
    assert retried["lawyer_review_needed"] is False
    assert retried["lawyer_decision"] is None
    assert retried["current_input"] is None
    assert retried["fact_law_attempts"][:3] == historical_attempts
    assert retried["fact_law_attempts"][-1] == {
        "attempt": 4,
        "outcome": "complete",
        "session_id": "retry-session",
    }
    assert check_facts_sufficient(retried) == "complete"


@pytest.mark.asyncio
async def test_lawyer_retry_starts_a_fresh_dependency_failure_streak():
    historical_attempts = [
        {"attempt": 1, "outcome": "dependency_failure", "session_id": "fresh-window"},
        {"attempt": 2, "outcome": "dependency_failure", "session_id": "fresh-window"},
        {"attempt": 3, "outcome": "dependency_failure", "session_id": "fresh-window"},
    ]
    state = make_consultation_state(
        session_id="fresh-window",
        facts_raw=["已有案情"],
        facts_structured={"behavior_sequence": ["发生争执"]},
        applied_laws=[],
        fact_law_loop_count=3,
        fact_law_attempts=historical_attempts,
        fact_law_termination_reason="dependency_failure_retry_exhausted",
        fact_law_failure_streak=3,
        fact_law_last_failure="dependency_failure",
        law_search_status="dependency_failure",
        workflow_status="degraded",
        lawyer_review_needed=True,
        awaiting_lawyer_review=True,
        lawyer_decision="revise_facts",
    )

    reviewed = await human_review_node(state)
    with patch("app.agents.fact_digger._extract_structured_facts", side_effect=_fixed_facts):
        restarted = await _fact_intake_workflow_node(reviewed)

    assert restarted["workflow_status"] is None
    assert restarted["fact_law_failure_streak"] == 0
    assert restarted["fact_law_last_failure"] is None
    assert restarted["fact_law_attempts"][:3] == historical_attempts

    restarted["law_search_status"] = "dependency_failure"
    with patch("app.agents.fact_digger._extract_structured_facts", side_effect=_fixed_facts):
        failed_again = await fact_digger_node(restarted)

    assert failed_again["fact_law_loop_count"] == 4
    assert failed_again["fact_law_failure_streak"] == 1
    assert failed_again["fact_law_last_failure"] == "dependency_failure"
    assert failed_again["workflow_status"] is None
    assert failed_again["fact_law_termination_reason"] is None
    assert failed_again["fact_law_attempts"][-1] == {
        "attempt": 4,
        "outcome": "dependency_failure",
        "session_id": "fresh-window",
    }
    assert check_facts_sufficient(failed_again) == "loop"


@pytest.mark.asyncio
async def test_alternating_non_fact_failures_share_one_finite_retry_window():
    state = make_consultation_state(
        session_id="alternating-failures",
        facts_raw=["已有案情"],
        facts_structured={"behavior_sequence": ["发生争执"]},
        applied_laws=[],
        law_search_status="no_law_match",
    )

    with patch("app.agents.fact_digger._extract_structured_facts", side_effect=_fixed_facts):
        state = await fact_digger_node(state)
        assert state.get("workflow_status") != "degraded"
        assert state["fact_law_failure_streak"] == 1

        state["law_search_status"] = "dependency_failure"
        state = await fact_digger_node(state)
        assert state.get("workflow_status") != "degraded"
        assert state["fact_law_failure_streak"] == 2

        state["law_search_status"] = "no_law_match"
        state = await fact_digger_node(state)

    assert state["workflow_status"] == "degraded"
    assert state["fact_law_failure_streak"] == 3
    assert state["fact_law_last_failure"] == "no_law_match"
    assert state["fact_law_termination_reason"] == "no_law_match_retry_exhausted"
    assert [item["outcome"] for item in state["fact_law_attempts"]] == [
        "no_law_match",
        "dependency_failure",
        "no_law_match",
    ]
    assert [item["session_id"] for item in state["fact_law_attempts"]] == [
        "alternating-failures",
        "alternating-failures",
        "alternating-failures",
    ]
    assert check_facts_sufficient(state) == "degraded"


@pytest.mark.asyncio
async def test_unverified_law_candidates_use_finite_non_fact_retry_window():
    """未连接 allowlist 的候选不能伪装成普通事实不足并耗尽十轮。"""
    state = make_consultation_state(
        session_id="unverified-only",
        facts_raw=["已有案情"],
        facts_structured={"behavior_sequence": ["发生争执"]},
        applied_laws=[
            {
                "article_number": "第999条",
                "required_elements": ["虚构要件"],
                "data_source": "rag_unverified",
            }
        ],
        law_search_status="no_law_match",
    )

    with patch(
        "app.agents.fact_digger._generate_follow_up_questions",
        new_callable=AsyncMock,
        return_value=["请补充事实"],
    ):
        for _ in range(3):
            state = await fact_coverage_node(state)

    assert state["workflow_status"] == "degraded"
    assert state["fact_law_loop_count"] == 3
    assert [item["outcome"] for item in state["fact_law_attempts"]] == ["no_law_match"] * 3
    assert state["fact_law_termination_reason"] == "no_law_match_retry_exhausted"
    assert check_facts_sufficient(state) == "degraded"
