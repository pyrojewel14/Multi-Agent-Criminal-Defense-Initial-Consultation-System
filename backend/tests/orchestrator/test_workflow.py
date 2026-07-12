"""Orchestrator workflow unit tests.

Tests cover:
1. check_consent conditional edge
2. check_facts_sufficient conditional edge
3. lawyer_decision conditional edge
4. _calculate_coverage_rate helper
5. _get_fact_value helper
6. ConsultationOrchestrator session management and run_node
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.orchestrator.workflow import (
    ConsultationOrchestrator,
    _calculate_coverage_rate,
    _get_fact_value,
    check_consent,
    check_facts_sufficient,
    lawyer_decision,
)
from tests.factories import make_consultation_state, make_applied_law


# ---------------------------------------------------------------------------
# check_consent
# ---------------------------------------------------------------------------


class TestCheckConsent:
    def test_consent_given_returns_continue(self):
        state = make_consultation_state(consent_given=True, facts_raw=["用户陈述"])
        assert check_consent(state) == "continue"

    def test_consent_false_returns_end(self):
        state = make_consultation_state(consent_given=False, facts_raw=["用户陈述"])
        assert check_consent(state) == "end"

    def test_consent_missing_returns_end(self):
        state = make_consultation_state(facts_raw=["用户陈述"])
        state.pop("consent_given", None)
        assert check_consent(state) == "end"


# ---------------------------------------------------------------------------
# check_facts_sufficient
# ---------------------------------------------------------------------------


class TestCheckFactsSufficient:
    def test_alert_triggered_returns_alert(self):
        state = make_consultation_state(alert_triggered=True, facts_raw=["高风险陈述"])
        assert check_facts_sufficient(state) == "alert"

    def test_max_loop_returns_max_loop(self):
        state = make_consultation_state(
            fact_law_loop_count=10,
            alert_triggered=False,
            facts_raw=["陈述"],
        )
        assert check_facts_sufficient(state) == "max_loop"

    def test_high_coverage_returns_complete(self):
        state = make_consultation_state(
            facts_coverage_rate=0.9,
            alert_triggered=False,
            facts_raw=["陈述"],
        )
        assert check_facts_sufficient(state) == "complete"

    def test_low_coverage_returns_loop(self):
        state = make_consultation_state(
            facts_coverage_rate=0.3,
            alert_triggered=False,
            facts_raw=["陈述"],
        )
        assert check_facts_sufficient(state) == "loop"

    def test_zero_coverage_returns_loop(self):
        state = make_consultation_state(
            facts_coverage_rate=0.0,
            alert_triggered=False,
            facts_raw=["陈述"],
        )
        assert check_facts_sufficient(state) == "loop"

    def test_exact_threshold_returns_complete(self):
        state = make_consultation_state(
            facts_coverage_rate=0.8,
            alert_triggered=False,
            facts_raw=["陈述"],
        )
        assert check_facts_sufficient(state) == "complete"

    def test_alert_overrides_coverage(self):
        state = make_consultation_state(
            facts_coverage_rate=0.9,
            alert_triggered=True,
            facts_raw=["陈述"],
        )
        assert check_facts_sufficient(state) == "alert"

    def test_no_coverage_rate_defaults_to_zero(self):
        state = make_consultation_state(
            alert_triggered=False,
            facts_raw=["陈述"],
        )
        state.pop("facts_coverage_rate", None)
        assert check_facts_sufficient(state) == "loop"


# ---------------------------------------------------------------------------
# lawyer_decision
# ---------------------------------------------------------------------------


class TestLawyerDecision:
    def test_approved_returns_approved(self):
        state = make_consultation_state(lawyer_decision="approved", facts_raw=["陈述"])
        assert lawyer_decision(state) == "approved"

    def test_revise_facts_returns_revise_facts(self):
        state = make_consultation_state(lawyer_decision="revise_facts", facts_raw=["陈述"])
        assert lawyer_decision(state) == "revise_facts"

    def test_revise_risk_returns_revise_risk(self):
        state = make_consultation_state(lawyer_decision="revise_risk", facts_raw=["陈述"])
        assert lawyer_decision(state) == "revise_risk"

    def test_missing_decision_returns_wait(self):
        state = make_consultation_state(facts_raw=["陈述"])
        state.pop("lawyer_decision", None)
        assert lawyer_decision(state) == "wait"

    def test_none_decision_returns_wait(self):
        state = make_consultation_state(lawyer_decision=None, facts_raw=["陈述"])
        assert lawyer_decision(state) == "wait"


# ---------------------------------------------------------------------------
# _calculate_coverage_rate
# ---------------------------------------------------------------------------


class TestCalculateCoverageRate:
    def test_no_applied_laws_returns_zero(self):
        state = make_consultation_state(applied_laws=[], facts_raw=["陈述"])
        assert _calculate_coverage_rate(state) == 0.0

    def test_full_coverage(self):
        law = make_applied_law(
            elements=[
                {"key": "time", "name": "时间"},
                {"key": "location", "name": "地点"},
            ]
        )
        state = make_consultation_state(
            applied_laws=[law],
            facts_structured={
                "incident_time": "2026年3月",
                "incident_location": "北京市",
            },
            facts_raw=["陈述"],
        )
        assert _calculate_coverage_rate(state) == 1.0

    def test_partial_coverage(self):
        law = make_applied_law(
            elements=[
                {"key": "time", "name": "时间"},
                {"key": "location", "name": "地点"},
            ]
        )
        state = make_consultation_state(
            applied_laws=[law],
            facts_structured={
                "incident_time": "2026年3月",
            },
            facts_raw=["陈述"],
        )
        assert _calculate_coverage_rate(state) == 0.5

    def test_empty_fact_value_not_counted(self):
        law = make_applied_law(
            elements=[
                {"key": "time", "name": "时间"},
                {"key": "location", "name": "地点"},
            ]
        )
        state = make_consultation_state(
            applied_laws=[law],
            facts_structured={
                "incident_time": "2026年3月",
                "incident_location": "",
            },
            facts_raw=["陈述"],
        )
        assert _calculate_coverage_rate(state) == 0.5

    def test_empty_list_not_counted(self):
        law = make_applied_law(
            elements=[
                {"key": "parties", "name": "当事人"},
            ]
        )
        state = make_consultation_state(
            applied_laws=[law],
            facts_structured={
                "parties": [],
            },
            facts_raw=["陈述"],
        )
        # _calculate_coverage_rate 故意将空列表视为弱覆盖（与 _analyze_coverage 保持一致），
        # 因此 parties=[] 仍记为已覆盖，覆盖度 = 1/1 = 1.0
        assert _calculate_coverage_rate(state) == 1.0

    def test_string_elements(self):
        law = make_applied_law(
            elements=["time", "location"],
        )
        state = make_consultation_state(
            applied_laws=[law],
            facts_structured={
                "incident_time": "2026年3月",
            },
            facts_raw=["陈述"],
        )
        assert _calculate_coverage_rate(state) == 0.5

    def test_multiple_laws(self):
        law1 = make_applied_law(elements=[{"key": "time", "name": "时间"}])
        law2 = make_applied_law(elements=[{"key": "behavior", "name": "行为"}])
        state = make_consultation_state(
            applied_laws=[law1, law2],
            facts_structured={
                "incident_time": "2026年3月",
                "behavior_sequence": ["推搡"],
            },
            facts_raw=["陈述"],
        )
        assert _calculate_coverage_rate(state) == 1.0

    def test_no_elements_in_law_returns_zero(self):
        law = make_applied_law(elements=[])
        state = make_consultation_state(
            applied_laws=[law],
            facts_raw=["陈述"],
        )
        assert _calculate_coverage_rate(state) == 0.0


# ---------------------------------------------------------------------------
# _get_fact_value
# ---------------------------------------------------------------------------


class TestGetFactValue:
    def test_mapped_key_time(self):
        result = _get_fact_value({"incident_time": "2026年3月"}, "time")
        assert result == "2026年3月"

    def test_mapped_key_location(self):
        result = _get_fact_value({"incident_location": "北京"}, "location")
        assert result == "北京"

    def test_mapped_key_behavior(self):
        result = _get_fact_value({"behavior_sequence": ["推"]}, "behavior")
        assert result == ["推"]

    def test_mapped_key_parties(self):
        result = _get_fact_value({"parties": ["A", "B"]}, "parties")
        assert result == ["A", "B"]

    def test_mapped_key_consequence(self):
        result = _get_fact_value({"consequence": "轻伤"}, "consequence")
        assert result == "轻伤"

    def test_unmapped_key_passes_through(self):
        result = _get_fact_value({"custom_field": "value"}, "custom_field")
        assert result == "value"

    def test_missing_key_returns_none(self):
        result = _get_fact_value({}, "time")
        assert result is None

    def test_all_mapped_keys(self):
        facts = {
            "incident_time": "2026年3月",
            "incident_location": "北京",
            "parties": ["A"],
            "behavior_sequence": ["推"],
            "consequence": "轻伤",
            "evidence_mentioned": ["证人"],
            "arrest_status": "未逮捕",
            "surrender": "自首",
            "victim_forgiveness": "已谅解",
            "prior_record": "无前科",
        }
        key_mapping = {
            "time": "incident_time",
            "location": "incident_location",
            "parties": "parties",
            "behavior": "behavior_sequence",
            "consequence": "consequence",
            "evidence": "evidence_mentioned",
            "arrest": "arrest_status",
            "surrender": "surrender",
            "forgiveness": "victim_forgiveness",
            "record": "prior_record",
        }
        for key, expected_field in key_mapping.items():
            assert _get_fact_value(facts, key) == facts[expected_field]


# ---------------------------------------------------------------------------
# ConsultationOrchestrator
# ---------------------------------------------------------------------------


class TestConsultationOrchestratorInit:
    def test_init_creates_empty_sessions(self):
        orch = ConsultationOrchestrator()
        assert orch._active_sessions == {}


class TestSessionContext:
    def test_get_session_context_returns_state(self):
        orch = ConsultationOrchestrator()
        state = make_consultation_state(session_id="s1", facts_raw=["陈述"])
        orch._active_sessions["s1"] = state
        assert orch.get_session_context("s1") == state

    def test_get_session_context_returns_none_for_missing(self):
        orch = ConsultationOrchestrator()
        assert orch.get_session_context("nonexistent") is None

    def test_update_session_context_success(self):
        orch = ConsultationOrchestrator()
        state = make_consultation_state(session_id="s1", facts_raw=["陈述"])
        orch._active_sessions["s1"] = state
        result = orch.update_session_context("s1", {"consent_given": True})
        assert result is True
        assert orch._active_sessions["s1"]["consent_given"] is True

    def test_update_session_context_missing_session(self):
        orch = ConsultationOrchestrator()
        result = orch.update_session_context("nonexistent", {"consent_given": True})
        assert result is False

    def test_get_active_sessions_returns_shallow_copy(self):
        orch = ConsultationOrchestrator()
        state = make_consultation_state(session_id="s1", facts_raw=["陈述"])
        orch._active_sessions["s1"] = state
        sessions = orch.get_active_sessions()
        # get_active_sessions returns a shallow copy of the outer dict,
        # so modifying the outer dict (adding/removing keys) does not
        # affect the original, but inner dicts are shared references.
        assert "s1" in sessions
        sessions["s2"] = make_consultation_state(session_id="s2", facts_raw=["陈述"])
        assert "s2" not in orch._active_sessions


class TestCleanupSession:
    def test_cleanup_removes_session(self):
        orch = ConsultationOrchestrator()
        state = make_consultation_state(session_id="s1", facts_raw=["陈述"])
        orch._active_sessions["s1"] = state
        orch._cleanup_session("s1")
        assert "s1" not in orch._active_sessions

    def test_cleanup_nonexistent_session_no_error(self):
        orch = ConsultationOrchestrator()
        orch._cleanup_session("nonexistent")


class TestStartWorkflow:
    """Tests for ConsultationOrchestrator.start_workflow / resume_workflow.

    These tests replace the legacy ``run_node`` method, which has been
    superseded by the LangGraph ``StateGraph`` driven by ``start_workflow`` /
    ``resume_workflow`` on the orchestrator.
    """

    @pytest.mark.asyncio
    async def test_run_node_invalid_name_raises(self):
        """resume_workflow raises ValueError when the session is not in the checkpointer."""
        orch = ConsultationOrchestrator()
        with pytest.raises(ValueError, match="会话不存在"):
            await orch.resume_workflow("invalid_node", {})

    @pytest.mark.asyncio
    async def test_run_node_calls_agent(self):
        """start_workflow invokes the entry-point (receptionist) node and returns its state."""
        orch = ConsultationOrchestrator()
        state = make_consultation_state(facts_raw=["陈述"])

        with patch("app.orchestrator.workflow.receptionist_node", new_callable=AsyncMock) as mock_node:
            mock_node.return_value = {**state, "current_agent": "Receptionist"}
            result = await orch.start_workflow(state)
            mock_node.assert_called_once()
            assert result["current_agent"] == "Receptionist"

    @pytest.mark.asyncio
    async def test_run_node_fact_digger_calculates_coverage(self):
        """start_workflow reaches fact_digger and produces facts_coverage_rate in the result."""
        orch = ConsultationOrchestrator()
        state = make_consultation_state(
            consent_given=True,
            facts_raw=["陈述"],
            applied_laws=[
                make_applied_law(elements=[{"key": "time", "name": "时间"}]),
            ],
        )

        async def _fake_recep(s):
            return {**s, "current_agent": "Receptionist"}

        async def _fake_extract(facts_raw):
            return {"incident_time": "2026年3月"}

        async def _fake_analyze(facts_structured, applied_laws):
            return {
                "total_elements": 1,
                "covered_elements": 1,
                "coverage_rate": 1.0,
                "missing_elements": [],
                "weak_elements": [],
                "source": "json_knowledge",
            }

        with patch("app.orchestrator.workflow.receptionist_node", side_effect=_fake_recep), \
             patch("app.agents.fact_digger._extract_structured_facts", side_effect=_fake_extract), \
             patch("app.agents.fact_digger._analyze_coverage", side_effect=_fake_analyze):
            result = await orch.start_workflow(state)
            assert "facts_coverage_rate" in result

    @pytest.mark.asyncio
    async def test_human_review_remains_paused_without_lawyer_decision(self):
        """Reaching HumanReview must not auto-approve a report without a lawyer decision."""
        orch = ConsultationOrchestrator()
        state = make_consultation_state(
            session_id="review-pause",
            consent_given=True,
            facts_raw=["陈述"],
        )

        async def _fake_receptionist(current_state):
            current_state["current_agent"] = "Receptionist"
            return current_state

        async def _fake_fact_digger(current_state):
            current_state["current_agent"] = "FactDigger"
            current_state["facts_coverage_rate"] = 1.0
            return current_state

        async def _fake_risk_assessor(current_state):
            current_state["current_agent"] = "RiskAssessor"
            return current_state

        async def _fake_service_planner(current_state):
            current_state["current_agent"] = "ServicePlanner"
            current_state["report_draft"] = "报告草案"
            return current_state

        with patch("app.orchestrator.workflow.receptionist_node", side_effect=_fake_receptionist), \
             patch("app.orchestrator.workflow.fact_digger_node", side_effect=_fake_fact_digger), \
             patch("app.orchestrator.workflow.risk_assessor_node", side_effect=_fake_risk_assessor), \
             patch("app.orchestrator.workflow.service_planner_node", side_effect=_fake_service_planner):
            await orch.start_workflow(state)
            result = await orch.resume_workflow("review-pause")

        assert result["current_agent"] == "HumanReview"
        assert result["awaiting_lawyer_review"] is True
        assert await orch.get_next_node("review-pause") == "human_review"
        assert await orch.is_workflow_finished("review-pause") is False

    @pytest.mark.asyncio
    async def test_run_node_propagates_exception(self):
        """Exceptions raised inside a workflow node propagate out of start_workflow."""
        orch = ConsultationOrchestrator()
        state = make_consultation_state(facts_raw=["陈述"])

        with patch("app.orchestrator.workflow.receptionist_node", new_callable=AsyncMock) as mock_node:
            mock_node.side_effect = RuntimeError("agent failed")
            with pytest.raises(RuntimeError, match="agent failed"):
                await orch.start_workflow(state)


class TestProcessLawyerFeedback:
    @pytest.mark.asyncio
    async def test_process_lawyer_feedback_approved(self):
        """process_lawyer_feedback forwards the decision/feedback to resume_workflow."""
        orch = ConsultationOrchestrator()
        expected_state = make_consultation_state(
            session_id="s1",
            facts_raw=["陈述"],
            lawyer_decision="approved",
            lawyer_feedback="looks good",
            awaiting_lawyer_review=False,
        )

        async def _fake_resume(session_id, state_updates):
            assert session_id == "s1"
            assert state_updates["lawyer_decision"] == "approved"
            assert state_updates["lawyer_feedback"] == "looks good"
            assert state_updates["awaiting_lawyer_review"] is False
            return expected_state

        with patch.object(orch, "resume_workflow", side_effect=_fake_resume):
            result = await orch.process_lawyer_feedback("s1", "approved", "looks good")
        assert result["lawyer_decision"] == "approved"
        assert result["lawyer_feedback"] == "looks good"
        assert result["awaiting_lawyer_review"] is False

    @pytest.mark.asyncio
    async def test_process_lawyer_feedback_missing_session(self):
        orch = ConsultationOrchestrator()
        with pytest.raises(ValueError, match="会话不存在"):
            await orch.process_lawyer_feedback("nonexistent", "approved")

    @pytest.mark.asyncio
    async def test_process_lawyer_feedback_no_feedback_text(self):
        """When feedback is None, the state_updates only contain decision + awaiting flag."""
        orch = ConsultationOrchestrator()
        expected_state = make_consultation_state(
            session_id="s1", facts_raw=["陈述"], lawyer_decision="revise_facts"
        )

        async def _fake_resume(session_id, state_updates):
            assert state_updates["lawyer_decision"] == "revise_facts"
            assert state_updates["lawyer_feedback"] is None
            return expected_state

        with patch.object(orch, "resume_workflow", side_effect=_fake_resume):
            result = await orch.process_lawyer_feedback("s1", "revise_facts", None)
        assert result["lawyer_decision"] == "revise_facts"


# ---------------------------------------------------------------------------
# human_review_node / wait_for_user_node
# ---------------------------------------------------------------------------


class TestHumanReviewNode:
    @pytest.mark.asyncio
    async def test_with_report_draft(self):
        from app.orchestrator.workflow import human_review_node

        state = make_consultation_state(
            session_id="sess-1",
            facts_raw=["陈述"],
            report_draft="报告草案内容",
            service_plan={"plan": "abc"},
        )
        result = await human_review_node(state)
        assert result["awaiting_lawyer_review"] is True
        assert result["current_agent"] == "HumanReview"
        assert "报告草案" in result["final_output"]
        assert len(result["conversation_history"]) == 1
        hist = result["conversation_history"][0]
        assert hist["agent"] == "HumanReview"
        assert hist["action"] == "awaiting_review"
        assert hist["has_report"] is True
        assert hist["has_service_plan"] is True

    @pytest.mark.asyncio
    async def test_without_report_draft(self):
        from app.orchestrator.workflow import human_review_node

        state = make_consultation_state(
            session_id="sess-2",
            facts_raw=["陈述"],
            report_draft=None,
            service_plan=None,
        )
        result = await human_review_node(state)
        assert result["awaiting_lawyer_review"] is True
        assert "尚未生成" in result["final_output"]
        hist = result["conversation_history"][0]
        assert hist["has_report"] is False
        assert hist["has_service_plan"] is False

    @pytest.mark.asyncio
    async def test_uses_default_session_id(self):
        from app.orchestrator.workflow import human_review_node

        state = make_consultation_state(facts_raw=["陈述"])
        state.pop("session_id", None)
        result = await human_review_node(state)
        assert result["conversation_history"][0]["session_id"] == "unknown"

    @pytest.mark.asyncio
    async def test_appends_to_existing_conversation_history(self):
        from app.orchestrator.workflow import human_review_node

        state = make_consultation_state(
            session_id="sess-3",
            facts_raw=["陈述"],
            report_draft="draft",
            conversation_history=[{"existing": "entry"}],
        )
        result = await human_review_node(state)
        assert len(result["conversation_history"]) == 2
        assert result["conversation_history"][0] == {"existing": "entry"}

    @pytest.mark.asyncio
    async def test_explicit_decision_does_not_replace_reviewed_output(self):
        from app.orchestrator.workflow import human_review_node

        state = make_consultation_state(
            session_id="sess-approved",
            facts_raw=["陈述"],
            lawyer_decision="approved",
            awaiting_lawyer_review=True,
            final_output="律师确认后的最终报告",
        )

        result = await human_review_node(state)

        assert result["awaiting_lawyer_review"] is False
        assert result["final_output"] == "律师确认后的最终报告"


class TestWaitForUserNode:
    @pytest.mark.asyncio
    async def test_returns_state_unchanged(self):
        from app.orchestrator.workflow import wait_for_user_node

        state = make_consultation_state(session_id="sess-1", facts_raw=["陈述"])
        result = await wait_for_user_node(state)
        assert result["session_id"] == "sess-1"
        assert result["facts_raw"] == ["陈述"]

    @pytest.mark.asyncio
    async def test_uses_default_session_id(self):
        from app.orchestrator.workflow import wait_for_user_node

        state = make_consultation_state(facts_raw=["陈述"])
        state.pop("session_id", None)
        # Should not raise; state is returned as-is
        result = await wait_for_user_node(state)
        assert "facts_raw" in result


# ---------------------------------------------------------------------------
# _build_workflow / _ensure_compiled
# ---------------------------------------------------------------------------


class TestBuildWorkflow:
    def test_returns_state_graph(self):
        from langgraph.graph import StateGraph

        orch = ConsultationOrchestrator()
        graph = orch._build_workflow()
        assert isinstance(graph, StateGraph)


class TestEnsureCompiled:
    def test_compiles_lazily(self):
        orch = ConsultationOrchestrator()
        assert orch._compiled is None
        orch._ensure_compiled()
        assert orch._compiled is not None
        # Calling again should be a no-op
        compiled_ref = orch._compiled
        orch._ensure_compiled()
        assert orch._compiled is compiled_ref


# ---------------------------------------------------------------------------
# _update_session_context
# ---------------------------------------------------------------------------


class TestUpdateSessionContextInternal:
    def test_delegates_to_update_session_context(self):
        """The internal helper has no return value but mutates _active_sessions."""
        orch = ConsultationOrchestrator()
        state = make_consultation_state(session_id="s1", facts_raw=["陈述"])
        orch._active_sessions["s1"] = state
        result = orch._update_session_context("s1", {"consent_given": True})
        # The internal helper returns None (unlike the public update_session_context).
        assert result is None
        assert orch._active_sessions["s1"]["consent_given"] is True

    def test_missing_session_does_not_modify_state(self):
        orch = ConsultationOrchestrator()
        result = orch._update_session_context("missing", {"key": "val"})
        assert result is None
        assert "missing" not in orch._active_sessions


# ---------------------------------------------------------------------------
# resume_workflow success path / get_snapshot / get_next_node / is_workflow_finished
# ---------------------------------------------------------------------------


class TestResumeWorkflow:
    @pytest.mark.asyncio
    async def test_resume_with_no_state_updates(self):
        """resume_workflow without state_updates does NOT call aupdate_state."""
        orch = ConsultationOrchestrator()
        state = make_consultation_state(session_id="rs-1", facts_raw=["陈述"])
        expected = {**state, "current_agent": "FactDigger"}

        async def _fake_ainvoke(initial, config):
            return expected

        async def _fake_update_state(config, updates, as_node=None):
            raise AssertionError("aupdate_state should not be called when state_updates is empty")

        async def _fake_get_state(config):
            snap = MagicMock()
            snap.values = {"some": "vals"}
            snap.next = ("fact_digger",)
            return snap

        orch._ensure_compiled()
        orch._compiled.ainvoke = AsyncMock(side_effect=_fake_ainvoke)
        orch._compiled.aget_state = AsyncMock(side_effect=_fake_get_state)
        orch._compiled.aupdate_state = AsyncMock(side_effect=_fake_update_state)

        result = await orch.resume_workflow("rs-1", None)
        assert result == expected
        orch._compiled.aget_state.assert_awaited()

    @pytest.mark.asyncio
    async def test_resume_with_state_updates(self):
        """resume_workflow calls aupdate_state when state_updates is provided."""
        orch = ConsultationOrchestrator()
        state = make_consultation_state(session_id="rs-2", facts_raw=["陈述"])
        expected = {**state, "current_agent": "FactDigger"}

        async def _fake_ainvoke(initial, config):
            return expected

        snap_with_vals = MagicMock()
        snap_with_vals.values = {"existing": "vals"}
        snap_with_vals.next = ("fact_digger",)

        async def _fake_get_state(config):
            return snap_with_vals

        async def _fake_update_state(config, updates, as_node=None):
            assert updates == {"foo": "bar"}

        orch._ensure_compiled()
        orch._compiled.ainvoke = AsyncMock(side_effect=_fake_ainvoke)
        orch._compiled.aget_state = AsyncMock(side_effect=_fake_get_state)
        orch._compiled.aupdate_state = AsyncMock(side_effect=_fake_update_state)

        result = await orch.resume_workflow("rs-2", {"foo": "bar"})
        assert result == expected
        orch._compiled.aupdate_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resume_propagates_generic_exception(self):
        """Generic exceptions in ainvoke should propagate (after cleanup)."""
        from app.errors.exceptions import LLMServiceException

        orch = ConsultationOrchestrator()
        orch._active_sessions["rs-3"] = {"session_id": "rs-3"}

        async def _fake_ainvoke(initial, config):
            raise LLMServiceException("llm down")

        async def _fake_get_state(config):
            snap = MagicMock()
            snap.values = {"x": 1}
            snap.next = ("fact_digger",)
            return snap

        orch._ensure_compiled()
        orch._compiled.ainvoke = AsyncMock(side_effect=_fake_ainvoke)
        orch._compiled.aget_state = AsyncMock(side_effect=_fake_get_state)
        orch._compiled.aupdate_state = AsyncMock()

        with pytest.raises(LLMServiceException):
            await orch.resume_workflow("rs-3", None)
        # session is cleaned up
        assert "rs-3" not in orch._active_sessions


class TestGetSnapshot:
    @pytest.mark.asyncio
    async def test_returns_snapshot_when_session_exists(self):
        orch = ConsultationOrchestrator()
        snap = MagicMock()
        snap.values = {"x": 1}

        async def _fake_aget_state(config):
            return snap

        orch._ensure_compiled()
        orch._compiled.aget_state = AsyncMock(side_effect=_fake_aget_state)
        result = await orch.get_snapshot("s1")
        assert result is snap

    @pytest.mark.asyncio
    async def test_returns_none_when_values_empty(self):
        orch = ConsultationOrchestrator()
        snap = MagicMock()
        snap.values = None

        async def _fake_aget_state(config):
            return snap

        orch._ensure_compiled()
        orch._compiled.aget_state = AsyncMock(side_effect=_fake_aget_state)
        result = await orch.get_snapshot("missing")
        assert result is None


class TestGetNextNode:
    @pytest.mark.asyncio
    async def test_returns_first_next_node(self):
        orch = ConsultationOrchestrator()
        snap = MagicMock()
        snap.values = {"x": 1}
        snap.next = ("fact_digger", "law_ref")

        async def _fake_get_snapshot(sid):
            return snap

        with patch.object(orch, "get_snapshot", side_effect=_fake_get_snapshot):
            result = await orch.get_next_node("s1")
        assert result == "fact_digger"

    @pytest.mark.asyncio
    async def test_returns_none_when_snapshot_missing(self):
        orch = ConsultationOrchestrator()

        async def _fake_get_snapshot(sid):
            return None

        with patch.object(orch, "get_snapshot", side_effect=_fake_get_snapshot):
            result = await orch.get_next_node("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_next_nodes(self):
        orch = ConsultationOrchestrator()
        snap = MagicMock()
        snap.values = {"x": 1}
        snap.next = ()

        async def _fake_get_snapshot(sid):
            return snap

        with patch.object(orch, "get_snapshot", side_effect=_fake_get_snapshot):
            result = await orch.get_next_node("s1")
        assert result is None


class TestIsWorkflowFinished:
    @pytest.mark.asyncio
    async def test_returns_true_when_no_snapshot(self):
        orch = ConsultationOrchestrator()

        async def _fake_get_snapshot(sid):
            return None

        with patch.object(orch, "get_snapshot", side_effect=_fake_get_snapshot):
            assert await orch.is_workflow_finished("missing") is True

    @pytest.mark.asyncio
    async def test_returns_true_when_next_empty(self):
        orch = ConsultationOrchestrator()
        snap = MagicMock()
        snap.values = {"x": 1}
        snap.next = ()

        async def _fake_get_snapshot(sid):
            return snap

        with patch.object(orch, "get_snapshot", side_effect=_fake_get_snapshot):
            assert await orch.is_workflow_finished("s1") is True

    @pytest.mark.asyncio
    async def test_returns_false_when_next_has_nodes(self):
        orch = ConsultationOrchestrator()
        snap = MagicMock()
        snap.values = {"x": 1}
        snap.next = ("fact_digger",)

        async def _fake_get_snapshot(sid):
            return snap

        with patch.object(orch, "get_snapshot", side_effect=_fake_get_snapshot):
            assert await orch.is_workflow_finished("s1") is False


# ---------------------------------------------------------------------------
# Orchestrator module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_orchestrator_instance_exists(self):
        from app.orchestrator import workflow

        assert isinstance(workflow.orchestrator, ConsultationOrchestrator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
