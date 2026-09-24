from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.models.user import ConsultationStatus
from app.orchestrator.workflow import ConsultationOrchestrator
from app.state.consultation_state import ConsultationState
from app.v1.service import consultation_service
from tests.factories import make_consultation_state


def test_default_orchestrator_exposes_process_scoped_checkpoint_boundary():
    orchestrator = ConsultationOrchestrator()

    assert orchestrator.checkpoint_persistence == "process"
    assert orchestrator.can_resume_after_restart is False


def test_unknown_injected_checkpointer_is_not_assumed_durable():
    class UnknownCheckpointer:
        pass

    orchestrator = ConsultationOrchestrator(checkpointer=UnknownCheckpointer())

    assert orchestrator.checkpoint_persistence == "process"
    assert orchestrator.can_resume_after_restart is False


def test_injected_checkpointer_is_durable_only_with_explicit_confirmation():
    class ConfirmedCheckpointer:
        pass

    orchestrator = ConsultationOrchestrator(
        checkpointer=ConfirmedCheckpointer(),
        persistent=True,
    )

    assert orchestrator.checkpoint_persistence == "persistent"
    assert orchestrator.can_resume_after_restart is True


def test_persistent_true_without_injected_checkpointer_is_rejected():
    with pytest.raises(ValueError, match="持久化"):
        ConsultationOrchestrator(persistent=True)


def test_memory_saver_cannot_be_declared_persistent():
    with pytest.raises(ValueError, match="MemorySaver"):
        ConsultationOrchestrator(checkpointer=MemorySaver(), persistent=True)


@pytest.mark.asyncio
async def test_session_state_never_falls_back_to_redis_or_memory():
    empty_snapshot = SimpleNamespace(values=None)
    with patch.object(
        consultation_service.orchestrator,
        "get_snapshot",
        new_callable=AsyncMock,
        return_value=empty_snapshot,
    ), patch.object(
        consultation_service.orchestrator,
        "get_session_context",
    ) as memory_get:
        assert await consultation_service.get_session_state("session-1") is None

    memory_get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "command_kwargs"),
    [
        ("approve", {"final_output": "不得写入"}),
        ("reject", {"target_node": "risk_assessor", "feedback": "不得推进"}),
    ],
)
async def test_review_command_is_rejected_outside_human_review_without_advancing_or_writing_db(
    action,
    command_kwargs,
):
    session_id = f"p1-003-invalid-{action}-position"

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        return state

    local_orchestrator = ConsultationOrchestrator()
    initial_state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-invalid-approve",
        consent_given=False,
    )
    db = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch.object(consultation_service, "orchestrator", local_orchestrator),
    ):
        await local_orchestrator.start_workflow(initial_state)
        before = await local_orchestrator.get_snapshot(session_id)

        with pytest.raises(consultation_service.LifecycleCommandConflictError):
            await consultation_service.execute_lifecycle_command(
                session_id,
                action,
                db=db,
                actor_id="lawyer-1",
                **command_kwargs,
            )

        after = await local_orchestrator.get_snapshot(session_id)

    assert after.next == before.next
    assert after.config["configurable"]["checkpoint_id"] == before.config["configurable"]["checkpoint_id"]
    db.commit.assert_not_awaited()
    db.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_graph_marks_repair_required_when_sqlite_commit_fails_and_retry_repairs():
    session_id = "p1-003-real-checkpoint-failure"

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        return state

    initial_state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-1",
        consent_given=False,
        conversation_history=[],
    )
    local_orchestrator = ConsultationOrchestrator()
    consultation = SimpleNamespace(status=None, lawyer_review_needed=True)
    db = MagicMock()
    db.commit = AsyncMock(side_effect=[RuntimeError("sqlite unavailable: secret"), None])
    db.rollback = AsyncMock()

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch.object(consultation_service, "orchestrator", local_orchestrator),
        patch.object(
            consultation_service,
            "_get_consultation_for_command",
            new_callable=AsyncMock,
            return_value=consultation,
        ),
    ):
        await local_orchestrator.start_workflow(initial_state)
        pending_before = (await local_orchestrator.get_snapshot(session_id)).next

        with pytest.raises(consultation_service.LifecycleConsistencyError) as exc_info:
            await consultation_service.execute_lifecycle_command(
                session_id,
                "close",
                db=db,
                actor_id="user-1",
                reason="private close reason",
            )

        failed_snapshot = await local_orchestrator.get_snapshot(session_id)
        assert failed_snapshot.next == pending_before
        assert failed_snapshot.values["workflow_status"] == "repair_required"
        assert failed_snapshot.values["repair_required"] is True
        assert failed_snapshot.values["consistency_error"] == {
            "action": "close",
            "failed_stage": "sqlite_audit_commit",
            "error_code": "audit_write_failed",
        }
        assert exc_info.value.action == "close"
        assert exc_info.value.failed_stage == "sqlite_audit_commit"
        assert exc_info.value.repair_required is True
        assert "secret" not in str(exc_info.value)
        assert "private close reason" not in str(exc_info.value)

        with pytest.raises(ValueError, match="修复"):
            await local_orchestrator.resume_workflow(session_id)

        repaired = await consultation_service.execute_lifecycle_command(
            session_id,
            "close",
            db=db,
            actor_id="user-1",
            reason="private close reason",
        )

        repaired_snapshot = await local_orchestrator.get_snapshot(session_id)

    assert repaired["workflow_status"] == "closed"
    assert repaired_snapshot.next == pending_before
    assert repaired_snapshot.values["repair_required"] is False
    assert repaired_snapshot.values["consistency_error"] is None
    close_events = [
        item
        for item in repaired_snapshot.values["conversation_history"]
        if item.get("action") == "session_closed"
    ]
    assert len(close_events) == 1
    assert db.commit.await_count == 2
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_human_review_approve_failure_keeps_advanced_position_and_repairs_without_replay():
    session_id = "p1-003-real-approve-failure"
    human_review_calls = 0

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        return state

    async def fact_intake(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "FactDigger"
        state["alert_triggered"] = False
        return state

    async def law_ref(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "LawRef"
        state["applied_laws"] = [{"article_number": "第二百六十四条"}]
        return state

    async def fact_coverage(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "FactDigger"
        state["facts_coverage_rate"] = 1.0
        state["fact_law_loop_count"] = 1
        return state

    async def risk_assessor(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "RiskAssessor"
        state["risk_assessment"] = {"risk_level": "medium"}
        return state

    async def service_planner(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "ServicePlanner"
        state["service_plan"] = {"next_step": "lawyer_review"}
        state["report_draft"] = "确定性报告"
        return state

    async def human_review(state: ConsultationState) -> ConsultationState:
        nonlocal human_review_calls
        human_review_calls += 1
        state["current_agent"] = "HumanReview"
        state["awaiting_lawyer_review"] = state.get("lawyer_decision") != "approved"
        history = list(state.get("conversation_history", []))
        history.append(
            {
                "agent": "HumanReview",
                "action": "approved" if state.get("lawyer_decision") == "approved" else "awaiting_review",
            }
        )
        state["conversation_history"] = history
        return state

    initial_state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-approve",
        consent_given=False,
        conversation_history=[],
        lawyer_decision=None,
    )
    local_orchestrator = ConsultationOrchestrator()
    consultation = SimpleNamespace(
        status=None,
        final_output=None,
        completed_at=None,
        lawyer_review_needed=True,
    )
    db = MagicMock()
    db.commit = AsyncMock(side_effect=[RuntimeError("private sqlite failure"), None])
    db.rollback = AsyncMock()

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch("app.orchestrator.workflow.fact_intake_node", fact_intake),
        patch("app.orchestrator.workflow.law_ref_node", law_ref),
        patch("app.orchestrator.workflow.fact_coverage_node", fact_coverage),
        patch("app.orchestrator.workflow.risk_assessor_node", risk_assessor),
        patch("app.orchestrator.workflow.service_planner_node", service_planner),
        patch("app.orchestrator.workflow.human_review_node", human_review),
        patch.object(consultation_service, "orchestrator", local_orchestrator),
        patch.object(
            consultation_service,
            "_get_consultation_for_command",
            new_callable=AsyncMock,
            return_value=consultation,
        ),
    ):
        await local_orchestrator.start_workflow(initial_state)
        await local_orchestrator.resume_workflow(session_id, {"consent_given": True})
        review_snapshot = await local_orchestrator.get_snapshot(session_id)
        assert review_snapshot.next == ("human_review",)

        with pytest.raises(consultation_service.LifecycleConsistencyError):
            await consultation_service.execute_lifecycle_command(
                session_id,
                "approve",
                db=db,
                actor_id="lawyer-1",
                final_output="批准后的报告",
            )

        failed_snapshot = await local_orchestrator.get_snapshot(session_id)
        assert failed_snapshot.next == ()
        assert failed_snapshot.next != review_snapshot.next
        assert failed_snapshot.values["workflow_status"] == "repair_required"
        assert failed_snapshot.values["consistency_error"]["action"] == "approve"
        calls_after_advance = human_review_calls

        with pytest.raises(ValueError, match="修复"):
            await local_orchestrator.resume_workflow(session_id)

        repaired = await consultation_service.execute_lifecycle_command(
            session_id,
            "approve",
            db=db,
            actor_id="lawyer-1",
            final_output="批准后的报告",
        )
        repaired_snapshot = await local_orchestrator.get_snapshot(session_id)

    assert repaired["workflow_status"] == "completed"
    assert repaired_snapshot.next == ()
    assert repaired_snapshot.values["repair_required"] is False
    assert repaired_snapshot.values["consistency_error"] is None
    assert human_review_calls == calls_after_advance
    assert consultation.status == ConsultationStatus.COMPLETED
    assert consultation.final_output == "批准后的报告"
    assert db.commit.await_count == 2


@pytest.mark.asyncio
async def test_persist_state_does_not_create_an_extra_checkpoint_or_change_pending_node():
    session_id = "p1-003-projection-only"

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        return state

    local_orchestrator = ConsultationOrchestrator()
    initial_state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-2",
        consent_given=False,
    )

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch.object(consultation_service, "orchestrator", local_orchestrator),
    ):
        result = await local_orchestrator.start_workflow(initial_state)
        before = await local_orchestrator.get_snapshot(session_id)
        await consultation_service.persist_state(session_id, result)
        after_noop = await local_orchestrator.get_snapshot(session_id)
        await consultation_service.persist_state(
            session_id,
            result,
            projection_updates={"lawyer_id": "lawyer-1"},
        )
        after_projection = await local_orchestrator.get_snapshot(session_id)

    assert after_noop.next == before.next
    assert after_noop.config["configurable"]["checkpoint_id"] == before.config["configurable"]["checkpoint_id"]
    assert after_projection.next == before.next
    assert after_projection.values["lawyer_id"] == "lawyer-1"


@pytest.mark.asyncio
async def test_approve_command_updates_checkpoint_and_sqlite_audit():
    before = {"session_id": "session-1", "consultation_id": "consultation-1"}
    after = {**before, "lawyer_decision": "approved", "final_output": "final"}
    consultation = SimpleNamespace(status=None, final_output=None, completed_at=None, lawyer_review_needed=True)
    db = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()

    with patch.object(
        consultation_service.orchestrator,
        "get_snapshot",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(values=before, next=("human_review",)),
    ), patch.object(
        consultation_service,
        "process_lawyer_review",
        new_callable=AsyncMock,
        return_value=after,
    ), patch.object(
        consultation_service,
        "_get_consultation_for_command",
        new_callable=AsyncMock,
        return_value=consultation,
    ):
        result = await consultation_service.execute_lifecycle_command(
            "session-1",
            "approve",
            db=db,
            actor_id="lawyer-1",
            final_output="final",
        )

    assert result is after
    assert consultation.final_output == "final"
    assert consultation.lawyer_review_needed is False
    assert consultation.status == ConsultationStatus.COMPLETED
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_lawyer_approval_delegates_to_application_command():
    with patch(
        "app.v1.router.lawyer.consultation_service.execute_lifecycle_command",
        new_callable=AsyncMock,
        return_value={"final_output": "final"},
    ) as command, patch(
        "app.v1.router.lawyer._workflow_session_id",
        new_callable=AsyncMock,
        return_value="workflow-1",
    ):
        from app.v1.router import lawyer

        # The route-level delegation contract is asserted without invoking FastAPI dependencies.
        await lawyer._approve_report_command(
            "workflow-1",
            final_output="final",
            feedback="ok",
            actor_id="lawyer-1",
            db=MagicMock(),
        )

    command.assert_awaited_once()
    assert command.call_args.kwargs["action"] == "approve"


@pytest.mark.asyncio
async def test_legacy_lawyer_rejection_delegates_to_application_command():
    with patch(
        "app.v1.router.lawyer.consultation_service.execute_lifecycle_command",
        new_callable=AsyncMock,
        return_value={"workflow_status": "in_progress"},
    ) as command, patch(
        "app.v1.router.lawyer._workflow_session_id",
        new_callable=AsyncMock,
        return_value="workflow-1",
    ):
        from app.v1.router import lawyer

        await lawyer._reject_session_command(
            "consultation-1",
            target_node="risk_assessor",
            feedback="revise",
            actor_id="lawyer-1",
            db=MagicMock(),
        )

    command.assert_awaited_once()
    assert command.call_args.kwargs["action"] == "reject"
    assert command.call_args.kwargs["target_node"] == "risk_assessor"
