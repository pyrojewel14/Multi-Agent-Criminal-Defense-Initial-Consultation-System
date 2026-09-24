import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.user import ConsultationStatus
from app.v1.service import consultation_service
from tests.factories import make_consultation_state


def test_idempotency_result_registry_has_a_hard_capacity_bound():
    coordinator = consultation_service._SessionCommandCoordinator(
        max_results=2,
        result_ttl_seconds=3600,
    )
    for index in range(3):
        coordinator.remember(
            f"session-{index}",
            "message",
            f"key-{index}",
            f"fingerprint-{index}",
            f"result-{index}",
        )

    assert coordinator.metrics()["cached_results"] == 2
    assert coordinator.replay(
        "session-0", "message", "key-0", "fingerprint-0"
    ) is None
    assert coordinator.replay(
        "session-2", "message", "key-2", "fingerprint-2"
    ) == "result-2"


@pytest.mark.asyncio
async def test_different_sessions_are_not_blocked_by_a_global_execution_lock():
    coordinator = consultation_service._SessionCommandCoordinator()
    active = 0
    max_active = 0

    async def command(session_id):
        nonlocal active, max_active
        async with coordinator.serialize(session_id):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(command("session-a"), command("session-b"))

    assert max_active == 2
    assert coordinator.metrics()["active_lock_entries"] == 0


@pytest.mark.asyncio
async def test_same_session_messages_are_serialized_without_losing_either_input():
    session_id = "p1-004-concurrent-messages"
    active_resumes = 0
    max_active_resumes = 0
    resumed_inputs: list[str] = []
    persisted_histories: list[list[dict]] = []

    async def resume_workflow(_session_id, state_updates):
        nonlocal active_resumes, max_active_resumes
        active_resumes += 1
        max_active_resumes = max(max_active_resumes, active_resumes)
        content = state_updates["current_input"]
        resumed_inputs.append(content)
        await asyncio.sleep(0.01)
        active_resumes -= 1
        return make_consultation_state(
            session_id=session_id,
            consultation_id="consultation-concurrent",
            consent_given=True,
            current_agent="FactDigger",
            final_output=f"回复:{content}",
            conversation_history=[],
        )

    async def update_workflow_state(_session_id, updates):
        persisted_histories.append(list(updates["conversation_history"]))
        return make_consultation_state(
            session_id=session_id,
            consultation_id="consultation-concurrent",
            consent_given=True,
            current_agent="FactDigger",
            conversation_history=list(updates["conversation_history"]),
        )

    state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-concurrent",
        consent_given=True,
        current_agent="FactDigger",
        conversation_history=[],
    )
    with (
        patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(values=state, next=("fact_intake",)),
        ),
        patch.object(
            consultation_service.orchestrator,
            "resume_workflow",
            side_effect=resume_workflow,
        ) as resume_mock,
        patch.object(
            consultation_service.orchestrator,
            "update_workflow_state",
            side_effect=update_workflow_state,
        ),
        patch.object(
            consultation_service.orchestrator,
            "is_workflow_finished",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(consultation_service.orchestrator, "update_session_context"),
    ):
        first, second = await asyncio.gather(
            consultation_service.process_message(
                session_id,
                "输入一",
                state,
                "FactDigger",
                idempotency_key="message-1",
            ),
            consultation_service.process_message(
                session_id,
                "输入二",
                state,
                "FactDigger",
                idempotency_key="message-2",
            ),
        )

    assert max_active_resumes == 1
    assert resumed_inputs == ["输入一", "输入二"]
    assert resume_mock.await_count == 2
    assert [first.response_content, second.response_content] == ["回复:输入一", "回复:输入二"]
    assert [history[0]["user_message"] for history in persisted_histories] == ["输入一", "输入二"]
    assert consultation_service.get_session_command_metrics()["active_lock_entries"] == 0


@pytest.mark.asyncio
async def test_waiting_message_refreshes_executing_agent_from_checkpoint_inside_lock():
    """第二个排队命令必须使用第一个命令推进后的 checkpoint 执行者。"""
    session_id = "p1-004-refresh-agent"
    checkpoint_state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-refresh-agent",
        consent_given=True,
        current_agent="Receptionist",
        conversation_history=[],
    )
    first_resume_started = asyncio.Event()
    release_first_resume = asyncio.Event()
    resume_count = 0

    async def get_snapshot(_session_id):
        return SimpleNamespace(values=dict(checkpoint_state), next=("fact_intake",))

    async def resume_workflow(_session_id, state_updates):
        nonlocal resume_count
        resume_count += 1
        if resume_count == 1:
            first_resume_started.set()
            await release_first_resume.wait()
            next_agent = "FactDigger"
        else:
            next_agent = "RiskAssessor"
        checkpoint_state["current_agent"] = next_agent
        checkpoint_state["final_output"] = f"回复:{state_updates['current_input']}"
        return dict(checkpoint_state)

    async def update_workflow_state(_session_id, updates):
        checkpoint_state.update(updates)
        return dict(checkpoint_state)

    first_db = MagicMock()
    first_db.add = MagicMock()
    first_db.commit = AsyncMock()
    first_db.rollback = AsyncMock()
    second_db = MagicMock()
    second_db.add = MagicMock()
    second_db.commit = AsyncMock()
    second_db.rollback = AsyncMock()
    stale_state = dict(checkpoint_state)

    with (
        patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            side_effect=get_snapshot,
        ),
        patch.object(
            consultation_service.orchestrator,
            "resume_workflow",
            side_effect=resume_workflow,
        ),
        patch.object(
            consultation_service.orchestrator,
            "update_workflow_state",
            side_effect=update_workflow_state,
        ),
        patch.object(
            consultation_service.orchestrator,
            "is_workflow_finished",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(consultation_service.orchestrator, "update_session_context"),
    ):
        first_task = asyncio.create_task(
            consultation_service.process_message(
                session_id,
                "输入一",
                stale_state,
                "Receptionist",
                idempotency_key="refresh-agent-1",
                sender_id="client-1",
                db=first_db,
            )
        )
        await first_resume_started.wait()
        second_task = asyncio.create_task(
            consultation_service.process_message(
                session_id,
                "输入二",
                stale_state,
                "Receptionist",
                idempotency_key="refresh-agent-2",
                sender_id="client-1",
                db=second_db,
            )
        )
        await asyncio.sleep(0)
        release_first_resume.set()
        first, second = await asyncio.gather(first_task, second_task)

    assert first.agent_name == "Receptionist"
    assert first.next_agent == "FactDigger"
    assert second.agent_name == "FactDigger"
    assert second.next_agent == "RiskAssessor"
    assert [entry["agent"] for entry in checkpoint_state["conversation_history"]] == [
        "Receptionist",
        "FactDigger",
    ]
    assert first_db.add.call_args_list[1].args[0].agent_name == "Receptionist"
    assert second_db.add.call_args_list[1].args[0].agent_name == "FactDigger"


@pytest.mark.asyncio
async def test_duplicate_message_key_replays_result_without_workflow_or_history_duplication():
    session_id = "p1-004-duplicate-message"
    result_state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-duplicate",
        consent_given=True,
        current_agent="FactDigger",
        final_output="只生成一次",
        conversation_history=[],
    )
    state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-duplicate",
        consent_given=True,
        current_agent="FactDigger",
        conversation_history=[],
    )

    async def delayed_resume(*_args, **_kwargs):
        await asyncio.sleep(0.01)
        return result_state

    with (
        patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(values=state, next=("fact_intake",)),
        ),
        patch.object(
            consultation_service.orchestrator,
            "resume_workflow",
            side_effect=delayed_resume,
        ) as resume_mock,
        patch.object(
            consultation_service.orchestrator,
            "update_workflow_state",
            new_callable=AsyncMock,
            return_value=result_state,
        ) as update_mock,
        patch.object(
            consultation_service.orchestrator,
            "is_workflow_finished",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(consultation_service.orchestrator, "update_session_context"),
    ):
        first, replay = await asyncio.gather(
            consultation_service.process_message(
                session_id,
                "相同输入",
                state,
                "FactDigger",
                idempotency_key="same-message-key",
            ),
            consultation_service.process_message(
                session_id,
                "相同输入",
                state,
                "FactDigger",
                idempotency_key="same-message-key",
            ),
        )

    assert resume_mock.await_count == 1
    assert update_mock.await_count == 1
    assert len(result_state["conversation_history"]) == 1
    assert replay is first
    assert replay.message_id == first.message_id
    assert replay.created_at == first.created_at


@pytest.mark.asyncio
async def test_duplicate_http_message_writes_user_and_agent_rows_once():
    session_id = "p1-004-duplicate-http-db"
    result_state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-http-db",
        consent_given=True,
        current_agent="FactDigger",
        final_output="服务回复",
        conversation_history=[],
    )
    state = dict(result_state)
    state["final_output"] = ""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()

    with (
        patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(values=state, next=("fact_intake",)),
        ),
        patch.object(
            consultation_service.orchestrator,
            "resume_workflow",
            new_callable=AsyncMock,
            return_value=result_state,
        ),
        patch.object(
            consultation_service.orchestrator,
            "update_workflow_state",
            new_callable=AsyncMock,
            return_value=result_state,
        ),
        patch.object(
            consultation_service.orchestrator,
            "is_workflow_finished",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(consultation_service.orchestrator, "update_session_context"),
    ):
        first = await consultation_service.process_message(
            session_id,
            "用户输入",
            state,
            "FactDigger",
            idempotency_key="http-db-key",
            sender_id="client-1",
            db=db,
        )
        replay = await consultation_service.process_message(
            session_id,
            "用户输入",
            state,
            "FactDigger",
            idempotency_key="http-db-key",
            sender_id="client-1",
            db=db,
        )

    assert replay is first
    assert db.add.call_count == 2
    assert [call.args[0].sender_type for call in db.add.call_args_list] == ["user", "agent"]
    assert [call.args[0].content for call in db.add.call_args_list] == ["用户输入", "服务回复"]
    assert db.commit.await_count == 1


@pytest.mark.asyncio
async def test_same_key_after_sqlite_failure_does_not_resume_or_append_history_again():
    session_id = "p1-004-http-db-failure"
    result_state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-http-db-failure",
        consent_given=True,
        current_agent="FactDigger",
        final_output="已推进的回复",
        conversation_history=[],
    )
    state = dict(result_state)
    state["final_output"] = ""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock(side_effect=RuntimeError("sqlite unavailable: secret"))
    db.rollback = AsyncMock()

    with (
        patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(values=state, next=("fact_intake",)),
        ),
        patch.object(
            consultation_service.orchestrator,
            "resume_workflow",
            new_callable=AsyncMock,
            return_value=result_state,
        ) as resume_mock,
        patch.object(
            consultation_service.orchestrator,
            "update_workflow_state",
            new_callable=AsyncMock,
            return_value=result_state,
        ) as update_mock,
        patch.object(
            consultation_service.orchestrator,
            "is_workflow_finished",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(consultation_service.orchestrator, "update_session_context"),
    ):
        first = await consultation_service.process_message(
            session_id,
            "只推进一次",
            state,
            "FactDigger",
            idempotency_key="db-failure-key",
            sender_id="client-1",
            db=db,
        )
        replay = await consultation_service.process_message(
            session_id,
            "只推进一次",
            state,
            "FactDigger",
            idempotency_key="db-failure-key",
            sender_id="client-1",
            db=db,
        )

    assert first.error is not None
    assert replay is first
    assert resume_mock.await_count == 1
    assert update_mock.await_count == 1
    assert len(result_state["conversation_history"]) == 1
    assert db.commit.await_count == 1
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_reusing_message_key_with_different_payload_is_rejected():
    session_id = "p1-004-key-conflict"
    state = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-conflict",
        consent_given=True,
        current_agent="FactDigger",
        final_output="响应",
        conversation_history=[],
    )

    with (
        patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(values=state, next=("fact_intake",)),
        ),
        patch.object(
            consultation_service.orchestrator,
            "resume_workflow",
            new_callable=AsyncMock,
            return_value=state,
        ) as resume_mock,
        patch.object(
            consultation_service.orchestrator,
            "update_workflow_state",
            new_callable=AsyncMock,
            return_value=state,
        ),
        patch.object(
            consultation_service.orchestrator,
            "is_workflow_finished",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(consultation_service.orchestrator, "update_session_context"),
    ):
        await consultation_service.process_message(
            session_id,
            "第一次",
            state,
            "FactDigger",
            idempotency_key="reused-key",
        )
        with pytest.raises(consultation_service.IdempotencyConflictError):
            await consultation_service.process_message(
                session_id,
                "被篡改的第二次",
                state,
                "FactDigger",
                idempotency_key="reused-key",
            )

    assert resume_mock.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "command_kwargs"),
    [
        ("approve", {"final_output": "最终报告"}),
        ("reject", {"target_node": "fact_digger", "feedback": "补充事实"}),
    ],
)
async def test_duplicate_review_command_replays_without_advancing_or_writing_twice(
    action,
    command_kwargs,
):
    session_id = f"p1-004-review-{action}"
    before = make_consultation_state(
        session_id=session_id,
        consultation_id=f"consultation-{action}",
        awaiting_lawyer_review=True,
        conversation_history=[],
    )
    after = dict(before)
    after["workflow_status"] = "completed" if action == "approve" else "in_progress"
    consultation = SimpleNamespace(
        status=ConsultationStatus.IN_PROGRESS,
        final_output=None,
        completed_at=None,
        lawyer_review_needed=True,
    )
    db = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    snapshot = SimpleNamespace(values=before, next=("human_review",))

    with (
        patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            new_callable=AsyncMock,
            return_value=snapshot,
        ),
        patch.object(
            consultation_service,
            "process_lawyer_review",
            new_callable=AsyncMock,
            return_value=after,
        ) as review_mock,
        patch.object(
            consultation_service.orchestrator,
            "update_workflow_state",
            new_callable=AsyncMock,
            return_value=after,
        ),
        patch.object(
            consultation_service,
            "_get_consultation_for_command",
            new_callable=AsyncMock,
            return_value=consultation,
        ),
    ):
        first = await consultation_service.execute_lifecycle_command(
            session_id,
            action,
            db=db,
            actor_id="lawyer-1",
            idempotency_key=f"review-key-{action}",
            **command_kwargs,
        )
        replay = await consultation_service.execute_lifecycle_command(
            session_id,
            action,
            db=db,
            actor_id="lawyer-1",
            idempotency_key=f"review-key-{action}",
            **command_kwargs,
        )

    assert replay is first
    assert first["command_processed_at"] == replay["command_processed_at"]
    assert review_mock.await_count == 1
    assert db.commit.await_count == 1


@pytest.mark.asyncio
async def test_failed_review_is_not_cached_so_same_key_can_repair_p1_003_state():
    session_id = "p1-004-review-repair"
    before = make_consultation_state(
        session_id=session_id,
        consultation_id="consultation-repair",
        awaiting_lawyer_review=True,
        conversation_history=[],
    )
    repair = dict(before)
    repair.update(
        workflow_status="repair_required",
        repair_required=True,
        consistency_error={"action": "approve"},
    )
    after = dict(before)
    after.update(workflow_status="completed", repair_required=False, consistency_error=None)
    snapshots = [
        SimpleNamespace(values=before, next=("human_review",)),
        SimpleNamespace(values=repair, next=()),
    ]
    consultation = SimpleNamespace(
        status=ConsultationStatus.IN_PROGRESS,
        final_output=None,
        completed_at=None,
        lawyer_review_needed=True,
    )
    db = MagicMock()
    db.commit = AsyncMock(side_effect=[RuntimeError("sqlite unavailable: secret"), None])
    db.rollback = AsyncMock()

    with (
        patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            new_callable=AsyncMock,
            side_effect=snapshots,
        ),
        patch.object(
            consultation_service,
            "process_lawyer_review",
            new_callable=AsyncMock,
            return_value=after,
        ) as review_mock,
        patch.object(
            consultation_service.orchestrator,
            "update_workflow_state",
            new_callable=AsyncMock,
            return_value=after,
        ),
        patch.object(
            consultation_service.orchestrator,
            "mark_repair_required",
            new_callable=AsyncMock,
        ),
        patch.object(
            consultation_service,
            "_get_consultation_for_command",
            new_callable=AsyncMock,
            return_value=consultation,
        ),
    ):
        with pytest.raises(consultation_service.LifecycleConsistencyError):
            await consultation_service.execute_lifecycle_command(
                session_id,
                "approve",
                db=db,
                actor_id="lawyer-1",
                final_output="最终报告",
                idempotency_key="repair-key",
            )
        repaired = await consultation_service.execute_lifecycle_command(
            session_id,
            "approve",
            db=db,
            actor_id="lawyer-1",
            final_output="最终报告",
            idempotency_key="repair-key",
        )

    assert repaired["workflow_status"] == "completed"
    assert review_mock.await_count == 1
    assert db.commit.await_count == 2
    db.rollback.assert_awaited_once()
