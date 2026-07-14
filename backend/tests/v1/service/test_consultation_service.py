"""Unit tests for ``app.v1.service.consultation_service``.

These tests exercise the consultation service helpers with the orchestrator,
database, and Redis cache dependencies mocked.  Both success and error paths
are covered for each public function.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.errors.exceptions import LLMTimeoutException
from app.state.consultation_state import validate_consultation_state
from app.v1.service import consultation_service
from app.v1.service.consultation_service import (
    ProcessMessageResult,
    create_consultation_record,
    generate_welcome_message,
    get_session_state,
    handle_high_risk_alert,
    persist_state,
    process_consent,
    process_lawyer_review,
    process_message,
    save_message_to_db,
    start_session,
)
from tests.factories import make_consultation_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state():
    """A baseline consultation state dict."""
    return make_consultation_state(
        session_id="sess-001",
        user_id="user-001",
        consent_given=True,
        facts_raw=[],
        current_agent="Receptionist",
    )


@pytest.fixture
def snapshot_with_values(sample_state):
    """A LangGraph StateSnapshot-like object with non-empty values."""
    snapshot = MagicMock()
    snapshot.values = sample_state
    return snapshot


@pytest.fixture
def empty_snapshot():
    """A snapshot whose ``values`` is empty (i.e. session does not exist)."""
    snapshot = MagicMock()
    snapshot.values = None
    return snapshot


# ---------------------------------------------------------------------------
# get_session_state
# ---------------------------------------------------------------------------


class TestGetSessionState:
    @pytest.mark.asyncio
    async def test_returns_snapshot_values_when_available(self, snapshot_with_values, sample_state):
        """If checkpointer has a snapshot, that wins over Redis and memory."""
        with patch.object(consultation_service.orchestrator, "get_snapshot", new_callable=AsyncMock) as mock_get_snap, \
             patch("app.v1.service.consultation_service.get_redis_cache_json", new_callable=AsyncMock) as mock_redis:
            mock_get_snap.return_value = snapshot_with_values
            mock_redis.return_value = None

            result = await get_session_state("sess-001")

        assert result == sample_state
        assert result is not sample_state
        mock_redis.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_snapshot_with_invalid_state_types(self):
        """Checkpointer 返回的动态字典必须经过真实的状态校验。"""
        snapshot = MagicMock()
        snapshot.values = {"session_id": 123}
        with patch.object(
            consultation_service.orchestrator,
            "get_snapshot",
            new_callable=AsyncMock,
            return_value=snapshot,
        ):
            with pytest.raises(ValueError, match="咨询状态无效"):
                await get_session_state("sess-001")

    @pytest.mark.asyncio
    async def test_falls_back_to_redis_dict(self, sample_state):
        """When the snapshot is empty, the dict from Redis is wrapped in ConsultationState."""
        empty = MagicMock()
        empty.values = None
        with patch.object(consultation_service.orchestrator, "get_snapshot", new_callable=AsyncMock) as mock_get_snap, \
             patch("app.v1.service.consultation_service.get_redis_cache_json", new_callable=AsyncMock) as mock_redis, \
             patch.object(consultation_service.orchestrator, "get_session_context", return_value=None) as mock_ctx:
            mock_get_snap.return_value = empty
            mock_redis.return_value = dict(sample_state)
            mock_ctx.return_value = None

            result = await get_session_state("sess-001")

        assert result is not None
        assert result.get("session_id") == "sess-001"

    @pytest.mark.asyncio
    async def test_rejects_non_dict_redis_value(self):
        """Redis 中的非字典值不是有效的咨询状态。"""
        empty = MagicMock()
        empty.values = None
        # Wrap in a non-dict container to exercise the ``else`` branch.
        non_dict_state = object()
        with patch.object(consultation_service.orchestrator, "get_snapshot", new_callable=AsyncMock) as mock_get_snap, \
             patch("app.v1.service.consultation_service.get_redis_cache_json", new_callable=AsyncMock) as mock_redis, \
             patch.object(consultation_service.orchestrator, "get_session_context", return_value=None) as mock_ctx:
            mock_get_snap.return_value = empty
            mock_redis.return_value = non_dict_state
            mock_ctx.return_value = None

            with pytest.raises(ValueError, match="咨询状态无效"):
                await get_session_state("sess-001")

    @pytest.mark.asyncio
    async def test_falls_back_to_memory_cache(self):
        """When both snapshot and Redis are empty, the in-memory cache is used."""
        empty = MagicMock()
        empty.values = None
        in_memory = {"session_id": "sess-001", "current_agent": "FactDigger"}
        with patch.object(consultation_service.orchestrator, "get_snapshot", new_callable=AsyncMock) as mock_get_snap, \
             patch("app.v1.service.consultation_service.get_redis_cache_json", new_callable=AsyncMock) as mock_redis, \
             patch.object(consultation_service.orchestrator, "get_session_context", return_value=in_memory) as mock_ctx:
            mock_get_snap.return_value = empty
            mock_redis.return_value = None
            mock_ctx.return_value = in_memory

            result = await get_session_state("sess-001")

        assert result is in_memory

    @pytest.mark.asyncio
    async def test_returns_none_when_nothing_found(self):
        empty = MagicMock()
        empty.values = None
        with patch.object(consultation_service.orchestrator, "get_snapshot", new_callable=AsyncMock) as mock_get_snap, \
             patch("app.v1.service.consultation_service.get_redis_cache_json", new_callable=AsyncMock) as mock_redis, \
             patch.object(consultation_service.orchestrator, "get_session_context", return_value=None) as mock_ctx:
            mock_get_snap.return_value = empty
            mock_redis.return_value = None
            mock_ctx.return_value = None

            result = await get_session_state("missing")

        assert result is None

    @pytest.mark.asyncio
    async def test_snapshot_with_empty_values_treated_as_missing(self):
        """An empty dict in ``values`` should be considered absent."""
        snapshot = MagicMock()
        snapshot.values = {}
        with patch.object(consultation_service.orchestrator, "get_snapshot", new_callable=AsyncMock) as mock_get_snap, \
             patch("app.v1.service.consultation_service.get_redis_cache_json", new_callable=AsyncMock) as mock_redis, \
             patch.object(consultation_service.orchestrator, "get_session_context", return_value=None) as mock_ctx:
            mock_get_snap.return_value = snapshot
            mock_redis.return_value = None
            mock_ctx.return_value = None

            result = await get_session_state("missing")

        assert result is None


# ---------------------------------------------------------------------------
# persist_state
# ---------------------------------------------------------------------------


class TestPersistState:
    @pytest.mark.asyncio
    async def test_updates_memory_and_redis(self, sample_state):
        """persist_state writes to in-memory cache and to Redis with TTL."""
        with patch.object(consultation_service.orchestrator, "update_session_context", return_value=True) as mock_upd, \
             patch("app.v1.service.consultation_service.set_redis_cache", new_callable=AsyncMock) as mock_set:
            await persist_state("sess-001", sample_state)

        mock_upd.assert_called_once_with("sess-001", sample_state)
        mock_set.assert_awaited_once()
        key, value, kwargs = mock_set.call_args.args[0], mock_set.call_args.args[1], mock_set.call_args.kwargs
        assert key == "session:sess-001"
        # 4th positional arg is the expire keyword
        assert mock_set.call_args.kwargs.get("expire") == 7200 or mock_set.call_args.args[3] == 7200


# ---------------------------------------------------------------------------
# handle_high_risk_alert
# ---------------------------------------------------------------------------


class TestHandleHighRiskAlert:
    @pytest.mark.asyncio
    async def test_appends_alert_to_history(self):
        result_state = validate_consultation_state({"conversation_history": []})
        with patch.object(consultation_service, "persist_state", new_callable=AsyncMock) as mock_persist:
            await handle_high_risk_alert("sess-001", result_state, "FactDigger")

        conversation_history = result_state.get("conversation_history", [])
        assert len(conversation_history) == 1
        entry = conversation_history[0]
        assert entry["agent"] == "FactDigger"
        assert entry["action"] == "high_risk_alert"
        assert "为保护您的权益" in entry["content"]
        mock_persist.assert_awaited_once_with("sess-001", result_state)

    @pytest.mark.asyncio
    async def test_creates_history_when_missing(self):
        result_state = validate_consultation_state({})
        with patch.object(consultation_service, "persist_state", new_callable=AsyncMock):
            await handle_high_risk_alert("sess-001", result_state, "Receptionist")

        assert "conversation_history" in result_state
        assert result_state.get("conversation_history", [])[0]["agent"] == "Receptionist"


# ---------------------------------------------------------------------------
# generate_welcome_message
# ---------------------------------------------------------------------------


class TestGenerateWelcomeMessage:
    @pytest.mark.asyncio
    async def test_returns_message_with_disclaimer(self):
        message = await generate_welcome_message("suspect")
        # Disclaimer prefix is always present
        assert "本内容为智能辅助生成" in message
        # Welcome wording is present
        assert "刑事辩护初期咨询系统" in message
        # Asks user to confirm
        assert "请回复" in message

    @pytest.mark.asyncio
    async def test_idempotent_disclaimer(self):
        """Calling twice returns the same string (disclaimer not double-injected)."""
        first = await generate_welcome_message("victim")
        second = await generate_welcome_message("victim")
        assert first == second


# ---------------------------------------------------------------------------
# create_consultation_record
# ---------------------------------------------------------------------------


class TestCreateConsultationRecord:
    @pytest.mark.asyncio
    async def test_creates_and_returns_consultation_id(self, mock_db_session):
        """The DB session receives a new Consultation and we return its id."""
        consultation = MagicMock()
        consultation.id = "consult-xyz"

        async def fake_refresh(obj):
            obj.id = "consult-xyz"

        mock_db_session.add = MagicMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock(side_effect=fake_refresh)

        with patch("app.v1.service.consultation_service.Consultation", return_value=consultation):
            consultation_id = await create_consultation_record(
                "sess-001", "user-001", "suspect", mock_db_session
            )

        assert consultation_id == "consult-xyz"
        mock_db_session.add.assert_called_once_with(consultation)
        mock_db_session.commit.assert_awaited()
        mock_db_session.refresh.assert_awaited_with(consultation)


# ---------------------------------------------------------------------------
# save_message_to_db
# ---------------------------------------------------------------------------


class TestSaveMessageToDb:
    @pytest.mark.asyncio
    async def test_returns_uuid_when_db_is_none(self):
        result = await save_message_to_db(
            consultation_id="consult-001",
            session_id="sess-001",
            content="hi",
            sender_type="user",
        )
        # Just make sure it returns a string that looks like a UUID
        assert isinstance(result, str)
        uuid.UUID(result)

    @pytest.mark.asyncio
    async def test_persists_message_when_db_provided(self, mock_db_session):
        message = MagicMock()
        message.id = "msg-001"

        async def fake_refresh(obj):
            obj.id = "msg-001"

        mock_db_session.add = MagicMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock(side_effect=fake_refresh)

        with patch("app.v1.service.consultation_service.ConsultationMessage", return_value=message):
            msg_id = await save_message_to_db(
                consultation_id="consult-001",
                session_id="sess-001",
                content="hello",
                sender_type="agent",
                agent_name="Receptionist",
                db=mock_db_session,
            )

        assert msg_id == "msg-001"
        mock_db_session.add.assert_called_once_with(message)
        mock_db_session.commit.assert_awaited()


# ---------------------------------------------------------------------------
# start_session
# ---------------------------------------------------------------------------


class TestStartSession:
    @pytest.mark.asyncio
    async def test_starts_workflow_and_persists_state(self, sample_state):
        with patch.object(consultation_service.orchestrator, "start_workflow", new_callable=AsyncMock) as mock_start, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock) as mock_persist:
            mock_start.return_value = sample_state

            result = await start_session(sample_state)

        assert result is sample_state
        mock_start.assert_awaited_once_with(sample_state)
        mock_persist.assert_awaited_once_with(sample_state["session_id"], sample_state)

    @pytest.mark.asyncio
    async def test_rejects_state_without_session_id_before_starting_workflow(self):
        """启动工作流前必须明确校验 session_id，避免稍后出现 KeyError。"""
        state = validate_consultation_state(make_consultation_state())
        state.pop("session_id", None)

        with patch.object(
            consultation_service.orchestrator,
            "start_workflow",
            new_callable=AsyncMock,
        ) as mock_start:
            with pytest.raises(ValueError, match="session_id"):
                await start_session(state)

        mock_start.assert_not_awaited()


# ---------------------------------------------------------------------------
# process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    @pytest.mark.asyncio
    async def test_success_appends_history(self, sample_state):
        sample_state["facts_raw"] = []
        resume_result = {
            "final_output": "这是回复",
            "current_agent": "FactDigger",
            "alert_triggered": False,
            "conversation_history": [],
        }
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service.orchestrator, "is_workflow_finished", new_callable=AsyncMock) as mock_done, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock) as mock_persist:
            mock_resume.return_value = resume_result
            mock_done.return_value = False

            result = await process_message("sess-001", "我的案件...", sample_state, "Receptionist")

        assert isinstance(result, ProcessMessageResult)
        assert result.response_content == "这是回复"
        assert result.next_agent == "FactDigger"
        assert result.alert_triggered is False
        assert result.is_workflow_finished is False
        assert result.error is None
        # FactDigger owns facts_raw append + masking; the service only passes this turn's input.
        state_updates = mock_resume.call_args.args[1]
        assert state_updates == {"current_input": "我的案件..."}
        # history should be appended to the result state
        assert len(resume_result["conversation_history"]) == 1
        mock_persist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_alert_branch_returns_alert_message(self, sample_state):
        sample_state["facts_raw"] = []
        resume_result = {
            "final_output": "ignored",
            "current_agent": "FactDigger",
            "alert_triggered": True,
            "conversation_history": [],
        }
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service.orchestrator, "is_workflow_finished", new_callable=AsyncMock) as mock_done, \
             patch.object(consultation_service, "handle_high_risk_alert", new_callable=AsyncMock) as mock_alert, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock) as mock_persist:
            mock_resume.return_value = resume_result
            mock_done.return_value = True

            result = await process_message("sess-001", "敏感陈述", sample_state, "FactDigger")

        assert result.alert_triggered is True
        assert "为保护您的权益" in result.response_content
        mock_alert.assert_awaited_once()
        # When alert is triggered, we should NOT call persist_state
        mock_persist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_result(self, sample_state):
        sample_state["facts_raw"] = []
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume:
            mock_resume.side_effect = RuntimeError("boom")

            result = await process_message("sess-001", "msg", sample_state, "Receptionist")

        assert result.error == "boom"
        assert result.response_content == ""
        assert result.next_agent == "Receptionist"
        assert result.result_state is None

    @pytest.mark.asyncio
    async def test_typed_app_exception_is_not_collapsed_into_generic_error(self, sample_state):
        sample_state["facts_raw"] = []
        with patch.object(
            consultation_service.orchestrator,
            "resume_workflow",
            new_callable=AsyncMock,
            side_effect=LLMTimeoutException("upstream timeout"),
        ):
            with pytest.raises(LLMTimeoutException):
                await process_message("sess-001", "msg", sample_state, "Receptionist")

    @pytest.mark.asyncio
    async def test_creates_conversation_history_if_missing(self, sample_state):
        sample_state["facts_raw"] = []
        resume_result = {
            "final_output": "ok",
            "current_agent": "FactDigger",
            "alert_triggered": False,
        # No conversation_history key at all
        }
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service.orchestrator, "is_workflow_finished", new_callable=AsyncMock) as mock_done, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock):
            mock_resume.return_value = resume_result
            mock_done.return_value = False

            result = await process_message("sess-001", "msg", sample_state, "Receptionist")

        assert "conversation_history" in resume_result
        assert len(resume_result["conversation_history"]) == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_current_agent_when_missing(self, sample_state):
        sample_state["facts_raw"] = []
        resume_result = {
            "final_output": "ok",
            "alert_triggered": False,
            "conversation_history": [],
        # current_agent missing -> should fall back to "Receptionist"
        }
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service.orchestrator, "is_workflow_finished", new_callable=AsyncMock) as mock_done, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock):
            mock_resume.return_value = resume_result
            mock_done.return_value = False

            result = await process_message("sess-001", "msg", sample_state, "Receptionist")

        assert result.next_agent == "Receptionist"


# ---------------------------------------------------------------------------
# process_consent
# ---------------------------------------------------------------------------


class TestProcessConsent:
    @pytest.mark.asyncio
    async def test_consent_given_resumes_workflow(self):
        updated_state = {"session_id": "sess-001", "consent_given": True, "identity_info": {"name": "X"}}
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock) as mock_persist:
            mock_resume.return_value = updated_state

            result = await process_consent(
                session_id="sess-001",
                consent_given=True,
                identity_info={"name": "X"},
            )

        assert result is updated_state
        # ``resume_workflow`` is called with positional args: (session_id, state_updates)
        state_updates = mock_resume.call_args.args[1]
        assert state_updates["consent_given"] is True
        assert state_updates["identity_info"] == {"name": "X"}
        mock_persist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_consent_not_given_still_calls_resume(self):
        """Per the implementation, both branches call resume_workflow."""
        updated_state = {"session_id": "sess-001", "consent_given": False}
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock) as mock_persist:
            mock_resume.return_value = updated_state

            result = await process_consent(session_id="sess-001", consent_given=False)

        assert result is updated_state
        state_updates = mock_resume.call_args.args[1]
        assert state_updates["consent_given"] is False
        # identity_info is omitted when not provided
        assert "identity_info" not in state_updates
        mock_persist.assert_awaited_once()


# ---------------------------------------------------------------------------
# process_lawyer_review
# ---------------------------------------------------------------------------


class TestProcessLawyerReview:
    @pytest.mark.asyncio
    async def test_approved_with_final_output(self):
        updated_state = {"session_id": "sess-001", "lawyer_decision": "approved", "final_output": "ok"}
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock) as mock_persist:
            mock_resume.return_value = updated_state

            result = await process_lawyer_review(
                session_id="sess-001",
                decision="approved",
                feedback="looks good",
                final_output="最终报告",
            )

        updates = mock_resume.call_args.args[1]
        assert updates["lawyer_decision"] == "approved"
        assert updates["lawyer_feedback"] == "looks good"
        assert updates["awaiting_lawyer_review"] is False
        assert updates["final_output"] == "最终报告"
        assert result is updated_state
        mock_persist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_revise_facts_drops_final_output(self):
        updated_state = {"session_id": "sess-001", "lawyer_decision": "revise_facts"}
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock):
            mock_resume.return_value = updated_state

            await process_lawyer_review(
                session_id="sess-001",
                decision="revise_facts",
                feedback="need more facts",
                final_output="this should be ignored",
            )

        updates = mock_resume.call_args.args[1]
        # final_output is only set when decision == "approved"
        assert "final_output" not in updates
        assert updates["lawyer_decision"] == "revise_facts"

    @pytest.mark.asyncio
    async def test_approved_without_final_output(self):
        updated_state = {"session_id": "sess-001"}
        with patch.object(consultation_service.orchestrator, "resume_workflow", new_callable=AsyncMock) as mock_resume, \
             patch.object(consultation_service, "persist_state", new_callable=AsyncMock):
            mock_resume.return_value = updated_state

            await process_lawyer_review(session_id="sess-001", decision="approved")

        updates = mock_resume.call_args.args[1]
        assert updates["lawyer_decision"] == "approved"
        # final_output is empty / None, so it's not added
        assert "final_output" not in updates
        assert updates["awaiting_lawyer_review"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
