"""Tests for uncovered branches in the consultation routes.

These tests focus on the remaining endpoints that ``test_consultation_api.py``
does not exercise in depth:

- POST /api/v1/sessions/{id}/message - send_message
- PUT  /api/v1/sessions/{id}/review  - lawyer_review
- /state, /close, /confirm-consent edge cases

All external dependencies (consultation_service, orchestrator, DB session) are
mocked so the suite runs without Redis, the LLM, or a real database.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.security.jwt import create_access_token
from app.errors.exceptions import LLMTimeoutException
from app.v1.service.consultation_service import ProcessMessageResult
from tests.factories import make_consultation_state


SESSIONS_PREFIX = "/api/v1/sessions"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_auth_headers():
    token = create_access_token(user_id="user-001", role="client")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def lawyer_auth_headers():
    token = create_access_token(user_id="lawyer-001", role="lawyer")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_session_state():
    return make_consultation_state(
        session_id="sess-001",
        user_id="user-001",
        consent_given=True,
        current_agent="FactDigger",
        lawyer_id="lawyer-001",
        facts_raw=["用户陈述"],
        conversation_history=[],
    )


@pytest.fixture
async def lawyer_app(test_app, monkeypatch):
    """Test app with ``require_lawyer``/``require_admin`` overridden to a fixed user.

    The original routes capture ``Depends(require_lawyer)`` (a function) at
    import time, so simply calling the JWT-protected endpoint would invoke
    the role check. We patch both the symbol in the route module and the
    dependency override so the role check is bypassed.
    """
    import app.security.rbac as rbac_module
    import app.v1.router.consultation.routes as consultation_routes_module

    async def _lawyer_override():
        return {"user_id": "lawyer-001", "role": "lawyer"}

    monkeypatch.setattr(
        consultation_routes_module, "require_lawyer", _lawyer_override, raising=False
    )
    test_app.dependency_overrides[rbac_module.require_lawyer] = _lawyer_override

    yield test_app



# ---------------------------------------------------------------------------
# POST /sessions/{id}/message
# ---------------------------------------------------------------------------


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_message_session_not_found(
        self, test_app, client_auth_headers, mock_db_session
    ):
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state:
            mock_get_state.return_value = None

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "hello"},
                    headers=client_auth_headers,
                )

        assert response.status_code == 404
        assert "会话不存在" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_send_message_wrong_user_forbidden(
        self, test_app, sample_session_state, mock_db_session
    ):
        other_token = create_access_token(user_id="other-user-999", role="client")
        other_headers = {"Authorization": f"Bearer {other_token}"}

        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "hi"},
                    headers=other_headers,
                )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_send_message_consent_required(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        sample_session_state["consent_given"] = False
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "hi"},
                    headers=client_auth_headers,
                )

        assert response.status_code == 403
        assert "请先确认隐私条款" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_send_message_processing_error(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_message",
            new_callable=AsyncMock,
        ) as mock_proc:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = ProcessMessageResult(
                response_content="",
                next_agent="FactDigger",
                alert_triggered=False,
                result_state=None,
                error="LLM failed",
            )

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "hi"},
                    headers=client_auth_headers,
                )

        assert response.status_code == 500
        assert "消息处理失败" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_send_message_preserves_typed_llm_timeout_response(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
            return_value=sample_session_state,
        ), patch(
            "app.v1.service.consultation_service.process_message",
            new_callable=AsyncMock,
            side_effect=LLMTimeoutException("upstream timeout"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=test_app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "hi"},
                    headers=client_auth_headers,
                )

        assert response.status_code == 504
        assert response.json() == {
            "error": {"code": "LLM_TIMEOUT", "message": "AI 服务响应超时，请稍后重试"}
        }

    @pytest.mark.asyncio
    async def test_send_message_no_result_state(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        """When result.error is None but result_state is None, still returns 500."""
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_message",
            new_callable=AsyncMock,
        ) as mock_proc:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = ProcessMessageResult(
                response_content="",
                next_agent="FactDigger",
                alert_triggered=False,
                result_state=None,
                error=None,
            )

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "hi"},
                    headers=client_auth_headers,
                )

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_send_message_alert_triggered_returns_403(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_message",
            new_callable=AsyncMock,
        ) as mock_proc:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = ProcessMessageResult(
                response_content="ignored",
                next_agent="FactDigger",
                alert_triggered=True,
                result_state=sample_session_state,
            )

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "bad statement"},
                    headers=client_auth_headers,
                )

        assert response.status_code == 403
        assert "为保护您的权益" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_send_message_success_persists_user_and_agent_messages(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_message",
            new_callable=AsyncMock,
        ) as mock_proc, patch(
            "app.v1.service.consultation_service.save_message_to_db",
            new_callable=AsyncMock,
        ) as mock_save:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = ProcessMessageResult(
                response_content="这是回复",
                next_agent="RiskAssessor",
                alert_triggered=False,
                result_state=sample_session_state,
                is_workflow_finished=False,
            )

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "用户消息"},
                    headers=client_auth_headers,
                )

            assert response.status_code == 200
            data = response.json()
            assert data["response_content"] == "这是回复"
            assert data["agent_name"] == "FactDigger"
            assert data["is_complete"] is False

        # The DB should have been called twice (user message + agent reply)
        assert mock_save.await_count == 2
        # First call is for the user
        first_call = mock_save.await_args_list[0]
        assert first_call.kwargs["sender_type"] == "user"
        assert first_call.kwargs["content"] == "用户消息"
        # Second call is for the agent
        second_call = mock_save.await_args_list[1]
        assert second_call.kwargs["sender_type"] == "agent"
        assert second_call.kwargs["content"] == "这是回复"

    @pytest.mark.asyncio
    async def test_send_message_success_without_agent_response(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        """When the agent returns empty content, only the user message is persisted."""
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_message",
            new_callable=AsyncMock,
        ) as mock_proc, patch(
            "app.v1.service.consultation_service.save_message_to_db",
            new_callable=AsyncMock,
        ) as mock_save:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = ProcessMessageResult(
                response_content="",
                next_agent="FactDigger",
                alert_triggered=False,
                result_state=sample_session_state,
            )

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/message",
                    json={"session_id": "sess-001", "content": "hi"},
                    headers=client_auth_headers,
                )

            assert response.status_code == 200
        # Only the user message is persisted (no agent reply)
        assert mock_save.await_count == 1


# ---------------------------------------------------------------------------
# PUT /sessions/{id}/review
# ---------------------------------------------------------------------------


class TestLawyerReview:
    @pytest.mark.asyncio
    async def test_lawyer_review_session_not_found(
        self, lawyer_app, lawyer_auth_headers, mock_db_session
    ):
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state:
            mock_get_state.return_value = None

            async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client:
                response = await client.put(
                    f"{SESSIONS_PREFIX}/sess-001/review",
                    json={"decision": "approved"},
                    headers=lawyer_auth_headers,
                )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_lawyer_review_not_awaiting(
        self, lawyer_app, lawyer_auth_headers, sample_session_state, mock_db_session
    ):
        sample_session_state["awaiting_lawyer_review"] = False
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client:
                response = await client.put(
                    f"{SESSIONS_PREFIX}/sess-001/review",
                    json={"decision": "approved"},
                    headers=lawyer_auth_headers,
                )

        assert response.status_code == 400
        assert "未在等待律师审核" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_lawyer_review_invalid_decision(
        self, lawyer_app, lawyer_auth_headers, sample_session_state, mock_db_session
    ):
        sample_session_state["awaiting_lawyer_review"] = True
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client:
                response = await client.put(
                    f"{SESSIONS_PREFIX}/sess-001/review",
                    json={"decision": "invalid_value"},
                    headers=lawyer_auth_headers,
                )

        # Pydantic validation rejects unknown Literal -> 422
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_lawyer_review_internal_error(
        self, lawyer_app, lawyer_auth_headers, sample_session_state, mock_db_session
    ):
        sample_session_state["awaiting_lawyer_review"] = True
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_lawyer_review",
            new_callable=AsyncMock,
        ) as mock_proc:
            mock_get_state.return_value = sample_session_state
            mock_proc.side_effect = RuntimeError("workflow failed")

            async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client:
                response = await client.put(
                    f"{SESSIONS_PREFIX}/sess-001/review",
                    json={"decision": "approved"},
                    headers=lawyer_auth_headers,
                )

        assert response.status_code == 500
        assert "审核处理失败" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_lawyer_review_approved_workflow_finished(
        self, lawyer_app, lawyer_auth_headers, sample_session_state, mock_db_session
    ):
        sample_session_state["awaiting_lawyer_review"] = True
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_lawyer_review",
            new_callable=AsyncMock,
        ) as mock_proc, patch(
            "app.v1.router.consultation.routes.orchestrator"
        ) as mock_orch:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = sample_session_state
            mock_orch.is_workflow_finished = AsyncMock(return_value=True)

            async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client:
                response = await client.put(
                    f"{SESSIONS_PREFIX}/sess-001/review",
                    json={"decision": "approved", "feedback": "ok", "final_output": "final report"},
                    headers=lawyer_auth_headers,
                )

            assert response.status_code == 200
            data = response.json()
            assert data["decision"] == "approved"
            assert data["next_agent"] is None  # Workflow finished
            assert data["feedback"] == "ok"

    @pytest.mark.asyncio
    async def test_lawyer_review_revise_routes_back(
        self, lawyer_app, lawyer_auth_headers, sample_session_state, mock_db_session
    ):
        sample_session_state["awaiting_lawyer_review"] = True
        result_state = dict(sample_session_state)
        result_state["current_agent"] = "FactDigger"
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_lawyer_review",
            new_callable=AsyncMock,
        ) as mock_proc, patch(
            "app.v1.router.consultation.routes.orchestrator"
        ) as mock_orch:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = result_state
            mock_orch.is_workflow_finished = AsyncMock(return_value=False)

            async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client:
                response = await client.put(
                    f"{SESSIONS_PREFIX}/sess-001/review",
                    json={"decision": "revise_facts", "feedback": "补充更多事实"},
                    headers=lawyer_auth_headers,
                )

            assert response.status_code == 200
            data = response.json()
            assert data["decision"] == "revise_facts"
            assert data["next_agent"] == "FactDigger"


# ---------------------------------------------------------------------------
# GET /sessions - additional branch coverage
# ---------------------------------------------------------------------------


class TestListSessionsExtra:
    @pytest.mark.asyncio
    async def test_list_sessions_lawyer_filters_by_assignment(
        self, test_app, lawyer_auth_headers
    ):
        # Build two sessions — one assigned to this lawyer, one assigned to someone else.
        my_session = make_consultation_state(
            session_id="sess-mine",
            user_id="client-001",
            lawyer_id="lawyer-001",
            consent_given=True,
            awaiting_lawyer_review=True,
        )
        other_session = make_consultation_state(
            session_id="sess-other",
            user_id="client-002",
            lawyer_id="lawyer-999",
            consent_given=True,
        )

        with patch(
            "app.v1.router.consultation.routes.orchestrator"
        ) as mock_orch:
            mock_orch.get_active_sessions.return_value = {
                "sess-mine": my_session,
                "sess-other": other_session,
            }

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}", headers=lawyer_auth_headers
                )

        assert response.status_code == 200
        ids = [s["session_id"] for s in response.json()["sessions"]]
        assert "sess-mine" in ids
        # Sessions assigned to other lawyers are filtered out
        assert "sess-other" not in ids

    @pytest.mark.asyncio
    async def test_list_sessions_admin_sees_all(
        self, test_app
    ):
        admin_token = create_access_token(user_id="admin-001", role="admin")
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        s1 = make_consultation_state(
            session_id="sess-a",
            user_id="client-001",
            lawyer_id="lawyer-999",
            consent_given=True,
        )
        s2 = make_consultation_state(
            session_id="sess-b",
            user_id="client-002",
            lawyer_id=None,
            consent_given=True,
        )

        with patch(
            "app.v1.router.consultation.routes.orchestrator"
        ) as mock_orch:
            mock_orch.get_active_sessions.return_value = {
                "sess-a": s1,
                "sess-b": s2,
            }

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}", headers=admin_headers
                )

        assert response.status_code == 200
        ids = [s["session_id"] for s in response.json()["sessions"]]
        assert "sess-a" in ids
        assert "sess-b" in ids

    @pytest.mark.asyncio
    async def test_list_sessions_with_risk_level(
        self, test_app
    ):
        admin_token = create_access_token(user_id="admin-001", role="admin")
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        state = make_consultation_state(
            session_id="sess-risk",
            user_id="client-001",
            consent_given=True,
            risk_assessment={"risk_level": "high"},
        )

        with patch(
            "app.v1.router.consultation.routes.orchestrator"
        ) as mock_orch:
            mock_orch.get_active_sessions.return_value = {"sess-risk": state}

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}", headers=admin_headers
                )

        data = response.json()
        assert data["total"] == 1
        item = data["sessions"][0]
        assert item["risk_level"] == "high"
        # When awaiting_lawyer_review is True, status is "active"
        assert item["status"] == "in_progress"


# ---------------------------------------------------------------------------
# /state - admin access and awaiting_lawyer_review branch
# ---------------------------------------------------------------------------


class TestGetSessionStateExtra:
    @pytest.mark.asyncio
    async def test_admin_can_access_other_users_state(
        self, test_app, sample_session_state
    ):
        admin_token = create_access_token(user_id="admin-001", role="admin")
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}/sess-001/state",
                    headers=admin_headers,
                )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_state_uses_defaults_for_missing_fields(
        self, test_app, client_auth_headers
    ):
        # state with minimal keys to exercise default branches
        minimal_state = {
            "session_id": "sess-001",
            "user_id": "user-001",
            "consent_given": True,
            "current_agent": "Receptionist",
            "facts_raw": [],
            "applied_laws": [],
            "conversation_history": [],
            "pending_questions": [],
        }
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state:
            mock_get_state.return_value = minimal_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.get(
                    f"{SESSIONS_PREFIX}/sess-001/state",
                    headers=client_auth_headers,
                )

        data = response.json()
        assert data["user_type"] == "suspect"
        assert data["status"] == "active"
        assert data["alert_triggered"] is False
        assert data["final_output"] is None


# ---------------------------------------------------------------------------
# /close - additional coverage
# ---------------------------------------------------------------------------


class TestCloseSessionExtra:
    @pytest.mark.asyncio
    async def test_close_session_no_request_body(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.persist_state",
            new_callable=AsyncMock,
        ) as mock_persist:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/close",
                    headers=client_auth_headers,
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # When reason is None, the message should not include a reason suffix.
        assert "原因" not in data["message"]
        mock_persist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_session_creates_history_when_missing(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        # Make conversation_history missing to exercise the "not in state" branch.
        sample_session_state.pop("conversation_history", None)
        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.persist_state",
            new_callable=AsyncMock,
        ) as mock_persist:
            mock_get_state.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/close",
                    json={"reason": "测试"},
                    headers=client_auth_headers,
                )

        assert response.status_code == 200
        # The state should now have a conversation_history with the closing entry.
        assert "conversation_history" in sample_session_state
        last = sample_session_state["conversation_history"][-1]
        assert last["action"] == "session_closed"
        assert last["reason"] == "测试"
        assert sample_session_state["current_agent"] == "END"


# ---------------------------------------------------------------------------
# /confirm-consent - extra branches (consultation lookup, decline path)
# ---------------------------------------------------------------------------


class TestConfirmConsentExtra:
    @pytest.mark.asyncio
    async def test_confirm_consent_updates_db_when_consultation_exists(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        sample_session_state["consultation_id"] = "consult-001"

        consultation = MagicMock()
        consultation.consent_given = False
        result_proxy = MagicMock()
        result_proxy.scalar_one_or_none.return_value = consultation
        mock_db_session.execute = AsyncMock(return_value=result_proxy)

        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_consent",
            new_callable=AsyncMock,
        ) as mock_proc:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/confirm-consent",
                    json={
                        "session_id": "sess-001",
                        "consent_given": True,
                        "consent_timestamp": "2026-01-01T00:00:00Z",
                        "consent_version": "1.0",
                    },
                    headers=client_auth_headers,
                )

        assert response.status_code == 200
        assert consultation.consent_given is True
        mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_confirm_consent_declined(
        self, test_app, client_auth_headers, sample_session_state, mock_db_session
    ):
        # ``sample_session_state`` has a ``consultation_id`` so the route will
        # try to load the consultation. Make ``db.execute`` awaitable.
        result_proxy = MagicMock()
        result_proxy.scalar_one_or_none.return_value = None
        mock_db_session.execute = AsyncMock(return_value=result_proxy)

        with patch(
            "app.v1.service.consultation_service.get_session_state",
            new_callable=AsyncMock,
        ) as mock_get_state, patch(
            "app.v1.service.consultation_service.process_consent",
            new_callable=AsyncMock,
        ) as mock_proc:
            mock_get_state.return_value = sample_session_state
            mock_proc.return_value = sample_session_state

            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
                response = await client.post(
                    f"{SESSIONS_PREFIX}/sess-001/confirm-consent",
                    json={
                        "session_id": "sess-001",
                        "consent_given": False,
                        "consent_timestamp": "2026-01-01T00:00:00Z",
                        "consent_version": "1.0",
                    },
                    headers=client_auth_headers,
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # The decline path should pass identity_info=None
        proc_kwargs = mock_proc.call_args.kwargs
        assert proc_kwargs["consent_given"] is False
        assert proc_kwargs["identity_info"] is None
        assert "如需继续" in data["next_prompt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
