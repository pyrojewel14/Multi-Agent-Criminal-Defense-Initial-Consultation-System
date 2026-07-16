"""Lawyer session router endpoint tests.

Tests cover:
1. GET /api/v1/lawyer/sessions - List assigned sessions
2. GET /api/v1/lawyer/sessions/{id} - Get session detail
3. PUT /api/v1/lawyer/sessions/{id}/report - Approve report
4. POST /api/v1/lawyer/sessions/{id}/reject - Reject session
5. POST /api/v1/lawyer/sessions/{id}/intervene - Manual intervention
6. GET /api/v1/lawyer/alerts - Get risk alerts
7. PUT /api/v1/lawyer/alerts/{id}/read - Mark alert as read
"""

import json
from datetime import datetime
from enum import Enum as PyEnum
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.security.jwt import create_access_token
from app.security.rbac import require_admin, require_lawyer


LAWYER_PREFIX = "/api/v1/lawyer"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StatusValue(str, PyEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


def _make_consultation(**overrides):
    """Build a SimpleNamespace standing in for a Consultation row."""
    consultation = SimpleNamespace(
        id="consult-001",
        client_id="client-001",
        assigned_lawyer_id="lawyer-001",
        user_type="suspect",
        consent_given=True,
        status=_StatusValue.IN_PROGRESS,
        facts_raw=json.dumps(["一段原始事实"]),
        facts_structured=json.dumps({"key": "value"}),
        applied_laws=json.dumps([{"article": "264"}]),
        risk_level="medium",
        alert_triggered=True,
        lawyer_review_needed=True,
        risk_assessment={"risk_type": "逃亡风险", "risk_level": "high"},
        report_draft="报告草稿",
        service_plan={"plan": "test"},
        final_output=None,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 2, 12, 0, 0),
        completed_at=None,
        alert_read=False,
    )
    for k, v in overrides.items():
        setattr(consultation, k, v)
    return consultation


def _make_client(**overrides):
    client = SimpleNamespace(
        id="client-001",
        username="clientuser",
        real_name="客户甲",
    )
    for k, v in overrides.items():
        setattr(client, k, v)
    return client


def _make_message(msg_id="msg-001", content="hello"):
    return SimpleNamespace(
        id=msg_id,
        sender_type="user",
        sender_id="client-001",
        content=content,
        agent_name=None,
        message_type="text",
        created_at=datetime(2026, 1, 1, 12, 0, 0),
    )


def _setup_db(mock_db_session, *, scalars_returns=None, scalar_one=None, scalar_one_or_none=None, n_calls=10):
    """Configure the mock DB to return a sequence of results for repeated .execute() calls.

    - scalars_returns: list of iterables for .scalars().all() (per call index, None to skip)
    - scalar_one: list of values for .scalar_one() (per call index, None to skip)
    - scalar_one_or_none: list of values for .scalar_one_or_none() (per call index, None to skip)
    - n_calls: number of expected calls (used to pad the lists)
    """
    if scalars_returns is not None:
        scalars_returns = list(scalars_returns) + [None] * (n_calls - len(scalars_returns))
    if scalar_one is not None and isinstance(scalar_one, list):
        scalar_one = list(scalar_one) + [None] * (n_calls - len(scalar_one))
    if scalar_one_or_none is not None and isinstance(scalar_one_or_none, list):
        scalar_one_or_none = list(scalar_one_or_none) + [None] * (n_calls - len(scalar_one_or_none))

    execute_call = {"i": 0}

    async def _execute(*args, **kwargs):
        idx = execute_call["i"]
        execute_call["i"] += 1
        result = MagicMock()
        # Default scalar_one_or_none to None so that "not consultation" checks work
        result.scalar_one_or_none.return_value = None
        if scalars_returns is not None and idx < len(scalars_returns):
            data = scalars_returns[idx]
            if data is not None:
                result.scalars.return_value.all.return_value = data
        if scalar_one is not None and idx < len(scalar_one):
            val = scalar_one[idx]
            if val is not None:
                result.scalar_one.return_value = val
        if scalar_one_or_none is not None and idx < len(scalar_one_or_none):
            val = scalar_one_or_none[idx]
            if val is not None:
                result.scalar_one_or_none.return_value = val
        return result

    mock_db_session.execute = AsyncMock(side_effect=_execute)
    return execute_call


@pytest.fixture
async def lawyer_app(test_app, monkeypatch):
    """Test app with ``require_lawyer``/``require_admin`` overridden to fixed users."""
    import app.security.rbac as rbac_module
    import app.v1.router.lawyer as lawyer_module
    import app.v1.router.lawyers as lawyers_module
    import app.v1.router.users as users_module
    import app.v1.router.knowledge_router as knowledge_module
    import app.v1.router.consultation_history as history_module
    from app.security.jwt import create_access_token

    async def _lawyer_override():
        return {"user_id": "lawyer-001", "role": "lawyer"}

    async def _admin_override():
        return {"user_id": "admin-001", "role": "admin"}

    # Replace the factory functions so the Depends() in the route sees our
    # async functions. Since ``Depends(require_lawyer)`` is evaluated at the
    # time of route registration (import time), we need to patch BEFORE
    # re-importing the route modules. Instead, we patch in place: replace the
    # ``require_lawyer`` and ``require_admin`` symbols in each module that
    # references them.
    for mod in [lawyer_module, lawyers_module, users_module, knowledge_module, history_module]:
        monkeypatch.setattr(mod, "require_lawyer", _lawyer_override, raising=False)
        monkeypatch.setattr(mod, "require_admin", _admin_override, raising=False)
    monkeypatch.setattr(history_module, "get_current_user", _lawyer_override, raising=False)

    # Also override the actual RoleChecker instance that the route captured.
    # Since Depends() in the route body was already evaluated at import time,
    # the route still references the original RoleChecker. We can override
    # via ``app.dependency_overrides`` using the *function* that produces the
    # RoleChecker (require_lawyer / require_admin themselves).
    test_app.dependency_overrides[rbac_module.require_lawyer] = _lawyer_override
    test_app.dependency_overrides[rbac_module.require_admin] = _admin_override

    yield test_app


@pytest.fixture
async def forbidden_lawyer_app(test_app, monkeypatch):
    """Test app where the auth check returns a 'other-lawyer' user.

    Used for testing 403/permission-denied paths.
    """
    import app.security.rbac as rbac_module
    import app.v1.router.lawyer as lawyer_module
    import app.v1.router.lawyers as lawyers_module
    import app.v1.router.users as users_module
    import app.v1.router.knowledge_router as knowledge_module
    import app.v1.router.consultation_history as history_module

    async def _lawyer_override():
        return {"user_id": "other-lawyer", "role": "lawyer"}

    async def _admin_override():
        return {"user_id": "admin-001", "role": "admin"}

    for mod in [lawyer_module, lawyers_module, users_module, knowledge_module, history_module]:
        monkeypatch.setattr(mod, "require_lawyer", _lawyer_override, raising=False)
        monkeypatch.setattr(mod, "require_admin", _admin_override, raising=False)
    monkeypatch.setattr(history_module, "get_current_user", _lawyer_override, raising=False)

    test_app.dependency_overrides[rbac_module.require_lawyer] = _lawyer_override
    test_app.dependency_overrides[rbac_module.require_admin] = _admin_override

    yield test_app


@pytest.fixture
async def client_role_app(test_app, monkeypatch):
    """Test app with a 'client' role user (should be denied from lawyer routes)."""
    import app.security.rbac as rbac_module
    import app.v1.router.lawyer as lawyer_module
    import app.v1.router.lawyers as lawyers_module
    import app.v1.router.users as users_module
    import app.v1.router.knowledge_router as knowledge_module
    import app.v1.router.consultation_history as history_module

    async def _lawyer_override():
        return {"user_id": "client-001", "role": "client"}

    async def _admin_override():
        return {"user_id": "client-001", "role": "client"}

    for mod in [lawyer_module, lawyers_module, users_module, knowledge_module, history_module]:
        monkeypatch.setattr(mod, "require_lawyer", _lawyer_override, raising=False)
        monkeypatch.setattr(mod, "require_admin", _admin_override, raising=False)
    monkeypatch.setattr(history_module, "get_current_user", _lawyer_override, raising=False)

    test_app.dependency_overrides[rbac_module.require_lawyer] = _lawyer_override
    test_app.dependency_overrides[rbac_module.require_admin] = _admin_override

    yield test_app


# ---------------------------------------------------------------------------
# GET /lawyer/sessions
# ---------------------------------------------------------------------------


class TestGetSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_success(self, lawyer_app, mock_db_session):
        """Lawyer can list assigned sessions with default pagination."""
        c = _make_consultation()
        client = _make_client()
        # The router runs:
        # 0) count_query -> scalar_one
        # 1) main query -> scalars
        # 2) per-row client lookup -> scalar_one_or_none
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [c], None],
            scalar_one=[1, None, None],
            scalar_one_or_none=[None, None, client],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{LAWYER_PREFIX}/sessions")

            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 200
            assert data["data"]["total"] == 1
            assert data["data"]["page"] == 1
            assert data["data"]["page_size"] == 20
            assert len(data["data"]["sessions"]) == 1
            assert data["data"]["sessions"][0]["client_username"] == "clientuser"

    @pytest.mark.asyncio
    async def test_list_sessions_with_status_filter(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [c], None],
            scalar_one=[1, None, None],
            scalar_one_or_none=[None, None, _make_client()],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{LAWYER_PREFIX}/sessions?status=in_progress&page=1&page_size=10",
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_sessions_invalid_status(self, lawyer_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{LAWYER_PREFIX}/sessions?status=invalid",
            )

            assert response.status_code == 400
            assert "无效" in response.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_list_sessions_with_needs_review_filter(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [c], None],
            scalar_one=[1, None, None],
            scalar_one_or_none=[None, None, _make_client()],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{LAWYER_PREFIX}/sessions?needs_review=true",
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_sessions_with_no_results(self, lawyer_app, mock_db_session):
        """Empty list when no sessions exist for the lawyer."""
        _setup_db(
            mock_db_session,
            scalars_returns=[None, [], None],
            scalar_one=[0, None, None],
            scalar_one_or_none=[None, None, None],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{LAWYER_PREFIX}/sessions")

            assert response.status_code == 200
            data = response.json()
            assert data["data"]["total"] == 0
            assert data["data"]["sessions"] == []


# ---------------------------------------------------------------------------
# GET /lawyer/sessions/{id}
# ---------------------------------------------------------------------------


class TestGetSessionDetail:
    @pytest.mark.asyncio
    async def test_get_session_detail_success(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        # 0) consultation lookup -> scalar_one_or_none
        # 1) client lookup -> scalar_one_or_none
        # 2) messages lookup -> scalars
        _setup_db(
            mock_db_session,
            scalars_returns=[None, None, [_make_message()]],
            scalar_one_or_none=[c, _make_client(), None],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{LAWYER_PREFIX}/sessions/consult-001",
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["id"] == "consult-001"
            assert data["facts_raw"] == ["一段原始事实"]
            assert data["facts_structured"] == {"key": "value"}
            assert data["applied_laws"] == [{"article": "264"}]
            assert len(data["conversation_history"]) == 1

    @pytest.mark.asyncio
    async def test_get_session_detail_not_found(self, lawyer_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{LAWYER_PREFIX}/sessions/missing",
            )
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_detail_forbidden(self, forbidden_lawyer_app, mock_db_session):
        """A consultation assigned to a different lawyer is forbidden."""
        c = _make_consultation(assigned_lawyer_id="lawyer-001")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=forbidden_lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{LAWYER_PREFIX}/sessions/consult-001",
            )
            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_session_detail_with_unparseable_json(self, lawyer_app, mock_db_session):
        """Invalid JSON in facts/applied_laws should be silently ignored."""
        c = _make_consultation(
            facts_raw="not-json",
            facts_structured="also-not-json",
            applied_laws="still-not-json",
        )
        _setup_db(
            mock_db_session,
            scalars_returns=[None, None, []],
            scalar_one_or_none=[c, _make_client(), None],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{LAWYER_PREFIX}/sessions/consult-001",
            )
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["facts_raw"] is None
            assert data["facts_structured"] is None
            assert data["applied_laws"] is None


# ---------------------------------------------------------------------------
# PUT /lawyer/sessions/{id}/report
# ---------------------------------------------------------------------------


class TestApproveReport:
    @pytest.mark.asyncio
    async def test_approve_report_success(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        _setup_db(mock_db_session, scalar_one_or_none=[c])
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYER_PREFIX}/sessions/consult-001/report",
                json={"final_output": "最终报告内容", "feedback": "已审核"},
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["success"] is True
            assert data["session_id"] == "consult-001"
            assert c.final_output == "最终报告内容"
            assert c.lawyer_review_needed is False
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_approve_report_not_found(self, lawyer_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYER_PREFIX}/sessions/missing/report",
                json={"final_output": "报告"},
            )
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_approve_report_forbidden(self, forbidden_lawyer_app, mock_db_session):
        c = _make_consultation(assigned_lawyer_id="lawyer-001")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=forbidden_lawyer_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYER_PREFIX}/sessions/consult-001/report",
                json={"final_output": "报告"},
            )
            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_approve_report_already_completed(self, lawyer_app, mock_db_session):
        c = _make_consultation(status=_StatusValue.COMPLETED)
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYER_PREFIX}/sessions/consult-001/report",
                json={"final_output": "报告"},
            )
            assert response.status_code == 400
            assert "已完成" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# POST /lawyer/sessions/{id}/reject
# ---------------------------------------------------------------------------


class TestRejectSession:
    @pytest.mark.asyncio
    async def test_reject_session_success(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        _setup_db(mock_db_session, scalar_one_or_none=[c])
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYER_PREFIX}/sessions/consult-001/reject",
                json={
                    "target_node": "fact_digger",
                    "reason": "事实需要补充",
                    "feedback": "请补充细节",
                },
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["success"] is True
            assert data["target_node"] == "fact_digger"
            assert c.lawyer_review_needed is False
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_reject_session_invalid_target_node(self, lawyer_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYER_PREFIX}/sessions/consult-001/reject",
                json={"target_node": "unknown_node"},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_reject_session_not_found(self, lawyer_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYER_PREFIX}/sessions/missing/reject",
                json={"target_node": "risk_assessor"},
            )
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_reject_session_forbidden(self, forbidden_lawyer_app, mock_db_session):
        c = _make_consultation(assigned_lawyer_id="lawyer-001")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=forbidden_lawyer_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYER_PREFIX}/sessions/consult-001/reject",
                json={"target_node": "fact_digger"},
            )
            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_reject_session_risk_assessor_node(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        _setup_db(mock_db_session, scalar_one_or_none=[c])
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYER_PREFIX}/sessions/consult-001/reject",
                json={"target_node": "risk_assessor"},
            )
            assert response.status_code == 200
            assert response.json()["data"]["target_node"] == "risk_assessor"


# ---------------------------------------------------------------------------
# POST /lawyer/sessions/{id}/intervene
# ---------------------------------------------------------------------------


class TestInterveneSession:
    @pytest.mark.asyncio
    async def test_intervene_session_success(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        _setup_db(mock_db_session, scalar_one_or_none=[c])
        mock_db_session.commit = AsyncMock()
        mock_db_session.add = MagicMock()

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYER_PREFIX}/sessions/consult-001/intervene",
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["success"] is True
            assert data["current_agent"] == "Lawyer"
            assert c.status == _StatusValue.IN_PROGRESS
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_intervene_session_not_found(self, lawyer_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYER_PREFIX}/sessions/missing/intervene",
            )
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_intervene_session_forbidden(self, forbidden_lawyer_app, mock_db_session):
        c = _make_consultation(assigned_lawyer_id="lawyer-001")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=forbidden_lawyer_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{LAWYER_PREFIX}/sessions/consult-001/intervene",
            )
            assert response.status_code == 403


# ---------------------------------------------------------------------------
# GET /lawyer/alerts
# ---------------------------------------------------------------------------


class TestGetAlerts:
    @pytest.mark.asyncio
    async def test_get_alerts_success(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        # 0) consultation list -> scalars
        # 1) per-row client lookup -> scalar_one_or_none
        _setup_db(
            mock_db_session,
            scalars_returns=[[c]],
            scalar_one_or_none=[_make_client()],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{LAWYER_PREFIX}/alerts")

            assert response.status_code == 200
            data = response.json()["data"]
            assert len(data) == 1
            assert data[0]["id"].startswith("alert_")
            assert data[0]["risk_type"] == "逃亡风险"
            assert data[0]["risk_level"] == "high"

    @pytest.mark.asyncio
    async def test_get_alerts_with_risk_level_filter(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        _setup_db(
            mock_db_session,
            scalars_returns=[[c]],
            scalar_one_or_none=[_make_client()],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(
                f"{LAWYER_PREFIX}/alerts?risk_level=high",
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_alerts_str_risk_assessment(self, lawyer_app, mock_db_session):
        """risk_assessment stored as a JSON string should be parsed."""
        c = _make_consultation(
            risk_assessment=json.dumps({"risk_type": "自伤风险", "risk_level": "critical"})
        )
        _setup_db(
            mock_db_session,
            scalars_returns=[[c]],
            scalar_one_or_none=[_make_client()],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{LAWYER_PREFIX}/alerts")
            assert response.status_code == 200
            data = response.json()["data"]
            assert data[0]["risk_type"] == "自伤风险"

    @pytest.mark.asyncio
    async def test_get_alerts_unparseable_risk_assessment(self, lawyer_app, mock_db_session):
        c = _make_consultation(risk_assessment="not-json")
        _setup_db(
            mock_db_session,
            scalars_returns=[[c]],
            scalar_one_or_none=[_make_client()],
        )

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{LAWYER_PREFIX}/alerts")
            assert response.status_code == 200
            data = response.json()["data"]
            # risk_type should fall back to "未知"
            assert data[0]["risk_type"] == "未知"


# ---------------------------------------------------------------------------
# PUT /lawyer/alerts/{id}/read
# ---------------------------------------------------------------------------


class TestMarkAlertRead:
    @pytest.mark.asyncio
    async def test_mark_alert_read_success(self, lawyer_app, mock_db_session):
        c = _make_consultation()
        _setup_db(mock_db_session, scalar_one_or_none=[c])
        mock_db_session.commit = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYER_PREFIX}/alerts/alert_consult-001/read",
            )

            assert response.status_code == 200
            assert c.alert_read is True
            mock_db_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_mark_alert_read_invalid_format(self, lawyer_app, mock_db_session):
        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYER_PREFIX}/alerts/bad_id/read",
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_mark_alert_read_not_found(self, lawyer_app, mock_db_session):
        _setup_db(mock_db_session, scalar_one_or_none=[None])

        async with AsyncClient(transport=ASGITransport(app=lawyer_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYER_PREFIX}/alerts/alert_missing/read",
            )
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_mark_alert_read_forbidden(self, forbidden_lawyer_app, mock_db_session):
        c = _make_consultation(assigned_lawyer_id="lawyer-001")
        _setup_db(mock_db_session, scalar_one_or_none=[c])

        async with AsyncClient(transport=ASGITransport(app=forbidden_lawyer_app), base_url="http://test") as client_http:
            response = await client_http.put(
                f"{LAWYER_PREFIX}/alerts/alert_consult-001/read",
            )
            assert response.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
