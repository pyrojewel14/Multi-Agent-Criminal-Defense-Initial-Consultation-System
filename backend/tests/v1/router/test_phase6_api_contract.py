"""Phase 6 的真实 FastAPI 鉴权、错误和核心接口契约。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.security.jwt import create_access_token
from tests.factories import make_consultation_state


def _headers(user_id: str, role: str) -> dict[str, str]:
    token = create_access_token(user_id=user_id, role=role)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_openapi_exposes_core_paths_and_bearer_security(test_app):
    schema = test_app.openapi()
    paths = schema["paths"]
    expected_paths = {
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/sessions",
        "/api/v1/sessions/{session_id}/message",
        "/api/v1/sessions/{session_id}/state",
        "/api/v1/sessions/{session_id}/report-draft",
        "/api/v1/sessions/{session_id}/review",
        "/api/v1/knowledge/list",
        "/api/v1/consultations/list",
    }
    assert expected_paths <= paths.keys()
    assert paths["/api/v1/auth/login"]["post"].get("security") is None
    assert paths["/api/v1/users/"]["get"]["security"] == [{"HTTPBearer": []}]
    assert paths["/api/v1/knowledge/list"]["get"]["security"] == [{"HTTPBearer": []}]
    assert paths["/api/v1/sessions/{session_id}/review"]["put"]["security"] == [
        {"HTTPBearer": []}
    ]


@pytest.mark.asyncio
async def test_lawyer_review_role_matrix_uses_real_jwt_dependency(test_app):
    url = "/api/v1/sessions/phase6/review"
    payload = {"decision": "approved"}
    with patch(
        "app.v1.service.consultation_service.get_session_state",
        new_callable=AsyncMock,
        return_value=None,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            anonymous = await client.put(url, json=payload)
            ordinary = await client.put(
                url, json=payload, headers=_headers("client-1", "client")
            )
            lawyer = await client.put(
                url, json=payload, headers=_headers("lawyer-1", "lawyer")
            )
            admin = await client.put(
                url, json=payload, headers=_headers("admin-1", "admin")
            )

    assert (anonymous.status_code, anonymous.json()["error"]["code"]) == (
        401,
        "UNAUTHORIZED",
    )
    assert (ordinary.status_code, ordinary.json()["error"]["code"]) == (
        403,
        "FORBIDDEN",
    )
    assert lawyer.status_code == 404
    assert admin.status_code == 404


@pytest.mark.asyncio
async def test_admin_route_role_matrix_uses_real_jwt_dependency(
    test_app, mock_db_session
):
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    mock_db_session.execute = AsyncMock(return_value=result)
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        anonymous = await client.get("/api/v1/users/")
        ordinary = await client.get(
            "/api/v1/users/", headers=_headers("client-1", "client")
        )
        lawyer = await client.get(
            "/api/v1/users/", headers=_headers("lawyer-1", "lawyer")
        )
        admin = await client.get(
            "/api/v1/users/", headers=_headers("admin-1", "admin")
        )

    assert anonymous.status_code == 401
    assert ordinary.status_code == 403
    assert lawyer.status_code == 403
    assert admin.status_code == 200


@pytest.mark.asyncio
async def test_report_draft_enforces_assignment_and_returns_live_state(test_app):
    state = make_consultation_state(
        session_id="session-report",
        consultation_id="consultation-report",
        user_id="client-1",
        lawyer_id="lawyer-1",
        report_draft="报告草案正文",
        service_plan={"summary": "建议预约律师面谈"},
        awaiting_lawyer_review=True,
    )
    url = "/api/v1/sessions/session-report/report-draft"
    with patch(
        "app.v1.service.consultation_service.get_session_state",
        new_callable=AsyncMock,
        return_value=state,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            assigned = await client.get(
                url, headers=_headers("lawyer-1", "lawyer")
            )
            other = await client.get(
                url, headers=_headers("lawyer-2", "lawyer")
            )
            ordinary = await client.get(
                url, headers=_headers("client-1", "client")
            )
            admin = await client.get(url, headers=_headers("admin-1", "admin"))

    assert assigned.status_code == 200
    assert assigned.json()["report_draft"] == "报告草案正文"
    assert other.status_code == 403
    assert ordinary.status_code == 403
    assert admin.status_code == 200


@pytest.mark.asyncio
async def test_session_state_rejects_unassigned_lawyer(test_app):
    state = make_consultation_state(
        session_id="session-owned",
        user_id="client-1",
        lawyer_id="lawyer-1",
    )
    url = "/api/v1/sessions/session-owned/state"
    with patch(
        "app.v1.service.consultation_service.get_session_state",
        new_callable=AsyncMock,
        return_value=state,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            owner = await client.get(url, headers=_headers("client-1", "client"))
            assigned = await client.get(
                url, headers=_headers("lawyer-1", "lawyer")
            )
            other = await client.get(
                url, headers=_headers("lawyer-2", "lawyer")
            )
            admin = await client.get(url, headers=_headers("admin-1", "admin"))

    assert owner.status_code == 200
    assert assigned.status_code == 200
    assert other.status_code == 403
    assert admin.status_code == 200


@pytest.mark.asyncio
async def test_client_state_reports_completed_after_lawyer_approval(test_app):
    state = make_consultation_state(
        session_id="session-approved",
        consultation_id="consultation-approved",
        user_id="client-1",
        lawyer_id="lawyer-1",
        current_agent="HumanReview",
        awaiting_lawyer_review=False,
        lawyer_decision="approved",
        final_output="律师审核后的最终报告",
    )
    with patch(
        "app.v1.service.consultation_service.get_session_state",
        new_callable=AsyncMock,
        return_value=state,
    ), patch(
        "app.v1.router.consultation.routes.orchestrator.is_workflow_finished",
        new_callable=AsyncMock,
        return_value=True,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/sessions/session-approved/state",
                headers=_headers("client-1", "client"),
            )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["final_output"] == "律师审核后的最终报告"


@pytest.mark.asyncio
async def test_unknown_signed_role_fails_closed(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/v1/sessions", headers=_headers("unknown-1", "bogus")
        )

    assert response.status_code == 401
    assert response.json() == {
        "error": {"code": "UNAUTHORIZED", "message": "无效的 Token 声明"}
    }


@pytest.mark.asyncio
async def test_http_and_validation_errors_share_error_envelope(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        not_found = await client.get("/api/v1/not-a-real-route")
        validation = await client.post(
            "/api/v1/sessions",
            json={},
            headers=_headers("client-1", "client"),
        )

    assert not_found.status_code == 404
    assert not_found.json()["error"] == {
        "code": "NOT_FOUND",
        "message": "Not Found",
    }
    assert validation.status_code == 422
    assert validation.json()["error"] == {
        "code": "VALIDATION_ERROR",
        "message": "请求参数校验失败",
    }
