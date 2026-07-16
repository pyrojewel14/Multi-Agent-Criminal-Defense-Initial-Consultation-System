"""WebSocket error-envelope contracts for typed application failures."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from app.errors.exceptions import LLMTimeoutException
from app.security.jwt import create_access_token
from app.v1.router.consultation.websocket import websocket_endpoint
from tests.factories import make_consultation_state


@pytest.mark.asyncio
async def test_websocket_returns_typed_llm_error_before_continuing():
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.query_params = {
        "token": create_access_token("ws-owner", "client")
    }
    websocket.headers = {}
    websocket.receive_text = AsyncMock(
        side_effect=[
            json.dumps({"type": "message", "content": "继续咨询"}, ensure_ascii=False),
            WebSocketDisconnect(),
        ]
    )
    heartbeat_task = MagicMock()
    state = make_consultation_state(
        session_id="ws-phase5", user_id="ws-owner", consent_given=True
    )

    def consume_heartbeat(coroutine):
        coroutine.close()
        return heartbeat_task

    with patch(
        "app.v1.router.consultation.websocket.consultation_service.get_session_state",
        new_callable=AsyncMock,
        return_value=state,
    ), patch(
        "app.v1.router.consultation.websocket.consultation_service.process_message",
        new_callable=AsyncMock,
        side_effect=LLMTimeoutException("upstream timeout"),
    ), patch(
        "app.v1.router.consultation.websocket.asyncio.create_task",
        side_effect=consume_heartbeat,
    ):
        await websocket_endpoint(websocket, "ws-phase5")

    sent_messages = [call.args[0] for call in websocket.send_json.await_args_list]
    assert {
        "type": "error",
        "content": "AI 服务响应超时，请稍后重试",
        "error_code": "LLM_TIMEOUT",
        "session_id": "ws-phase5",
    } in sent_messages
    heartbeat_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_rejects_anonymous_before_accepting():
    websocket = MagicMock()
    websocket.query_params = {}
    websocket.headers = {}
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()

    await websocket_endpoint(websocket, "ws-protected")

    websocket.accept.assert_not_awaited()
    websocket.close.assert_awaited_once_with(code=4401, reason="请先登录")


@pytest.mark.asyncio
async def test_websocket_rejects_other_client():
    websocket = MagicMock()
    websocket.query_params = {
        "token": create_access_token("other-client", "client")
    }
    websocket.headers = {}
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    state = make_consultation_state(
        session_id="ws-protected", user_id="owner-client"
    )

    with patch(
        "app.v1.router.consultation.websocket.consultation_service.get_session_state",
        new_callable=AsyncMock,
        return_value=state,
    ):
        await websocket_endpoint(websocket, "ws-protected")

    websocket.accept.assert_not_awaited()
    websocket.close.assert_awaited_once_with(code=4403, reason="无权访问此会话")
