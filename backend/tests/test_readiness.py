"""API 存活与重排序器 readiness 边界测试。"""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.mark.asyncio
async def test_readiness_distinguishes_api_liveness_from_reranker_error():
    reranker_state = {
        "status": "error",
        "available": False,
        "model": "test-model",
        "error": "RuntimeError: broken weights",
        "type": "cross_encoder",
    }
    with patch("main.reorder_service", create=True) as reorder_service:
        reorder_service.readiness.return_value = reranker_state
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            health_response = await client.get("/health")
            readiness_response = await client.get("/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy"}
    assert readiness_response.status_code == 200
    assert readiness_response.json() == {
        "status": "degraded",
        "api": "ready",
        "dependencies": {
            "reranker": reranker_state,
            "checkpoint": {"persistence": "process", "restart_recovery": False},
        },
    }
