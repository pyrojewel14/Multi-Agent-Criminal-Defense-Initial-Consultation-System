"""Integration tests for the registered FastAPI exception handlers."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.errors.exceptions import UnauthorizedException
from app.errors.register import register_exception_handlers


def _error_app() -> FastAPI:
    app = FastAPI()

    @app.get("/app-error")
    async def app_error() -> None:
        raise UnauthorizedException("token expired")

    @app.get("/unexpected")
    async def unexpected_error() -> None:
        raise RuntimeError("internal detail must stay private")

    @app.get("/validated/{item_id}")
    async def validated(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    register_exception_handlers(app)
    return app


@pytest.mark.asyncio
async def test_registered_handlers_keep_a_stable_error_envelope():
    transport = ASGITransport(app=_error_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        app_error = await client.get("/app-error")
        validation_error = await client.get("/validated/not-an-int")
        unexpected_error = await client.get("/unexpected")

    assert app_error.status_code == 401
    assert app_error.json() == {
        "error": {"code": "UNAUTHORIZED", "message": "身份验证失败，请重新登录"}
    }
    assert validation_error.status_code == 422
    assert validation_error.json() == {
        "error": {"code": "VALIDATION_ERROR", "message": "请求参数校验失败"}
    }
    assert unexpected_error.status_code == 500
    assert unexpected_error.json() == {
        "error": {"code": "INTERNAL_ERROR", "message": "系统内部错误，请稍后重试"}
    }
    assert "internal detail" not in unexpected_error.text
