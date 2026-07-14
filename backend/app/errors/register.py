from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors.exceptions import AppException
from app.errors.handlers import (
    app_exception_handler,
    validation_exception_handler,
    fallback_exception_handler,
)
from app.utils.logger import get_logger

_logger = get_logger("ErrorHandlers")


async def _registered_app_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """适配 Starlette 的通用异常处理器签名。"""
    if not isinstance(exc, AppException):
        return await fallback_exception_handler(request, exc)
    return await app_exception_handler(request, exc)


def register_exception_handlers(app: FastAPI) -> None:
    """为 FastAPI 应用注册全部异常处理器。

    处理层级：
        1. AppException → 自定义状态码和统一错误响应
        2. RequestValidationError → 422 和 VALIDATION_ERROR
        3. Exception（兜底）→ 500 和 INTERNAL_ERROR

    应在全部路由挂载完成后，于应用启动阶段调用一次。
    """
    app.add_exception_handler(AppException, _registered_app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, fallback_exception_handler)

    _logger.info("Exception handlers registered")
