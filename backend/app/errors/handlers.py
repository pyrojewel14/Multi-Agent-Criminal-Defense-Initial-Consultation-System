from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors.codes import ErrorCode
from app.errors.exceptions import AppException
from app.utils.logger import get_logger

_logger = get_logger("Errors")


async def app_exception_handler(
    request: Request, exc: AppException
) -> JSONResponse:
    """处理所有 AppException 子类的标准化错误响应。

    将异常的 code 和 message 映射到格式：
        {"error": {"code": "...", "message": "..."}}

    Args:
        request: FastAPI 请求对象。
        exc: AppException 异常实例。

    Returns:
        JSONResponse 响应对象。
    """
    _logger.warning(
        "【app_exception_handler】AppException: code=%s status=%d path=%s",
        exc.code.value,
        exc.status_code,
        request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code.value,
                "message": exc.message,
            }
        },
    )


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """处理 Pydantic/FastAPI 请求验证错误（422）。

    注册用于 RequestValidationError，记录哪些字段验证失败。

    Args:
        request: FastAPI 请求对象。
        exc: 验证异常实例。

    Returns:
        JSONResponse 响应对象。
    """
    from fastapi.exceptions import RequestValidationError

    if isinstance(exc, RequestValidationError):
        _logger.warning(
            "【validation_exception_handler】验证错误 on %s: %s",
            request.url.path,
            exc.errors(),
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": "请求参数校验失败",
            }
        },
    )


_HTTP_ERROR_CODES = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    429: ErrorCode.RATE_LIMITED,
}


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """将 FastAPI/Starlette HTTPException 映射为统一错误响应。"""
    code = _HTTP_ERROR_CODES.get(exc.status_code)
    if code is None:
        code = ErrorCode.INTERNAL_ERROR if exc.status_code >= 500 else ErrorCode.HTTP_ERROR

    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    _logger.warning(
        "【http_exception_handler】HTTPException: code=%s status=%d path=%s",
        code.value,
        exc.status_code,
        request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code.value, "message": message}},
        headers=exc.headers,
    )


async def fallback_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """未处理异常的兜底处理器。

    返回 500 和通用消息；完整的堆栈跟踪会被记录，
    但不会暴露给客户端。

    Args:
        request: FastAPI 请求对象。
        exc: 异常实例。

    Returns:
        JSONResponse 响应对象。
    """
    import traceback

    _logger.error(
        "【fallback_exception_handler】未处理的异常 on %s: %s\n%s",
        request.url.path,
        str(exc),
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "系统内部错误，请稍后重试",
            }
        },
    )
