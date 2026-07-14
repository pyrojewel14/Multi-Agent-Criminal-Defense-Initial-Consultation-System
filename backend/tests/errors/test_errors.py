import pytest

from app.errors.codes import ErrorCode
from app.errors.exceptions import (
    AppException,
    ConsentRequiredException,
    HighRiskAlertException,
    LawRetrievalFailedException,
    LLMServiceException,
    LLMTimeoutException,
    UnauthorizedException,
)
from app.errors.handlers import (
    app_exception_handler,
    fallback_exception_handler,
    validation_exception_handler,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_all_expected_values_exist(self):
        expected = [
            "INTERNAL_ERROR",
            "VALIDATION_ERROR",
            "NOT_FOUND",
            "CONSENT_REQUIRED",
            "HIGH_RISK_ALERT",
            "LAW_RETRIEVAL_FAILED",
            "UNAUTHORIZED",
            "FORBIDDEN",
            "PII_DETECTED",
            "LLM_SERVICE_ERROR",
            "LLM_TIMEOUT",
        ]
        for name in expected:
            assert hasattr(ErrorCode, name)

    def test_enum_values_are_strings(self):
        for member in ErrorCode:
            assert isinstance(member.value, str)

    def test_enum_value_equals_name(self):
        assert ErrorCode.INTERNAL_ERROR.value == "INTERNAL_ERROR"
        assert ErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"
        assert ErrorCode.UNAUTHORIZED.value == "UNAUTHORIZED"


class TestAppException:
    """Tests for AppException and subclasses."""

    def test_app_exception_construction_with_defaults(self):
        """AppException requires class-level attributes; test via subclass."""
        exc = UnauthorizedException(detail="test detail")
        assert exc.code == ErrorCode.UNAUTHORIZED
        assert exc.status_code == 401
        assert exc.message == "身份验证失败，请重新登录"
        assert exc.detail == "test detail"

    def test_app_exception_override_message_via_base_class(self):
        """AppException base class allows overriding message via keyword arg."""
        exc = AppException(
            detail="some detail",
            code=ErrorCode.UNAUTHORIZED,
            status_code=401,
            message="自定义消息",
        )
        assert exc.message == "自定义消息"
        assert exc.code == ErrorCode.UNAUTHORIZED
        assert exc.status_code == 401

    def test_consent_required_exception(self):
        exc = ConsentRequiredException(session_id="sess123")
        assert exc.code == ErrorCode.CONSENT_REQUIRED
        assert exc.status_code == 403
        assert exc.message == "请先完成隐私条款确认"
        assert exc.detail is not None
        assert "sess123" in exc.detail

    def test_high_risk_alert_exception(self):
        exc = HighRiskAlertException(session_id="sess456", risk_detail="自证其罪")
        assert exc.code == ErrorCode.HIGH_RISK_ALERT
        assert exc.status_code == 403
        assert exc.detail is not None
        assert "sess456" in exc.detail
        assert "自证其罪" in exc.detail

    def test_law_retrieval_failed_exception(self):
        exc = LawRetrievalFailedException(detail="connection refused")
        assert exc.code == ErrorCode.LAW_RETRIEVAL_FAILED
        assert exc.status_code == 502
        assert exc.message == "法条检索服务暂不可用，请稍后重试"

    def test_llm_service_exception(self):
        exc = LLMServiceException(detail="api error")
        assert exc.code == ErrorCode.LLM_SERVICE_ERROR
        assert exc.status_code == 502

    def test_llm_timeout_exception(self):
        exc = LLMTimeoutException(detail="timeout after 30s")
        assert exc.code == ErrorCode.LLM_TIMEOUT
        assert exc.status_code == 504

    def test_exception_is_subclass_of_exception(self):
        exc = UnauthorizedException(detail="")
        assert isinstance(exc, AppException)
        assert isinstance(exc, Exception)


class TestExceptionHandlers:
    """Tests for exception handler functions."""

    @pytest.mark.asyncio
    async def test_app_exception_handler_returns_correct_status_and_json(self):
        from unittest.mock import MagicMock

        request = MagicMock()
        request.url.path = "/test"
        exc = UnauthorizedException(detail="bad token")

        response = await app_exception_handler(request, exc)
        assert response.status_code == 401
        body = response.body
        import json

        data = json.loads(bytes(body))
        assert "error" in data
        assert data["error"]["code"] == "UNAUTHORIZED"
        assert data["error"]["message"] == "身份验证失败，请重新登录"

    @pytest.mark.asyncio
    async def test_fallback_exception_handler_returns_500(self):
        from unittest.mock import MagicMock

        request = MagicMock()
        request.url.path = "/test"
        exc = RuntimeError("unexpected")

        response = await fallback_exception_handler(request, exc)
        assert response.status_code == 500
        import json

        data = json.loads(bytes(response.body))
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert data["error"]["message"] == "系统内部错误，请稍后重试"

    @pytest.mark.asyncio
    async def test_validation_exception_handler_returns_422(self):
        from unittest.mock import MagicMock

        request = MagicMock()
        request.url.path = "/test"
        exc = Exception("validation error")

        response = await validation_exception_handler(request, exc)
        assert response.status_code == 422
        import json

        data = json.loads(bytes(response.body))
        assert data["error"]["code"] == "VALIDATION_ERROR"
