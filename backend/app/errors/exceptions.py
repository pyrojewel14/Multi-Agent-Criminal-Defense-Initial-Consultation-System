from app.errors.codes import ErrorCode
from app.infrastructure.logging import get_logger


class AppException(Exception):  # noqa: N818 - 保留现有公开异常类型名称
    """应用层异常的基类。

    在实例化时自动以 ERROR 级别记录日志。子类声明静态类属性
    code、status_code 和 message；基类 __init__ 默认使用这些属性，
    但允许实例级覆盖。

    Attributes:
        code: 机器可读的错误代码（ErrorCode 枚举）。
        status_code: 返回给客户端的 HTTP 状态码。
        message: 对客户端暴露的人类可读消息。
        detail: 内部诊断字符串（不会暴露给客户端）。
    """

    code: ErrorCode
    status_code: int
    message: str

    def __init__(
        self,
        detail: str | None = None,
        code: ErrorCode | None = None,
        status_code: int | None = None,
        message: str | None = None,
    ):
        """初始化应用异常。

        Args:
            detail: 内部诊断详情。
            code: 错误代码。
            status_code: HTTP 状态码。
            message: 错误消息。
        """
        self.code = code or self.__class__.code
        self.status_code = status_code or self.__class__.status_code
        self.message = message or self.__class__.message
        self.detail = detail

        _logger = get_logger("Errors")
        _logger.error(
            "【__init__】[%s] %s | status=%d detail=%s",
            self.code.value,
            self.message,
            self.status_code,
            self.detail or "-",
        )
        super().__init__(self.message)


class ConsentRequiredException(AppException):
    """当客户端尝试在未获得隐私同意的情况下继续操作时抛出。"""

    code = ErrorCode.CONSENT_REQUIRED
    status_code = 403
    message = "请先完成隐私条款确认"

    def __init__(self, session_id: str):
        """初始化隐私同意异常。

        Args:
            session_id: 会话 ID。
        """
        super().__init__(detail=f"Consent not given for session: {session_id}")


class HighRiskAlertException(AppException):
    """当检测到高风险语句（自证其罪等）时抛出。"""

    code = ErrorCode.HIGH_RISK_ALERT
    status_code = 403
    message = "为保护您的权益，此部分内容建议直接与律师单独沟通"

    def __init__(self, session_id: str, risk_detail: str = ""):
        """初始化高风险告警异常。

        Args:
            session_id: 会话 ID。
            risk_detail: 风险详情。
        """
        super().__init__(detail=f"High-risk statement in session {session_id}: {risk_detail}")


class LawRetrievalFailedException(AppException):
    """当法律知识库检索失败时抛出。"""

    code = ErrorCode.LAW_RETRIEVAL_FAILED
    status_code = 502
    message = "法条检索服务暂不可用，请稍后重试"

    def __init__(self, detail: str = ""):
        """初始化法律检索失败异常。

        Args:
            detail: 详细错误信息。
        """
        super().__init__(detail=detail)


class LLMServiceException(AppException):
    """当上游 LLM API 返回错误时抛出。"""

    code = ErrorCode.LLM_SERVICE_ERROR
    status_code = 502
    message = "AI 服务暂不可用，请稍后重试"

    def __init__(self, detail: str = ""):
        """初始化 LLM 服务异常。

        Args:
            detail: 详细错误信息。
        """
        super().__init__(detail=detail)


class LLMTimeoutException(AppException):
    """当上游 LLM API 超时时抛出。"""

    code = ErrorCode.LLM_TIMEOUT
    status_code = 504
    message = "AI 服务响应超时，请稍后重试"

    def __init__(self, detail: str = ""):
        """初始化 LLM 超时异常。

        Args:
            detail: 详细错误信息。
        """
        super().__init__(detail=detail)


class UnauthorizedException(AppException):
    """当 JWT 令牌缺失、无效或过期时抛出。"""

    code = ErrorCode.UNAUTHORIZED
    status_code = 401
    message = "身份验证失败，请重新登录"

    def __init__(self, detail: str = ""):
        """初始化未授权异常。

        Args:
            detail: 详细错误信息。
        """
        super().__init__(detail=detail)
