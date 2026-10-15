"""以固定诊断类别限制检索错误输出，未知类名不能携带私密数据。"""

from __future__ import annotations

# 只接受项目使用的固定类别，不按命名格式、异常继承或错误消息放行未知名称。
_SAFE_ERROR_TYPES = frozenset(
    {
        "OtherError",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "TimeoutError",
        "OSError",
        "FileNotFoundError",
        "PermissionError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "BrokenPipeError",
        "ImportError",
        "ModuleNotFoundError",
        "KeyError",
        "IndexError",
        "OverflowError",
        "MemoryError",
        "JSONDecodeError",
        "HTTPError",
        "HTTPStatusError",
        "RequestError",
        "TransportError",
        "NetworkError",
        "ConnectError",
        "ReadError",
        "WriteError",
        "CloseError",
        "ProxyError",
        "ProtocolError",
        "LocalProtocolError",
        "RemoteProtocolError",
        "UnsupportedProtocol",
        "DecodingError",
        "TooManyRedirects",
        "InvalidURL",
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "SSLError",
        "gaierror",
        "NotFoundError",
        "InvalidDimensionException",
        "LawKnowledgeDataError",
        "SessionBudgetExceeded",
        "RerankerBusyError",
        "RerankerClosedError",
        "CancelledError",
    }
)


def safe_error_type(error: BaseException | str | None) -> str | None:
    """保留固定异常类别；未知名称或非预期值统一返回 OtherError，不读取消息。"""
    if error is None:
        return None
    name = type(error).__name__ if isinstance(error, BaseException) else error
    return name if type(name) is str and name in _SAFE_ERROR_TYPES else "OtherError"
