from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """包含机器可读错误码和用户可读消息的单条错误信息。"""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """所有异常处理器统一返回的错误响应结构。

    响应格式：
        {"error": {"code": "...", "message": "..."}}
    """

    error: ErrorDetail
