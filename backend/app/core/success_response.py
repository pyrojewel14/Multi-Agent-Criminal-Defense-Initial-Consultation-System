from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


def success_response(
    message: str = "success", data=None, status_code: int = 200
) -> JSONResponse:
    """成功响应体。

    Args:
        message: 响应消息。
        data: 响应数据。
        status_code: HTTP 状态码，同时写入响应体 code 字段。

    Returns:
        JSONResponse。
    """
    response = {
        "code": status_code,
        "message": message,
        "data": data,
    }
    return JSONResponse(status_code=status_code, content=jsonable_encoder(response))
