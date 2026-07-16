from typing import List, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.security.jwt import decode_token
from app.utils.logger import get_logger

_logger = get_logger("RBAC")

security = HTTPBearer(auto_error=False)
VALID_ROLES = {"admin", "lawyer", "client"}


def _user_info_from_payload(payload: dict) -> Optional[dict]:
    """从已验证的 access token payload 构建受信任用户信息。"""
    user_id = payload.get("sub")
    role = payload.get("role")
    if payload.get("type") != "access" or not isinstance(user_id, str) or not user_id:
        return None
    if role not in VALID_ROLES:
        return None
    return {"user_id": user_id, "role": role}


async def get_optional_user_from_header(request: Request) -> Optional[dict]:
    """从请求头中尝试解析可选的用户信息。

    Args:
        request: FastAPI 请求对象。

    Returns:
        用户信息字典，解析失败返回 None。
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        _logger.debug("【get_optional_user_from_header】请求缺少Authorization头或格式错误: %s", request.url.path)
        return None

    token = auth_header.replace("Bearer ", "")
    payload = decode_token(token)

    if not payload:
        _logger.debug("【get_optional_user_from_header】Token解码失败")
        return None

    user_info = _user_info_from_payload(payload)
    if user_info is None:
        _logger.debug("【get_optional_user_from_header】Token声明不完整或角色无效")
        return None
    _logger.debug(
        "【get_optional_user_from_header】从Token解析用户信息: user_id=%s, role=%s",
        user_info["user_id"],
        user_info["role"],
    )
    return user_info


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """获取当前已认证的用户信息。

    Args:
        credentials: HTTP Bearer 凭证依赖。

    Returns:
        用户信息字典，包含 user_id 和 role。

    Raises:
        HTTPException: 未提供凭证或凭证无效时抛出 401 错误。
    """
    if not credentials:
        _logger.warning("【get_current_user】认证失败: 未提供凭证")
        raise HTTPException(status_code=401, detail="请先登录")

    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        _logger.warning("【get_current_user】认证失败: Token无效或已过期")
        raise HTTPException(status_code=401, detail="无效或过期的 Token")

    user_info = _user_info_from_payload(payload)
    if user_info is None:
        _logger.warning("【get_current_user】认证失败: Token声明不完整或角色无效")
        raise HTTPException(status_code=401, detail="无效的 Token 声明")
    _logger.info("【get_current_user】用户认证成功: user_id=%s, role=%s", user_info["user_id"], user_info["role"])
    return user_info


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> Optional[dict]:
    """获取可选的用户信息，未认证时返回 None。

    Args:
        credentials: HTTP Bearer 凭证依赖。

    Returns:
        用户信息字典，未认证时返回 None。
    """
    if not credentials:
        return None

    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        return None
    return _user_info_from_payload(payload)


class RoleChecker:
    """角色权限检查器。

    用于检查当前用户是否具有访问特定端点所需的角色权限。
    """

    def __init__(self, allowed_roles: List[str]):
        """初始化角色检查器。

        Args:
            allowed_roles: 允许访问的角色列表。
        """
        self.allowed_roles = allowed_roles
        _logger.debug("【__init__】RoleChecker初始化: allowed_roles=%s", allowed_roles)

    async def __call__(self, user: dict = Depends(get_current_user)) -> dict:
        """检查用户角色是否具有访问权限。

        Args:
            user: 当前用户信息字典，由 Depends(get_current_user) 自动注入。

        Returns:
            用户信息字典，权限验证通过后返回。

        Raises:
            HTTPException: 权限不足时抛出 403 错误。
        """
        user_role = user.get("role")
        if user_role not in self.allowed_roles:
            _logger.warning(
                "【__call__】权限不足: user_id=%s, user_role=%s, required_roles=%s",
                user["user_id"],
                user_role,
                self.allowed_roles,
            )
            raise HTTPException(status_code=403, detail=f"权限不足，需要角色: {', '.join(self.allowed_roles)}")

        _logger.info(
            "【__call__】权限验证通过: user_id=%s, role=%s, endpoint_roles=%s",
            user["user_id"],
            user_role,
            self.allowed_roles,
        )
        return user


def require_roles(allowed_roles: List[str]) -> RoleChecker:
    """创建具有指定角色要求的检查器。

    Args:
        allowed_roles: 允许访问的角色列表。

    Returns:
        RoleChecker 实例。
    """
    return RoleChecker(allowed_roles)


_admin_checker = RoleChecker(["admin"])
_lawyer_checker = RoleChecker(["admin", "lawyer"])


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """校验当前用户是否具有管理员角色。"""
    return await _admin_checker(user)


async def require_lawyer(user: dict = Depends(get_current_user)) -> dict:
    """校验当前用户是否具有律师或管理员角色。"""
    return await _lawyer_checker(user)


async def get_user_from_request(request: Request) -> Optional[dict]:
    """从请求状态中获取用户信息。

    Args:
        request: FastAPI 请求对象。

    Returns:
        用户信息字典，不存在返回 None。
    """
    return getattr(request.state, "user", None)


async def attach_user_to_request(request: Request, call_next):
    """中间件：尝试解析请求中的用户信息并附加到请求状态。

    Args:
        request: FastAPI 请求对象。
        call_next: 下一个中间件处理器。

    Returns:
        响应对象。
    """
    user = await get_optional_user_from_header(request)
    if user:
        _logger.debug(
            "【attach_user_to_request】中间件解析用户: user_id=%s, role=%s, path=%s",
            user["user_id"],
            user["role"],
            request.url.path,
        )
        request.state.user = user
    else:
        _logger.debug("【attach_user_to_request】中间件未解析到用户: path=%s", request.url.path)

    response = await call_next(request)
    return response


ADMIN_ROLES = ["admin"]
LAWYER_ROLES = ["lawyer"]
CLIENT_ROLES = ["client"]
ADMIN_LAWYER_ROLES = ["admin", "lawyer"]
