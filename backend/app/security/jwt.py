from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt

from app.infrastructure.logging import get_logger
from app.security.config import get_jwt_config

_logger = get_logger("Security.JWT")


def hash_password(password: str) -> str:
    """对密码进行哈希处理。

    Args:
        password: 明文密码。

    Returns:
        哈希处理后的密码字符串。
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码是否正确。

    Args:
        plain_password: 明文密码。
        hashed_password: 哈希后的密码。

    Returns:
        密码匹配返回 True，否则返回 False。
    """
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(user_id: str, role: str) -> str:
    """创建访问令牌。

    Args:
        user_id: 用户 ID。
        role: 用户角色。

    Returns:
        编码后的 JWT 访问令牌。
    """
    config = get_jwt_config()
    expire = datetime.utcnow() + timedelta(minutes=config.get_access_token_expire_minutes())
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, config.get_secret_key(), algorithm=config.get_algorithm())


def create_refresh_token(user_id: str) -> str:
    """创建刷新令牌。

    Args:
        user_id: 用户 ID。

    Returns:
        编码后的 JWT 刷新令牌。
    """
    config = get_jwt_config()
    expire = datetime.utcnow() + timedelta(days=config.get_refresh_token_expire_days())
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, config.get_secret_key(), algorithm=config.get_algorithm())


def decode_token(token: str) -> Optional[dict]:
    """解码并验证 JWT 令牌。

    Args:
        token: JWT 令牌字符串。

    Returns:
        解码后的 payload 字典，验证失败返回 None。
    """
    config = get_jwt_config()
    try:
        payload = jwt.decode(token, config.get_secret_key(), algorithms=[config.get_algorithm()])
        return payload
    except jwt.ExpiredSignatureError:
        _logger.warning("【decode_token】Token 已过期")
        return None
    except jwt.InvalidTokenError as e:
        _logger.warning("【decode_token】无效的 Token: %s", str(e))
        return None


def get_token_expiry(token: str) -> Optional[datetime]:
    """获取令牌过期时间（不解验证签名）。

    Args:
        token: JWT 令牌字符串。

    Returns:
        令牌的过期时间，解析失败返回 None。
    """
    config = get_jwt_config()
    try:
        payload = jwt.decode(
            token, config.get_secret_key(), algorithms=[config.get_algorithm()], options={"verify_exp": False}
        )
        exp = payload.get("exp")
        if exp:
            return datetime.fromtimestamp(exp)
        return None
    except Exception:
        return None
