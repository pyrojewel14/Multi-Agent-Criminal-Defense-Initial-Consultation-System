from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.infrastructure.logging import get_logger

_logger = get_logger("Security.JWTConfig")


class JWTConfigError(Exception):
    """JWT 配置异常。"""

    pass


class JWTConfig(BaseSettings):
    """JWT 配置类，使用 pydantic-settings 管理配置。"""

    model_config = SettingsConfigDict(
        env_prefix="JWT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    secret_key: str = Field(default="CHANGE_ME_IN_PRODUCTION", description="JWT 签名密钥")

    algorithm: str = Field(default="HS256", description="JWT 签名算法")

    access_token_expire_minutes: int = Field(default=15, ge=1, le=1440, description="访问令牌有效期（分钟）")

    refresh_token_expire_days: int = Field(default=7, ge=1, le=90, description="刷新令牌有效期（天）")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """验证 JWT 密钥。

        Args:
            v: 待验证的密钥值。

        Returns:
            验证通过返回密钥值。

        Raises:
            无直接异常，通过日志输出警告。
        """
        if v == "CHANGE_ME_IN_PRODUCTION":
            _logger.warning(
                "[SECURITY WARNING] 使用默认 JWT 密钥在生产环境是危险的！"
                "请设置 JWT_SECRET_KEY 环境变量，使用安全的随机字符串。"
                '生成方法: python -c "import secrets; print(secrets.token_hex(32))"'
            )
        elif len(v) < 32:
            _logger.warning(
                "[SECURITY WARNING] JWT 密钥长度不足（< 32 字符）。这是不安全的，请使用更长的密钥（至少 32 个字符）。"
            )
        return v

    def get_secret_key(self) -> str:
        """获取 JWT 密钥。

        Returns:
            JWT 签名密钥。
        """
        return self.secret_key

    def get_algorithm(self) -> str:
        """获取 JWT 算法。

        Returns:
            JWT 签名算法。
        """
        return self.algorithm

    def get_access_token_expire_minutes(self) -> int:
        """获取访问令牌过期时间。

        Returns:
            访问令牌有效期（分钟）。
        """
        return self.access_token_expire_minutes

    def get_refresh_token_expire_days(self) -> int:
        """获取刷新令牌过期时间。

        Returns:
            刷新令牌有效期（天）。
        """
        return self.refresh_token_expire_days


jwt_config: Optional[JWTConfig] = None


def get_jwt_config() -> JWTConfig:
    """获取 JWT 配置的全局单例。

    Returns:
        JWT 配置实例。

    Raises:
        JWTConfigError: 配置加载失败时抛出。
    """
    global jwt_config

    if jwt_config is None:
        try:
            jwt_config = JWTConfig()
            _logger.info("【get_jwt_config】JWT 配置加载成功")
        except Exception as e:
            _logger.error("【get_jwt_config】JWT 配置加载失败: %s", e)
            raise JWTConfigError(f"JWT 配置加载失败: {e}") from e

    return jwt_config


def reload_jwt_config() -> JWTConfig:
    """重新加载 JWT 配置。

    Returns:
        重新加载的配置实例。
    """
    global jwt_config
    jwt_config = JWTConfig()
    _logger.info("【reload_jwt_config】JWT 配置已重新加载")
    return jwt_config
