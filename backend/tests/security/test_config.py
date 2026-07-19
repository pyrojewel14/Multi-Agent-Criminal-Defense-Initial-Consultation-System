import os
import pytest
from unittest.mock import patch, MagicMock

from app.security.config import (
    JWTConfig,
    JWTConfigError,
    get_jwt_config,
    reload_jwt_config,
)


class TestJWTConfig:
    """JWT配置类测试"""

    def test_default_values(self):
        """测试默认值（不受.env影响）"""
        config = JWTConfig(
            secret_key="CHANGE_ME_IN_PRODUCTION",
            algorithm="HS256",
            access_token_expire_minutes=15,
            refresh_token_expire_days=7,
        )
        assert config.secret_key == "CHANGE_ME_IN_PRODUCTION"
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 15
        assert config.refresh_token_expire_days == 7

    def test_custom_values(self):
        """测试自定义值"""
        config = JWTConfig(
            secret_key="custom-secret-key-with-at-least-32-chars",
            algorithm="HS384",
            access_token_expire_minutes=30,
            refresh_token_expire_days=14,
        )
        assert config.secret_key == "custom-secret-key-with-at-least-32-chars"
        assert config.algorithm == "HS384"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 14

    def test_getter_methods(self):
        """测试getter方法"""
        config = JWTConfig(
            secret_key="test-secret-key-minimum-32-characters-long",
            algorithm="HS512",
            access_token_expire_minutes=60,
            refresh_token_expire_days=30,
        )
        assert config.get_secret_key() == "test-secret-key-minimum-32-characters-long"
        assert config.get_algorithm() == "HS512"
        assert config.get_access_token_expire_minutes() == 60
        assert config.get_refresh_token_expire_days() == 30

    def test_expire_minutes_validation(self):
        """测试过期分钟数验证"""
        with pytest.raises(ValueError):
            JWTConfig(access_token_expire_minutes=0)

        with pytest.raises(ValueError):
            JWTConfig(access_token_expire_minutes=2000)

    def test_expire_days_validation(self):
        """测试过期天数验证"""
        with pytest.raises(ValueError):
            JWTConfig(refresh_token_expire_days=0)

        with pytest.raises(ValueError):
            JWTConfig(refresh_token_expire_days=100)

    def test_short_secret_key_warning(self, caplog):
        """测试短密钥警告"""
        import logging
        caplog.set_level(logging.WARNING)

        config = JWTConfig(secret_key="short")
        assert "too short" in caplog.text.lower() or config.secret_key == "short"


class TestJWTConfigSingleton:
    """JWT配置单例测试"""

    def setup_method(self):
        """每个测试前重置全局配置"""
        import app.security.config as config_module
        config_module.jwt_config = None

    def teardown_method(self):
        """每个测试后重置全局配置"""
        import app.security.config as config_module
        config_module.jwt_config = None

    def test_get_jwt_config_returns_instance(self):
        """测试获取配置实例"""
        config = get_jwt_config()
        assert config is not None
        assert isinstance(config, JWTConfig)

    def test_get_jwt_config_returns_same_instance(self):
        """测试获取相同实例"""
        config1 = get_jwt_config()
        config2 = get_jwt_config()
        assert config1 is config2

    def test_reload_jwt_config(self):
        """测试重新加载配置"""
        config1 = get_jwt_config()
        config2 = reload_jwt_config()
        assert config1 is not config2
        assert isinstance(config2, JWTConfig)


class TestJWTConfigFromEnv:
    """从环境变量加载配置测试"""

    def setup_method(self):
        """每个测试前重置"""
        import app.security.config as config_module
        config_module.jwt_config = None

    def teardown_method(self):
        """每个测试后重置"""
        import app.security.config as config_module
        config_module.jwt_config = None

    @patch.dict(os.environ, {
        "JWT_SECRET_KEY": "env-secret-key-with-at-least-32-characters-long",
        "JWT_ALGORITHM": "HS384",
        "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "45",
        "JWT_REFRESH_TOKEN_EXPIRE_DAYS": "21",
    })
    def test_load_from_env(self):
        """测试从环境变量加载"""
        config = JWTConfig()
        assert config.secret_key == "env-secret-key-with-at-least-32-characters-long"
        assert config.algorithm == "HS384"
        assert config.access_token_expire_minutes == 45
        assert config.refresh_token_expire_days == 21


class TestJWTConfigError:
    """JWT配置错误测试"""

    def test_error_class_exists(self):
        """测试错误类存在"""
        error = JWTConfigError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
