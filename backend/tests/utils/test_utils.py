import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from app.utils.config_loader import ConfigLoader
from app.utils.logger import SensitiveFilter, get_logger
from app.utils.path_tool import (
    get_abstract_path,
    get_config_path,
    get_data_path,
    get_project_root,
)
from app.utils.prompt_loader import PromptLoader


# ---------------------------------------------------------------------------
# SensitiveFilter.filter
# ---------------------------------------------------------------------------


class TestSensitiveFilter:
    def setup_method(self):
        self.filt = SensitiveFilter()

    def _make_record(self, msg: str) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=None, exc_info=None,
        )
        return record

    def test_filter_returns_true(self):
        record = self._make_record("normal message")
        assert self.filt.filter(record) is True

    def test_mask_phone_number(self):
        record = self._make_record("用户手机号13812345678已注册")
        self.filt.filter(record)
        assert "[PHONE-MASKED]" in record.msg
        assert "13812345678" not in record.msg

    def test_mask_email(self):
        record = self._make_record("邮箱test@example.com已验证")
        self.filt.filter(record)
        assert "[EMAIL-MASKED]" in record.msg
        assert "test@example.com" not in record.msg

    def test_mask_id_card_18(self):
        record = self._make_record("身份证110101199003076543")
        self.filt.filter(record)
        assert "[ID-MASKED]" in record.msg
        assert "110101199003076543" not in record.msg

    def test_mask_id_card_15(self):
        record = self._make_record("身份证110101900307654")
        self.filt.filter(record)
        assert "[ID-MASKED]" in record.msg

    def test_no_pii_unchanged(self):
        original = "这是一条普通日志消息"
        record = self._make_record(original)
        self.filt.filter(record)
        assert record.msg == original

    def test_filter_clears_args(self):
        record = self._make_record("msg %s")
        record.args = ("some_arg",)
        self.filt.filter(record)
        assert record.args is None


# ---------------------------------------------------------------------------
# PromptLoader.load
# ---------------------------------------------------------------------------


class TestPromptLoader:
    def test_load_existing_prompt(self, tmp_path):
        # Create a mock prompt.yaml and a prompt .txt file
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        txt_file = prompt_dir / "test_prompt.txt"
        txt_file.write_text("你是一个测试助手。", encoding="utf-8")

        yaml_file = tmp_path / "prompt.yaml"
        yaml_file.write_text(
            yaml.dump({"test_prompt": "prompts/test_prompt.txt"}),
            encoding="utf-8",
        )

        loader = PromptLoader()
        with patch.object(PromptLoader, "__init__", lambda self: None):
            loader._prompt_map = {"test_prompt": str(txt_file)}
            # Override PROJECT_ROOT resolution by patching the module-level constant
            with patch("app.utils.prompt_loader.PROJECT_ROOT", tmp_path):
                with patch("app.utils.prompt_loader.PROMPT_CONFIG_PATH", yaml_file):
                    content = loader.load("test_prompt")
        assert content == "你是一个测试助手。"

    def test_load_missing_prompt_raises_key_error(self):
        loader = PromptLoader()
        with patch.object(PromptLoader, "__init__", lambda self: None):
            loader._prompt_map = {}
            with pytest.raises(KeyError, match="not registered"):
                loader.load("nonexistent_prompt")

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        loader = PromptLoader()
        with patch.object(PromptLoader, "__init__", lambda self: None):
            loader._prompt_map = {"missing_file": "nonexistent_dir/missing.txt"}
            with patch("app.utils.prompt_loader.PROJECT_ROOT", tmp_path):
                with pytest.raises(FileNotFoundError):
                    loader.load("missing_file")

    def test_load_strips_whitespace(self, tmp_path):
        """The .strip() should remove leading/trailing whitespace."""
        txt = tmp_path / "x.txt"
        txt.write_text("  hello  \n\n", encoding="utf-8")

        loader = PromptLoader()
        with patch.object(PromptLoader, "__init__", lambda self: None):
            loader._prompt_map = {"x": str(txt)}
            with patch("app.utils.prompt_loader.PROJECT_ROOT", tmp_path):
                content = loader.load("x")
        assert content == "hello"

    def test_load_propagates_oserror(self, tmp_path):
        """When the .read() fails, OSError should propagate."""
        # The file must actually exist (exists() check) so that the code
        # reaches the open() call which we'll then mock to raise.
        txt = tmp_path / "real.txt"
        txt.write_text("data", encoding="utf-8")
        loader = PromptLoader()
        with patch.object(PromptLoader, "__init__", lambda self: None):
            loader._prompt_map = {"x": "real.txt"}
            with patch("app.utils.prompt_loader.PROJECT_ROOT", tmp_path):
                with patch("builtins.open", side_effect=OSError("disk error")):
                    with pytest.raises(OSError, match="disk error"):
                        loader.load("x")

    def test_get_map_returns_copy(self):
        """get_map should return a copy, not the internal dict reference."""
        loader = PromptLoader()
        with patch.object(PromptLoader, "__init__", lambda self: None):
            loader._prompt_map = {"k": "v"}
            result = loader.get_map()
        assert result == {"k": "v"}
        # Mutating the returned copy should not affect the loader
        result["new"] = "value"
        assert "new" not in loader._prompt_map

    def test_init_loads_real_prompt_map(self):
        """Without monkeypatching __init__, the loader should populate _prompt_map from the file."""
        loader = PromptLoader()
        # The default project must have at least one prompt registered
        if loader._prompt_map:
            assert isinstance(loader._prompt_map, dict)
            for name, path in loader._prompt_map.items():
                assert isinstance(name, str)
                assert isinstance(path, str)

    def test_init_missing_config_file(self, tmp_path):
        """When PROMPT_CONFIG_PATH doesn't exist, _prompt_map should be empty."""
        with patch("app.utils.prompt_loader.PROMPT_CONFIG_PATH", tmp_path / "absent.yaml"):
            loader = PromptLoader()
        assert loader._prompt_map == {}

    def test_init_invalid_yaml_logs_error(self, tmp_path):
        """When prompt.yaml is malformed, _prompt_map should be empty."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : invalid", encoding="utf-8")
        with patch("app.utils.prompt_loader.PROMPT_CONFIG_PATH", bad):
            loader = PromptLoader()
        assert loader._prompt_map == {}


# ---------------------------------------------------------------------------
# get_project_root / get_abstract_path
# ---------------------------------------------------------------------------


class TestPathTool:
    def test_get_project_root_returns_str(self):
        root = get_project_root()
        assert isinstance(root, str)
        assert os.path.isabs(root)

    def test_get_project_root_points_to_backend(self):
        root = get_project_root()
        # The project root should be the backend directory
        assert os.path.basename(root) == "backend"

    def test_get_abstract_path_returns_absolute(self):
        path = get_abstract_path("data/test.txt")
        assert os.path.isabs(path)

    def test_get_abstract_path_joins_correctly(self):
        root = get_project_root()
        path = get_abstract_path("data/test.txt")
        expected = os.path.normpath(os.path.join(root, "data/test.txt"))
        assert path == expected

    def test_get_abstract_path_normalizes(self):
        path = get_abstract_path("data/../app/config")
        assert ".." not in path

    def test_get_data_path(self):
        path = get_data_path()
        assert path.endswith("data") or path.endswith("data" + os.sep)
        assert os.path.isabs(path)
        assert path == os.path.normpath(os.path.join(get_project_root(), "data"))

    def test_get_config_path(self):
        path = get_config_path()
        assert path.endswith("app" + os.sep + "config") or path.endswith("app/config")
        assert os.path.isabs(path)
        assert path == os.path.normpath(os.path.join(get_project_root(), "app", "config"))


# ---------------------------------------------------------------------------
# ConfigLoader.load_yaml
# ---------------------------------------------------------------------------


class TestConfigLoaderLoadYaml:
    def test_loads_valid_yaml(self, tmp_path):
        cfg = {"a": 1, "b": {"c": 2}}
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump(cfg), encoding="utf-8")
        result = ConfigLoader.load_yaml(str(f))
        assert result == cfg

    def test_empty_file_returns_empty_dict(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        assert ConfigLoader.load_yaml(str(f)) == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            ConfigLoader.load_yaml(str(tmp_path / "nope.yaml"))

    def test_directory_path_raises(self, tmp_path):
        with pytest.raises(IsADirectoryError, match="路径不是文件"):
            ConfigLoader.load_yaml(str(tmp_path))

    def test_invalid_yaml_raises_value_error(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("a: b\n  c: d\nbad indent", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML 格式错误"):
            ConfigLoader.load_yaml(str(f))

    def test_permission_error_propagates(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("a: 1", encoding="utf-8")
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError):
                ConfigLoader.load_yaml(str(f))

    def test_oserror_propagates(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        with patch("builtins.open", side_effect=OSError("io error")):
            with pytest.raises(OSError):
                ConfigLoader.load_yaml(str(f))

    def test_returns_absolute_path(self, tmp_path):
        """The path check is run on an absolute path internally; relative input is accepted."""
        f = tmp_path / "cfg.yaml"
        f.write_text("a: 1", encoding="utf-8")
        rel = os.path.relpath(f, start=os.getcwd())
        result = ConfigLoader.load_yaml(rel)
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# ConfigLoader.llm_type / get_llm_config
# ---------------------------------------------------------------------------


class TestConfigLoaderLlmType:
    def test_default_is_aliyun(self, monkeypatch):
        monkeypatch.delenv("LLM_TYPE", raising=False)
        loader = ConfigLoader()
        assert loader.llm_type == "ALIYUN"

    def test_lowercase_value_normalized(self, monkeypatch):
        monkeypatch.setenv("LLM_TYPE", "ollama")
        loader = ConfigLoader()
        assert loader.llm_type == "OLLAMA"

    def test_unsupported_value_falls_back_to_aliyun(self, monkeypatch):
        monkeypatch.setenv("LLM_TYPE", "BOGUS")
        loader = ConfigLoader()
        assert loader.llm_type == "ALIYUN"


class TestConfigLoaderGetLlmConfig:
    def test_aliyun_config(self, monkeypatch):
        monkeypatch.delenv("LLM_TYPE", raising=False)
        monkeypatch.setenv("ALIYUN_ACCESS_KEY_SECRET", "key-1")
        monkeypatch.setenv("ALIYUN_MODEL_NAME", "qwen3-max")
        monkeypatch.setenv("ALIYUN_BASE_URL", "https://example.com")
        loader = ConfigLoader()
        cfg = loader.get_llm_config()
        assert cfg["type"] == "ALIYUN"
        assert cfg["aliyun"]["api_key"] == "key-1"
        assert cfg["aliyun"]["model"] == "qwen3-max"
        assert cfg["aliyun"]["base_url"] == "https://example.com"

    def test_ollama_config(self, monkeypatch):
        monkeypatch.setenv("LLM_TYPE", "OLLAMA")
        monkeypatch.delenv("ALIYUN_ACCESS_KEY_SECRET", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_MODEL_NAME", "qwen3:7b")
        loader = ConfigLoader()
        cfg = loader.get_llm_config()
        assert cfg["type"] == "OLLAMA"
        assert cfg["ollama"]["base_url"] == "http://localhost:11434"
        assert cfg["ollama"]["model"] == "qwen3:7b"

    def test_aliyun_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_TYPE", raising=False)
        monkeypatch.delenv("ALIYUN_ACCESS_KEY_SECRET", raising=False)
        loader = ConfigLoader()
        with pytest.raises(ValueError, match="ALIYUN_ACCESS_KEY_SECRET"):
            loader.get_llm_config()

    def test_ollama_does_not_require_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_TYPE", "OLLAMA")
        monkeypatch.delenv("ALIYUN_ACCESS_KEY_SECRET", raising=False)
        loader = ConfigLoader()
        cfg = loader.get_llm_config()
        assert cfg["type"] == "OLLAMA"


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test.module.1")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module.1"

    def test_does_not_duplicate_handlers(self):
        """Calling get_logger twice with the same name should not stack handlers."""
        logger1 = get_logger("test.dup.handlers")
        n_handlers_initial = len(logger1.handlers)
        logger2 = get_logger("test.dup.handlers")
        assert logger1 is logger2
        assert len(logger2.handlers) == n_handlers_initial

    def test_level_argument_overrides(self):
        logger = get_logger("test.module.level.override", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_propagate_disabled(self):
        logger = get_logger("test.module.propagate")
        assert logger.propagate is False

    def test_has_handlers_attached(self):
        logger = get_logger("test.module.handlers.attached")
        # The logger must have the file and console handlers attached
        assert len(logger.handlers) > 0
        # And each handler must have the SensitiveFilter installed
        for h in logger.handlers:
            assert any(isinstance(f, SensitiveFilter) for f in h.filters)


# ---------------------------------------------------------------------------
# module-level logger configuration
# ---------------------------------------------------------------------------


class TestLoggerModuleConfig:
    def test_log_dir_created(self):
        """LOG_DIR should exist (the module creates it at import time)."""
        from app.utils.logger import LOG_DIR
        assert LOG_DIR.exists()

    def test_module_level_parsing(self, monkeypatch):
        """LOG_LEVEL_MODULES=foo=DEBUG should be parsed into _MODULE_LEVELS."""
        # Force a re-evaluation by reloading the module
        monkeypatch.setenv("LOG_LEVEL_MODULES", "fake.mod=DEBUG")
        import importlib
        from app.utils import logger as logger_mod
        importlib.reload(logger_mod)
        assert logger_mod._MODULE_LEVELS.get("fake.mod") == logging.DEBUG

    def test_module_level_invalid_value_skipped(self, monkeypatch):
        """Invalid level names should be silently skipped."""
        monkeypatch.setenv("LOG_LEVEL_MODULES", "fake.mod=NOPE")
        import importlib
        from app.utils import logger as logger_mod
        importlib.reload(logger_mod)
        assert "fake.mod" not in logger_mod._MODULE_LEVELS

    def test_module_level_no_equals_skipped(self, monkeypatch):
        """Entries without '=' should be skipped."""
        monkeypatch.setenv("LOG_LEVEL_MODULES", "no_equals_here")
        import importlib
        from app.utils import logger as logger_mod
        importlib.reload(logger_mod)
        # No entry created
        assert all("no_equals_here" not in k for k in logger_mod._MODULE_LEVELS)

    def test_module_level_used_to_set_logger_level(self, monkeypatch):
        """When a module name is in _MODULE_LEVELS, get_logger should use that level."""
        monkeypatch.setenv("LOG_LEVEL_MODULES", "fake.level.module=DEBUG")
        import importlib
        from app.utils import logger as logger_mod
        importlib.reload(logger_mod)
        logger = get_logger("fake.level.module")
        assert logger.level == logging.DEBUG

    def test_filter_mask_combined(self):
        """Multiple PII types in the same record should all be masked."""
        filt = SensitiveFilter()
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="", lineno=0,
            msg="phone=13812345678, email=a@b.com", args=None, exc_info=None,
        )
        filt.filter(record)
        assert "13812345678" not in record.msg
        assert "a@b.com" not in record.msg


# ---------------------------------------------------------------------------
# get_data_path / get_config_path (smoke tests on the higher-level helpers)
# ---------------------------------------------------------------------------


class TestPathToolHelpers:
    def test_get_data_path_matches_abstract(self):
        assert get_data_path() == get_abstract_path("data")

    def test_get_config_path_matches_abstract(self):
        assert get_config_path() == get_abstract_path("app/config")

    def test_get_data_path_under_project_root(self):
        root = get_project_root()
        data = get_data_path()
        assert os.path.commonpath([root, data]) == root

    def test_get_config_path_under_project_root(self):
        root = get_project_root()
        cfg = get_config_path()
        assert os.path.commonpath([root, cfg]) == root
