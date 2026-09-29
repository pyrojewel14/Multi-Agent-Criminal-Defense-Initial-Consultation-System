import os

import yaml
from dotenv import load_dotenv

from app.infrastructure.logging import get_logger

load_dotenv()


class ConfigLoader:
    _logger = get_logger("Utils.ConfigLoader")

    @staticmethod
    def load_yaml(config_path: str, encoding: str = "utf-8") -> dict:
        """从 YAML 文件加载配置。

        Args:
            config_path: YAML 配置文件路径
            encoding: 文件编码，默认 utf-8

        Returns:
            解析后的配置字典，空文件返回空字典 {}

        Raises:
            FileNotFoundError: 配置文件不存在
            IsADirectoryError: 路径是目录而非文件
            ValueError: YAML 格式错误
            PermissionError: 无读取权限
        """
        path = os.path.abspath(config_path)

        if not os.path.exists(path):
            ConfigLoader._logger.error("配置文件不存在: %s", path)
            raise FileNotFoundError(f"配置文件不存在: {path}")

        if not os.path.isfile(path):
            ConfigLoader._logger.error("路径不是文件: %s", path)
            raise IsADirectoryError(f"路径不是文件: {path}")

        try:
            with open(path, "r", encoding=encoding) as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            ConfigLoader._logger.error("YAML 格式错误 %s: %s", path, e)
            raise ValueError(f"YAML 格式错误 {path}: {e}")
        except PermissionError:
            ConfigLoader._logger.error("无读取权限: %s", path)
            raise
        except OSError as e:
            ConfigLoader._logger.error("读取文件失败 %s: %s", path, e)
            raise

        if config is None:
            ConfigLoader._logger.warning("配置文件为空: %s", path)
            return {}

        ConfigLoader._logger.debug("成功加载配置: %s", path)
        return config

    @property
    def llm_type(self) -> str:
        """当前激活的 LLM 后端类型。

        Returns:
            'ALIYUN' 或 'OLLAMA'

        Raises:
            ValueError: LLM_TYPE 设置为不支持的值
        """
        llm_type = os.getenv("LLM_TYPE", "ALIYUN").upper()

        if llm_type not in ("ALIYUN", "OLLAMA"):
            ConfigLoader._logger.warning("LLM_TYPE 值不支持: %s，将使用 ALIYUN", llm_type)
            return "ALIYUN"

        return llm_type

    def get_llm_config(self) -> dict:
        """获取 LLM 配置信息。

        Returns:
            {
                "type": "ALIYUN" | "OLLAMA",
                "aliyun": {...},
                "ollama": {...}
            }

        Raises:
            ValueError: 必需的环境变量未设置
        """
        llm_type = self.llm_type

        api_key = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
        if llm_type == "ALIYUN" and not api_key:
            ConfigLoader._logger.error("ALIYUN 模式需要 ALIYUN_ACCESS_KEY_SECRET 环境变量")
            raise ValueError("环境变量 ALIYUN_ACCESS_KEY_SECRET 未设置，无法使用 ALIYUN LLM 后端")

        cfg = {
            "type": llm_type,
            "aliyun": {
                "api_key": api_key,
                "model": os.getenv("ALIYUN_MODEL_NAME", "qwen3-max"),
                "base_url": os.getenv(
                    "ALIYUN_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            },
            "ollama": {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "model": os.getenv("OLLAMA_MODEL_NAME", "qwen3.5:0.8b"),
            },
        }

        active_model = cfg[llm_type.lower()]["model"]
        ConfigLoader._logger.info("LLM 配置已加载: type=%s, model=%s", llm_type, active_model)

        return cfg


config_loader = ConfigLoader()
