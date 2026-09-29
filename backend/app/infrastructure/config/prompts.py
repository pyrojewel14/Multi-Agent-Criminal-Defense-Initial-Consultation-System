from pathlib import Path

import yaml

from app.infrastructure.logging import get_logger

_logger = get_logger("Utils.PromptLoader")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_CONFIG_PATH = PROJECT_ROOT / "app" / "config" / "prompt.yaml"


class PromptLoader:
    """从文本文件加载各 Agent 的 System Prompts。

    Prompts 保存在独立的 .txt 文件中（符合编码规范），
    由 config/prompt.yaml 中的 YAML 注册表索引。
    """

    def __init__(self):
        """初始化提示词加载器。"""
        self._logger = get_logger("Utils.PromptLoader")
        self._prompt_map: dict[str, str] = {}

        if not PROMPT_CONFIG_PATH.exists():
            self._logger.error("【__init__】Prompt 配置文件不存在: %s", PROMPT_CONFIG_PATH)
            return

        try:
            with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
                self._prompt_map = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            self._logger.error("【__init__】加载 prompt 配置失败 %s: %s", PROMPT_CONFIG_PATH, e)
            self._prompt_map = {}

        self._logger.debug("【__init__】已加载 prompt 映射: %d 条", len(self._prompt_map))

    def load(self, name: str) -> str:
        """加载并返回指定名称的提示词文本。

        Args:
            name: prompt.yaml 注册表中的键（如 'receptionist_prompt'）。

        Returns:
            对应 .txt 文件的完整文本内容。

        Raises:
            KeyError: 名称未在 prompt.yaml 中注册。
            FileNotFoundError: 引用的 .txt 文件不存在。
            OSError: 文件无法读取。
        """
        if name not in self._prompt_map:
            self._logger.error("【load】Prompt '%s' 在配置中未找到", name)
            raise KeyError(f"Prompt '{name}' not registered in {PROMPT_CONFIG_PATH}")

        relative_path = self._prompt_map[name]
        full_path = PROJECT_ROOT / relative_path

        if not full_path.exists():
            self._logger.error("【load】Prompt 文件缺失: %s (来自键 '%s')", full_path, name)
            raise FileNotFoundError(f"Prompt file not found: {full_path} (key: {name})")

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError as e:
            self._logger.error("【load】读取 prompt '%s' 失败: %s", name, e)
            raise

        self._logger.debug("【load】已加载 prompt '%s' from %s", name, relative_path)
        return content

    def get_map(self) -> dict[str, str]:
        """返回提示词名称到相对路径映射的副本。

        Returns:
            提示词映射字典。
        """
        return dict(self._prompt_map)


prompt_loader = PromptLoader()
