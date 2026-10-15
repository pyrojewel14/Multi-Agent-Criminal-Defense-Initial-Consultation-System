"""显式、不可变的模型配置；创建资源后不再读取进程环境。"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field


def _integer(name: str, value: int, lower: int, upper: int | None = None) -> None:
    """拒绝布尔值与隐式截断，仅允许范围内的整数。"""
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"invalid integer setting: {name}")


def _env_integer(env: Mapping[str, str], name: str, default: int) -> int:
    """兼容整数形式的小数表示，但不截断非整数配置。"""
    value = float(env.get(name, str(default)))
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"invalid integer setting: {name}")
    return int(value)


@dataclass(frozen=True)
class ChatConfig:
    """一次聊天模型构造使用的供应商、端点、凭据与采样参数快照。"""

    provider: str = "ALIYUN"
    model_name: str = "qwen3-max"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str | None = field(default=None, repr=False)
    temperature: float | None = None
    streaming: bool = False
    top_p: float | None = None

    @classmethod
    def from_env(
        cls,
        values: Mapping[str, str] | None = None,
        *,
        temperature: float | None = None,
        streaming: bool = False,
        top_p: float | None = None,
    ) -> ChatConfig:
        """复制环境映射并沿用既有默认值；显式映射不回退进程环境。"""
        values = dict(os.environ if values is None else values)
        provider = values.get("LLM_TYPE", "ALIYUN").upper()
        if provider not in {"ALIYUN", "OLLAMA"}:
            provider = "ALIYUN"
        if provider == "OLLAMA":
            return cls(
                provider=provider,
                model_name=values.get("OLLAMA_MODEL_NAME", "qwen3.5:0.8b"),
                base_url=values.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=temperature,
                streaming=streaming,
                top_p=top_p,
            )
        api_key = values.get("ALIYUN_ACCESS_KEY_SECRET")
        return cls(
            provider=provider,
            model_name=values.get("ALIYUN_MODEL_NAME", "qwen3-max"),
            base_url=values.get("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=api_key,
            temperature=temperature,
            streaming=streaming,
            top_p=top_p,
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    """上传和查询共享的向量模型配置，凭据参与身份但不出现在 repr。"""

    provider: str = "OLLAMA"
    model_name: str = "qwen3-embedding:0.6b"
    base_url: str = "http://localhost:11434"
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> EmbeddingConfig:
        """复制环境映射，保留向量模型的供应商与凭据优先级。"""
        values = dict(os.environ if values is None else values)
        provider = values.get("EMBED_MODEL_TYPE", "OLLAMA").upper()
        if provider == "OLLAMA":
            return cls(
                provider=provider,
                model_name=values.get("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b"),
                base_url=values.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
        if provider != "ALIYUN":
            raise ValueError(f"不支持的 EMBED_MODEL_TYPE: {provider}，可选值: OLLAMA, ALIYUN")
        api_key = values.get("DASHSCOPE_API_KEY") or values.get("ALIYUN_ACCESS_KEY_SECRET")
        return cls(
            provider=provider,
            model_name=values.get("ALIYUN_EMBED_MODEL_NAME", "text-embedding-v4"),
            base_url=(
                values.get("DASHSCOPE_BASE_URL")
                or values.get("ALIYUN_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            api_key=api_key,
        )


@dataclass(frozen=True)
class RerankerConfig:
    """保留评分模型与输入协议配置，加载权重由具体评分组件负责。"""

    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    local_path: str = "./data/models/Qwen/Qwen3-Reranker-0.6B"
    cache_dir: str = "./data/models"
    max_length: int = 512
    device: str = "auto"
    positive_token: str = "yes"
    negative_token: str = "no"
    instruction: str | None = None
    prefix_template: str = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix_template: str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    input_template: str = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        _integer("max_length", self.max_length, 1)
        _integer("max_concurrency", self.max_concurrency, 1)
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("invalid reranker device")
        for name in (
            "model_name",
            "local_path",
            "cache_dir",
            "positive_token",
            "negative_token",
            "prefix_template",
            "suffix_template",
            "input_template",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid reranker setting: {name}")
        if self.instruction is not None and not isinstance(self.instruction, str):
            raise ValueError("invalid reranker instruction")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RerankerConfig:
        """将现有模型环境变量映射为配置，不下载模型或分配设备。"""
        values = dict(os.environ if env is None else env)
        return cls(
            model_name=values.get("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B"),
            local_path=values.get("RERANKER_MODEL_PATH", "./data/models/Qwen/Qwen3-Reranker-0.6B"),
            cache_dir=values.get("MODEL_CACHE_DIR", "./data/models"),
            max_length=_env_integer(values, "RERANKER_MAX_LENGTH", 512),
            device=values.get("LAW_FULL_RERANK_DEVICE", "auto"),
            instruction=values.get("RERANKER_INSTRUCTION"),
            max_concurrency=_env_integer(values, "RERANKER_MAX_CONCURRENCY", 1),
        )
