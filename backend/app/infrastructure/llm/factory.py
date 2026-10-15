"""只按显式配置构造模型；缓存与生命周期由 ModelManager 统一管理。"""

from __future__ import annotations

from typing import List
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.errors.exceptions import LLMServiceException
from app.infrastructure.llm.config import ChatConfig, EmbeddingConfig, RerankerConfig
from app.infrastructure.logging import get_logger

_logger = get_logger("Factory")


def _ollama_client_kwargs(base_url: str) -> dict:
    """仅为回环地址禁用环境代理，远程服务保留环境代理能力。"""
    return {"trust_env": False} if urlparse(base_url).hostname in {"localhost", "127.0.0.1", "::1"} else {}


class DashScopeEmbeddingsWrapper(Embeddings):
    """用 OpenAI 兼容接口执行阿里云向量请求，不读取环境或持有缓存。"""

    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        """按显式端点与凭据创建连接；省略凭据时直接拒绝。"""
        from openai import OpenAI

        if not api_key:
            raise ValueError("未设置 API Key，请设置 DASHSCOPE_API_KEY 或 ALIYUN_ACCESS_KEY_SECRET 环境变量")
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def close(self):
        """成功关闭后移除连接引用，失败时保留以便再次清理。"""
        client = getattr(self, "client", None)
        if client is not None:
            client.close()
            self.client = None

    def __del__(self):
        """析构只尽力清理已经存在的同步连接。"""
        try:
            self.close()
        except Exception:
            pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """按既有十条批次请求向量，供应商错误沿用 typed LLM 异常。"""
        batch_size = 10
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(model=self.model_name, input=batch)
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as exc:
                _logger.error(
                    "【embed_documents】DashScope embedding 批量错误 (batch %d-%d): %s", i, i + len(batch), exc
                )
                raise LLMServiceException(detail=f"DashScope embedding 批量错误: {exc}") from exc
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """获取单条向量，保留原有供应商错误适配行为。"""
        try:
            response = self.client.embeddings.create(model=self.model_name, input=text)
            return response.data[0].embedding
        except Exception as exc:
            _logger.error("【embed_query】DashScope embedding 错误: %s", exc)
            raise LLMServiceException(detail=f"DashScope embedding 错误: {exc}") from exc


class ChatModelFactory:
    """按配置构造聊天模型，不缓存实例或关闭调用方资源。"""

    def create(self, config: ChatConfig) -> BaseChatModel:
        """保留供应商既有传参语义；采样参数 None 与零分别处理。"""
        if config.provider == "OLLAMA":
            from langchain_ollama import ChatOllama

            constructor = ChatOllama
            params = {
                "model": config.model_name,
                "base_url": config.base_url,
                "client_kwargs": _ollama_client_kwargs(config.base_url),
            }
        elif config.provider == "ALIYUN":
            if not config.api_key:
                raise ValueError("未设置 API Key，请设置 ALIYUN_ACCESS_KEY_SECRET 环境变量")
            from langchain_community.chat_models.tongyi import ChatTongyi

            constructor = ChatTongyi
            params = {"model": config.model_name, "api_key": config.api_key}
            if config.base_url:
                params["base_url"] = config.base_url
        else:
            raise ValueError("unsupported chat backend")
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.streaming:
            params["streaming"] = config.streaming
        if config.top_p is not None:
            params["top_p"] = config.top_p
        return constructor(**params)


class EmbedModelFactory:
    """按显式模型与端点创建向量客户端。"""

    def create(self, config: EmbeddingConfig) -> Embeddings:
        """创建独立模型实例，不通过环境回退改变已组装配置。"""
        if config.provider == "OLLAMA":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(
                model=config.model_name, base_url=config.base_url, client_kwargs=_ollama_client_kwargs(config.base_url)
            )
        if config.provider == "ALIYUN":
            return DashScopeEmbeddingsWrapper(
                model_name=config.model_name, api_key=config.api_key, base_url=config.base_url
            )
        raise ValueError("unsupported embedding backend")


class RerankerFactory:
    """构造轻量评分器；权重加载由首次评分触发。"""

    def create(self, config: RerankerConfig, *, kind: str = "causal_lm"):
        """拒绝未知评分器类型，不降级到另一个模型。"""
        from app.infrastructure.llm.rerankers import CrossEncoderReranker, QwenReranker

        if kind == "causal_lm":
            return QwenReranker(config)
        if kind == "cross_encoder":
            return CrossEncoderReranker(config)
        raise ValueError("unknown reranker type")
