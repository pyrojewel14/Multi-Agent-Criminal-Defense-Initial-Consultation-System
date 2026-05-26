import os
from abc import ABC, abstractmethod
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.errors.exceptions import LLMServiceException
from app.utils.config_loader import config_loader
from app.utils.logger import get_logger

load_dotenv()

_logger = get_logger("Factory")


class DashScopeEmbeddingsWrapper(Embeddings):
    """阿里云 DashScope 嵌入模型封装。

    支持文本嵌入功能，将输入文本转换为向量表示。
    使用 OpenAI 兼容接口调用阿里云百炼 embedding 服务。
    """

    def __init__(self, model_name: str = "text-embedding-v4", api_key: str = None):
        """初始化 DashScope 嵌入模型。

        Args:
            model_name: 模型名称，默认为 text-embedding-v4。
            api_key: API 密钥，如不提供则从环境变量获取。
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要安装 openai 库: pip install openai")

        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_ACCESS_KEY_SECRET")

        if not self.api_key:
            raise ValueError("未设置 API Key，请设置 DASHSCOPE_API_KEY 或 ALIYUN_ACCESS_KEY_SECRET 环境变量")

        base_url = (
            os.getenv("DASHSCOPE_BASE_URL")
            or os.getenv("ALIYUN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def close(self):
        """关闭 OpenAI 客户端，释放连接池资源。"""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
            _logger.info("【close】DashScopeEmbeddingsWrapper 客户端已关闭")

    def __del__(self):
        """析构时自动关闭客户端。"""
        try:
            self.close()
        except Exception:
            pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文本转换为嵌入向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            嵌入向量列表，每个向量为 float 列表。
        """
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(model=self.model_name, input=batch)
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                _logger.error("【embed_documents】DashScope embedding 批量错误 (batch %d-%d): %s", i, i + len(batch), e)
                raise LLMServiceException(detail=f"DashScope embedding 批量错误: {e}") from e

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为嵌入向量。

        Args:
            text: 待嵌入的查询文本。

        Returns:
            嵌入向量，为 float 列表。
        """
        try:
            response = self.client.embeddings.create(model=self.model_name, input=text)
            return response.data[0].embedding
        except Exception as e:
            _logger.error("【embed_query】DashScope embedding 错误: %s", e)
            raise LLMServiceException(detail=f"DashScope embedding 错误: {e}") from e


class BaseModelFactory(ABC):
    """基础模型工厂抽象类。

    提供模型创建的标准接口和配置获取方法。
    """

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """生成模型实例。"""

    def _get_llm_config(self) -> dict:
        """从配置加载器获取 LLM 配置。

        Returns:
            LLM 配置字典。
        """
        return config_loader.get_llm_config()


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂。

    支持阿里云百炼和 Ollama 两种后端。
    提供模型缓存，同一配置的模型实例会被复用。

    统一接口：
    - create_model() - 创建模型实例（支持所有参数）
    - create_streaming_model() - 快捷方法，创建流式响应模型
    - create_precise_model() - 快捷方法，创建精确响应模型

    注意：为保持向后兼容，generator() 方法保留但标记为废弃。
    """

    def __init__(self):
        """初始化聊天模型工厂。"""
        super().__init__()
        self._model_cache = {}  # 模型实例缓存

    def generator(self, streaming: bool = True, top_p: float = 0.7) -> Optional[BaseChatModel]:
        """创建聊天模型实例（带 streaming 配置）- 已废弃，请使用 create_streaming_model()。

        Args:
            streaming: 是否启用流式输出。
            top_p: 采样参数。

        Returns:
            聊天模型实例。
        """
        import warnings

        warnings.warn(
            "generator() 已废弃，请使用 create_streaming_model() 或 create_model()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_streaming_model(top_p=top_p)

    def create_model(
        self,
        temperature: Optional[float] = None,
        streaming: bool = False,
        top_p: Optional[float] = None,
    ) -> BaseChatModel:
        """创建聊天模型实例（统一接口）。

        支持自定义 temperature、streaming 和 top_p 参数。
        相同配置的模型实例会被缓存复用。

        Args:
            temperature: 采样温度 (0.0-1.0)，与 top_p 二选一。
            streaming: 是否启用流式输出。
            top_p: Top-P 采样参数，与 temperature 二选一。

        Returns:
            配置好的聊天模型实例。
        """
        return self._create_model(temperature=temperature, streaming=streaming, top_p=top_p)

    def create_streaming_model(self, top_p: float = 0.7) -> BaseChatModel:
        """创建流式聊天模型实例（快捷方法）。

        相当于 create_model(streaming=True, top_p=top_p)

        Args:
            top_p: Top-P 采样参数，默认 0.7。

        Returns:
            流式聊天模型实例。
        """
        return self.create_model(streaming=True, top_p=top_p)

    def create_precise_model(self, temperature: float = 0.1) -> BaseChatModel:
        """创建精确聊天模型实例（快捷方法）。

        相当于 create_model(streaming=False, temperature=temperature)

        Args:
            temperature: 采样温度，默认 0.1。

        Returns:
            精确聊天模型实例。
        """
        return self.create_model(streaming=False, temperature=temperature)

    def clear_cache(self):
        """清空模型缓存，释放已缓存的模型实例。"""
        self._model_cache.clear()
        _logger.info("【clear_cache】模型缓存已清空")

    def close(self):
        """关闭工厂，释放所有模型资源。"""
        for cache_key, model in self._model_cache.items():
            # ChatTongyi / ChatOllama 底层可能持有 httpx 客户端
            client = getattr(model, "client", None)
            if client and hasattr(client, "close"):
                try:
                    client.close()
                except Exception as e:
                    _logger.warning("【close】关闭模型客户端失败 (%s): %s", cache_key, e)
        self._model_cache.clear()
        _logger.info("【close】ChatModelFactory 已关闭，所有模型资源已释放")

    def _create_model(
        self,
        temperature: Optional[float] = None,
        streaming: bool = False,
        top_p: Optional[float] = None,
    ) -> BaseChatModel:
        """根据配置创建聊天模型（带缓存）。

        Args:
            temperature: 采样温度。
            streaming: 是否启用流式输出。
            top_p: Top-P 采样参数。

        Returns:
            聊天模型实例。
        """
        # 先获取配置，确保缓存键包含完整的模型标识
        config = self._get_llm_config()
        llm_type = config.get("type", "ALIYUN").upper()

        # 根据后端类型获取对应的模型名称
        if llm_type == "OLLAMA":
            model_name = config.get("ollama", {}).get("model") or os.getenv("OLLAMA_MODEL_NAME", "qwen3:7b")
        else:
            model_name = config.get("aliyun", {}).get("model") or os.getenv("ALIYUN_MODEL_NAME", "qwen3-max")

        # 生成包含完整模型标识的缓存键
        cache_key = f"{llm_type}:{model_name}:{streaming}:{temperature}:{top_p}"

        # 检查缓存
        if cache_key in self._model_cache:
            _logger.debug("【_create_model】从缓存返回模型: %s", cache_key)
            return self._model_cache[cache_key]

        # 创建新实例
        if llm_type == "OLLAMA":
            model = self._create_ollama_model(config, model_name, temperature, streaming, top_p)
        else:
            model = self._create_aliyun_model(config, model_name, temperature, streaming, top_p)

        # 缓存模型实例
        self._model_cache[cache_key] = model
        _logger.info("【_create_model】创建并缓存新模型: %s", cache_key)

        return model

    def _create_ollama_model(
        self,
        config: dict,
        model_name: str,
        temperature: Optional[float],
        streaming: bool,
        top_p: Optional[float],
    ) -> BaseChatModel:
        """创建 Ollama 聊天模型。

        Args:
            config: 模型配置字典。
            model_name: 模型名称。
            temperature: 采样温度。
            streaming: 是否启用流式输出。
            top_p: 采样参数。

        Returns:
            ChatOllama 实例。
        """
        from langchain_ollama import ChatOllama

        base_url = config.get("ollama", {}).get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        _logger.info("【_create_ollama_model】ChatModel 使用 Ollama: model=%s, base_url=%s", model_name, base_url)

        params = {"model": model_name, "base_url": base_url}
        if temperature is not None:
            params["temperature"] = temperature
        if streaming:
            params["streaming"] = streaming
        if top_p is not None:
            params["top_p"] = top_p

        return ChatOllama(**params)

    def _create_aliyun_model(
        self,
        config: dict,
        model_name: str,
        temperature: Optional[float],
        streaming: bool,
        top_p: Optional[float],
    ) -> BaseChatModel:
        """创建阿里云百炼聊天模型。

        Args:
            config: 模型配置字典。
            model_name: 模型名称。
            temperature: 采样温度。
            streaming: 是否启用流式输出。
            top_p: 采样参数。

        Returns:
            ChatTongyi 实例。
        """
        from langchain_community.chat_models.tongyi import ChatTongyi

        api_key = config.get("aliyun", {}).get("api_key") or os.getenv("ALIYUN_ACCESS_KEY_SECRET")
        base_url = config.get("aliyun", {}).get("base_url") or os.getenv("ALIYUN_BASE_URL")

        _logger.info("【_create_aliyun_model】ChatModel 使用阿里云百炼: model=%s", model_name)

        params = {"model": model_name, "api_key": api_key}
        if base_url:
            params["base_url"] = base_url
        if temperature is not None:
            params["temperature"] = temperature
        if streaming:
            params["streaming"] = streaming
        if top_p is not None:
            params["top_p"] = top_p

        return ChatTongyi(**params)


class EmbedModelFactory(BaseModelFactory):
    """嵌入模型工厂。

    支持 Ollama 和阿里云百炼两种后端。
    提供模型缓存，同一配置的模型实例会被复用。
    """

    def __init__(self):
        """初始化嵌入模型工厂。"""
        super().__init__()
        self._model_cache = {}

    def generator(self) -> Optional[Embeddings]:
        """生成嵌入模型实例 - 已废弃，请使用 create_embedding_model()。"""
        import warnings

        warnings.warn(
            "generator() 已废弃，请使用 create_embedding_model()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_embedding_model()

    def create_embedding_model(self) -> Optional[Embeddings]:
        """根据 EMBED_MODEL_TYPE 生成对应的嵌入模型（带缓存）。

        Returns:
            嵌入模型实例。

        Raises:
            ValueError: 当 EMBED_MODEL_TYPE 不支持时。
        """
        embed_type = os.getenv("EMBED_MODEL_TYPE", "OLLAMA").upper()
        cache_key = embed_type

        if cache_key in self._model_cache:
            _logger.debug("【create_embedding_model】从缓存返回嵌入模型: %s", cache_key)
            return self._model_cache[cache_key]

        if embed_type == "OLLAMA":
            model = self._create_ollama_embeddings()
        elif embed_type == "ALIYUN":
            model = self._create_aliyun_embeddings()
        else:
            raise ValueError(f"不支持的 EMBED_MODEL_TYPE: {embed_type}，可选值: OLLAMA, ALIYUN")

        self._model_cache[cache_key] = model
        return model

    def close(self):
        """关闭工厂，释放所有嵌入模型资源。"""
        for cache_key, model in self._model_cache.items():
            if isinstance(model, DashScopeEmbeddingsWrapper):
                try:
                    model.close()
                except Exception as e:
                    _logger.warning("【close】关闭嵌入模型客户端失败 (%s): %s", cache_key, e)
        self._model_cache.clear()
        _logger.info("【close】EmbedModelFactory 已关闭，所有嵌入模型资源已释放")

    def _create_ollama_embeddings(self) -> Embeddings:
        """创建 Ollama 嵌入模型。

        Returns:
            OllamaEmbeddings 实例。
        """
        from langchain_ollama import OllamaEmbeddings

        model_name = os.getenv("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        _logger.info("【_create_ollama_embeddings】EmbedModel 使用 Ollama: model=%s, base_url=%s", model_name, base_url)

        return OllamaEmbeddings(model=model_name, base_url=base_url)

    def _create_aliyun_embeddings(self) -> DashScopeEmbeddingsWrapper:
        """创建阿里云百炼嵌入模型。

        Returns:
            DashScopeEmbeddingsWrapper 实例。
        """
        model_name = os.getenv("ALIYUN_EMBED_MODEL_NAME", "text-embedding-v4")
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_ACCESS_KEY_SECRET")

        _logger.info("【_create_aliyun_embeddings】EmbedModel 使用阿里云百炼: model=%s", model_name)

        return DashScopeEmbeddingsWrapper(model_name=model_name, api_key=api_key)


chat_model_factory = ChatModelFactory()
embed_model_factory = EmbedModelFactory()

# 延迟初始化，避免导入时创建模型实例导致事件循环问题
_chat_model = None
_embed_model = None


def __getattr__(name):
    global _chat_model, _embed_model
    if name == "chat_model":
        if _chat_model is None:
            _chat_model = chat_model_factory.create_streaming_model()
        return _chat_model
    if name == "embed_model":
        if _embed_model is None:
            _embed_model = embed_model_factory.create_embedding_model()
        return _embed_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
