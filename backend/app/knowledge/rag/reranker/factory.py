from typing import Dict, Type

from app.infrastructure.logging import get_logger
from app.knowledge.rag.reranker.base import BaseReranker, RerankerConfig
from app.knowledge.rag.reranker.causal_lm import CausalLMReranker
from app.knowledge.rag.reranker.cross_encoder import CrossEncoderReranker

_logger = get_logger("RAG.RerankerFactory")


class RerankerFactory:
    """重排序器工厂类，负责创建和管理重排序器实例。"""

    _registry: Dict[str, Type[BaseReranker]] = {}

    @classmethod
    def register(cls, name: str, reranker_class: Type[BaseReranker]) -> None:
        """注册重排序器类型。

        Args:
            name: 重排序器名称。
            reranker_class: 重排序器类。
        """
        cls._registry[name] = reranker_class
        _logger.debug("【register】已注册重排序器: %s -> %s", name, reranker_class.__name__)

    @classmethod
    def create(cls, config: RerankerConfig | None = None, reranker_type: str = "causal_lm") -> BaseReranker:
        """创建重排序器实例。

        Args:
            config: 重排序模型配置。
            reranker_type: 重排序器类型。

        Returns:
            重排序器实例。

        Raises:
            ValueError: 当 reranker_type 未注册时抛出。
        """
        if config is None:
            config = RerankerConfig.from_env()

        if reranker_type not in cls._registry:
            _logger.error("【create】未知的重排序器类型: %s", reranker_type)
            raise ValueError(f"未知的重排序器类型: {reranker_type}")

        _logger.info("【create】创建重排序器: type=%s, model=%s", reranker_type, config.model_name)
        return cls._registry[reranker_type](config)


RerankerFactory.register("causal_lm", CausalLMReranker)
RerankerFactory.register("cross_encoder", CrossEncoderReranker)
