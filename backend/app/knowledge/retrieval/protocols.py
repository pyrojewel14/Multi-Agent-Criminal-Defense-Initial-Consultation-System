"""检索组件的异步能力边界，不负责选择数据源或执行最终排序。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import ChannelResult, RetrievalRequest


class Retriever(Protocol):
    """返回保留执行状态的单通道召回结果。"""

    async def retrieve(self, request: RetrievalRequest) -> ChannelResult:
        """按显式请求范围召回，失败须由结果状态或异常保留。"""
        ...


class QueryExpander(Protocol):
    """为额外召回生成扩展文本，不改写原查询。"""

    async def expand(self, query: str, *, timeout_s: float) -> str:
        """在阶段超时预算内生成文本，失败向调用方传播。"""
        ...


class Reranker(Protocol):
    """仅评分，返回与输入正文顺序一一对应的分数。"""

    async def score(self, query: str, documents: Sequence[str], *, timeout_s: float) -> Sequence[float]:
        """保留原查询并执行评分，不在组件内重排或截断候选集合。"""
        ...
