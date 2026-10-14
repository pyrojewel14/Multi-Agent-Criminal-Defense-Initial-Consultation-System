"""统一管理召回池和最终融合，为领域核验及可选模型阶段保留注入边界。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from app.infrastructure.observability.tracing import SessionBudgetExceeded

from .config import RetrievalConfig
from .fusion import choose_stable_representative, document_key, rrf_fuse
from .models import ChannelResult, RecallPool, RetrievalCandidate, RetrievalRequest, SearchResult, StageStatus
from .protocols import QueryExpander, Reranker, Retriever


class RetrievalPipeline:
    """同一流程支持向量、混合召回和显式未配置向量的领域候选池。"""

    def __init__(
        self, vector: Retriever | None, *, bm25: Retriever | None = None,
        expander: QueryExpander | None = None, reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
        key: Callable[[RetrievalCandidate], str] = document_key,
        choose_representative: Callable[[Sequence[RetrievalCandidate]], RetrievalCandidate] = choose_stable_representative,
    ):
        self.vector = vector
        self.bm25 = bm25
        self.expander = expander
        self.reranker = reranker
        self.config = config if config is not None else RetrievalConfig()
        self.key = key
        self.choose_representative = choose_representative

    @staticmethod
    def _guard(request: RetrievalRequest) -> str | None:
        """空查询及无授权范围在任何检索和最终融合前结束。"""
        if not request.scope.user_id and not request.scope.include_public:
            return "skipped_unauthorized"
        if not request.query.strip():
            return "skipped_empty"
        return None

    @staticmethod
    def _record(status: dict[str, Any], name: str, stage: StageStatus) -> None:
        """对外状态保持扁平字符串与错误类别，通道内部保留类型化状态。"""
        status[name] = stage.state
        if stage.error_type:
            status[f"{name}_error"] = stage.error_type
        status["degraded"] = (
            status.get("degraded", False) or stage.degraded
            or stage.state in {"failed", "timeout", "busy"}
        )

    async def _retrieve(self, name: str, retriever: Retriever | None,
                        request: RetrievalRequest) -> ChannelResult:
        """保留通道异常类别；外层取消不转换成降级结果。"""
        if retriever is None:
            return ChannelResult(name, (), {}, StageStatus("unconfigured"))
        try:
            result = await retriever.retrieve(request)
            if result.name != name:
                raise ValueError("retriever channel name does not match pipeline stage")
            return result
        except SessionBudgetExceeded:
            raise
        except TimeoutError as exc:
            return ChannelResult(name, (), {}, StageStatus("timeout", type(exc).__name__, True))
        except Exception as exc:
            return ChannelResult(name, (), {}, StageStatus("failed", type(exc).__name__, True))

    async def recall(
        self, request: RetrievalRequest, *, extra_channels: Mapping[str, ChannelResult] | None = None,
    ) -> RecallPool:
        """收集完整召回池，允许领域服务在最终排序前核验和补充候选。"""
        guard = self._guard(request)
        if guard:
            return RecallPool({}, {name: guard for name in ("vector", "bm25", "hyde", "rerank")}, False)
        channels = dict(extra_channels or {})
        reserved = {"vector", "bm25", "hyde", "rerank", "degraded", "recall_counts",
                    "mode", "method", "candidate_count"}
        # 阶段状态、阶段元数据和自动生成的错误字段不能由补充通道覆盖。
        if any(name in reserved or name.endswith("_error")
               or name.startswith(("vector_", "bm25_", "hyde_", "rerank_")) for name in channels):
            raise ValueError("extra channel conflicts with a pipeline stage or status field")
        if any(name != channel.name for name, channel in channels.items()):
            raise ValueError("extra channel name does not match its mapping key")
        status: dict[str, Any] = {"degraded": False}
        vector_request = replace(request, recall_k=request.top_k) if self.config.mode == "vector" else request
        channels["vector"] = await self._retrieve("vector", self.vector, vector_request)
        channels["bm25"] = (
            ChannelResult("bm25", (), {}, StageStatus("disabled")) if self.config.mode == "vector"
            else await self._retrieve("bm25", self.bm25, request)
        )
        for name, channel in channels.items():
            self._record(status, name, channel.status)
        # 可选模型阶段由后续实现接入；当前不虚报已执行。
        status["hyde"] = "unconfigured" if self.config.hyde else "disabled"
        status["rerank"] = "unconfigured" if self.config.rerank else "disabled"
        status["recall_counts"] = {name: len(channel.candidates) for name, channel in channels.items()}
        failed = channels["vector"].status.state in {"failed", "timeout", "busy"}
        return RecallPool(channels, status, failed)

    async def finalize(self, request: RetrievalRequest, pool: RecallPool) -> SearchResult:
        """融合完整候选池，仅在最终出口按请求数量截断并赋排名。"""
        if self._guard(request):
            return SearchResult((), pool.retrieval_status, pool.dependency_failed)
        candidates = rrf_fuse(pool.channels, key=self.key, choose_representative=self.choose_representative)
        final = tuple(replace(candidate, rank=rank)
                      for rank, candidate in enumerate(candidates[:request.top_k], 1))
        return SearchResult(final, pool.retrieval_status, pool.dependency_failed)

    async def search(self, request: RetrievalRequest) -> SearchResult:
        """顺序执行共享召回与最终排序，不另外维护搜索算法。"""
        return await self.finalize(request, await self.recall(request))
