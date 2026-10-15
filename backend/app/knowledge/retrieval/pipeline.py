"""统一管理召回池和最终融合，为领域核验及可选模型阶段保留注入边界。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from app.infrastructure.observability.tracing import SessionBudgetExceeded, trace_span, trace_store

from .config import RetrievalConfig
from .errors import safe_error_type
from .fusion import choose_stable_representative, document_key, rrf_fuse
from .models import ChannelResult, RecallPool, RetrievalCandidate, RetrievalRequest, SearchResult, StageStatus
from .protocols import QueryExpander, Reranker, Retriever
from .rerankers import RerankerBusyError


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
            return ChannelResult(name, (), {}, StageStatus("timeout", safe_error_type(exc), True))
        except Exception as exc:
            return ChannelResult(name, (), {}, StageStatus("failed", safe_error_type(exc), True))

    async def _hyde(self, request: RetrievalRequest) -> tuple[ChannelResult, int]:
        """整个扩展链共享 deadline，失败只丢弃额外向量通道。"""
        if self.config.mode != "hybrid" or not self.config.hyde:
            return ChannelResult("hyde", (), {}, StageStatus("disabled")), 0
        if self.expander is None or self.vector is None:
            return ChannelResult("hyde", (), {}, StageStatus("unconfigured")), 0
        if self.config.hyde_budget == 0:
            return ChannelResult("hyde", (), {}, StageStatus("budget_exhausted", degraded=True)), 0

        async def expand_and_retrieve() -> ChannelResult:
            text = await self.expander.expand(request.query, timeout_s=self.config.hyde_timeout)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty HyDE output")
            result = await self._retrieve("vector", self.vector,
                                          replace(request, query=text, recall_k=self.config.hyde_k))
            if result.status.state in {"failed", "timeout", "busy", "budget_exhausted"}:
                return ChannelResult("hyde", (), {}, replace(result.status, degraded=True))
            # 复用同一向量实现时，将候选历史中的通道身份一起改写，防止距离串线。
            candidates = tuple(replace(
                candidate,
                recall_ranks={"hyde" if name == "vector" else name: value
                              for name, value in candidate.recall_ranks.items()},
                recall_scores={"hyde" if name == "vector" else name: value
                               for name, value in candidate.recall_scores.items()},
                origins=tuple(replace(origin, channel="hyde") if origin.channel == "vector" else origin
                              for origin in candidate.origins),
            ) for candidate in result.candidates)
            return ChannelResult("hyde", candidates, result.scores, result.status)

        try:
            return await asyncio.wait_for(expand_and_retrieve(), timeout=self.config.hyde_timeout), 1
        except SessionBudgetExceeded as exc:
            stage = StageStatus("budget_exhausted", safe_error_type(exc), True)
        except TimeoutError as exc:
            stage = StageStatus("timeout", safe_error_type(exc), True)
        except Exception as exc:
            stage = StageStatus("failed", safe_error_type(exc), True)
        return ChannelResult("hyde", (), {}, stage), 1

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
        channels["hyde"], status["hyde_attempt_count"] = await self._hyde(request)
        for name, channel in channels.items():
            self._record(status, name, channel.status)
        status["rerank"] = "unconfigured" if self.config.rerank else "disabled"
        status["recall_counts"] = {name: len(channel.candidates) for name, channel in channels.items()}
        failed = channels["vector"].status.state in {"failed", "timeout", "busy"}
        return RecallPool(channels, status, failed)

    async def finalize(self, request: RetrievalRequest, pool: RecallPool) -> SearchResult:
        """完整融合后仅重排一次，失败保留融合顺序，最后截断并赋排名。"""
        if self._guard(request):
            return SearchResult((), pool.retrieval_status, pool.dependency_failed)
        candidates = rrf_fuse(pool.channels, key=self.key, choose_representative=self.choose_representative)
        status = dict(pool.retrieval_status)
        status["candidate_count"] = len(candidates)
        if status.get("method") == "article_id_index":
            stage = StageStatus("skipped_exact")
        elif not self.config.rerank:
            stage = StageStatus("disabled")
        elif self.reranker is None:
            stage = StageStatus("unconfigured")
        elif not candidates:
            stage = StageStatus("skipped_empty")
        else:
            try:
                with trace_span(trace_store, event_type="retrieval", name="full_rerank") as event:
                    try:
                        values = await asyncio.wait_for(self.reranker.score(
                            request.query, [candidate.content for candidate in candidates],
                            timeout_s=self.config.rerank_timeout), timeout=self.config.rerank_timeout)
                        if len(values) != len(candidates) or any(
                            not isinstance(value, (int, float)) or not math.isfinite(value)
                            or (getattr(self.reranker, "probability_scores", False) and not 0 <= value <= 1)
                            for value in values
                        ):
                            raise ValueError("invalid reranker scores")
                    except Exception:
                        event.outcome = "degraded"
                        raise
                candidates = tuple(sorted(
                    (replace(candidate, rerank_score=float(score)) for candidate, score in zip(candidates, values)),
                    key=lambda candidate: -candidate.rerank_score,
                ))
                stage = StageStatus("executed")
            except SessionBudgetExceeded:
                raise
            except TimeoutError as exc:
                stage = StageStatus("timeout", safe_error_type(exc), True)
            except RerankerBusyError as exc:
                stage = StageStatus("busy", safe_error_type(exc), True)
            except Exception as exc:
                stage = StageStatus("failed", safe_error_type(exc), True)
        self._record(status, "rerank", stage)
        final = tuple(replace(candidate, rank=rank)
                      for rank, candidate in enumerate(candidates[:request.top_k], 1))
        return SearchResult(final, status, pool.dependency_failed)

    async def search(self, request: RetrievalRequest) -> SearchResult:
        """顺序执行共享召回与最终排序，不另外维护搜索算法。"""
        return await self.finalize(request, await self.recall(request))
