"""协调法律身份、索引边界和领域核验，最终排序只交给统一 Pipeline。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace

from app.infrastructure.observability.tracing import (
    SessionBudgetExceeded,
    current_trace_context,
    session_budget,
    trace_span,
    trace_store,
)
from app.knowledge.retrieval.errors import safe_error_type
from app.knowledge.retrieval.models import (
    ChannelResult,
    RecallPool,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalScope,
    SearchResult,
    StageStatus,
)
from app.knowledge.retrieval.pipeline import RetrievalPipeline

from .identifiers import extract_article_number, normalize_article_number
from .keyword import LawKeywordRetriever
from .models import LawSearchResult, plain
from .repository import LawRepository
from .verification import LawVerifier


class LawSearchService:
    """一次请求核验完整候选池，再统一融合、重排和最终 Top 5。"""

    def __init__(
        self,
        repository: LawRepository,
        pipeline: RetrievalPipeline,
        *,
        preflight: Callable[[], Awaitable[RetrievalPipeline | None]] | None = None,
        required_index: bool = False,
        rerank_protocol: str | None = None,
        integrity_metadata: Mapping[str, str | None] | None = None,
    ):
        self.repository, self.pipeline = repository, pipeline
        self.preflight, self.required_index = preflight, required_index
        self.rerank_protocol = rerank_protocol
        self.integrity_metadata = dict(integrity_metadata or {})
        self.keyword = LawKeywordRetriever(repository)
        self.verifier = LawVerifier(repository)

    def _view(
        self,
        pipeline: RetrievalPipeline,
        *,
        skip_rag: bool = False,
        choose: Callable[[Sequence[RetrievalCandidate]], RetrievalCandidate] | None = None,
    ) -> RetrievalPipeline:
        """只复制轻量流程视图，共享模型和缓存不保存请求核验或身份状态。"""
        return RetrievalPipeline(
            None if skip_rag else pipeline.vector,
            bm25=None if skip_rag else pipeline.bm25,
            expander=None if skip_rag else pipeline.expander,
            reranker=pipeline.reranker,
            config=replace(pipeline.config, hyde=False) if skip_rag else pipeline.config,
            key=lambda candidate: (
                f"{self.repository.source_id}:{normalize_article_number(candidate.metadata.get('article_number', '')) or candidate.candidate_id}"
            ),
            choose_representative=choose or pipeline.choose_representative,
        )

    async def search(
        self, query: str, *, user_id: str | None, keyword_terms: Sequence[str] | None = None
    ) -> LawSearchResult:
        """保留查询、失败类别和原始来源，缺少身份时仅使用受控资产。"""
        context = current_trace_context()
        session_id = str(context["session_id"]) if context and context.get("session_id") else None
        if (
            query.strip()
            and user_id
            and session_id
            and (self.preflight is not None or self.pipeline.vector is not None)
        ):
            session_budget.reserve_call(session_id)
        with trace_span(
            trace_store, event_type="rag", name="retrieve_document", cache_hit=False, metadata={"query": query}
        ) as event:
            result = await self._search(
                query, user_id=user_id, keyword_terms=tuple(keyword_terms) if keyword_terms is not None else None
            )
            if result.search.dependency_failed:
                event.outcome = "degraded"
            with trace_span(trace_store, event_type="rag_ranked_result_set", name="retrieve_documents"):
                pass
            for rank, candidate in enumerate(result.search.candidates[:5], 1):
                with trace_span(
                    trace_store,
                    event_type="rag_ranked_result",
                    name="returned_document",
                    attempt=rank,
                    metadata={"content": candidate.content, "origin": candidate.source_id},
                ):
                    pass
            return result

    async def _search(
        self, query: str, *, user_id: str | None, keyword_terms: Sequence[str] | None = None
    ) -> LawSearchResult:
        """依次执行索引预检、条号分流、候选核验和统一排序。"""
        query = query.strip()
        request = RetrievalRequest(
            query, RetrievalScope(self.repository.source_id, user_id, True), self.pipeline.config.recall_k, 5
        )
        if not query:
            return LawSearchResult(
                SearchResult(
                    (), {name: "skipped_empty" for name in ("vector", "bm25", "hyde", "rerank", "keyword")}, False
                ),
                {},
            )

        # 预检失败后仍可返回受控 JSON 候选，但必须保留依赖失败及降级状态。
        pipeline, integrity_error = self.pipeline, None
        if user_id and self.preflight is not None:
            try:
                prepared = await self.preflight()
                pipeline = prepared if prepared is not None else pipeline
            except SessionBudgetExceeded:
                raise
            except Exception as exc:
                integrity_error = safe_error_type(exc)
        elif user_id and self.required_index and pipeline.vector is None:
            integrity_error = "ValueError"

        # 精确条号直接读取受控资产；已发生的索引预检错误仍随结果返回。
        exact = normalize_article_number(extract_article_number(query))
        if exact:
            status = {name: "skipped_exact" for name in ("vector", "bm25", "hyde", "rerank", "keyword")}
            status.update(method="article_id_index", mode=pipeline.config.mode, degraded=bool(integrity_error))
            if self.preflight is not None and not integrity_error and user_id:
                status.update(integrity_verified=True, **self.integrity_metadata)
            if integrity_error:
                status.update(integrity_error=integrity_error)
            article = self.repository.get(exact)
            candidates = (self.keyword.candidate(article, exact=True),) if article is not None else ()
            verified = self.verifier.verify_pool(
                RecallPool(
                    {"exact": ChannelResult("exact", candidates, {}, StageStatus("executed"))},
                    status,
                    bool(integrity_error),
                )
            )
            final = tuple(
                replace(candidate, rank=rank)
                for rank, candidate in enumerate(verified.pool.channels["exact"].candidates, 1)
            )
            return LawSearchResult(SearchResult(final, status, bool(integrity_error)), verified.verifications)

        # 自然语言查询先收集全部通道；缺少身份或预检失败时关闭 RAG 通道。
        skip_rag = not user_id or integrity_error is not None
        view = self._view(pipeline, skip_rag=skip_rag)
        keyword = self.keyword.recall(query, request.recall_k, terms=keyword_terms)
        pool = await view.recall(request, extra_channels={"keyword": keyword})
        status = plain(pool.retrieval_status)
        status.update(
            mode=pipeline.config.mode, method="hybrid_index" if pipeline.config.mode == "hybrid" else "vector_index"
        )
        if self.preflight is not None and not integrity_error and user_id:
            status.update(integrity_verified=True, **self.integrity_metadata)
        channels = dict(pool.channels)
        if skip_rag:
            state = "failed" if integrity_error else "skipped_unauthorized"
            for name in ("vector", "bm25", "hyde"):
                status[name] = state
                channels[name] = ChannelResult(name, (), {}, StageStatus(state, integrity_error, bool(integrity_error)))
        if integrity_error:
            status.update(integrity_error=integrity_error, degraded=True)
        pool = RecallPool(channels, status, pool.dependency_failed or integrity_error is not None)

        # 在去重与截断前逐个核验，保留同一条文在不同来源中的独立资格。
        verified = self.verifier.verify_pool(pool)

        def choose(candidates: Sequence[RetrievalCandidate]) -> RetrievalCandidate:
            """可信代表仅影响正文和资格，不改变通道排名的 RRF 贡献。"""
            return min(
                candidates,
                key=lambda candidate: (
                    verified.verifications[candidate.candidate_id].source == "rag_unverified",
                    not verified.verifications[candidate.candidate_id].coverage_eligible,
                    candidate.source_id,
                    candidate.candidate_id,
                ),
            )

        # 代表选择只决定正文和资格；相关性仍由统一融合、重排及 Top K 决定。
        final_view = self._view(pipeline, skip_rag=skip_rag, choose=choose)
        final = await final_view.finalize(request, verified.pool)
        if self.rerank_protocol and final.retrieval_status.get("rerank") == "executed":
            final = replace(final, retrieval_status={**final.retrieval_status, "rerank_protocol": self.rerank_protocol})
        return LawSearchResult(final, verified.verifications)
