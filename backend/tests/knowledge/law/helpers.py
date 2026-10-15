"""为 Agent 回归显式注入隔离召回组件，保持真实领域核验和统一排序。"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from app.knowledge.law.models import LawSearchResult
from app.knowledge.law.repository import LawRepository
from app.knowledge.law.service import LawSearchService
from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.models import ChannelResult, RecallPool, RetrievalCandidate, SearchResult, StageStatus
from app.knowledge.retrieval.pipeline import RetrievalPipeline


@contextmanager
def patch_law_recall(*, return_value=None, side_effect=None, dependency_failed=False, **ignored):
    recalled = AsyncMock(return_value=[] if return_value is None else return_value, side_effect=side_effect)

    class InjectedRetriever:
        async def retrieve(self, request):
            laws = await recalled({"behavior_sequence": [request.query], "consequence": ""}, request.scope.user_id)
            failed = dependency_failed
            candidates = tuple(
                RetrievalCandidate(f"test-rag-{index}", request.scope.source_id, law.get("content", ""), law)
                for index, law in enumerate(laws)
            )
            return ChannelResult(
                "vector", candidates, {}, StageStatus("failed" if failed else "executed", degraded=failed)
            )

    def build(*, law_data, **kwargs):
        repository = LawRepository(law_data)
        return LawSearchService(
            repository, RetrievalPipeline(InjectedRetriever(), config=RetrievalConfig(rerank=False))
        )

    with patch("app.consultation.agents.legal_research.build_law_search_service", build):
        yield recalled


@contextmanager
def patch_law_keyword(*, return_value=None, side_effect=None, **ignored):
    from app.knowledge.law.keyword import LawKeywordRetriever

    called = AsyncMock(return_value=[] if return_value is None else return_value, side_effect=side_effect)

    def recall(retriever, query, k, **kwargs):
        if side_effect is not None:
            raise side_effect
        laws = called.return_value
        candidates = tuple(retriever.candidate(law, score=law.get("relevance_score", 1)) for law in laws)
        return ChannelResult("keyword", candidates, {}, StageStatus("executed"))

    with patch.object(LawKeywordRetriever, "recall", recall):
        yield called


def make_law_service(law_data, *, laws=(), error=None):
    """构建真实法律服务，仅替换外部向量召回边界并返回请求记录。"""
    requests = []

    class Vector:
        async def retrieve(self, request):
            requests.append(request)
            if error is not None:
                raise error
            candidates = tuple(
                RetrievalCandidate(f"row-{i}", request.scope.source_id, law["content"], law)
                for i, law in enumerate(laws)
            )
            return ChannelResult("vector", candidates, {}, StageStatus("executed"))

    return LawSearchService(
        LawRepository(law_data), RetrievalPipeline(Vector(), config=RetrievalConfig(rerank=False))
    ), requests


def verified_laws(laws, article_index):
    """把核验样例转为 typed pool，断言使用正式领域输出。"""
    from app.knowledge.law.verification import LawVerifier

    repository = LawRepository(
        {
            "chapters": [
                {"chapter": article.get("chapter", ""), "articles": [article]} for article in article_index.values()
            ]
        }
    )
    candidates = tuple(
        RetrievalCandidate(
            str(index),
            repository.source_id,
            law.get("content", ""),
            law,
            rank=law.get("rank"),
            fusion_score=law.get("fusion_score"),
            rerank_score=law.get("rerank_score"),
            recall_ranks=law.get("recall_ranks", {}),
            recall_scores=law.get("recall_scores", {}),
        )
        for index, law in enumerate(laws)
    )
    pool = RecallPool({"vector": ChannelResult("vector", candidates, {}, StageStatus("executed"))}, {}, False)
    verified = LawVerifier(repository).verify_pool(pool)
    return LawSearchResult(
        SearchResult(verified.pool.channels["vector"].candidates, {}, False), verified.verifications
    ).as_laws()


async def keyword_laws(facts, law_data):
    """通过正式法律服务检验受控关键词，不注入向量数据源。"""
    from app.knowledge.law.keyword import facts_query, facts_terms

    service = LawSearchService(LawRepository(law_data), RetrievalPipeline(None, config=RetrievalConfig(rerank=False)))
    result = await service.search(facts_query(facts), user_id=None, keyword_terms=facts_terms(facts))
    return result.as_laws()
