"""法律服务只在核验后执行一次最终排序。"""

import asyncio
import json
from dataclasses import replace

import pytest

from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.models import ChannelResult, RetrievalCandidate, StageStatus
from app.knowledge.retrieval.pipeline import RetrievalPipeline


class Vector:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    async def retrieve(self, request):
        self.calls.append(request)
        await asyncio.sleep(0)
        return ChannelResult("vector", tuple(self.candidates), {}, StageStatus("executed"))


class Scorer:
    probability_scores = True

    def __init__(self):
        self.calls = []

    async def score(self, query, documents, *, timeout_s):
        self.calls.append((query, tuple(documents)))
        return [1.0 if "盗窃" in body else 0.1 for body in documents]


def service(*, profile="snapshot", vector=None, scorer=None, preflight=None):
    from app.knowledge.law.corpus import load_criminal_law_data
    from app.knowledge.law.repository import LawRepository
    from app.knowledge.law.service import LawSearchService

    repository = LawRepository(load_criminal_law_data(profile=profile))
    config = RetrievalConfig(rerank=scorer is not None)
    pipeline = RetrievalPipeline(vector, reranker=scorer, config=config)
    return LawSearchService(repository, pipeline, preflight=preflight, required_index=profile == "full")


@pytest.mark.asyncio
async def test_snapshot_without_index_is_controlled_keyword_not_upload_library():
    result = await service().search("盗窃", user_id="alice")
    assert result.search.candidates
    assert result.search.retrieval_status["vector"] == "unconfigured"
    assert not result.search.dependency_failed
    assert all(v.source == "json_keyword" and v.coverage_eligible for v in result.verifications.values())
    assert json.dumps(result.as_laws(), ensure_ascii=False)


@pytest.mark.asyncio
async def test_missing_user_skips_bound_rag_but_keeps_keyword():
    vector = Vector([])
    result = await service(vector=vector).search("盗窃", user_id=None)
    assert result.search.candidates and vector.calls == []
    assert result.search.retrieval_status["vector"] == "skipped_unauthorized"
    assert not result.search.dependency_failed


@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected", [("第133条之1", "第133条之一"), ("第133条", "第133条"), ("第199条", None), ("第999条", None), ("第264条 第133条", "第264条")])
async def test_exact_article_never_expands_or_reranks(query, expected):
    vector, scorer = Vector([]), Scorer()
    result = await service(profile="full", vector=vector, scorer=scorer).search(query, user_id=None)
    assert [v.article_id for cid, v in result.verifications.items() if cid in {c.candidate_id for c in result.search.candidates}] == ([expected] if expected else [])
    assert vector.calls == scorer.calls == []
    assert all(result.search.retrieval_status[name] == "skipped_exact" for name in ("vector", "bm25", "hyde", "rerank", "keyword"))


@pytest.mark.asyncio
async def test_failed_full_preflight_retains_keyword_and_dependency_status():
    async def broken():
        raise ValueError("manifest mismatch")

    vector = Vector([])
    result = await service(profile="full", vector=vector, preflight=broken).search("盗窃", user_id="alice")
    assert result.search.candidates and result.search.dependency_failed
    assert result.search.retrieval_status["integrity_error"] == "ValueError"
    assert vector.calls == []


@pytest.mark.asyncio
async def test_all_channels_verify_before_once_final_rerank_and_top_five():
    from app.knowledge.law.corpus import load_criminal_law_data
    from app.knowledge.law.repository import LawRepository
    from app.knowledge.law.service import LawSearchService

    repository = LawRepository(load_criminal_law_data(profile="full"))
    rows = []
    for i, article in enumerate(list(repository.article_index.values())[:12]):
        rows.append(RetrievalCandidate(str(i), repository.source_id, f"{article['article_number']} {article['content']}", {**article, "content_sha256": article["text_provenance"]["content_sha256"]}))
    vector, scorer = Vector(rows), Scorer()
    pipeline = RetrievalPipeline(vector, reranker=scorer, config=RetrievalConfig(rerank=True))
    result = await LawSearchService(repository, pipeline).search("盗窃", user_id="alice")
    assert len(scorer.calls) == 1 and len(scorer.calls[0][1]) > 5
    assert len(result.search.candidates) == 5
    assert result.search.retrieval_status["keyword"] == "executed"
    assert all(c.candidate_id in result.verifications for c in result.search.candidates)
    assert all("data_source" in c.metadata for c in result.search.candidates)


@pytest.mark.asyncio
async def test_empty_query_keeps_complete_status_without_preflight():
    async def forbidden():
        pytest.fail("empty input must not check external dependencies")

    result = await service(preflight=forbidden).search(" ", user_id="alice")
    assert not result.search.candidates
    assert all(result.search.retrieval_status[name] == "skipped_empty" for name in ("vector", "bm25", "hyde", "rerank"))


@pytest.mark.asyncio
async def test_registry_consumes_service_result_and_keeps_read_constraint():
    from app.consultation.agents.legal_research import LegalToolRegistry

    selected = service()
    registry = LegalToolRegistry({}, "alice", selected.repository.law_data, search_service=selected)
    assert (await registry._get_article("第264条"))["article"] is None
    result = await registry._search_laws("盗窃")
    assert result["candidates"][0]["source"] == "json_keyword"
    assert result["candidates"][0]["origins"]
    article = (await registry._get_article("第264条"))["article"]
    assert article["coverage_eligible"] is True
    assert article["required_elements"]


@pytest.mark.asyncio
async def test_fact_keyword_terms_survive_full_query_without_changing_vector_input():
    vector = Vector([])
    query = "殴打 轻伤 他殴打他人造成轻伤"
    result = await service(vector=vector).search(query, user_id="alice", keyword_terms=["殴打", "轻伤", "他殴打他人造成轻伤"])
    assert "第234条" in {v.article_id for v in result.verifications.values()}
    assert vector.calls[0].query == query


@pytest.mark.asyncio
async def test_registry_current_unverified_candidate_does_not_inherit_previous_eligibility():
    from app.consultation.agents.legal_research import LegalToolRegistry
    from tests.knowledge.law.helpers import make_law_service

    selected = service(profile="full")
    registry = LegalToolRegistry({}, "alice", selected.repository.law_data, search_service=selected)
    first = await registry._search_laws("盗窃")
    assert first["candidates"][0]["coverage_eligible"] is True
    unverified, _ = make_law_service(selected.repository.law_data, laws=[{"article_number": "第264条", "content": "第264条 错误正文"}])
    registry.search_service = unverified
    second = await registry._search_laws("无法识别")
    assert second["candidates"][0]["source"] == "rag_unverified"
    assert second["candidates"][0]["coverage_eligible"] is False
    assert second["candidates"][0]["provenance"]["verified"] is False


@pytest.mark.asyncio
async def test_exact_integrity_failure_keeps_only_first_controlled_article_and_partial_status():
    from app.consultation.agents.legal_research import LegalToolRegistry

    async def broken():
        raise ValueError("manifest mismatch")

    vector, scorer = Vector([]), Scorer()
    selected = service(profile="full", vector=vector, scorer=scorer, preflight=broken)
    registry = LegalToolRegistry({}, "alice", selected.repository.law_data, search_service=selected)
    status, observation, _ = await registry.execute("search_laws", {"query": "第264条 第999条"}, 1)
    assert status == "partial_dependency_failure"
    assert [candidate["article_id"] for candidate in observation["candidates"]] == ["第264条"]
    assert observation["candidates"][0]["source"] == "json_keyword"
    assert observation["rag_dependency_failed"] is True
    assert observation["retrieval_status"]["integrity_error"] == "ValueError"
    assert vector.calls == scorer.calls == []
    assert all(observation["retrieval_status"][name] == "skipped_exact" for name in ("bm25", "hyde", "rerank", "keyword"))


@pytest.mark.asyncio
async def test_concurrent_searches_share_components_but_not_verification_chooser():
    from app.knowledge.law.corpus import load_criminal_law_data
    from app.knowledge.law.repository import LawRepository
    from app.knowledge.law.service import LawSearchService

    repo = LawRepository(load_criminal_law_data(profile="snapshot"))
    class DynamicVector:
        async def retrieve(self, request):
            number = "第264条" if request.query == "盗窃" else "第234条"
            article = repo.article_index[number]
            await asyncio.sleep(0)
            candidate = RetrievalCandidate(number, repo.source_id, article["content"], article)
            return ChannelResult("vector", (candidate,), {}, StageStatus("executed"))
    pipeline = RetrievalPipeline(DynamicVector(), config=RetrievalConfig(rerank=False))
    chooser = pipeline.choose_representative
    selected = LawSearchService(repo, pipeline)
    first, second = await asyncio.gather(selected.search("盗窃", user_id="alice"), selected.search("伤害", user_id="bob"))
    assert first.verifications["第264条"].article_id == "第264条"
    assert second.verifications["第234条"].article_id == "第234条"
    assert "第234条" not in first.verifications and "第264条" not in second.verifications
    assert pipeline.choose_representative is chooser
