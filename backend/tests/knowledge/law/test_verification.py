"""检索相关性不能提升法律来源或覆盖资格。"""

import json
from dataclasses import replace

import pytest

from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.models import ChannelResult, RecallPool, RetrievalCandidate, StageStatus
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from tests.knowledge.law.test_search_service import Vector


def fixture_repository():
    from app.knowledge.law.corpus import load_criminal_law_data
    from app.knowledge.law.repository import LawRepository

    return LawRepository(load_criminal_law_data(profile="full"))


def raw(repository, number, *, cid="rag", content=None, **metadata):
    article = repository.article_index[number]
    return RetrievalCandidate(
        cid,
        repository.source_id,
        content or f"{article['article_number']} {article['content']}",
        {**article, "content_sha256": article["text_provenance"]["content_sha256"], **metadata},
    )


def test_failed_rag_origin_survives_trusted_keyword_representative():
    from app.knowledge.law.verification import LawVerifier
    from app.knowledge.law.keyword import LawKeywordRetriever

    repo = fixture_repository()
    bad = raw(repo, "第264条", content="第264条 不一致正文")
    keyword = LawKeywordRetriever(repo).recall("盗窃", 20)
    pool = RecallPool(
        {"vector": ChannelResult("vector", (bad,), {}, StageStatus("executed")), "keyword": keyword}, {}, False
    )
    verified = LawVerifier(repo).verify_pool(pool)
    assert verified.verifications["rag"].source == "rag_unverified"
    assert verified.verifications["rag"].coverage_eligible is False
    assert verified.verifications["rag"].provenance["verified"] is False
    assert any(v.source == "json_keyword" and v.coverage_eligible for v in verified.verifications.values())
    with pytest.raises(TypeError):
        verified.verifications["rag"].provenance["verified"] = True


@pytest.mark.asyncio
async def test_service_dedupe_does_not_wash_failed_origin_or_choose_by_source_rank():
    from app.knowledge.law.service import LawSearchService

    repo = fixture_repository()
    bad = raw(repo, "第264条", content="第264条 不一致正文")
    vector = Vector([bad])
    result = await LawSearchService(repo, RetrievalPipeline(vector, config=RetrievalConfig(rerank=False))).search(
        "盗窃", user_id="alice"
    )
    assert result.as_laws()[0]["data_source"] == "json_keyword"
    assert result.as_laws()[0]["required_elements"]
    assert result.verifications["rag"].source == "rag_unverified"
    assert any(origin.candidate_id == "rag" for origin in result.search.candidates[0].origins)
    assert (
        next(origin for origin in result.as_laws()[0]["origins"] if origin["candidate_id"] == "rag")["verification"][
            "source"
        ]
        == "rag_unverified"
    )
    json.dumps(result.as_laws(), ensure_ascii=False)


@pytest.mark.parametrize(
    "field,value", [("corpus_version", "wrong"), ("corpus_sha256", "wrong"), ("content_sha256", "wrong")]
)
def test_full_candidate_binding_failure_never_supplies_elements(field, value):
    from app.knowledge.law.verification import LawVerifier

    repo = fixture_repository()
    candidate = raw(repo, "第264条", **{field: value})
    pool = RecallPool({"vector": ChannelResult("vector", (candidate,), {}, StageStatus("executed"))}, {}, False)
    result = LawVerifier(repo).verify_pool(pool)
    assert result.verifications["rag"].source == "rag_unverified"
    assert not result.verifications["rag"].required_elements


def test_full_text_only_is_readable_without_coverage_eligibility(monkeypatch):
    from app.knowledge.law.verification import LawVerifier

    monkeypatch.setenv("LAW_FULL_DEMO_ANNOTATIONS", "off")
    repo = fixture_repository()
    candidate = raw(repo, "第133条之一")
    pool = RecallPool({"vector": ChannelResult("vector", (candidate,), {}, StageStatus("executed"))}, {}, False)
    result = LawVerifier(repo).verify_pool(pool)
    assert result.verifications["rag"].source == "text_only"
    assert not result.verifications["rag"].required_elements
