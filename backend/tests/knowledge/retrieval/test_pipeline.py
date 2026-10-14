"""The pipeline exposes an untruncated recall pool and explicit dependency states."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.models import ChannelResult, RetrievalCandidate, RetrievalRequest, RetrievalScope, StageStatus
from app.knowledge.retrieval.pipeline import RetrievalPipeline


def request(query="query", user="alice", public=False, top_k=5):
    return RetrievalRequest(query, RetrievalScope("source", user, public), 20, top_k)


def channel(name, count=8, state="executed"):
    return ChannelResult(name, tuple(RetrievalCandidate(f"{name}-{i}", "source", f"text-{name}-{i}", {"row_id": f"{name}-{i}"}) for i in range(count)), {}, StageStatus(state, degraded=state == "failed"))


def retriever(result=None, error=None):
    return type("FakeRetriever", (), {"retrieve": AsyncMock(return_value=result, side_effect=error)})()


async def test_recall_keeps_all_candidates_and_only_finalize_truncates():
    vector, bm25 = retriever(channel("vector")), retriever(channel("bm25"))
    pipeline = RetrievalPipeline(vector, bm25=bm25, config=RetrievalConfig(rerank=False))
    pool = await pipeline.recall(request())
    assert sum(len(c.candidates) for c in pool.channels.values()) == 16
    result = await pipeline.finalize(request(), pool)
    assert len(result.candidates) == 5
    assert [c.rank for c in result.candidates] == [1, 2, 3, 4, 5]
    assert all(c.fusion_score > 0 for c in result.candidates)
    assert await pipeline.search(request()) == result
    assert vector.retrieve.call_args.args[0].recall_k == 20


async def test_vector_mode_uses_final_k_and_disables_bm25():
    vector, bm25 = retriever(channel("vector")), retriever(channel("bm25"))
    result = await RetrievalPipeline(vector, bm25=bm25, config=RetrievalConfig(mode="vector", rerank=False)).search(request(top_k=3))
    assert len(result.candidates) == 3
    assert vector.retrieve.call_args.args[0].recall_k == 3
    bm25.retrieve.assert_not_awaited()
    assert result.retrieval_status["bm25"] == "disabled"


@pytest.mark.parametrize("req,state", [(request(" \n\t"), "skipped_empty"), (request(user=None), "skipped_unauthorized")])
async def test_guard_does_no_io_and_does_not_return_extra_candidates(req, state):
    vector, bm25 = retriever(channel("vector")), retriever(channel("bm25"))
    pipeline = RetrievalPipeline(vector, bm25=bm25)
    pool = await pipeline.recall(req, extra_channels={"keyword": channel("keyword")})
    result = await pipeline.finalize(req, pool)
    assert result.candidates == () and not result.dependency_failed
    assert result.retrieval_status["vector"] == state
    vector.retrieve.assert_not_awaited()
    bm25.retrieve.assert_not_awaited()


async def test_primary_failure_survives_usable_supplement():
    pipeline = RetrievalPipeline(retriever(error=RuntimeError("index unavailable")), bm25=retriever(channel("bm25")))
    pool = await pipeline.recall(request(), extra_channels={"keyword": channel("keyword", 1)})
    result = await pipeline.finalize(request(), pool)
    assert result.candidates and result.dependency_failed
    assert result.retrieval_status["vector"] == "failed"
    assert result.retrieval_status["vector_error"] == "RuntimeError"
    assert result.retrieval_status["degraded"] is True


async def test_failed_channel_status_marks_primary_dependency_failure():
    result = await RetrievalPipeline(retriever(channel("vector", 0, "failed"))).search(request())
    assert result.dependency_failed
    assert result.candidates == ()


async def test_bm25_failure_is_degraded_without_required_dependency_failure():
    result = await RetrievalPipeline(retriever(channel("vector")), bm25=retriever(error=OSError("snapshot unavailable"))).search(request())
    assert not result.dependency_failed and result.candidates
    assert result.retrieval_status["bm25"] == "failed"
    assert result.retrieval_status["bm25_error"] == "OSError"
    assert result.retrieval_status["degraded"] is True


async def test_explicit_optional_timeout_is_degraded_even_if_component_omits_flag():
    timeout = ChannelResult("bm25", (), {}, StageStatus("timeout", "TimeoutError"))
    result = await RetrievalPipeline(retriever(channel("vector")), bm25=retriever(timeout)).search(request())
    assert result.retrieval_status["degraded"] is True
    assert not result.dependency_failed


async def test_real_vector_and_bm25_share_identity_in_one_pipeline(tmp_path):
    import chromadb

    from app.knowledge.retrieval.bm25 import BM25Retriever
    from app.knowledge.retrieval.vector import ChromaVectorRetriever
    from app.knowledge.storage.chroma import ChromaSource, ChromaStore, StoredRow

    client = chromadb.PersistentClient(path=str(tmp_path))
    store = ChromaStore.create(ChromaSource("source", client, "pipeline-integration"))
    texts = ["盗窃他人财物", "危险驾驶机动车", "故意伤害他人身体", "贩卖毒品", "诈骗公私财物"]
    store.upsert_rows([StoredRow(str(i), text, {"user_id": "alice"}) for i, text in enumerate(texts)],
                      embeddings=[[1.0, 0.0]] * 5)
    vector = ChromaVectorRetriever(store, AsyncMock(return_value=[1.0, 0.0]), expected_dimension=2)
    pipeline = RetrievalPipeline(vector, bm25=BM25Retriever(store), config=RetrievalConfig(rerank=False))
    result = await pipeline.search(request(query="盗窃"))
    assert len(result.candidates) == 5
    assert result.candidates[0].content == "盗窃他人财物"
    assert result.candidates[0].recall_ranks["bm25"] == 1
    assert set(result.candidates[0].recall_ranks) == {"vector", "bm25"}
    assert {o.channel for o in result.candidates[0].origins} == {"vector", "bm25"}
    assert not result.dependency_failed


async def test_explicit_unconfigured_vector_supports_domain_pool():
    pipeline = RetrievalPipeline(None)
    pool = await pipeline.recall(request(user=None, public=True), extra_channels={"keyword": channel("keyword", 2)})
    result = await pipeline.finalize(request(user=None, public=True), pool)
    assert len(result.candidates) == 2 and not result.dependency_failed
    assert result.retrieval_status["vector"] == "unconfigured"
    assert result.retrieval_status["keyword"] == "executed"
    assert result.retrieval_status["degraded"] is False


@pytest.mark.parametrize("enabled,expected", [(False, "disabled"), (True, "unconfigured")])
async def test_optional_capabilities_keep_disabled_or_unconfigured_state(enabled, expected):
    result = await RetrievalPipeline(retriever(channel("vector", 0)), config=RetrievalConfig(hyde=enabled, rerank=enabled)).search(request())
    assert result.retrieval_status["hyde"] == expected
    assert result.retrieval_status["rerank"] == expected


async def test_outer_cancellation_propagates_without_supplement():
    with pytest.raises(asyncio.CancelledError):
        await RetrievalPipeline(retriever(error=asyncio.CancelledError())).recall(request())


async def test_session_budget_control_propagates_instead_of_dependency_failure():
    from app.infrastructure.observability.tracing import SessionBudgetExceeded

    error = SessionBudgetExceeded(session_id="session", dimension="calls")
    bm25 = retriever(channel("bm25"))
    with pytest.raises(SessionBudgetExceeded):
        await RetrievalPipeline(retriever(error=error), bm25=bm25).recall(request())
    bm25.retrieve.assert_not_awaited()


async def test_main_vector_timeout_retains_timeout_state_and_dependency_failure():
    result = await RetrievalPipeline(retriever(error=TimeoutError())).search(request())
    assert result.retrieval_status["vector"] == "timeout"
    assert result.retrieval_status["vector_error"] == "TimeoutError"
    assert result.dependency_failed


@pytest.mark.parametrize("name", [
    "vector", "bm25", "hyde", "rerank", "degraded", "recall_counts", "mode", "method",
    "candidate_count", "vector_error", "bm25_error", "hyde_error", "rerank_error",
    "keyword_error", "hyde_attempt_count", "rerank_protocol", "vector_distance", "bm25_cache",
])
async def test_extra_channel_collision_is_rejected_before_retrieval(name):
    vector = retriever(channel("vector"))
    bm25 = retriever(channel("bm25"))
    with pytest.raises(ValueError, match="channel"):
        await RetrievalPipeline(vector, bm25=bm25).recall(request(), extra_channels={name: channel(name)})
    vector.retrieve.assert_not_awaited()
    bm25.retrieve.assert_not_awaited()
