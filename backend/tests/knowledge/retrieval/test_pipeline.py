"""The pipeline exposes an untruncated recall pool and explicit dependency states."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.models import (
    ChannelResult,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalScope,
    StageStatus,
)
from app.knowledge.retrieval.pipeline import RetrievalPipeline


def request(query="query", user="alice", public=False, top_k=5):
    return RetrievalRequest(query, RetrievalScope("source", user, public), 20, top_k)


def channel(name, count=8, state="executed"):
    return ChannelResult(
        name,
        tuple(
            RetrievalCandidate(f"{name}-{i}", "source", f"text-{name}-{i}", {"row_id": f"{name}-{i}"})
            for i in range(count)
        ),
        {},
        StageStatus(state, degraded=state == "failed"),
    )


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
    result = await RetrievalPipeline(vector, bm25=bm25, config=RetrievalConfig(mode="vector", rerank=False)).search(
        request(top_k=3)
    )
    assert len(result.candidates) == 3
    assert vector.retrieve.call_args.args[0].recall_k == 3
    bm25.retrieve.assert_not_awaited()
    assert result.retrieval_status["bm25"] == "disabled"


@pytest.mark.parametrize(
    "req,state", [(request(" \n\t"), "skipped_empty"), (request(user=None), "skipped_unauthorized")]
)
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
    result = await RetrievalPipeline(
        retriever(channel("vector")), bm25=retriever(error=OSError("snapshot unavailable"))
    ).search(request())
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
    store.upsert_rows(
        [StoredRow(str(i), text, {"user_id": "alice"}) for i, text in enumerate(texts)], embeddings=[[1.0, 0.0]] * 5
    )
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
    result = await RetrievalPipeline(
        retriever(channel("vector", 0)), config=RetrievalConfig(hyde=enabled, rerank=enabled)
    ).search(request())
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


@pytest.mark.parametrize(
    "name",
    [
        "vector",
        "bm25",
        "hyde",
        "rerank",
        "degraded",
        "recall_counts",
        "mode",
        "method",
        "candidate_count",
        "vector_error",
        "bm25_error",
        "hyde_error",
        "rerank_error",
        "keyword_error",
        "hyde_attempt_count",
        "rerank_protocol",
        "vector_distance",
        "bm25_cache",
    ],
)
async def test_extra_channel_collision_is_rejected_before_retrieval(name):
    vector = retriever(channel("vector"))
    bm25 = retriever(channel("bm25"))
    with pytest.raises(ValueError, match="channel"):
        await RetrievalPipeline(vector, bm25=bm25).recall(request(), extra_channels={name: channel(name)})
    vector.retrieve.assert_not_awaited()
    bm25.retrieve.assert_not_awaited()


async def test_hyde_adds_isolated_vector_channel_without_replacing_original_query():
    from app.knowledge.retrieval.models import CandidateOrigin

    original = RetrievalCandidate(
        "same",
        "source",
        "stored evidence",
        {"row_id": "row"},
        recall_ranks={"vector": 1},
        recall_scores={"vector": 0.2},
        origins=(CandidateOrigin("source", "same", "vector", {"row_id": "row"}),),
    )
    calls = []

    class Vector:
        async def retrieve(self, req):
            calls.append(req)
            return ChannelResult(
                "vector", (original,), {"same": 0.2 if req.query == "query" else 0.7}, StageStatus("executed")
            )

    expander = type("Expander", (), {"expand": AsyncMock(return_value="hypothetical text")})()
    pipeline = RetrievalPipeline(
        Vector(),
        bm25=retriever(channel("bm25", 1)),
        expander=expander,
        config=RetrievalConfig(hyde=True, hyde_k=3, rerank=False),
    )
    pool = await pipeline.recall(request())
    result = await pipeline.finalize(request(), pool)
    assert [(r.query, r.recall_k, r.scope) for r in calls] == [
        ("query", 20, request().scope),
        ("hypothetical text", 3, request().scope),
    ]
    assert pool.channels["hyde"].name == "hyde"
    extra = pool.channels["hyde"].candidates[0]
    assert extra.recall_ranks == {"hyde": 1} and extra.recall_scores == {"hyde": 0.2}
    assert {o.channel for o in extra.origins} == {"hyde"}
    same = next(c for c in result.candidates if c.candidate_id == "same")
    assert same.content == "stored evidence" and same.recall_scores == {"vector": 0.2, "hyde": 0.7}
    assert {o.channel for o in same.origins} == {"vector", "hyde"}
    assert result.retrieval_status["hyde_attempt_count"] == 1
    expander.expand.assert_awaited_once_with("query", timeout_s=10.0)


@pytest.mark.parametrize(
    "config,state",
    [
        (RetrievalConfig(hyde=False), "disabled"),
        (RetrievalConfig(hyde=True, hyde_budget=0), "budget_exhausted"),
        (RetrievalConfig(mode="vector", hyde=True), "disabled"),
    ],
)
async def test_hyde_skip_conditions_never_generate(config, state):
    expander = type("Expander", (), {"expand": AsyncMock(return_value="hypothesis")})()
    result = await RetrievalPipeline(retriever(channel("vector", 1)), expander=expander, config=config).search(
        request()
    )
    assert result.retrieval_status["hyde"] == state and result.retrieval_status["hyde_attempt_count"] == 0
    expander.expand.assert_not_awaited()


@pytest.mark.parametrize(
    "phase,error,state",
    [
        ("generate", OSError(), "failed"),
        ("generate", TimeoutError(), "timeout"),
        ("retrieve", OSError(), "failed"),
        ("retrieve", TimeoutError(), "timeout"),
    ],
)
async def test_hyde_failure_drops_only_extra_channel(phase, error, state):
    calls = 0

    class Vector:
        async def retrieve(self, req):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise error
            return channel("vector", 2)

    expander = type(
        "Expander",
        (),
        {"expand": AsyncMock(return_value="hypothesis", side_effect=error if phase == "generate" else None)},
    )()
    result = await RetrievalPipeline(
        Vector(), bm25=retriever(channel("bm25", 2)), expander=expander, config=RetrievalConfig(hyde=True, rerank=False)
    ).search(request())
    assert len(result.candidates) == 4 and not result.dependency_failed
    assert result.retrieval_status["hyde"] == state and result.retrieval_status["hyde_error"] == type(error).__name__
    assert result.retrieval_status["degraded"] is True and result.retrieval_status["recall_counts"]["hyde"] == 0
    assert result.retrieval_status["hyde_attempt_count"] == 1


@pytest.mark.parametrize("phase", ["generate", "retrieve"])
async def test_hyde_deadline_covers_generation_and_extra_vector(phase):
    cancelled = asyncio.Event()

    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    class Vector:
        async def retrieve(self, req):
            if req.query == "hypothesis" and phase == "retrieve":
                await blocked()
            return channel("vector", 1)

    class Expander:
        async def expand(self, _query, *, timeout_s):
            if phase == "generate":
                await blocked()
            return "hypothesis"

    result = await RetrievalPipeline(
        Vector(), expander=Expander(), config=RetrievalConfig(hyde=True, hyde_timeout=0.01, rerank=False)
    ).search(request())
    assert result.retrieval_status["hyde"] == "timeout" and len(result.candidates) == 1 and cancelled.is_set()


@pytest.mark.parametrize("phase", ["generate", "retrieve"])
async def test_hyde_budget_exhaustion_is_optional_and_external_cancel_is_not(phase):
    from app.infrastructure.observability.tracing import SessionBudgetExceeded

    for error, state in [
        (SessionBudgetExceeded(session_id="s", dimension="tokens"), "budget_exhausted"),
        (asyncio.CancelledError(), None),
    ]:

        class Vector:
            async def retrieve(self, req):
                if phase == "retrieve" and req.query == "hypothesis":
                    raise error
                return channel("vector", 1)

        expander = type(
            "Expander",
            (),
            {"expand": AsyncMock(return_value="hypothesis", side_effect=error if phase == "generate" else None)},
        )()
        pipeline = RetrievalPipeline(Vector(), expander=expander, config=RetrievalConfig(hyde=True, rerank=False))
        if state is None:
            with pytest.raises(asyncio.CancelledError):
                await pipeline.search(request())
        else:
            result = await pipeline.search(request())
            assert result.retrieval_status["hyde"] == state and result.retrieval_status["degraded"]
            assert not result.dependency_failed and len(result.candidates) == 1


async def test_rerank_scores_complete_combined_pool_once_with_original_query_before_topk():
    scorer = type("Scorer", (), {"score": AsyncMock(return_value=[0.1, 0.2, 0.3, 0.9])})()
    pipeline = RetrievalPipeline(retriever(channel("vector", 2)), bm25=retriever(channel("bm25", 2)), reranker=scorer)
    result = await pipeline.search(request(top_k=1))
    assert (
        result.candidates[0].candidate_id == "vector-1"
        and result.candidates[0].rerank_score == 0.9
        and result.candidates[0].rank == 1
    )
    assert result.retrieval_status["candidate_count"] == 4 and result.retrieval_status["rerank"] == "executed"
    scorer.score.assert_awaited_once_with(
        "query", ["text-bm25-0", "text-vector-0", "text-bm25-1", "text-vector-1"], timeout_s=30.0
    )


@pytest.mark.parametrize(
    "scores", [[0.1], [float("nan"), 1], [float("inf"), 1], ["invalid", 1], [1.1, 0.3], [-0.1, 0.3]]
)
async def test_invalid_probability_scores_preserve_fusion_order(scores):
    scorer = type("Scorer", (), {"probability_scores": True, "score": AsyncMock(return_value=scores)})()
    result = await RetrievalPipeline(retriever(channel("vector", 2)), reranker=scorer).search(request())
    assert [c.candidate_id for c in result.candidates] == ["vector-0", "vector-1"] and all(
        c.rerank_score is None for c in result.candidates
    )
    assert (
        result.retrieval_status["rerank"] == "failed"
        and result.retrieval_status["degraded"]
        and result.retrieval_status["rerank_error"] == "ValueError"
    )


async def test_non_probability_scores_allow_unbounded_finite_values_and_ties_are_stable():
    scorer = type("Scorer", (), {"score": AsyncMock(return_value=[-3, 7, 7])})()
    result = await RetrievalPipeline(retriever(channel("vector", 3)), reranker=scorer).search(request())
    assert [c.candidate_id for c in result.candidates] == [
        "vector-1",
        "vector-2",
        "vector-0",
    ] and result.retrieval_status["rerank"] == "executed"


@pytest.mark.parametrize("error,state", [(TimeoutError(), "timeout"), (OSError(), "failed")])
async def test_rerank_failures_preserve_fusion_and_trace_degraded(monkeypatch, error, state):
    from app.infrastructure.observability.tracing import BoundedTraceStore
    from app.knowledge.retrieval import pipeline as module

    store = BoundedTraceStore()
    monkeypatch.setattr(module, "trace_store", store)
    scorer = type("Scorer", (), {"score": AsyncMock(side_effect=error)})()
    result = await RetrievalPipeline(retriever(channel("vector", 2)), reranker=scorer).search(request())
    assert [c.candidate_id for c in result.candidates] == ["vector-0", "vector-1"]
    assert (
        result.retrieval_status["rerank"] == state
        and result.retrieval_status["degraded"]
        and result.retrieval_status["rerank_error"] == type(error).__name__
    )
    event = next(e for e in store.events() if e.name == "full_rerank")
    assert event.event_type == "retrieval" and event.outcome == "degraded"


async def test_rerank_busy_is_distinct_and_empty_pool_does_not_score():
    from app.knowledge.retrieval.rerankers import RerankerBusyError

    scorer = type("Scorer", (), {"score": AsyncMock(side_effect=RerankerBusyError())})()
    result = await RetrievalPipeline(retriever(channel("vector", 1)), reranker=scorer).search(request())
    assert result.retrieval_status["rerank"] == "busy" and result.retrieval_status["degraded"]
    scorer.score.reset_mock()
    empty = await RetrievalPipeline(retriever(channel("vector", 0)), reranker=scorer).search(request())
    assert empty.retrieval_status["rerank"] == "skipped_empty" and empty.retrieval_status["candidate_count"] == 0
    scorer.score.assert_not_awaited()


async def test_rerank_whole_stage_deadline_and_outer_cancellation():
    class Scorer:
        async def score(self, query, documents, *, timeout_s):
            await asyncio.Event().wait()

    pipeline = RetrievalPipeline(
        retriever(channel("vector", 1)), reranker=Scorer(), config=RetrievalConfig(rerank_timeout=0.01)
    )
    result = await pipeline.search(request())
    assert result.retrieval_status["rerank"] == "timeout" and result.candidates
    pipeline = RetrievalPipeline(
        retriever(channel("vector", 1)),
        reranker=type("Scorer", (), {"score": AsyncMock(side_effect=asyncio.CancelledError())})(),
    )
    with pytest.raises(asyncio.CancelledError):
        await pipeline.search(request())


@pytest.mark.parametrize("state", ["failed", "timeout", "busy", "budget_exhausted"])
async def test_failed_hyde_channel_cannot_leak_partial_candidates(state):
    class Vector:
        async def retrieve(self, req):
            return channel("vector", 2, state) if req.query == "hypothesis" else channel("vector", 1)

    expander = type("Expander", (), {"expand": AsyncMock(return_value="hypothesis")})()
    pool = await RetrievalPipeline(Vector(), expander=expander, config=RetrievalConfig(hyde=True)).recall(request())
    assert pool.channels["hyde"].candidates == ()
    assert pool.channels["hyde"].scores == {} and pool.retrieval_status["hyde"] == state
    assert pool.retrieval_status["degraded"] and not pool.dependency_failed


async def test_hyde_combined_deadline_is_not_reset_after_generation():
    class Vector:
        async def retrieve(self, req):
            if req.query == "hypothesis":
                await asyncio.sleep(0.04)
            return channel("vector", 1)

    class Expander:
        async def expand(self, _query, *, timeout_s):
            await asyncio.sleep(0.04)
            return "hypothesis"

    result = await RetrievalPipeline(
        Vector(), expander=Expander(), config=RetrievalConfig(hyde=True, hyde_timeout=0.06, rerank=False)
    ).search(request())
    assert result.retrieval_status["hyde"] == "timeout" and len(result.candidates) == 1


async def test_rerank_includes_hyde_and_domain_channels_in_the_single_full_pool():
    class Vector:
        async def retrieve(self, req):
            return ChannelResult(
                "vector",
                (RetrievalCandidate(req.query, "source", req.query, {"row_id": req.query}),),
                {},
                StageStatus("executed"),
            )

    expander = type("Expander", (), {"expand": AsyncMock(return_value="hypothesis")})()
    scorer = type("Scorer", (), {"score": AsyncMock(return_value=[0.1, 0.9, 0.2, 0.3])})()
    pipeline = RetrievalPipeline(
        Vector(),
        bm25=retriever(channel("bm25", 1)),
        expander=expander,
        reranker=scorer,
        config=RetrievalConfig(hyde=True),
    )
    pool = await pipeline.recall(request(top_k=1), extra_channels={"keyword": channel("keyword", 1)})
    result = await pipeline.finalize(request(top_k=1), pool)
    assert result.retrieval_status["candidate_count"] == 4 and len(result.candidates) == 1
    assert result.candidates[0].candidate_id == "hypothesis"
    scorer.score.assert_awaited_once_with(
        "query", ["text-bm25-0", "hypothesis", "text-keyword-0", "query"], timeout_s=30.0
    )


async def test_rerank_session_budget_exception_remains_control_flow():
    from app.infrastructure.observability.tracing import SessionBudgetExceeded

    scorer = type(
        "Scorer", (), {"score": AsyncMock(side_effect=SessionBudgetExceeded(session_id="s", dimension="calls"))}
    )()
    with pytest.raises(SessionBudgetExceeded):
        await RetrievalPipeline(retriever(channel("vector", 1)), reranker=scorer).search(request())
