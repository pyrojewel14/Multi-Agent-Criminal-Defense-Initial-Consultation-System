"""Shared retrieval records preserve failures and independent immutable snapshots."""

from dataclasses import FrozenInstanceError, replace
import pytest

from app.knowledge.retrieval import models


def test_empty_result_preserves_dependency_status():
    status = {"vector": models.StageStatus("failed", "MissingCollection", True)}
    result = models.SearchResult((), status, True)
    pool = models.RecallPool({}, status, True)

    status.clear()

    for record in (result, pool):
        assert record.dependency_failed is True
        assert record.retrieval_status["vector"].state == "failed"
        assert record.retrieval_status["vector"].error_type == "MissingCollection"
        assert record.retrieval_status["vector"].degraded is True
    assert result.candidates == ()


def test_candidate_containers_are_not_shared():
    metadata = {"labels": ["original"], "provenance": {"steps": [{"valid": False}]}}
    ranks = {"vector": 2}
    scores = {"vector": 0.25}
    origin = models.CandidateOrigin("source", "row-1", "vector", metadata)
    origins = [origin]
    candidate = models.RetrievalCandidate(
        "row-1", "source", "text", metadata,
        recall_ranks=ranks, recall_scores=scores, origins=origins,
    )
    other = models.RetrievalCandidate("row-2", "source", "other", {})
    metadata["labels"].append("changed")
    metadata["provenance"]["steps"][0]["valid"] = True
    ranks["vector"] = 99
    scores.clear()
    origins.clear()

    assert candidate.metadata["labels"] == ("original",)
    assert candidate.metadata["provenance"]["steps"][0]["valid"] is False
    assert origin.metadata["provenance"]["steps"][0]["valid"] is False
    assert candidate.recall_ranks == {"vector": 2}
    assert candidate.recall_scores == {"vector": 0.25}
    assert candidate.origins == (origin,)
    assert other.recall_ranks == {}
    assert other.recall_scores == {}
    assert other.origins == ()
    with pytest.raises(TypeError):
        candidate.metadata["provenance"]["steps"][0]["valid"] = True
    with pytest.raises(TypeError):
        candidate.recall_ranks["vector"] = 1


def test_result_channel_containers_are_independent_snapshots():
    candidate = models.RetrievalCandidate("a", "source", "text", {"tags": {"a"}})
    candidates = [candidate]
    scores = {"a": 0.9}
    channel = models.ChannelResult("vector", candidates, scores, models.StageStatus("executed"))
    channels = {"vector": channel}
    status = {"stages": ["vector"], "details": {"available": True}}
    pool = models.RecallPool(channels, status, False)
    result = models.SearchResult(candidates, status, False)
    channels.clear()
    candidates.clear()
    scores.clear()
    status["stages"].append("other")
    status["details"]["available"] = False

    assert pool.channels["vector"].candidates == (candidate,)
    assert channel.scores == {"a": 0.9}
    assert result.candidates == (candidate,)
    assert result.retrieval_status["stages"] == ("vector",)
    assert pool.retrieval_status["details"]["available"] is True
    assert candidate.metadata["tags"] == frozenset({"a"})
    with pytest.raises(TypeError):
        pool.channels["other"] = channel
    with pytest.raises(TypeError):
        channel.scores["a"] = 1.0


def test_request_and_ranking_updates_require_new_records():
    scope = models.RetrievalScope("source", "user-a", True)
    request = models.RetrievalRequest("original query", scope, 20, 5)
    candidate = models.RetrievalCandidate("a", "source", "text", {})
    ranked = replace(candidate, rank=1, fusion_score=0.02, rerank_score=0.75)

    assert request.query == "original query"
    assert request.scope.user_id == "user-a"
    assert (ranked.rank, ranked.fusion_score, ranked.rerank_score) == (1, 0.02, 0.75)
    assert (candidate.rank, candidate.fusion_score, candidate.rerank_score) == (None, None, None)
    assert "rag_verified" not in candidate.metadata
    with pytest.raises(FrozenInstanceError):
        request.query = "changed"
    with pytest.raises(FrozenInstanceError):
        scope.user_id = "user-b"
    with pytest.raises(FrozenInstanceError):
        candidate.rank = 1


@pytest.mark.parametrize("state", [
    "executed", "disabled", "unconfigured", "skipped_empty", "skipped_unauthorized",
    "skipped_exact", "failed", "timeout", "busy", "budget_exhausted",
])
def test_stage_status_accepts_shared_execution_states(state):
    assert models.StageStatus(state).state == state


def test_stage_status_rejects_unknown_execution_state():
    with pytest.raises(ValueError):
        models.StageStatus("success-ish")


@pytest.mark.parametrize("field,value", [
    ("recall_k", 0), ("recall_k", True), ("recall_k", 2.5),
    ("top_k", -1), ("top_k", False), ("top_k", 1.5),
])
def test_request_rejects_nonpositive_or_noninteger_counts(field, value):
    settings = {"recall_k": 20, "top_k": 5, field: value}
    with pytest.raises(ValueError):
        models.RetrievalRequest("query", models.RetrievalScope("source", None, True), **settings)


def test_request_top_k_can_exceed_one_channel_recall():
    request = models.RetrievalRequest("query", models.RetrievalScope("source", None, True), 5, 10)
    assert (request.recall_k, request.top_k) == (5, 10)
