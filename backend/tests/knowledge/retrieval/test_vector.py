"""Parameterized vector retrieval preserves scope, identity, and native distances."""

import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import chromadb
import pytest

from app.knowledge.retrieval.models import RetrievalRequest, RetrievalScope
from app.knowledge.retrieval.vector import ChromaVectorRetriever, candidate_from_row, candidate_identity, scope_where
from app.knowledge.storage.chroma import ChromaSource, ChromaStore, StoredRow


def request(source_id="source", user_id="a", include_public=False, query="query"):
    return RetrievalRequest(query, RetrievalScope(source_id, user_id, include_public), 10, 5)


@pytest.fixture
def store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    store = ChromaStore.create(ChromaSource("source", client, "documents"), metadata={"hnsw:space": "cosine"})
    store.upsert_rows([
        StoredRow("private-a", "A private", {"user_id": "a", "is_public": False, "page": 2}),
        StoredRow("private-b", "B private", {"user_id": "b", "is_public": False}),
        StoredRow("public", "public", {"user_id": "b", "is_public": True}),
    ], embeddings=[[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    return store


async def test_two_real_collections_with_same_id_are_isolated(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    results = []
    for index, text in enumerate(["first corpus", "second corpus"]):
        sid = f"source-{index}"
        store = ChromaStore.create(ChromaSource(sid, client, f"documents-{index}"))
        store.upsert_rows([StoredRow("same-id", text, {"is_public": True})], embeddings=[[1.0, 0.0]])
        retriever = ChromaVectorRetriever(store, AsyncMock(return_value=[1.0, 0.0]), expected_dimension=2)
        results.append(await retriever.retrieve(request(sid, None, True)))
    assert [r.candidates[0].content for r in results] == ["first corpus", "second corpus"]
    assert results[0].candidates[0].candidate_id != results[1].candidates[0].candidate_id
    assert all(r.status.state == "executed" for r in results)


async def test_scopes_match_snapshots_and_do_not_leak_between_concurrent_requests(store):
    embed = AsyncMock(return_value=[1.0, 0.0])
    retriever = ChromaVectorRetriever(store, embed, expected_dimension=2)
    requests = [request(), request(user_id="b"), request(include_public=True), request(user_id=None, include_public=True)]
    results = await asyncio.gather(*(retriever.retrieve(r) for r in requests))
    assert [{c.content for c in result.candidates} for result in results] == [
        {"A private"}, {"B private", "public"}, {"A private", "public"}, {"public"},
    ]
    for req, result in zip(requests, results):
        snapshot = store.read_snapshot(where=scope_where(req.scope))
        assert {c.content for c in result.candidates} == {r.content for r in snapshot.rows}


async def test_native_distance_metadata_and_rank_are_preserved(store):
    result = await ChromaVectorRetriever(store, AsyncMock(return_value=[1.0, 0.0]), expected_dimension=2).retrieve(request(include_public=True))
    native = store.query_by_vector([1.0, 0.0], k=10, where=scope_where(request(include_public=True).scope))
    for rank, (candidate, row) in enumerate(zip(result.candidates, native), 1):
        assert candidate.metadata == {**row.metadata, "row_id": row.row_id}
        assert candidate.recall_ranks == {"vector": rank}
        assert candidate.recall_scores == {"vector": row.distance}
        assert result.scores[candidate.candidate_id] == row.distance
        assert candidate.origins[0].metadata == row.metadata
        assert candidate.origins[0].candidate_id == candidate.candidate_id
    assert result.candidates[1].recall_scores["vector"] == pytest.approx(0.4)


@pytest.mark.parametrize("req,state", [(request(user_id=None), "skipped_unauthorized"), (request(query=" \n\t "), "skipped_empty")])
async def test_skipped_requests_do_not_embed_or_read(req, state):
    store = Mock(source=ChromaSource("source", Mock(), "documents"))
    embed = AsyncMock()
    result = await ChromaVectorRetriever(store, embed, expected_dimension=2).retrieve(req)
    assert result.status.state == state
    assert result.candidates == () and result.scores == {}
    embed.assert_not_awaited()
    store.query_by_vector.assert_not_called()


async def test_wrong_source_fails_before_embedding_or_io(store):
    embed = AsyncMock()
    with pytest.raises(ValueError, match="source"):
        await ChromaVectorRetriever(store, embed, expected_dimension=2).retrieve(request("wrong-source"))
    embed.assert_not_awaited()


@pytest.mark.parametrize("vector", [[], [1.0], [float("nan"), 0.0], [float("inf"), 0.0], [True, 0.0], ["1", 0.0], [[1.0], [0.0]], [1e300, 0.0]])
async def test_invalid_embedding_fails_before_io(vector):
    store = Mock(source=ChromaSource("source", Mock(), "documents"))
    with pytest.raises(ValueError, match="embedding"):
        await ChromaVectorRetriever(store, AsyncMock(return_value=vector), expected_dimension=2).retrieve(request())
    store.query_by_vector.assert_not_called()


async def test_unknown_dimension_uses_native_collection_validation(store):
    with pytest.raises(Exception, match="dimension"):
        await ChromaVectorRetriever(store, AsyncMock(return_value=[1.0, 0.0, 0.0]), expected_dimension=None).retrieve(request())


async def test_embedding_errors_and_cancellation_propagate(store):
    for error in [RuntimeError("embedding unavailable"), asyncio.CancelledError()]:
        with pytest.raises(type(error)):
            await ChromaVectorRetriever(store, AsyncMock(side_effect=error), expected_dimension=2).retrieve(request())


async def test_query_error_propagates_and_io_runs_on_worker_thread():
    event_thread = threading.get_ident()
    store = Mock(source=ChromaSource("source", Mock(), "documents"))
    threads = []

    def fail(*args, **kwargs):
        threads.append(threading.get_ident())
        raise RuntimeError("collection unavailable")

    store.query_by_vector.side_effect = fail
    with pytest.raises(RuntimeError, match="collection unavailable"):
        await ChromaVectorRetriever(store, AsyncMock(return_value=[1.0, 0.0]), expected_dimension=2).retrieve(request())
    assert threads and threads[0] != event_thread


async def test_empty_authorized_collection_is_executed(store):
    result = await ChromaVectorRetriever(store, AsyncMock(return_value=[1.0, 0.0]), expected_dimension=2).retrieve(request(user_id="absent"))
    assert result.status.state == "executed"
    assert result.candidates == ()


def test_scope_where_refuses_no_permission():
    with pytest.raises(PermissionError):
        scope_where(RetrievalScope("source", None, False))


def test_candidate_identity_includes_source_row_and_text_without_paths():
    original = candidate_identity("source", "id", "text")
    assert len({original, candidate_identity("other", "id", "text"), candidate_identity("source", "other", "text"), candidate_identity("source", "id", "changed")}) == 4
    assert original == candidate_identity("source", "id", "text")


def test_candidate_from_row_shares_identity_and_preserves_original_metadata():
    row = StoredRow("actual-id", "text", {"row_id": "untrusted-metadata-id", "page": 3})
    vector = candidate_from_row(row, source_id="source", channel="vector")
    bm25 = candidate_from_row(row, source_id="source", channel="bm25")
    assert vector.candidate_id == bm25.candidate_id == candidate_identity("source", "actual-id", "text")
    assert vector.metadata == {"row_id": "actual-id", "page": 3}
    assert vector.origins[0].metadata == row.metadata
    assert vector.origins[0].channel == "vector"
    assert bm25.origins[0].channel == "bm25"
    assert vector.recall_ranks == vector.recall_scores == {}


async def test_original_query_is_passed_to_injected_embedding(store):
    embed = AsyncMock(return_value=[1.0, 0.0])
    query = "  original query\n"
    await ChromaVectorRetriever(store, embed, expected_dimension=2).retrieve(request(query=query))
    embed.assert_awaited_once_with(query)


@pytest.mark.parametrize("dimension", [True, 0, -1, 2.5])
def test_invalid_expected_dimension_rejected(store, dimension):
    with pytest.raises(ValueError, match="dimension"):
        ChromaVectorRetriever(store, AsyncMock(), expected_dimension=dimension)
