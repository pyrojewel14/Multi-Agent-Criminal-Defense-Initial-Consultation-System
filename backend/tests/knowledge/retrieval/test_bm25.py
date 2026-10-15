"""BM25 reads fresh authorized snapshots and only returns positive matches."""

import asyncio
import threading
from unittest.mock import Mock

import chromadb
import pytest

from app.knowledge.retrieval import bm25, tokenization
from app.knowledge.retrieval.bm25 import BM25Retriever
from app.knowledge.retrieval.models import RetrievalRequest, RetrievalScope
from app.knowledge.retrieval.tokenization import chinese_tokens
from app.knowledge.storage.chroma import ChromaSource, ChromaStore, StoredRow


TEXTS = ["刑法264条盗窃公私财物", "诈骗FRAUD公私财物", "故意伤害他人身体", "醉酒驾驶机动车", "贩卖毒品有关规定"]


def request(source="source", user="alice", public=False, query="盗窃", k=10):
    return RetrievalRequest(query, RetrievalScope(source, user, public), k, 5)


@pytest.fixture
def store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    store = ChromaStore.create(ChromaSource("source", client, "bm25-corpus"))
    store.upsert_rows(
        [
            StoredRow(str(i), text, {"user_id": "alice", "is_public": False, "version": 1})
            for i, text in enumerate(TEXTS)
        ],
        embeddings=[[1.0, 0.0]] * 5,
    )
    return store


@pytest.mark.parametrize("query,expected", [("盗窃", TEXTS[0]), ("264", TEXTS[0]), ("fraud", TEXTS[1])])
async def test_chinese_english_and_numbers_match_without_spaces(store, query, expected):
    result = await BM25Retriever(store).retrieve(request(query=query))
    assert result.status.state == "executed"
    assert result.candidates[0].content == expected
    assert all(score > 0 for score in result.scores.values())
    item = result.candidates[0]
    assert item.recall_ranks == {"bm25": 1}
    assert item.recall_scores["bm25"] == result.scores[item.candidate_id]
    assert item.metadata["row_id"] == str(TEXTS.index(expected))
    assert item.origins[0].channel == "bm25"
    assert "row_id" not in item.origins[0].metadata


@pytest.mark.parametrize("query", ["!!!。！？", "quantumxyz"])
async def test_no_positive_match_never_backfills_arbitrary_candidates(store, query):
    result = await BM25Retriever(store).retrieve(request(query=query))
    assert result.candidates == () and result.scores == {}
    assert result.status.state == "executed"


def test_tokenizer_keeps_cjk_characters_bigrams_and_english_numbers():
    assert chinese_tokens("盗窃 FRAUD 264！") == ["盗", "窃", "盗窃", "fraud", "264"]
    assert chinese_tokens("？！") == []


async def test_same_source_user_and_public_scopes_do_not_leak(store):
    store.upsert_rows(
        [
            StoredRow("bob", "盗窃privatebob", {"user_id": "bob", "is_public": False}),
            StoredRow("pub", "盗窃public", {"user_id": "bob", "is_public": True}),
        ],
        embeddings=[[1.0, 0.0]] * 2,
    )
    # Keep the matching term rare within every tested scope, so positive-only BM25 applies.
    store.upsert_rows(
        [StoredRow(f"filler-{i}", text, {"user_id": "owner", "is_public": True}) for i, text in enumerate(TEXTS[1:])],
        embeddings=[[1.0, 0.0]] * 4,
    )
    store.upsert_rows(
        [
            StoredRow(f"bob-filler-{i}", text, {"user_id": "bob", "is_public": False})
            for i, text in enumerate(TEXTS[1:])
        ],
        embeddings=[[1.0, 0.0]] * 4,
    )
    retriever = BM25Retriever(store)
    requests = [request(), request(user="bob"), request(public=True), request(user=None, public=True)]
    results = await asyncio.gather(*(retriever.retrieve(req) for req in requests))
    assert [set(c.metadata["row_id"] for c in r.candidates) for r in results] == [
        {"0"},
        {"bob", "pub"},
        {"0", "pub"},
        {"pub"},
    ]


async def test_two_sources_with_same_row_id_keep_identity_separate(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    results = []
    for index, text in enumerate(["盗窃first", "盗窃second"]):
        source = f"source-{index}"
        store = ChromaStore.create(ChromaSource(source, client, f"bm25-{index}"))
        store.upsert_rows(
            [StoredRow(str(i), text if i == 0 else TEXTS[i], {"is_public": True}) for i in range(5)],
            embeddings=[[1.0, 0.0]] * 5,
        )
        results.append(await BM25Retriever(store).retrieve(request(source, user=None, public=True)))
    assert [r.candidates[0].content for r in results] == ["盗窃first", "盗窃second"]
    assert results[0].candidates[0].candidate_id != results[1].candidates[0].candidate_id


async def test_content_and_metadata_changes_refresh_warm_cache(store):
    retriever = BM25Retriever(store)
    before = await retriever.retrieve(request())
    store.upsert_rows(
        [StoredRow("0", "侵占他人财物", {"user_id": "alice", "is_public": False, "version": 2})],
        embeddings=[[1.0, 0.0]],
    )
    after = await retriever.retrieve(request(query="侵占"))
    assert after.candidates[0].content == "侵占他人财物"
    assert after.candidates[0].metadata["version"] == 2
    assert after.candidates[0].candidate_id != before.candidates[0].candidate_id
    store.upsert_rows(
        [StoredRow("0", "侵占他人财物", {"user_id": "alice", "is_public": False, "version": 3})],
        embeddings=[[1.0, 0.0]],
    )
    current = await retriever.retrieve(request(query="侵占"))
    assert current.candidates[0].metadata["version"] == 3


async def test_snapshot_failure_never_returns_warm_cache(store, monkeypatch):
    retriever = BM25Retriever(store)
    assert (await retriever.retrieve(request())).candidates
    monkeypatch.setattr(store, "read_snapshot", Mock(side_effect=OSError("snapshot unavailable")))
    with pytest.raises(OSError, match="snapshot unavailable"):
        await retriever.retrieve(request())


async def test_deletion_empties_scope_without_old_results(store):
    retriever = BM25Retriever(store)
    assert (await retriever.retrieve(request())).candidates
    store.delete_rows(where={"user_id": "alice"})
    assert (await retriever.retrieve(request())).candidates == ()


async def test_cache_reuses_unchanged_index_and_invalidates_tokenizer_version(store, monkeypatch):
    builds = []
    real = bm25.BM25Okapi

    def build(tokens):
        builds.append(tokens)
        return real(tokens)

    monkeypatch.setattr(bm25, "BM25Okapi", build)
    retriever = BM25Retriever(store)
    await retriever.retrieve(request(k=1))
    await retriever.retrieve(request(k=3))
    assert len(builds) == 1
    monkeypatch.setattr(tokenization, "TOKENIZER_VERSION", "next-version")
    await retriever.retrieve(request())
    assert len(builds) == 2


async def test_cache_is_bounded_and_evicted_scope_rebuilds(store, monkeypatch):
    builds = []
    real = bm25.BM25Okapi

    def build(tokens):
        builds.append(tokens)
        return real(tokens)

    monkeypatch.setattr(bm25, "BM25Okapi", build)
    retriever = BM25Retriever(store, cache_size=1)
    await retriever.retrieve(request())
    await retriever.retrieve(request(public=True))
    await retriever.retrieve(request())
    assert len(builds) == 3


@pytest.mark.parametrize(
    "req,state", [(request(query=" \n"), "skipped_empty"), (request(user=None), "skipped_unauthorized")]
)
async def test_guard_prevents_snapshot_io(req, state):
    store = Mock(source=ChromaSource("source", Mock(), "documents"))
    result = await BM25Retriever(store).retrieve(req)
    assert result.status.state == state and result.candidates == ()
    store.read_snapshot.assert_not_called()


async def test_wrong_source_is_rejected_before_snapshot_read(store, monkeypatch):
    read = Mock()
    monkeypatch.setattr(store, "read_snapshot", read)
    with pytest.raises(ValueError, match="source"):
        await BM25Retriever(store).retrieve(request(source="wrong"))
    read.assert_not_called()


async def test_snapshot_read_and_scoring_run_outside_event_loop(store, monkeypatch):
    threads = []
    event_thread = threading.get_ident()
    read = store.read_snapshot

    def observed(**kwargs):
        threads.append(threading.get_ident())
        return read(**kwargs)

    monkeypatch.setattr(store, "read_snapshot", observed)
    assert (await BM25Retriever(store).retrieve(request())).candidates
    assert threads and threads[0] != event_thread


async def test_all_punctuation_documents_do_not_build_invalid_index(store):
    store.delete_rows(where={"user_id": "alice"})
    store.upsert_rows([StoredRow("punctuation", "！？！", {"user_id": "alice"})], embeddings=[[1.0, 0.0]])
    result = await BM25Retriever(store).retrieve(request())
    assert result.status.state == "executed" and result.candidates == ()


async def test_concurrent_same_scope_builds_one_index(store, monkeypatch):
    import time

    builds = []
    real = bm25.BM25Okapi

    def slow_build(tokens):
        builds.append(tokens)
        time.sleep(0.02)
        return real(tokens)

    monkeypatch.setattr(bm25, "BM25Okapi", slow_build)
    retriever = BM25Retriever(store)
    results = await asyncio.gather(*(retriever.retrieve(request()) for _ in range(4)))
    assert len(builds) == 1
    assert all(result.candidates[0].content == TEXTS[0] for result in results)


async def test_punctuation_row_does_not_shift_matching_document(store):
    store.upsert_rows([StoredRow("punctuation", "……！！！", {"user_id": "alice"})], embeddings=[[1.0, 0.0]])
    result = await BM25Retriever(store).retrieve(request())
    assert result.candidates[0].content == TEXTS[0]
    assert result.candidates[0].metadata["row_id"] == "0"
