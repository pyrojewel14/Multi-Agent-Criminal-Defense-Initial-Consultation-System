"""Chinese BM25 and scoped cache contracts against real isolated Chroma."""

import asyncio
import time
from unittest.mock import patch

import chromadb
import pytest
from langchain_chroma import Chroma

from app.infrastructure.config.rag import chroma_config
from app.knowledge.rag.retrievers.hybrid_retriever import HybridRetriever

TEXTS = [
    "刑法第264条规定，盗窃公私财物。",
    "刑法第266条规定，诈骗公私财物，FRAUD。",
    "刑法第263条规定，以暴力抢劫公私财物。",
    "刑法第234条规定，故意伤害他人身体。",
    "刑法第133条之一规定，醉酒驾驶机动车。",
]


@pytest.fixture
def corpus(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "bm25"))
    collection = client.get_or_create_collection("bm25-scope", embedding_function=None)
    collection.add(
        ids=[f"alice-{index}" for index in range(5)],
        documents=TEXTS,
        embeddings=[[1.0, 0.0]] * 5,
        metadatas=[
            {"user_id": "alice", "is_public": False, "source": f"source-{index}"}
            for index in range(5)
        ],
    )
    store = Chroma(client=client, collection_name="bm25-scope", embedding_function=None)
    yield HybridRetriever(store), collection
    client.delete_collection("bm25-scope")


def add_public(collection):
    collection.add(
        ids=["public"],
        documents=["公开的法律参考资料。"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"user_id": "owner", "is_public": True, "source": "public-source"}],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected"), [("盗窃", TEXTS[0]), ("264", TEXTS[0]), ("fraud", TEXTS[1])])
async def test_chinese_numbers_and_english_match_inside_unspaced_documents(corpus, query, expected):
    hybrid, _ = corpus
    retriever = await hybrid.get_bm25_retriever("alice")
    results = await retriever.ainvoke(query)
    assert results[0].page_content == expected
    scores = retriever.vectorizer.get_scores(retriever.preprocess_func(query))
    assert max(scores) > 0


@pytest.mark.asyncio
async def test_unchanged_scope_reuses_index_with_fresh_documents_and_current_k(corpus, monkeypatch):
    hybrid, _ = corpus
    first = await hybrid.get_bm25_retriever("alice")
    first.docs[0].metadata["source"] = "caller-mutation"
    first.k = 0
    monkeypatch.setitem(chroma_config, "k", 1)
    second = await hybrid.get_bm25_retriever("alice")

    assert first.vectorizer is second.vectorizer
    assert first is not second
    assert second.k == 1
    assert {doc.metadata["source"] for doc in second.docs} == {f"source-{index}" for index in range(5)}
    assert (await second.ainvoke("盗窃"))[0].page_content == TEXTS[0]


@pytest.mark.asyncio
async def test_content_update_without_count_change_rebuilds_index(corpus):
    hybrid, collection = corpus
    first = await hybrid.get_bm25_retriever("alice")
    changed = "侵占他人财物的行为。"
    collection.update(ids=["alice-0"], documents=[changed], embeddings=[[1.0, 0.0]])
    second = await hybrid.get_bm25_retriever("alice")

    assert second.vectorizer is not first.vectorizer
    assert (await second.ainvoke("侵占"))[0].page_content == changed
    assert TEXTS[0] not in [doc.page_content for doc in second.docs]


@pytest.mark.asyncio
async def test_metadata_update_without_content_change_refreshes_snapshot(corpus):
    hybrid, collection = corpus
    first = await hybrid.get_bm25_retriever("alice")
    collection.update(ids=["alice-0"], metadatas=[{"source": "updated-source"}])
    second = await hybrid.get_bm25_retriever("alice")

    assert second.vectorizer is not first.vectorizer
    assert (await second.ainvoke("盗窃"))[0].metadata["source"] == "updated-source"


@pytest.mark.asyncio
async def test_added_document_rebuilds_index_and_can_be_retrieved(corpus):
    hybrid, collection = corpus
    first = await hybrid.get_bm25_retriever("alice")
    collection.add(
        ids=["new"], documents=["贩卖毒品的规定。"], embeddings=[[1.0, 0.0]],
        metadatas=[{"user_id": "alice", "is_public": False}],
    )
    second = await hybrid.get_bm25_retriever("alice")

    assert second.vectorizer is not first.vectorizer
    assert len(second.docs) == 6
    assert (await second.ainvoke("毒品"))[0].page_content == "贩卖毒品的规定。"


@pytest.mark.asyncio
async def test_deleted_document_is_not_returned_from_warm_cache(corpus):
    hybrid, collection = corpus
    first = await hybrid.get_bm25_retriever("alice")
    collection.delete(ids=["alice-0"])
    second = await hybrid.get_bm25_retriever("alice")

    assert second.vectorizer is not first.vectorizer
    assert len(second.docs) == 4
    assert TEXTS[0] not in [doc.page_content for doc in await second.ainvoke("盗窃")]


@pytest.mark.asyncio
async def test_empty_scope_drops_old_index_and_accepts_later_upload(corpus):
    hybrid, collection = corpus
    await hybrid.get_bm25_retriever("alice")
    collection.delete(where={"user_id": "alice"})
    assert await hybrid.get_bm25_retriever("alice") is None
    collection.add(
        ids=["new"], documents=["重新上传的文档。"], embeddings=[[1.0, 0.0]],
        metadatas=[{"user_id": "alice", "is_public": False}],
    )
    current = await hybrid.get_bm25_retriever("alice")
    assert [doc.page_content for doc in current.docs] == ["重新上传的文档。"]


@pytest.mark.asyncio
async def test_user_and_public_scopes_keep_separate_cache_entries(corpus):
    hybrid, collection = corpus
    add_public(collection)
    collection.add(
        ids=["bob"], documents=["另一用户的私有文档。"], embeddings=[[1.0, 0.0]],
        metadatas=[{"user_id": "bob", "is_public": False, "source": "bob-source"}],
    )
    private = await hybrid.get_bm25_retriever("alice", False)
    inclusive = await hybrid.get_bm25_retriever("alice", True)
    bob = await hybrid.get_bm25_retriever("bob", True)
    public = await hybrid.get_bm25_retriever(None, True)
    private_again = await hybrid.get_bm25_retriever("alice", False)

    assert private_again.vectorizer is private.vectorizer
    assert private.vectorizer is not inclusive.vectorizer
    assert {doc.metadata["user_id"] for doc in private.docs} == {"alice"}
    assert {doc.metadata["user_id"] for doc in inclusive.docs} == {"alice", "owner"}
    assert {doc.metadata["user_id"] for doc in bob.docs} == {"bob", "owner"}
    assert {doc.metadata["user_id"] for doc in public.docs} == {"owner"}


@pytest.mark.asyncio
async def test_revoking_public_visibility_removes_warm_cached_document(corpus):
    hybrid, collection = corpus
    add_public(collection)
    public = await hybrid.get_bm25_retriever("bob", True)
    assert len(public.docs) == 1
    collection.update(ids=["public"], metadatas=[{"is_public": False}])

    assert await hybrid.get_bm25_retriever("bob", True) is None
    assert await hybrid.get_bm25_retriever(None, True) is None
    owner = await hybrid.get_bm25_retriever("owner", False)
    assert [doc.page_content for doc in owner.docs] == ["公开的法律参考资料。"]


@pytest.mark.asyncio
async def test_transfer_between_users_updates_both_warm_scopes(corpus):
    hybrid, collection = corpus
    await hybrid.get_bm25_retriever("alice")
    assert await hybrid.get_bm25_retriever("bob") is None
    collection.update(ids=["alice-0"], metadatas=[{"user_id": "bob"}])

    alice = await hybrid.get_bm25_retriever("alice")
    bob = await hybrid.get_bm25_retriever("bob")
    assert TEXTS[0] not in [doc.page_content for doc in alice.docs]
    assert [doc.page_content for doc in bob.docs] == [TEXTS[0]]


@pytest.mark.asyncio
async def test_failed_scope_read_does_not_serve_cached_data(corpus):
    hybrid, _ = corpus
    await hybrid.get_bm25_retriever("alice")
    with patch.object(hybrid.vectors_store, "get", side_effect=RuntimeError("scope unavailable")):
        with pytest.raises(RuntimeError, match="scope unavailable"):
            await hybrid.get_bm25_retriever("alice")


@pytest.mark.asyncio
async def test_scope_without_indexable_tokens_falls_back_after_content_change(corpus):
    hybrid, collection = corpus
    await hybrid.get_bm25_retriever("alice")
    collection.update(
        ids=[f"alice-{index}" for index in range(5)],
        documents=["……！！！"] * 5,
        embeddings=[[1.0, 0.0]] * 5,
    )
    assert await hybrid.get_bm25_retriever("alice") is None


@pytest.mark.asyncio
async def test_mixed_punctuation_document_does_not_shift_bm25_result_mapping(corpus):
    hybrid, collection = corpus
    collection.add(
        ids=["punctuation"], documents=["……！！！"], embeddings=[[1.0, 0.0]],
        metadatas=[{"user_id": "alice", "is_public": False}],
    )
    retriever = await hybrid.get_bm25_retriever("alice")
    assert (await retriever.ainvoke("盗窃"))[0].page_content == TEXTS[0]
    assert len(retriever.vectorizer.get_scores(retriever.preprocess_func("盗窃"))) == len(retriever.docs)


@pytest.mark.asyncio
async def test_cache_evicts_least_recently_used_scope_after_eight_entries(corpus):
    hybrid, collection = corpus
    add_public(collection)
    first = await hybrid.get_bm25_retriever("user-0", True)
    oldest = await hybrid.get_bm25_retriever("user-1", True)
    for index in range(2, 8):
        await hybrid.get_bm25_retriever(f"user-{index}", True)
    refreshed = await hybrid.get_bm25_retriever("user-0", True)
    assert refreshed.vectorizer is first.vectorizer
    await hybrid.get_bm25_retriever("user-8", True)
    assert (await hybrid.get_bm25_retriever("user-0", True)).vectorizer is first.vectorizer
    assert (await hybrid.get_bm25_retriever("user-1", True)).vectorizer is not oldest.vectorizer


@pytest.mark.asyncio
async def test_identical_public_corpus_is_not_shared_between_users_or_stores(corpus):
    hybrid, collection = corpus
    add_public(collection)
    alice = await hybrid.get_bm25_retriever("user-a", True)
    bob = await hybrid.get_bm25_retriever("user-b", True)
    other_store = await HybridRetriever(hybrid.vectors_store).get_bm25_retriever("user-a", True)

    assert alice.vectorizer is not bob.vectorizer
    assert alice.vectorizer is not other_store.vectorizer


@pytest.mark.asyncio
async def test_concurrent_same_scope_builds_one_index(corpus):
    from langchain_community.retrievers import BM25Retriever

    hybrid, _ = corpus
    original = BM25Retriever.from_documents

    def slow_build(*args, **kwargs):
        time.sleep(0.03)
        return original(*args, **kwargs)

    with patch.object(BM25Retriever, "from_documents", side_effect=slow_build) as build:
        results = await asyncio.gather(*(hybrid.get_bm25_retriever("alice") for _ in range(4)))
    assert all(result.vectorizer is results[0].vectorizer for result in results)
    assert build.call_count == 1
