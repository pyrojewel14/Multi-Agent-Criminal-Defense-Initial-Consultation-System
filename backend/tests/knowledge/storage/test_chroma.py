"""Storage contracts use real isolated Chroma collections and explicit vectors."""

import chromadb
import pytest

from app.knowledge.storage.chroma import ChromaSource, ChromaStore, StoredRow


@pytest.fixture
def source(tmp_path):
    return ChromaSource("test-corpus", chromadb.PersistentClient(path=str(tmp_path)), "documents")


def test_open_missing_collection_does_not_create(source):
    before = source.client.list_collections()
    with pytest.raises(Exception, match="exist"):
        ChromaStore.open_existing(source)
    assert source.client.list_collections() == before == []


def test_explicit_create_upsert_read_delete_preserves_metadata(source):
    store = ChromaStore.create(source, metadata={"hnsw:space": "cosine"})
    metadata = {"user_id": "a", "is_public": False, "page": 3, "rating": 0.5}
    row = StoredRow("same-id", "first text", metadata)
    metadata["page"] = 99
    store.upsert_rows([row], embeddings=[[1.0, 0.0]])
    opened = ChromaStore.open_existing(source)
    rows = opened.get_by_metadata(where={"user_id": "a"})
    assert rows == (row,)
    assert rows[0].metadata["page"] == 3
    with pytest.raises(TypeError):
        rows[0].metadata["page"] = 99
    hits = opened.query_by_vector([0.6, 0.8], k=5, where={"user_id": "a"})
    assert hits[0].content == row.content
    assert hits[0].metadata == row.metadata
    assert hits[0].distance == pytest.approx(0.4)
    opened.upsert_rows([StoredRow("same-id", "changed", row.metadata)], embeddings=[[0.0, 1.0]])
    assert opened.get_by_metadata(where={"user_id": "a"})[0].content == "changed"
    opened.delete_rows(where={"user_id": "a"})
    assert opened.get_by_metadata(where={"user_id": "a"}) == ()


def test_snapshot_fingerprint_tracks_content_metadata_and_source(source):
    store = ChromaStore.create(source)
    rows = [StoredRow("z", "z text", {"is_public": True}), StoredRow("a", "a text", {"is_public": True})]
    store.upsert_rows(rows, embeddings=[[1.0, 0.0], [0.0, 1.0]])
    first = store.read_snapshot(where={"is_public": True})
    assert [r.row_id for r in first.rows] == ["a", "z"]
    store.delete_rows(where={"is_public": True})
    store.upsert_rows(list(reversed(rows)), embeddings=[[0.0, 1.0], [1.0, 0.0]])
    assert store.read_snapshot(where={"is_public": True}) == first
    other = ChromaStore.open_existing(ChromaSource("other-source", source.client, source.collection_name))
    assert other.read_snapshot(where={"is_public": True}).fingerprint != first.fingerprint
    store.upsert_rows([StoredRow("a", "new text", {"is_public": True})], embeddings=[[0.0, 1.0]])
    changed = store.read_snapshot(where={"is_public": True})
    assert changed.fingerprint != first.fingerprint
    store.upsert_rows([StoredRow("a", "new text", {"is_public": True, "tag": "new"})], embeddings=[[0.0, 1.0]])
    assert store.read_snapshot(where={"is_public": True}).fingerprint != changed.fingerprint
    assert store.read_snapshot(where={"user_id": "absent"}).rows == ()


@pytest.mark.parametrize("vectors", [[], [[1.0]], [[1.0, 0.0], [1.0]], [[float("nan"), 0.0], [1.0, 0.0]], [[float("inf"), 0.0], [1.0, 0.0]], [[True, 0.0], [1.0, 0.0]], [[1e300, 0.0], [1.0, 0.0]]])
def test_upsert_rejects_invalid_vectors_before_writing(source, vectors):
    store = ChromaStore.create(source)
    rows = [StoredRow("a", "a", {"is_public": True}), StoredRow("b", "b", {"is_public": True})]
    with pytest.raises(ValueError, match="embedding"):
        store.upsert_rows(rows, embeddings=vectors)
    assert source.client.get_collection(source.collection_name).count() == 0


def test_empty_upsert_and_empty_metadata_work_without_embedding_function(source):
    store = ChromaStore.create(source)
    store.upsert_rows([], embeddings=[])
    store.upsert_rows([StoredRow("a", "a", {})], embeddings=[[1.0, 0.0]])
    assert source.client.get_collection(source.collection_name).get()["documents"] == ["a"]


@pytest.mark.parametrize("operation", ["query", "get", "snapshot", "delete"])
def test_empty_filter_is_rejected(source, operation):
    store = ChromaStore.create(source)
    with pytest.raises(ValueError, match="where"):
        if operation == "query":
            store.query_by_vector([1.0, 0.0], k=2, where={})
        elif operation == "get":
            store.get_by_metadata(where={})
        elif operation == "snapshot":
            store.read_snapshot(where={})
        else:
            store.delete_rows(where={})


def test_open_and_create_disable_default_embedding_function():
    from unittest.mock import Mock

    client = Mock()
    source = ChromaSource("source", client, "documents")
    ChromaStore.open_existing(source)
    client.get_collection.assert_called_once_with(name="documents", embedding_function=None)
    ChromaStore.create(source)
    client.create_collection.assert_called_once_with(name="documents", embedding_function=None, metadata=None)
