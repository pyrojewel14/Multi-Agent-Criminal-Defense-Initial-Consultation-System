"""Document boundaries use isolated storage and no external model."""

from io import BytesIO
from pathlib import Path

import chromadb
import pytest
from fastapi import UploadFile
from langchain_core.documents import Document

from app.knowledge.documents.fingerprints import FingerprintStore
from app.knowledge.documents.processor import DocumentProcessor
from app.knowledge.documents.repository import DocumentRepository
from app.knowledge.documents.service import DocumentService
from app.knowledge.documents.splitters.text import AsyncTextSplitter
from app.knowledge.documents.summary import DocumentSummaryService
from app.knowledge.retrieval.models import SearchResult
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from app.knowledge.retrieval.vector import ChromaVectorRetriever
from app.knowledge.storage.chroma import ChromaSource, ChromaStore


@pytest.fixture
def resources(tmp_path):
    client = chromadb.EphemeralClient()
    store = ChromaStore.create(ChromaSource("documents-test", client, "documents-test"))
    fingerprints = FingerprintStore(str(tmp_path / "fingerprints"))
    repository = DocumentRepository(store, fingerprints)

    async def embed(texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(text):
        return [1.0, 0.0]

    processor = DocumentProcessor(AsyncTextSplitter(chunk_size=50, chunk_overlap=0))
    pipeline = RetrievalPipeline(ChromaVectorRetriever(store, embed_query, 2))
    service = DocumentService(repository, processor, embed, pipeline)
    yield store, fingerprints, repository, processor, service
    client.delete_collection("documents-test")


def upload(content=b"This is a document", filename="case.txt"):
    return UploadFile(file=BytesIO(content), filename=filename, size=len(content))


def test_processor_only_prepares_owned_chunks(resources, tmp_path):
    store, _, _, processor, _ = resources
    path = tmp_path / "case.txt"
    path.write_text("owned document", encoding="utf-8")
    chunks = processor.prepare_sync(str(path), filename="case.txt", user_id="alice", md5="file-md5", is_public=False)
    assert chunks[0].page_content == "owned document"
    assert chunks[0].metadata["original_filename"] == "case.txt"
    assert chunks[0].metadata["source"] == "case.txt"
    assert chunks[0].metadata["user_id"] == "alice"
    assert store.get_by_metadata(where={"user_id": "alice"}) == ()


@pytest.mark.asyncio
async def test_upload_deduplicates_and_exposes_user_scoped_management(resources):
    store, fingerprints, repo, _, service = resources
    await service.handle_add_vector_single(upload(), "alice")
    await service.handle_add_vector_single(upload(), "alice")
    assert len(store.get_by_metadata(where={"user_id": "alice"})) == 1
    records = await fingerprints.get_all_md5_records("alice")
    assert len(records) == 1
    assert records[0]["filename"] == "case.txt"
    assert await repo.get_user_documents("bob") == []
    listing = await repo.get_user_documents("alice")
    filename = listing[0]["filename"]
    assert filename == "case.txt"
    assert listing[0]["chunk_count"] == 1
    assert (await repo.get_document_detail("alice", filename))["content"] == "This is a document"
    assert (await repo.get_document_chunks("alice", filename))["total_chunks"] == 1
    assert await repo.delete_by_filename("alice", "case.txt") is True
    assert store.get_by_metadata(where={"user_id": "alice"}) == ()
    assert await fingerprints.get_all_md5_records("alice") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"source": "/tmp/old-case.txt"},
        {"source": r"C:\uploads\old-case.txt"},
        {"filename": "old-case.txt"},
    ],
)
async def test_legacy_management_uses_same_filename_for_list_and_detail(resources, metadata):
    from app.knowledge.storage.chroma import StoredRow

    store, _, repo, _, _ = resources
    store.upsert_rows(
        [StoredRow("legacy", "legacy content", {**metadata, "user_id": "alice"})], embeddings=[[1.0, 0.0]]
    )
    assert (await repo.get_user_documents("alice"))[0]["filename"] == "old-case.txt"
    assert (await repo.get_document_detail("alice", "old-case.txt"))["content"] == "legacy content"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["second_chunk", "file_after_append"])
async def test_fingerprint_failure_rolls_back_upload_and_retry_is_complete(resources, failure_stage):
    store, fingerprints, _, _, service = resources
    await service.handle_add_vector_single(upload(b"existing document", "existing.txt"), "alice")
    await service.handle_add_vector_single(upload(b"other user document"), "bob")
    existing_ids = {row.row_id for row in store.get_by_metadata(where={"user_id": "alice"})}
    existing_hashes = await fingerprints.get_all_chunk_md5("alice")
    original_chunk, original_file = fingerprints.save_chunk_md5, fingerprints.save_md5_hex
    calls = 0

    async def save_chunk(*args):
        nonlocal calls
        calls += 1
        if failure_stage == "second_chunk" and calls == 2:
            raise OSError("fingerprint write failed")
        await original_chunk(*args)

    async def save_file(*args):
        await original_file(*args)
        if failure_stage == "file_after_append":
            raise OSError("fingerprint write failed")

    fingerprints.save_chunk_md5, fingerprints.save_md5_hex = save_chunk, save_file
    content = b"A" * 50 + b"B" * 50
    with pytest.raises(OSError, match="fingerprint write failed"):
        await service.handle_add_vector_single(upload(content), "alice")
    assert {row.row_id for row in store.get_by_metadata(where={"user_id": "alice"})} == existing_ids
    assert await fingerprints.get_all_chunk_md5("alice") == existing_hashes
    assert [record["filename"] for record in await fingerprints.get_all_md5_records("alice")] == ["existing.txt"]
    assert len(store.get_by_metadata(where={"user_id": "bob"})) == 1
    fingerprints.save_chunk_md5, fingerprints.save_md5_hex = original_chunk, original_file
    await service.handle_add_vector_single(upload(content), "alice")
    await service.handle_add_vector_single(upload(content), "alice")
    assert len(store.get_by_metadata(where={"user_id": "alice"})) == 3
    assert len(await fingerprints.get_all_chunk_md5("alice")) == 3
    assert len(await fingerprints.get_all_md5_records("alice")) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_target", ["vectors", "fingerprints"])
async def test_failed_compensation_reports_error_and_retry_recovers_all_chunks(resources, cleanup_target):
    store, fingerprints, _, _, service = resources
    original_save, original_delete = fingerprints.save_md5_hex, store.delete_rows
    original_rollback = fingerprints.rollback_upload

    async def fail_file(*args):
        raise OSError("file fingerprint unavailable")

    def fail_delete(**kwargs):
        raise OSError("rollback unavailable")

    rollback_calls = 0

    async def fail_rollback(*args):
        nonlocal rollback_calls
        rollback_calls += 1
        if rollback_calls == 2:
            raise OSError("rollback unavailable")
        await original_rollback(*args)

    fingerprints.save_md5_hex = fail_file
    if cleanup_target == "vectors":
        store.delete_rows = fail_delete
    else:
        fingerprints.rollback_upload = fail_rollback
    content = b"A" * 50 + b"B" * 50
    with pytest.raises(Exception, match="rollback unavailable"):
        await service.handle_add_vector_single(upload(content), "alice")
    assert await fingerprints.get_all_md5_records("alice") == []
    if cleanup_target == "fingerprints":
        assert len(await fingerprints.get_all_chunk_md5("alice")) == 2
    fingerprints.save_md5_hex, store.delete_rows = original_save, original_delete
    fingerprints.rollback_upload = original_rollback
    await service.handle_add_vector_single(upload(content), "alice")
    assert len(store.get_by_metadata(where={"user_id": "alice"})) == 2
    assert len(await fingerprints.get_all_chunk_md5("alice")) == 2
    assert len(await fingerprints.get_all_md5_records("alice")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining_vectors", [0, 1])
async def test_appended_file_hash_with_failed_rollback_cannot_hide_missing_vectors(resources, remaining_vectors):
    store, fingerprints, _, _, service = resources
    original_save = fingerprints.save_md5_hex
    original_rollback = fingerprints.rollback_upload
    original_delete = store.delete_rows
    file_appended = False

    async def append_then_fail(*args):
        nonlocal file_appended
        await original_save(*args)
        file_appended = True
        raise OSError("file append failed afterwards")

    async def rollback_unavailable(*args):
        if file_appended:
            raise OSError("fingerprint rollback unavailable")
        await original_rollback(*args)

    def partial_delete(*, where):
        rows = store.get_by_metadata(where=where)
        store._collection.delete(ids=[rows[0].row_id])
        raise OSError("partial vector cleanup")

    fingerprints.save_md5_hex = append_then_fail
    fingerprints.rollback_upload = rollback_unavailable
    if remaining_vectors:
        store.delete_rows = partial_delete
    content = b"A" * 50 + b"B" * 50
    with pytest.raises(RuntimeError, match="fingerprint rollback unavailable"):
        await service.handle_add_vector_single(upload(content), "alice")
    assert len(store.get_by_metadata(where={"user_id": "alice"})) == remaining_vectors
    assert len(await fingerprints.get_all_md5_records("alice")) == 1
    # The residual file fingerprint is not proof that vectors are complete.
    with pytest.raises(Exception, match="fingerprint rollback unavailable"):
        await service.handle_add_vector_single(upload(content), "alice")
    fingerprints.save_md5_hex = original_save
    fingerprints.rollback_upload = original_rollback
    store.delete_rows = original_delete
    await service.handle_add_vector_single(upload(content), "alice")
    assert len(store.get_by_metadata(where={"user_id": "alice"})) == 2
    assert len(await fingerprints.get_all_chunk_md5("alice")) == 2
    assert len(await fingerprints.get_all_md5_records("alice")) == 1

    async def unexpected_embedding(texts):
        raise AssertionError("complete duplicate must not embed again")

    service.embed_documents = unexpected_embedding
    await service.handle_add_vector_single(upload(content), "alice")


@pytest.mark.asyncio
async def test_embedding_failure_leaves_no_fingerprint_or_tempfile(resources):
    _, fingerprints, _, processor, service = resources
    paths = []
    original = processor.prepare_sync

    def prepare(path, **kwargs):
        paths.append(path)
        return original(path, **kwargs)

    processor.prepare_sync = prepare

    async def fail(texts):
        raise RuntimeError("embedding unavailable")

    service.embed_documents = fail
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await service.handle_add_vector_single(upload(), "alice")
    assert paths and all(not Path(path).exists() for path in paths)
    assert await fingerprints.get_all_md5_records("alice") == []
    assert await fingerprints.get_all_chunk_md5("alice") == set()


@pytest.mark.asyncio
async def test_search_requires_identity_even_for_public_documents(resources):
    _, _, _, _, service = resources
    await service.handle_add_vector_single(upload(), "alice", is_public=True)
    assert (await service.search("document", user_id=None)).candidates == ()
    assert (await service.search("document", user_id="bob", include_public=False)).candidates == ()
    result = await service.search("document", user_id="bob", include_public=True)
    assert len(result.candidates) == 1
    assert result.candidates[0].source_id == "documents-test"


@pytest.mark.asyncio
async def test_summary_consumes_result_without_retrieval(resources):
    _, _, _, _, service = resources
    await service.handle_add_vector_single(upload(), "alice")
    result = await service.search("document", user_id="alice")
    calls = []

    class Model:
        async def ainvoke(self, data):
            calls.append(data)
            return "summary text"

    summary = await DocumentSummaryService(Model()).get_documents_and_summary("document", result)
    assert summary["summary"] == "summary text"
    assert len(summary["documents"]) == 1
    assert len(calls) == 1
    assert "This is a document" in calls[0]["context"]
    assert (
        await DocumentSummaryService(Model()).rag_summary("empty", SearchResult((), {}, False))
        == "抱歉，我没有找到相关的信息。"
    )


@pytest.mark.asyncio
async def test_sse_preserves_progress_payload_and_reports_write_failure(resources):
    import json

    store, fingerprints, _, _, service = resources

    async def fail(texts):
        raise RuntimeError("embedding unavailable")

    service.embed_documents = fail
    events = [
        json.loads(item.split("data: ", 1)[1])
        async for item in service.handle_add_vector_multiple_stream([upload()], "alice")
    ]
    assert [event["event_type"] for event in events] == ["start", "slicing_completed", "writing", "error", "finish"]
    assert events[1]["chunk_count"] == 1
    assert events[3]["step"] == "writing"
    assert events[3]["error_message"] == "embedding unavailable"
    assert events[-1]["progress"] == 100
    assert events[-1]["success_count"] == 0
    assert events[-1]["failed_count"] == 1
    assert store.get_by_metadata(where={"user_id": "alice"}) == ()
    assert await fingerprints.get_all_md5_records("alice") == []


@pytest.mark.asyncio
async def test_sse_multiple_files_retains_serial_write_and_deduplication(resources):
    import json

    store, fingerprints, _, _, service = resources
    events = [
        json.loads(item.split("data: ", 1)[1])
        async for item in service.handle_add_vector_multiple_stream([upload(), upload()], "alice", is_public=True)
    ]
    assert events[-1]["event_type"] == "finish"
    assert events[-1]["success_count"] == 2
    assert events[-1]["failed_count"] == 0
    assert len(store.get_by_metadata(where={"user_id": "alice"})) == 1
    assert len(await fingerprints.get_all_md5_records("alice")) == 1


@pytest.mark.asyncio
async def test_single_upload_preserves_validation_for_missing_and_unknown_size(resources, monkeypatch):
    from fastapi import HTTPException
    from app.knowledge.documents import service as service_module

    *_, service = resources
    with pytest.raises(HTTPException) as missing:
        await service.handle_add_vector_single(upload(filename=None), "alice")
    assert missing.value.status_code == 400
    unknown = upload()
    unknown.size = None
    assert await service.handle_add_vector_single(unknown, "alice") == "case.txt"
    monkeypatch.setattr(service_module, "MAX_FILE_SIZE", 3)
    too_large = upload(b"1234")
    too_large.size = None
    with pytest.raises(HTTPException) as large:
        await service.handle_add_vector_single(too_large, "alice")
    assert large.value.status_code == 400


@pytest.mark.asyncio
async def test_deleted_document_can_be_uploaded_again(resources):
    store, fingerprints, repo, _, service = resources
    await service.handle_add_vector_single(upload(), "alice")
    assert await repo.delete_by_filename("alice", "case.txt") is True
    assert await fingerprints.get_all_chunk_md5("alice") == set()
    await service.handle_add_vector_single(upload(), "alice")
    assert len(store.get_by_metadata(where={"user_id": "alice"})) == 1


@pytest.mark.asyncio
async def test_delete_user_removes_chunk_records_and_preserves_other_user(resources):
    store, fingerprints, repo, _, service = resources
    await service.handle_add_vector_single(upload(), "alice")
    await service.handle_add_vector_single(upload(b"another document"), "bob")
    await repo.delete_user_documents("alice")
    assert store.get_by_metadata(where={"user_id": "alice"}) == ()
    assert await fingerprints.get_all_chunk_md5("alice") == set()
    assert not Path(fingerprints._get_md5_store_dir("alice")).exists()
    assert len(store.get_by_metadata(where={"user_id": "bob"})) == 1


@pytest.mark.asyncio
async def test_failed_upload_can_retry_with_same_content(resources):
    store, fingerprints, _, _, service = resources
    original = service.embed_documents

    async def fail(texts):
        raise RuntimeError("embedding unavailable")

    service.embed_documents = fail
    with pytest.raises(RuntimeError):
        await service.handle_add_vector_single(upload(), "alice")
    assert await fingerprints.get_all_md5_records("alice") == []
    service.embed_documents = original
    await service.handle_add_vector_single(upload(), "alice")
    assert len(store.get_by_metadata(where={"user_id": "alice"})) == 1
    assert len(await fingerprints.get_all_md5_records("alice")) == 1


@pytest.mark.asyncio
async def test_summary_limits_individual_documents_and_handles_timeout():
    import asyncio
    from app.knowledge.retrieval.models import RetrievalCandidate

    result = SearchResult(
        tuple(RetrievalCandidate(str(i), "documents", f"document-{i}", {}) for i in range(4)), {}, False
    )
    seen = []

    class Model:
        async def ainvoke(self, data):
            seen.append(data["context"])
            return "combined answer" if "以下是多个文档" in data["context"] else "individual answer"

    summary = await DocumentSummaryService(Model()).get_documents_and_summary("query", result)
    assert summary["summary"] == "combined answer"
    assert len(summary["documents"]) == 4
    assert len(seen) == 4
    assert not any("document-3" in context for context in seen)

    class SlowModel:
        async def ainvoke(self, data):
            await asyncio.sleep(0.05)

    timed_out = await DocumentSummaryService(SlowModel(), timeout_s=0.001).get_documents_and_summary("query", result)
    assert timed_out["summary"] == "抱歉，生成摘要超时，请稍后再试。"
    assert len(timed_out["documents"]) == 4


@pytest.mark.asyncio
async def test_sse_slice_failure_cleans_temporary_file(resources):
    import json

    *_, processor, service = resources
    paths = []

    def fail(path, **kwargs):
        paths.append(path)
        raise ValueError("parser unavailable")

    processor.prepare_sync = fail
    events = [
        json.loads(item.split("data: ", 1)[1])
        async for item in service.handle_add_vector_multiple_stream([upload()], "alice")
    ]
    assert [event["event_type"] for event in events] == ["start", "error", "finish"]
    assert events[1]["step"] == "slicing"
    assert events[1]["error_message"] == "parser unavailable"
    assert paths and all(not Path(path).exists() for path in paths)
