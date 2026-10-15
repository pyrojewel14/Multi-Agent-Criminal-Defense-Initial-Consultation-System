"""Regression tests for document upload metadata validation."""

from io import BytesIO

import chromadb
import pytest
from fastapi import HTTPException, UploadFile

from app.knowledge.documents import service as document_service
from app.knowledge.storage.chroma import ChromaSource, ChromaStore
from app.knowledge.wiring import KnowledgeResources, build_document_services


@pytest.fixture
def service(tmp_path):
    client = chromadb.EphemeralClient()
    store = ChromaStore.create(ChromaSource("metadata-validation", client, "metadata-validation"))

    async def embed_documents(texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(text):
        return [1.0, 0.0]

    resources = KnowledgeResources()
    service = build_document_services(
        store=store,
        md5_directory=tmp_path,
        embed_documents=embed_documents,
        embed_query=embed_query,
        resources=resources,
        env={"LAW_FULL_RERANK_ENABLED": "false"},
    )
    yield service
    resources.close()
    client.delete_collection("metadata-validation")


@pytest.mark.asyncio
async def test_single_upload_rejects_missing_filename(service):
    upload = UploadFile(file=BytesIO(b"content"), filename=None, size=None)
    with pytest.raises(HTTPException, match="文件名不能为空") as exc_info:
        await service.handle_add_vector_single(upload, "user-1")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_single_upload_accepts_unknown_size_when_content_is_within_limit(service):
    upload = UploadFile(file=BytesIO(b"content"), filename="case.txt", size=None)
    assert await service.handle_add_vector_single(upload, "user-1") == "case.txt"
    rows = service.repository.store.get_by_metadata(where={"user_id": "user-1"})
    assert len(rows) == 1
    assert rows[0].content == "content"
    assert rows[0].metadata["original_filename"] == "case.txt"


@pytest.mark.asyncio
async def test_single_upload_checks_content_length_when_reported_size_is_unknown(service, monkeypatch):
    upload = UploadFile(file=BytesIO(b"1234"), filename="case.txt", size=None)
    monkeypatch.setattr(document_service, "MAX_FILE_SIZE", 3)
    with pytest.raises(HTTPException, match="文件大小不能超过") as exc_info:
        await service.handle_add_vector_single(upload, "user-1")
    assert exc_info.value.status_code == 400
