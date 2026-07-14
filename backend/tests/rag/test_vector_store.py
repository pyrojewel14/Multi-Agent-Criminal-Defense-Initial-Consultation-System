"""Unit tests for ``app.rag.vector_store.VectorStoreService``.

The ChromaDB-backed ``__init__`` is bypassed by patching every external
collaborator. We focus on the orchestration methods that the service exposes.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test fixture: build a VectorStoreService with all heavy dependencies mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Return a VectorStoreService with all backing services mocked."""
    with patch("app.rag.vector_store.get_abstract_path") as gap, \
         patch("app.rag.vector_store.os.makedirs"), \
         patch("app.rag.vector_store.VectorStoreService._create_chroma_client") as create_client, \
         patch("app.rag.vector_store.Chroma") as chroma_cls, \
         patch("app.rag.vector_store.MD5Store") as md5_cls, \
         patch("app.rag.vector_store.HybridRetriever") as hr_cls, \
         patch("app.rag.vector_store.DocumentProcessor") as dp_cls, \
         patch("app.rag.vector_store.embed_model"):
        gap.return_value = "/tmp/chroma"
        create_client.return_value = MagicMock()
        chroma_instance = MagicMock()
        chroma_cls.return_value = chroma_instance
        md5_instance = MagicMock()
        md5_cls.return_value = md5_instance
        hr_instance = MagicMock()
        hr_cls.return_value = hr_instance
        dp_instance = MagicMock()
        dp_cls.return_value = dp_instance

        from app.rag.vector_store import VectorStoreService

        svc = VectorStoreService()
        svc.vectors_store = chroma_instance
        svc.md5_store = md5_instance
        svc.hybrid_retriever = hr_instance
        svc.document_processor = dp_instance
        yield svc


# ---------------------------------------------------------------------------
# get_vector_store singleton
# ---------------------------------------------------------------------------


class TestGetVectorStore:
    def test_returns_singleton(self):
        import app.rag.vector_store as vs_mod

        # Reset module-level singleton for the test
        vs_mod._vector_store_instance = None
        with patch("app.rag.vector_store.VectorStoreService") as vs_cls:
            vs_cls.return_value = "singleton"
            a = vs_mod.get_vector_store()
            b = vs_mod.get_vector_store()
        assert a == b == "singleton"
        vs_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Wrapped helpers
# ---------------------------------------------------------------------------


class TestGetRetriever:
    @pytest.mark.asyncio
    async def test_delegates_to_hybrid(self, service):
        service.hybrid_retriever.get_retriever = AsyncMock(return_value="ret")
        result = await service.get_retriever("query", "u1", True)
        assert result == "ret"
        service.hybrid_retriever.get_retriever.assert_awaited_once_with("query", "u1", True)


class TestGetBm25Retriever:
    @pytest.mark.asyncio
    async def test_delegates_to_hybrid(self, service):
        service.hybrid_retriever.get_bm25_retriever = AsyncMock(return_value="bm25")
        result = await service.get_bm25_retriever("u1")
        assert result == "bm25"
        service.hybrid_retriever.get_bm25_retriever.assert_awaited_once_with("u1")


class TestGetDynamicWeights:
    def test_delegates_to_hybrid(self, service):
        with patch("app.rag.vector_store.HybridRetriever.get_dynamic_weights", return_value=[0.5, 0.5]) as gd:
            assert service.get_dynamic_weights("q") == [0.5, 0.5]
            gd.assert_called_once_with("q")


class TestGetAllDocuments:
    @pytest.mark.asyncio
    async def test_delegates(self, service):
        service.hybrid_retriever._get_all_documents = AsyncMock(return_value=["docs"])
        result = await service._get_all_documents()
        assert result == ["docs"]


# ---------------------------------------------------------------------------
# MD5 wrapper methods
# ---------------------------------------------------------------------------


class TestMd5Wrappers:
    @pytest.mark.asyncio
    async def test_check_md5_hex(self, service):
        service.md5_store.check_md5_hex = AsyncMock(return_value=True)
        assert await service.check_md5_hex("m", "u") is True
        service.md5_store.check_md5_hex.assert_awaited_once_with("m", "u")

    @pytest.mark.asyncio
    async def test_save_md5_hex(self, service):
        service.md5_store.save_md5_hex = AsyncMock()
        await service.save_md5_hex("m", "f", "of", "u")
        service.md5_store.save_md5_hex.assert_awaited_once_with("m", "f", "of", "u")

    def test_save_md5_hex_sync(self, service):
        service.md5_store.save_md5_hex_sync = MagicMock()
        service.save_md5_hex_sync("m", "f", "of", "u")
        service.md5_store.save_md5_hex_sync.assert_called_once_with("m", "f", "of", "u")

    @pytest.mark.asyncio
    async def test_get_md5_info(self, service):
        service.md5_store.get_md5_info = AsyncMock(return_value={"md5": "x"})
        result = await service.get_md5_info("u", "x")
        assert result == {"md5": "x"}

    @pytest.mark.asyncio
    async def test_get_md5_info_handles_exception(self, service):
        service.md5_store.get_md5_info = AsyncMock(side_effect=RuntimeError("boom"))
        result = await service.get_md5_info("u", "x")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_md5_records(self, service):
        service.md5_store.get_all_md5_records = AsyncMock(return_value=[{"md5": "a"}])
        result = await service.get_all_md5_records("u")
        assert result == [{"md5": "a"}]

    @pytest.mark.asyncio
    async def test_get_all_md5_records_handles_exception(self, service):
        service.md5_store.get_all_md5_records = AsyncMock(side_effect=RuntimeError("x"))
        assert await service.get_all_md5_records("u") == []


# ---------------------------------------------------------------------------
# Delete operations
# ---------------------------------------------------------------------------


class TestDeleteAll:
    @pytest.mark.asyncio
    async def test_delete_all_documents_clears_everything(self, service):
        service.vectors_store.delete = MagicMock()
        service.md5_store.clear_all = AsyncMock()
        await service.delete_all_documents()
        # delete was called via asyncio.to_thread -> MagicMock still recorded
        service.md5_store.clear_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_all_documents_propagates(self, service):
        service.vectors_store.delete = MagicMock(side_effect=RuntimeError("x"))
        service.md5_store.clear_all = AsyncMock()
        with pytest.raises(RuntimeError):
            await service.delete_all_documents()


class TestDeleteUserDocuments:
    @pytest.mark.asyncio
    async def test_delegates(self, service):
        service.delete_user_md5 = AsyncMock()
        await service.delete_user_documents("u1")
        service.delete_user_md5.assert_awaited_once_with("u1", delete_documents=True)

    @pytest.mark.asyncio
    async def test_propagates(self, service):
        service.delete_user_md5 = AsyncMock(side_effect=RuntimeError("x"))
        with pytest.raises(RuntimeError):
            await service.delete_user_documents("u1")


class TestDeleteUserMd5:
    @pytest.mark.asyncio
    async def test_delete_documents_true(self, service):
        service.vectors_store.delete = MagicMock()
        service.md5_store.delete_user_md5 = AsyncMock()
        await service.delete_user_md5("u1", delete_documents=True)
        # delete was called via to_thread, but the mock's behavior is recorded
        service.md5_store.delete_user_md5.assert_awaited_once_with("u1")

    @pytest.mark.asyncio
    async def test_delete_documents_false(self, service):
        service.vectors_store.delete = MagicMock()
        service.md5_store.delete_user_md5 = AsyncMock()
        await service.delete_user_md5("u1", delete_documents=False)
        service.vectors_store.delete.assert_not_called()
        service.md5_store.delete_user_md5.assert_awaited_once_with("u1")


class TestDeleteByFilename:
    @pytest.mark.asyncio
    async def test_success(self, service):
        service.md5_store.delete_by_filename = AsyncMock(return_value="md5-1")
        service.vectors_store.delete = MagicMock()
        result = await service.delete_by_filename("u1", "file.txt", delete_documents=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_md5_returns_false(self, service):
        service.md5_store.delete_by_filename = AsyncMock(return_value=None)
        result = await service.delete_by_filename("u1", "file.txt")
        assert result is False
        service.vectors_store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_delete_documents(self, service):
        service.md5_store.delete_by_filename = AsyncMock(return_value="md5-1")
        service.vectors_store.delete = MagicMock()
        result = await service.delete_by_filename("u1", "f.txt", delete_documents=False)
        assert result is True
        service.vectors_store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_returns_false(self, service):
        service.md5_store.delete_by_filename = AsyncMock(side_effect=RuntimeError("x"))
        result = await service.delete_by_filename("u1", "f.txt")
        assert result is False


class TestDeleteSingleMd5:
    @pytest.mark.asyncio
    async def test_success(self, service):
        service.md5_store.delete_single_md5 = AsyncMock(return_value=True)
        service.vectors_store.delete = MagicMock()
        result = await service.delete_single_md5("u1", "m1")
        assert result is True

    @pytest.mark.asyncio
    async def test_md5_not_found(self, service):
        service.md5_store.delete_single_md5 = AsyncMock(return_value=False)
        result = await service.delete_single_md5("u1", "m1")
        assert result is False

    @pytest.mark.asyncio
    async def test_exception_returns_false(self, service):
        service.md5_store.delete_single_md5 = AsyncMock(side_effect=RuntimeError("x"))
        result = await service.delete_single_md5("u1", "m1")
        assert result is False


# ---------------------------------------------------------------------------
# Document listing / detail
# ---------------------------------------------------------------------------


class TestGetUserDocuments:
    @pytest.mark.asyncio
    async def test_aggregates_chunks_per_file(self, service):
        service.vectors_store.get = MagicMock(return_value={
            "ids": ["id1", "id2", "id3"],
            "metadatas": [
                {"source": "a.txt", "user_id": "u1", "created_at": "t1"},
                {"source": "a.txt", "user_id": "u1", "created_at": "t1"},
                {"source": "b.txt", "user_id": "u1", "created_at": "t2"},
            ],
            "documents": ["chunk1 content here", "chunk2 content", "other"],
        })
        result = await service.get_user_documents("u1")
        assert len(result) == 2
        a = next(d for d in result if d["filename"] == "a.txt")
        b = next(d for d in result if d["filename"] == "b.txt")
        assert a["chunk_count"] == 2
        assert b["chunk_count"] == 1

    @pytest.mark.asyncio
    async def test_no_user_id(self, service):
        service.vectors_store.get = MagicMock(return_value={"ids": [], "metadatas": [], "documents": []})
        result = await service.get_user_documents(None)
        assert result == []

    @pytest.mark.asyncio
    async def test_uses_original_filename_for_display(self, service):
        service.vectors_store.get = MagicMock(return_value={
            "ids": ["id1"],
            "metadatas": [{"source": "a.txt", "original_filename": "原始.txt", "user_id": "u1"}],
            "documents": ["c"],
        })
        result = await service.get_user_documents("u1")
        assert result[0]["original_filename"] == "原始.txt"

    @pytest.mark.asyncio
    async def test_exception_propagates(self, service):
        service.vectors_store.get = MagicMock(side_effect=RuntimeError("x"))
        with pytest.raises(RuntimeError):
            await service.get_user_documents("u1")

    @pytest.mark.asyncio
    async def test_falls_back_to_filename_metadata(self, service):
        # When 'source' metadata is missing, fall back to 'filename' field.
        service.vectors_store.get = MagicMock(return_value={
            "ids": ["id1"],
            "metadatas": [{"filename": "b.txt", "user_id": "u1"}],
            "documents": ["c"],
        })
        result = await service.get_user_documents("u1")
        assert result[0]["filename"] == "b.txt"

    @pytest.mark.asyncio
    async def test_falls_back_to_unknown(self, service):
        # When neither source nor filename metadata exists, use 'unknown'.
        service.vectors_store.get = MagicMock(return_value={
            "ids": ["id1"],
            "metadatas": [{"user_id": "u1"}],
            "documents": ["c"],
        })
        result = await service.get_user_documents("u1")
        assert result[0]["filename"] == "unknown"


class TestGetDocumentDetail:
    @pytest.mark.asyncio
    async def test_returns_detail(self, service):
        service.vectors_store.get = MagicMock(return_value={
            "ids": ["id1", "id2"],
            "metadatas": [
                {"source": "a.txt", "user_id": "u1", "created_at": "t1"},
                {"source": "a.txt", "user_id": "u1", "created_at": "t1"},
            ],
            "documents": ["chunk1", "chunk2"],
        })
        result = await service.get_document_detail("u1", "a.txt")
        assert result is not None
        assert result["chunk_count"] == 2
        assert "chunk1" in result["content"]

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, service):
        service.vectors_store.get = MagicMock(return_value={"ids": [], "metadatas": [], "documents": []})
        result = await service.get_document_detail("u1", "nope.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_string_source(self, service):
        service.vectors_store.get = MagicMock(return_value={
            "ids": ["id1"],
            "metadatas": [{"source": 123, "user_id": "u1"}],
            "documents": ["c"],
        })
        result = await service.get_document_detail("u1", "123")
        assert result is not None
        assert result["chunk_count"] == 1

    @pytest.mark.asyncio
    async def test_exception_propagates(self, service):
        service.vectors_store.get = MagicMock(side_effect=RuntimeError("x"))
        with pytest.raises(RuntimeError):
            await service.get_document_detail("u1", "a.txt")


class TestGetDocumentChunks:
    @pytest.mark.asyncio
    async def test_returns_chunks(self, service):
        service.vectors_store.get = MagicMock(return_value={
            "ids": ["id1", "id2"],
            "metadatas": [
                {"source": "a.txt", "user_id": "u1"},
                {"source": "a.txt", "user_id": "u1"},
            ],
            "documents": ["c1", "c2"],
        })
        result = await service.get_document_chunks("u1", "a.txt")
        assert result["filename"] == "a.txt"
        assert result["total_chunks"] == 2
        assert len(result["chunks"]) == 2

    @pytest.mark.asyncio
    async def test_no_match(self, service):
        service.vectors_store.get = MagicMock(return_value={"ids": [], "metadatas": [], "documents": []})
        result = await service.get_document_chunks("u1", "nope.txt")
        assert result["total_chunks"] == 0
        assert result["chunks"] == []

    @pytest.mark.asyncio
    async def test_exception_propagates(self, service):
        service.vectors_store.get = MagicMock(side_effect=RuntimeError("x"))
        with pytest.raises(RuntimeError):
            await service.get_document_chunks("u1", "a.txt")


# ---------------------------------------------------------------------------
# File document helpers
# ---------------------------------------------------------------------------


class TestGetFileDocument:
    @pytest.mark.asyncio
    async def test_async_delegates(self, service):
        service.document_processor.get_file_document = AsyncMock(return_value=["doc"])
        result = await service.get_file_document("/path/to/file")
        assert result == ["doc"]

    def test_sync_delegates(self, service):
        service.document_processor.get_file_document_sync = MagicMock(return_value=["doc"])
        result = service.get_file_document_sync("/path/to/file")
        assert result == ["doc"]


class TestSplitDocumentsSync:
    def test_delegates(self, service):
        service.document_processor.split_documents_sync = MagicMock(return_value=["d1", "d2"])
        result = service.split_documents_sync(["d1", "d2"])
        assert result == ["d1", "d2"]


class TestGetDocument:
    @pytest.mark.asyncio
    async def test_delegates(self, service):
        service.document_processor.get_document = AsyncMock()
        await service.get_document(files=["f"], user_id="u1", is_public=False)
        service.document_processor.get_document.assert_awaited_once()


# ---------------------------------------------------------------------------
# _create_chroma_client
# ---------------------------------------------------------------------------


class TestCreateChromaClient:
    def test_returns_client(self):
        import sys
        import types
        # Inject a fake chromadb module that exposes PersistentClient
        fake_chromadb = types.ModuleType("chromadb")
        setattr(fake_chromadb, "PersistentClient", MagicMock(return_value="client"))
        sys.modules["chromadb"] = fake_chromadb
        try:
            from app.rag.vector_store import VectorStoreService
            result = VectorStoreService._create_chroma_client("/tmp/dir")
        finally:
            del sys.modules["chromadb"]
        assert result == "client"

    def test_retries_on_keyerror(self):
        import sys
        import types
        fake_chromadb = types.ModuleType("chromadb")
        persistent_mock = MagicMock(side_effect=[KeyError, KeyError, "client"])
        setattr(fake_chromadb, "PersistentClient", persistent_mock)
        sys.modules["chromadb"] = fake_chromadb
        try:
            from app.rag.vector_store import VectorStoreService
            result = VectorStoreService._create_chroma_client("/tmp/dir")
        finally:
            del sys.modules["chromadb"]
        assert result == "client"
        assert persistent_mock.call_count == 3

    def test_raises_after_three_keyerrors(self):
        import sys
        import types
        fake_chromadb = types.ModuleType("chromadb")
        persistent_mock = MagicMock(side_effect=KeyError("x"))
        setattr(fake_chromadb, "PersistentClient", persistent_mock)
        sys.modules["chromadb"] = fake_chromadb
        try:
            from app.rag.vector_store import VectorStoreService
            with pytest.raises(KeyError):
                VectorStoreService._create_chroma_client("/tmp/dir")
        finally:
            del sys.modules["chromadb"]
