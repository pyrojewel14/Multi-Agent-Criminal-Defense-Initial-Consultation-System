"""Knowledge router endpoint tests.

Tests cover the admin knowledge management endpoints:
1. POST /api/v1/knowledge/add/single - Upload single file
2. POST /api/v1/knowledge/add/multiple - Upload multiple files
3. POST /api/v1/knowledge/add/multiple/stream - Stream upload progress
4. DELETE /api/v1/knowledge/clean - Clean all vectors
5. DELETE /api/v1/knowledge/md5/clear - Clear all MD5
6. DELETE /api/v1/knowledge/md5/delete/{md5} - Delete single MD5
7. DELETE /api/v1/knowledge/delete/filename - Delete by filename
8. GET /api/v1/knowledge/md5/list - List MD5 records
9. GET /api/v1/knowledge/md5/{md5} - Get MD5 info
10. GET /api/v1/knowledge/list - List documents
11. GET /api/v1/knowledge/detail - Get document detail
12. GET /api/v1/knowledge/chunks - Get document chunks
"""

import json
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


KNOWLEDGE_PREFIX = "/api/v1/knowledge"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_md5_record(md5="abc123", filename="test.pdf", is_public=False):
    return {
        "md5": md5,
        "filename": filename,
        "original_filename": filename,
        "is_public": is_public,
        "user_id": "admin-001",
        "created_at": "2026-01-01T00:00:00",
        "chunk_count": 10,
    }


def _make_knowledge_document(**overrides):
    doc = {
        "id": "doc-001",
        "filename": "test.pdf",
        "original_filename": "test.pdf",
        "user_id": "admin-001",
        "chunk_count": 10,
        "preview": "first chunk content...",
        "created_at": "2026-01-01T00:00:00",
    }
    doc.update(overrides)
    return doc


def _make_doc_detail(filename="test.pdf"):
    return {
        "id": "doc-001",
        "filename": filename,
        "user_id": "admin-001",
        "chunk_count": 5,
        "content": "full document content...",
        "created_at": "2026-01-01T00:00:00",
    }


def _make_doc_chunks(filename="test.pdf"):
    return {
        "filename": filename,
        "total_chunks": 5,
        "chunks": [
            {"index": 0, "content": "chunk 1", "metadata": {}},
            {"index": 1, "content": "chunk 2", "metadata": {}},
        ],
    }


@pytest.fixture
async def admin_app(test_app, monkeypatch):
    """Test app with ``require_admin`` overridden to a fixed admin user."""
    import app.security.rbac as rbac_module
    import app.v1.router.knowledge_router as knowledge_module
    import app.core.rate_limit as rate_limit_module

    async def _admin_override():
        return {"user_id": "admin-001", "role": "admin"}

    monkeypatch.setattr(knowledge_module, "require_admin", _admin_override, raising=False)
    test_app.dependency_overrides[rbac_module.require_admin] = _admin_override

    # Bypass rate limiter by overriding the rate_limit module's dep factory
    # so the generated dependency is a no-op.
    async def _noop(request):
        return None

    def _noop_factory(limit=None, window=None):
        return _noop

    monkeypatch.setattr(rate_limit_module, "rate_limit", _noop_factory)
    # Also patch the reference inside the knowledge module (since the Depends
    # was evaluated at import time)
    monkeypatch.setattr(knowledge_module, "rate_limit", _noop_factory, raising=False)

    yield test_app


@pytest.fixture
def mock_knowledge_service():
    """Build a mock KnowledgeService that records calls."""
    service = MagicMock()
    service.handle_add_vector_single = AsyncMock(return_value="uploaded.pdf")
    service.handle_add_vector_multiple = AsyncMock(return_value=["file1.pdf", "file2.pdf"])

    async def _stream(*args, **kwargs):
        yield "event: start\ndata: {\"event_type\": \"start\"}\n\n"
        yield "event: finish\ndata: {\"event_type\": \"finish\"}\n\n"
    service.handle_add_vector_multiple_stream = _stream

    service.clean_all = AsyncMock()
    service.handle_clear_all_md5 = AsyncMock()
    service.handle_delete_single_md5 = AsyncMock(return_value=True)
    service.handle_delete_by_filename = AsyncMock(return_value=True)
    service.handle_get_md5_info = AsyncMock(return_value=_make_md5_record())
    service.handle_get_all_md5_records = AsyncMock(return_value=[_make_md5_record()])
    service.handle_get_user_knowledge = AsyncMock(return_value=[_make_knowledge_document()])
    service.handle_get_document_detail = AsyncMock(return_value=_make_doc_detail())
    service.handle_get_document_chunks = AsyncMock(return_value=_make_doc_chunks())
    return service


# ---------------------------------------------------------------------------
# POST /knowledge/add/single
# ---------------------------------------------------------------------------


class TestAddVectorSingle:
    @pytest.mark.asyncio
    async def test_add_single_success(self, admin_app, mock_knowledge_service):
        admin_app.dependency_overrides[
            __import__("app.v1.service.knowledge_service", fromlist=["get_knowledge_service"]).get_knowledge_service
        ] = lambda: mock_knowledge_service

        # Build a multipart upload
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        files = {"file": ("test.pdf", io.BytesIO(b"PDF-content"), "application/pdf")}

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{KNOWLEDGE_PREFIX}/add/single",
                files=files,
                params={"is_public": "false"},
            )

            assert response.status_code == 200
            mock_knowledge_service.handle_add_vector_single.assert_awaited()


# ---------------------------------------------------------------------------
# POST /knowledge/add/multiple
# ---------------------------------------------------------------------------


class TestAddVectorMultiple:
    @pytest.mark.asyncio
    async def test_add_multiple_success(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        files = [
            ("files", ("file1.pdf", io.BytesIO(b"PDF-1"), "application/pdf")),
            ("files", ("file2.pdf", io.BytesIO(b"PDF-2"), "application/pdf")),
        ]

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.post(
                f"{KNOWLEDGE_PREFIX}/add/multiple",
                files=files,
                params={"is_public": "true"},
            )

            assert response.status_code == 200
            mock_knowledge_service.handle_add_vector_multiple.assert_awaited()


# ---------------------------------------------------------------------------
# DELETE /knowledge/clean
# ---------------------------------------------------------------------------


class TestCleanAllVectors:
    @pytest.mark.asyncio
    async def test_clean_all_success(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(f"{KNOWLEDGE_PREFIX}/clean")

            assert response.status_code == 200
            mock_knowledge_service.clean_all.assert_awaited()


# ---------------------------------------------------------------------------
# DELETE /knowledge/md5/clear
# ---------------------------------------------------------------------------


class TestClearAllMd5:
    @pytest.mark.asyncio
    async def test_clear_all_md5_with_documents(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(
                f"{KNOWLEDGE_PREFIX}/md5/clear?delete_documents=true"
            )

            assert response.status_code == 200
            assert "MD5 记录和知识库文档" in response.json()["message"]
            mock_knowledge_service.handle_clear_all_md5.assert_awaited_with(True)

    @pytest.mark.asyncio
    async def test_clear_all_md5_without_documents(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(
                f"{KNOWLEDGE_PREFIX}/md5/clear?delete_documents=false"
            )

            assert response.status_code == 200
            assert "保留知识库文档" in response.json()["message"]
            mock_knowledge_service.handle_clear_all_md5.assert_awaited_with(False)


# ---------------------------------------------------------------------------
# DELETE /knowledge/md5/delete/{md5}
# ---------------------------------------------------------------------------


class TestDeleteSingleMd5:
    @pytest.mark.asyncio
    async def test_delete_md5_with_documents(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(
                f"{KNOWLEDGE_PREFIX}/md5/delete/abc123?delete_documents=true"
            )

            assert response.status_code == 200
            assert "abc123" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_delete_md5_without_documents(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(
                f"{KNOWLEDGE_PREFIX}/md5/delete/abc123?delete_documents=false"
            )

            assert response.status_code == 200
            assert "保留知识库文档" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_delete_md5_not_found(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        mock_knowledge_service.handle_delete_single_md5 = AsyncMock(return_value=False)
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(
                f"{KNOWLEDGE_PREFIX}/md5/delete/nonexistent"
            )

            assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /knowledge/delete/filename
# ---------------------------------------------------------------------------


class TestDeleteByFilename:
    @pytest.mark.asyncio
    async def test_delete_by_filename_with_documents(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(
                f"{KNOWLEDGE_PREFIX}/delete/filename?filename=test.pdf&delete_documents=true"
            )

            assert response.status_code == 200
            assert "test.pdf" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_delete_by_filename_without_documents(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(
                f"{KNOWLEDGE_PREFIX}/delete/filename?filename=test.pdf&delete_documents=false"
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_by_filename_not_found(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        mock_knowledge_service.handle_delete_by_filename = AsyncMock(return_value=False)
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.delete(
                f"{KNOWLEDGE_PREFIX}/delete/filename?filename=nonexistent.pdf"
            )

            assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /knowledge/md5/list
# ---------------------------------------------------------------------------


class TestGetAllMd5Records:
    @pytest.mark.asyncio
    async def test_get_all_md5_records(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{KNOWLEDGE_PREFIX}/md5/list")

            assert response.status_code == 200
            data = response.json()
            assert data["data"]["total_count"] == 1
            assert len(data["data"]["records"]) == 1


# ---------------------------------------------------------------------------
# GET /knowledge/md5/{md5}
# ---------------------------------------------------------------------------


class TestGetMd5Info:
    @pytest.mark.asyncio
    async def test_get_md5_info_success(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{KNOWLEDGE_PREFIX}/md5/abc123")

            assert response.status_code == 200
            assert response.json()["data"]["md5"] == "abc123"

    @pytest.mark.asyncio
    async def test_get_md5_info_not_found(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        mock_knowledge_service.handle_get_md5_info = AsyncMock(return_value=None)
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{KNOWLEDGE_PREFIX}/md5/missing")

            assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /knowledge/list
# ---------------------------------------------------------------------------


class TestGetAllKnowledgeList:
    @pytest.mark.asyncio
    async def test_get_all_knowledge_list(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{KNOWLEDGE_PREFIX}/list")

            assert response.status_code == 200
            data = response.json()
            assert data["data"]["total_count"] == 1


# ---------------------------------------------------------------------------
# GET /knowledge/detail
# ---------------------------------------------------------------------------


class TestGetDocumentDetail:
    @pytest.mark.asyncio
    async def test_get_document_detail_success(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{KNOWLEDGE_PREFIX}/detail?filename=test.pdf")

            assert response.status_code == 200
            assert response.json()["data"]["filename"] == "test.pdf"


# ---------------------------------------------------------------------------
# GET /knowledge/chunks
# ---------------------------------------------------------------------------


class TestGetDocumentChunks:
    @pytest.mark.asyncio
    async def test_get_document_chunks_success(self, admin_app, mock_knowledge_service):
        from app.v1.router.knowledge_router import get_knowledge_service
        admin_app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

        async with AsyncClient(transport=ASGITransport(app=admin_app), base_url="http://test") as client_http:
            response = await client_http.get(f"{KNOWLEDGE_PREFIX}/chunks?filename=test.pdf")

            assert response.status_code == 200
            assert response.json()["data"]["total_chunks"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
