"""Regression tests for knowledge upload metadata validation."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

from app.v1.service import knowledge_service
from app.v1.service.knowledge_service import KnowledgeService


@pytest.mark.asyncio
async def test_single_upload_rejects_missing_filename():
    upload = UploadFile(file=BytesIO(b"content"), filename=None, size=None)

    with patch.object(knowledge_service, "get_vector_store", return_value=MagicMock()):
        with pytest.raises(HTTPException, match="文件名不能为空") as exc_info:
            await KnowledgeService().handle_add_vector_single(upload, "user-1")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_single_upload_accepts_unknown_size_when_content_is_within_limit():
    upload = UploadFile(file=BytesIO(b"content"), filename="case.txt", size=None)
    store = MagicMock()
    store.get_document = AsyncMock()
    mime = MagicMock()
    mime.from_buffer.return_value = "text/plain"

    with patch.object(knowledge_service, "get_vector_store", return_value=store), patch.object(
        knowledge_service.magic, "Magic", return_value=mime
    ):
        result = await KnowledgeService().handle_add_vector_single(upload, "user-1")

    assert result == "case.txt"
    store.get_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_upload_checks_content_length_when_reported_size_is_unknown():
    upload = UploadFile(file=BytesIO(b"1234"), filename="case.txt", size=None)

    with patch.object(knowledge_service, "MAX_FILE_SIZE", 3), patch.object(
        knowledge_service, "get_vector_store", return_value=MagicMock()
    ):
        with pytest.raises(HTTPException, match="文件大小不能超过") as exc_info:
            await KnowledgeService().handle_add_vector_single(upload, "user-1")

    assert exc_info.value.status_code == 400
