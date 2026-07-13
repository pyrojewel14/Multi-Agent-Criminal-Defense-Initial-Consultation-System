"""Regression tests for document splitting boundaries."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.document_handler.processor import DocumentProcessor


@pytest.mark.asyncio
async def test_async_split_boundary_uses_sync_splitter_contract():
    """The processor must not send an async splitter method to ``to_thread``."""
    processor = object.__new__(DocumentProcessor)
    expected = [Document(page_content="split")]
    processor.spliter = MagicMock()
    processor.spliter.split_documents_sync.return_value = expected

    split_documents = getattr(processor, "_split_documents", None)
    assert split_documents is not None

    result = await split_documents([Document(page_content="source")])

    assert result == expected
    processor.spliter.split_documents_sync.assert_called_once()
