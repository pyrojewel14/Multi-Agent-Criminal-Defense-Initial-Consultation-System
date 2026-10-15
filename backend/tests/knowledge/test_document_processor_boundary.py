"""上传准备在线程池调用同步 splitter，不能把 coroutine 当分块写入。"""

import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from app.knowledge.documents.processor import DocumentProcessor
from app.knowledge.documents.service import DocumentService


@pytest.mark.asyncio
async def test_async_upload_boundary_uses_sync_splitter_contract():
    caller_thread = threading.get_ident()
    worker_threads = []
    expected = [Document(page_content="split")]
    splitter = MagicMock()
    splitter.split_documents = AsyncMock(side_effect=AssertionError("async splitter cannot run in to_thread"))

    def split_sync(documents):
        worker_threads.append(threading.get_ident())
        return expected

    splitter.split_documents_sync.side_effect = split_sync
    processor = DocumentProcessor(splitter)
    processor.get_file_document_sync = MagicMock(return_value=[Document(page_content="source")])
    service = object.__new__(DocumentService)
    service.processor = processor
    service._write_documents = AsyncMock()
    await service._ingest_content(b"source", "case.txt", "alice", False)
    splitter.split_documents_sync.assert_called_once()
    splitter.split_documents.assert_not_called()
    assert worker_threads and worker_threads[0] != caller_thread
    documents = service._write_documents.await_args.args[0]
    assert [doc.page_content for doc in documents] == ["split"]
    assert documents[0].metadata["user_id"] == "alice"
    assert not Path(processor.get_file_document_sync.call_args.args[0]).exists()
