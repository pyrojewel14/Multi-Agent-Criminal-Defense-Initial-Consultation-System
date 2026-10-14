import json
import threading

from langchain_core.documents import Document

from app.knowledge.documents.events import SliceResult, SSEEvent
from app.knowledge.documents.tasks import TaskQueue


class TestSSEEvent:
    def test_to_sse_format(self):
        event = SSEEvent(event_type="response", message="hello")
        sse_str = event.to_sse()
        assert sse_str.startswith("event: progress\ndata: ")
        assert sse_str.endswith("\n\n")
        data_part = sse_str[len("event: progress\ndata: ") : -2]
        payload = json.loads(data_part)
        assert payload["event_type"] == "response"
        assert payload["message"] == "hello"

    def test_to_sse_omits_none_fields(self):
        event = SSEEvent(event_type="error", message="fail", file_index=None, filename=None)
        payload_str = event.to_sse()
        data_part = payload_str[len("event: progress\ndata: ") : -2]
        payload = json.loads(data_part)
        assert "file_index" not in payload
        assert "filename" not in payload

    def test_to_sse_includes_non_none_optional_fields(self):
        event = SSEEvent(event_type="response", message="ok", file_index=2, filename="test.pdf")
        data_part = event.to_sse()[len("event: progress\ndata: ") : -2]
        payload = json.loads(data_part)
        assert payload["file_index"] == 2
        assert payload["filename"] == "test.pdf"


# ---------------------------------------------------------------------------
# SliceResult.success_result / error_result
# ---------------------------------------------------------------------------


class TestSliceResult:
    def test_success_result(self):
        docs = [Document(page_content="doc1"), Document(page_content="doc2")]
        result = SliceResult.success_result(file_index=1, filename="a.txt", documents=docs, md5="abc123")
        assert result.file_index == 1
        assert result.filename == "a.txt"
        assert result.documents == docs
        assert result.md5 == "abc123"
        assert result.success is True
        assert result.chunk_count == 2
        assert result.error is None

    def test_error_result(self):
        result = SliceResult.error_result(file_index=3, filename="b.txt", error="parse failed")
        assert result.file_index == 3
        assert result.filename == "b.txt"
        assert result.success is False
        assert result.error == "parse failed"
        assert result.documents == []
        assert result.chunk_count == 0

    def test_to_dict(self):
        docs = [Document(page_content="x")]
        result = SliceResult.success_result(file_index=0, filename="c.txt", documents=docs, md5="d4e5")
        d = result.to_dict()
        assert d["success"] is True
        assert d["chunk_count"] == 1
        assert d["md5"] == "d4e5"


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------


class TestTaskQueue:
    def test_put_and_get(self):
        q = TaskQueue()
        q.put("item1")
        q.put("item2")
        assert q.get(timeout=1) == "item1"
        assert q.get(timeout=1) == "item2"

    def test_task_done_and_completed_count(self):
        q = TaskQueue()
        q.set_total_count(2)
        q.put("a")
        q.put("b")
        q.get(timeout=1)
        q.task_done()
        assert q.get_completed_count() == 1
        q.get(timeout=1)
        q.task_done()
        assert q.get_completed_count() == 2

    def test_is_finished(self):
        q = TaskQueue()
        q.set_total_count(1)
        assert q.is_finished() is False
        q.put("x")
        q.get(timeout=1)
        q.task_done()
        assert q.is_finished() is False  # not set_finished yet
        q.set_finished()
        assert q.is_finished() is True

    def test_is_finished_before_set_finished(self):
        q = TaskQueue()
        q.set_total_count(1)
        q.put("x")
        q.get(timeout=1)
        q.task_done()
        # completed_count == total_count but _finished is False
        assert q.is_finished() is False

    def test_thread_safety(self):
        q = TaskQueue(maxsize=10)
        q.set_total_count(5)
        results = []

        def producer():
            for i in range(5):
                q.put(f"item-{i}")

        def consumer():
            for _ in range(5):
                item = q.get(timeout=2)
                results.append(item)
                q.task_done()

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert len(results) == 5
        assert q.get_completed_count() == 5

    def test_empty_and_full(self):
        q = TaskQueue(maxsize=1)
        assert q.empty() is True
        q.put("x")
        assert q.full() is True
        assert q.empty() is False


# ---------------------------------------------------------------------------
# RerankerFactory.register / create
