import json
import threading

import pytest
from langchain_core.documents import Document

from app.rag.legal_text_splitter import LegalArticleSplitter
from app.rag.retrievers.empty_retriever import EmptyRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.reranker.base import BaseReranker, RerankerConfig
from app.rag.reranker.factory import RerankerFactory
from app.rag.rag_service import _deduplicate_documents
from app.rag.sse_models import SSEEvent, SliceResult
from app.rag.task_queue import TaskQueue


# ---------------------------------------------------------------------------
# LegalArticleSplitter._is_structured_legal_doc
# ---------------------------------------------------------------------------

class TestIsStructuredLegalDoc:
    def setup_method(self):
        self.splitter = LegalArticleSplitter()

    def test_structured_json_list_with_law_text(self):
        content = json.dumps([{"law_text": "第一条 ...", "article": "第一条"}])
        doc = Document(page_content=content, metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is True

    def test_structured_json_dict_with_law_text(self):
        content = json.dumps({"law_text": "第一条 ...", "article": "第一条"})
        doc = Document(page_content=content, metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is True

    def test_source_type_legal_json(self):
        doc = Document(page_content="anything", metadata={"source_type": "legal_json"})
        assert self.splitter._is_structured_legal_doc(doc) is True

    def test_plain_text_not_structured(self):
        doc = Document(page_content="这是一段普通法律文本，没有JSON结构。", metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is False

    def test_json_without_law_text_not_structured(self):
        content = json.dumps([{"title": "标题", "body": "内容"}])
        doc = Document(page_content=content, metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is False

    def test_invalid_json_not_structured(self):
        doc = Document(page_content="{invalid json", metadata={})
        assert self.splitter._is_structured_legal_doc(doc) is False


# ---------------------------------------------------------------------------
# LegalArticleSplitter._extract_legal_metadata
# ---------------------------------------------------------------------------

class TestExtractLegalMetadata:
    def setup_method(self):
        self.splitter = LegalArticleSplitter(enable_metadata_extraction=True)

    def test_extract_article_index(self):
        text = "根据刑法第232条的规定……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["article_index"] == 232

    def test_extract_charge_from_parentheses(self):
        text = "故意杀人罪（故意杀人）处死刑……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["charge"] == "故意杀人"

    def test_existing_charge_not_overwritten(self):
        text = "故意伤害罪（故意伤害）……"
        metadata = {"charge": "故意杀人罪"}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["charge"] == "故意杀人罪"

    def test_death_related_category(self):
        text = "致人死亡的，处十年以上有期徒刑。"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "death_related"

    def test_injury_related_category(self):
        text = "致人重伤的，处三年以上十年以下有期徒刑。"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "injury_related"

    def test_property_crime_category(self):
        text = "盗窃公私财物，数额较大的……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "property_crime"

    def test_danger_related_category(self):
        text = "危害公共安全的……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "danger_related"

    def test_general_category(self):
        text = "国家工作人员利用职务上的便利……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["legal_category"] == "general"

    def test_mitigated_article_type(self):
        text = "情节较轻的，可以从轻处罚。"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["article_type"] == "mitigated"

    def test_default_article_type_is_standard(self):
        text = "国家工作人员利用职务上的便利……"
        metadata = {}
        result = self.splitter._extract_legal_metadata(text, metadata)
        assert result["article_type"] == "standard"


# ---------------------------------------------------------------------------
# HybridRetriever.get_dynamic_weights
# ---------------------------------------------------------------------------

class TestGetDynamicWeights:
    def test_no_query_returns_default(self):
        weights = HybridRetriever.get_dynamic_weights(None)
        assert weights == [0.8, 0.2]

    def test_empty_query_returns_default(self):
        weights = HybridRetriever.get_dynamic_weights("")
        assert weights == [0.8, 0.2]

    def test_short_query_favors_bm25(self):
        weights = HybridRetriever.get_dynamic_weights("防卫过当")
        assert weights[0] == 0.3
        assert weights[1] == 0.7

    def test_medium_query(self):
        weights = HybridRetriever.get_dynamic_weights("a" * 30)
        assert weights[0] == 0.6
        assert weights[1] == 0.4

    def test_long_query_favors_vector(self):
        weights = HybridRetriever.get_dynamic_weights("a" * 100)
        assert weights[0] == 0.8
        assert weights[1] == 0.2

    def test_very_long_query(self):
        weights = HybridRetriever.get_dynamic_weights("a" * 250)
        assert weights[0] == 0.9
        assert weights[1] == 0.1

    def test_weights_sum_to_one(self):
        for query in [None, "", "短", "a" * 30, "a" * 100, "a" * 250]:
            weights = HybridRetriever.get_dynamic_weights(query)
            assert abs(sum(weights) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# EmptyRetriever
# ---------------------------------------------------------------------------

class TestEmptyRetriever:
    def test_get_relevant_documents_returns_empty(self):
        retriever = EmptyRetriever()
        result = retriever._get_relevant_documents("test query")
        assert result == []

    async def test_aget_relevant_documents_returns_empty(self):
        retriever = EmptyRetriever()
        result = await retriever._aget_relevant_documents("test query")
        assert result == []


# ---------------------------------------------------------------------------
# SSEEvent.to_sse
# ---------------------------------------------------------------------------

class TestSSEEvent:
    def test_to_sse_format(self):
        event = SSEEvent(event_type="response", message="hello")
        sse_str = event.to_sse()
        assert sse_str.startswith("event: progress\ndata: ")
        assert sse_str.endswith("\n\n")
        data_part = sse_str[len("event: progress\ndata: "):-2]
        payload = json.loads(data_part)
        assert payload["event_type"] == "response"
        assert payload["message"] == "hello"

    def test_to_sse_omits_none_fields(self):
        event = SSEEvent(event_type="error", message="fail", file_index=None, filename=None)
        payload_str = event.to_sse()
        data_part = payload_str[len("event: progress\ndata: "):-2]
        payload = json.loads(data_part)
        assert "file_index" not in payload
        assert "filename" not in payload

    def test_to_sse_includes_non_none_optional_fields(self):
        event = SSEEvent(event_type="response", message="ok", file_index=2, filename="test.pdf")
        data_part = event.to_sse()[len("event: progress\ndata: "):-2]
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
# ---------------------------------------------------------------------------

class TestRerankerFactory:
    def setup_method(self):
        # Save original registry to restore after each test
        self._original_registry = dict(RerankerFactory._registry)

    def teardown_method(self):
        RerankerFactory._registry = self._original_registry

    def test_register_and_create(self):
        class MockReranker(BaseReranker):
            async def _load_model(self):
                pass

            async def _format_pairs(self, query, documents):
                return []

            async def _compute_scores(self, pairs):
                return []

        RerankerFactory.register("mock", MockReranker)
        config = RerankerConfig(model_name="test-model")
        instance = RerankerFactory.create(config=config, reranker_type="mock")
        assert isinstance(instance, MockReranker)
        assert instance.config.model_name == "test-model"

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError, match="未知的重排序器类型"):
            RerankerFactory.create(reranker_type="nonexistent_type")


# ---------------------------------------------------------------------------
# _deduplicate_documents
# ---------------------------------------------------------------------------

class TestDeduplicateDocuments:
    def test_removes_duplicate_content(self):
        doc1 = Document(page_content="刑法第232条的规定")
        doc2 = Document(page_content="刑法第232条的规定")  # duplicate
        doc3 = Document(page_content="刑法第233条的规定")
        result = _deduplicate_documents([doc1, doc2, doc3])
        assert len(result) == 2
        assert result[0].page_content == "刑法第232条的规定"
        assert result[1].page_content == "刑法第233条的规定"

    def test_all_unique(self):
        docs = [Document(page_content="内容A"), Document(page_content="内容B")]
        result = _deduplicate_documents(docs)
        assert len(result) == 2

    def test_empty_list(self):
        result = _deduplicate_documents([])
        assert result == []

    def test_all_duplicates(self):
        docs = [Document(page_content="相同内容"), Document(page_content="相同内容"), Document(page_content="相同内容")]
        result = _deduplicate_documents(docs)
        assert len(result) == 1

    def test_dedup_uses_first_200_chars(self):
        long_text_a = "A" * 300
        long_text_b = "A" * 300
        doc_a = Document(page_content=long_text_a)
        doc_b = Document(page_content=long_text_b)
        result = _deduplicate_documents([doc_a, doc_b])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# HybridRetriever.get_bm25_retriever / get_retriever / _get_all_documents
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock, patch


def _make_hybrid(vectors_store=None):
    if vectors_store is None:
        vectors_store = MagicMock()
    return HybridRetriever(vectors_store), vectors_store


class TestGetBm25Retriever:
    @pytest.mark.asyncio
    async def test_no_user_no_public_returns_none(self):
        hr, _ = _make_hybrid()
        assert await hr.get_bm25_retriever(user_id=None, include_public=False) is None

    @pytest.mark.asyncio
    async def test_user_with_no_documents_returns_none(self):
        hr, vs = _make_hybrid()
        vs.get = MagicMock(return_value={"documents": [], "metadatas": []})
        assert await hr.get_bm25_retriever(user_id="u1", include_public=False) is None

    @pytest.mark.asyncio
    async def test_user_with_documents_returns_bm25(self):
        hr, vs = _make_hybrid()
        vs.get = MagicMock(return_value={
            "documents": ["doc1", "doc2"],
            "metadatas": [{"user_id": "u1"}, {"user_id": "u1"}],
        })
        with patch("app.rag.retrievers.hybrid_retriever.BM25Retriever") as bm25:
            bm25.from_documents = MagicMock(return_value="bm25-instance")
            result = await hr.get_bm25_retriever(user_id="u1", include_public=False)
        assert result == "bm25-instance"
        bm25.from_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_with_include_public_uses_scoped_filter(self):
        hr, vs = _make_hybrid()
        vs.get = MagicMock(return_value={"documents": ["doc"], "metadatas": [{"user_id": "u1"}]})
        with patch("app.rag.retrievers.hybrid_retriever.BM25Retriever") as bm25:
            bm25.from_documents = MagicMock(return_value="bm25-instance")
            await hr.get_bm25_retriever(user_id="u1", include_public=True)
        args, kwargs = vs.get.call_args
        assert kwargs["where"] == {"$or": [{"user_id": "u1"}, {"is_public": True}]}

    @pytest.mark.asyncio
    async def test_only_public_uses_is_public_filter(self):
        hr, vs = _make_hybrid()
        vs.get = MagicMock(return_value={"documents": ["doc"], "metadatas": [{"is_public": True}]})
        with patch("app.rag.retrievers.hybrid_retriever.BM25Retriever") as bm25:
            bm25.from_documents = MagicMock(return_value="bm25-instance")
            await hr.get_bm25_retriever(user_id=None, include_public=True)
        args, kwargs = vs.get.call_args
        assert kwargs["where"] == {"is_public": True}

    @pytest.mark.asyncio
    async def test_dedup_repeated_content(self):
        hr, vs = _make_hybrid()
        vs.get = MagicMock(return_value={
            "documents": ["same", "same", "different"],
            "metadatas": [{"id": 1}, {"id": 2}, {"id": 3}],
        })
        with patch("app.rag.retrievers.hybrid_retriever.BM25Retriever") as bm25:
            bm25.from_documents = MagicMock(return_value="bm25")
            await hr.get_bm25_retriever(user_id="u1")
        # The bm25 retriever should have been built with only 2 unique docs
        args, kwargs = bm25.from_documents.call_args
        assert len(kwargs["documents"]) == 2


class TestGetRetriever:
    @pytest.mark.asyncio
    async def test_no_user_no_public_returns_empty(self):
        from app.rag.retrievers.empty_retriever import EmptyRetriever

        hr, vs = _make_hybrid()
        result = await hr.get_retriever(user_id=None, include_public=False)
        assert isinstance(result, EmptyRetriever)

    @pytest.mark.asyncio
    async def test_user_only_short_query_returns_ensemble(self):
        hr, vs = _make_hybrid()
        # Pretend we have at least one doc
        vs.get = MagicMock(return_value={"documents": ["d1"], "metadatas": [{"user_id": "u1"}]})
        with patch("app.rag.retrievers.hybrid_retriever.BM25Retriever") as bm25, \
             patch("app.rag.retrievers.hybrid_retriever.EnsembleRetriever") as ens:
            bm25.from_documents = MagicMock(return_value="bm25")
            ens.return_value = "ensemble"
            result = await hr.get_retriever(query="短的查询", user_id="u1", include_public=False)
        assert result == "ensemble"
        ens.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_only_long_query_returns_vector_retriever(self):
        hr, vs = _make_hybrid()
        long_query = "x" * 250
        result = await hr.get_retriever(query=long_query, user_id="u1", include_public=False)
        # Should NOT use EnsembleRetriever when query is long
        vs.as_retriever.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_only_no_bm25_returns_vector_retriever(self):
        hr, vs = _make_hybrid()
        # No docs => no BM25
        vs.get = MagicMock(return_value={"documents": [], "metadatas": []})
        with patch("app.rag.retrievers.hybrid_retriever.BM25Retriever") as bm25:
            bm25.from_documents = MagicMock()
            result = await hr.get_retriever(query="q", user_id="u1", include_public=False)
        # Should fall back to vector retriever when BM25 is None
        assert result is not None
        bm25.from_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_query_includes_filter(self):
        hr, vs = _make_hybrid()
        long_query = "x" * 250
        await hr.get_retriever(query=long_query, user_id="u1", include_public=False)
        # Inspect the search_kwargs passed to as_retriever
        args, kwargs = vs.as_retriever.call_args
        assert "filter" in kwargs["search_kwargs"]
        assert kwargs["search_kwargs"]["filter"] == {"user_id": "u1"}


class TestGetAllDocuments:
    @pytest.mark.asyncio
    async def test_returns_documents(self):
        hr, vs = _make_hybrid()
        vs.get = MagicMock(return_value={
            "documents": ["d1", "d2"],
            "metadatas": [{"id": 1}, {"id": 2}],
        })
        result = await hr._get_all_documents()
        assert len(result) == 2
        assert result[0].page_content == "d1"
        assert result[1].page_content == "d2"
