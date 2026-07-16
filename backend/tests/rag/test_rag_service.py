"""Unit tests for ``app.rag.rag_service.RagService``.

All external dependencies (vector store, retriever, reranker, chat model) are
mocked. Tests focus on the orchestration logic: HyDE generation, document
retrieval, reordering, summarization, and the empty-result / error paths.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.rag_service import RagService, _configure_hyde_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(content: str, source: str = "src.txt") -> Document:
    return Document(page_content=content, metadata={"source": source, "original_filename": source})


def _make_service(user_id: str | None = "u1", include_public=True, thinking_callback=None):
    """Build a RagService instance with every external dependency mocked."""
    with patch("app.rag.rag_service.get_vector_store") as get_vs, \
         patch("app.rag.rag_service.get_chat_model", return_value=MagicMock()), \
         patch("app.rag.rag_service.prompt_loader") as pl:
        pl.load.return_value = "summary-prompt"
        # Mock vector store service
        vs = MagicMock()
        vs.get_dynamic_weights.return_value = [0.8, 0.2]
        retriever = AsyncMock()
        retriever.ainvoke = AsyncMock(return_value=[])
        vs.get_retriever = AsyncMock(return_value=retriever)
        get_vs.return_value = vs
        svc = RagService(
            user_id=user_id,
            include_public=include_public,
            thinking_callback=thinking_callback,
        )
    return svc, vs, retriever


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_builds_chain(self):
        svc, _, _ = _make_service()
        assert svc.chain is not None
        assert svc.hyde_prompt_template is not None
        assert svc.user_id == "u1"
        assert svc.include_public is True
        assert svc.retriever is None

    def test_init_without_thinking_callback(self):
        svc, _, _ = _make_service()
        assert svc.thinking_callback is None

    def test_local_ollama_hyde_model_has_bounded_output(self):
        class LocalOllamaModel:
            __module__ = "langchain_ollama.chat_models"

            def __init__(self):
                self.model_copy = MagicMock(return_value="bounded-model")

        model = LocalOllamaModel()

        assert _configure_hyde_model(model) == "bounded-model"
        model.model_copy.assert_called_once_with(update={"reasoning": False, "num_predict": 64})


# ---------------------------------------------------------------------------
# initialize_retriever
# ---------------------------------------------------------------------------


class TestInitializeRetriever:
    @pytest.mark.asyncio
    async def test_creates_retriever_when_none(self):
        svc, vs, retriever = _make_service()
        await svc.initialize_retriever("query")
        vs.get_retriever.assert_awaited_once()
        assert svc.retriever is retriever

    @pytest.mark.asyncio
    async def test_idempotent_when_already_initialized(self):
        svc, vs, _ = _make_service()
        # pre-populate retriever
        svc.retriever = MagicMock()
        await svc.initialize_retriever("query")
        vs.get_retriever.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_thinking_callback_invoked(self):
        calls = []

        async def cb(payload):
            calls.append(payload)

        svc, _, _ = _make_service(thinking_callback=cb)
        await svc.initialize_retriever("query")
        assert len(calls) == 1
        assert calls[0]["type"] == "thinking"
        assert calls[0]["stage"] == "retrieval"


# ---------------------------------------------------------------------------
# generate_hypothetical_document
# ---------------------------------------------------------------------------


class TestGenerateHypotheticalDocument:
    @pytest.mark.asyncio
    async def test_returns_chain_output(self):
        svc, _, _ = _make_service()
        # Build a fake chain by overriding the operator overload on the
        # hyde_prompt_template.
        svc.hyde_prompt_template = MagicMock()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="hypo content")
        svc.hyde_prompt_template.__or__.return_value.__or__.return_value = mock_chain
        result = await svc.generate_hypothetical_document("query")
        assert result == "hypo content"

    @pytest.mark.asyncio
    async def test_falls_back_to_query_on_error(self):
        svc, _, _ = _make_service()
        svc.hyde_prompt_template = MagicMock()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        svc.hyde_prompt_template.__or__.return_value.__or__.return_value = mock_chain
        result = await svc.generate_hypothetical_document("original-query")
        assert result == "original-query"


# ---------------------------------------------------------------------------
# retrieve_document
# ---------------------------------------------------------------------------


class TestRetrieveDocument:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_user_id(self):
        svc, _, _ = _make_service(user_id=None)
        result = await svc.retrieve_document("query")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_documents_from_retriever(self):
        svc, _, retriever = _make_service()
        docs = [_make_doc("doc1"), _make_doc("doc2")]
        retriever.ainvoke.return_value = docs
        result = await svc.retrieve_document("query")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_deduplicates_results(self):
        svc, _, retriever = _make_service()
        retriever.ainvoke.return_value = [
            _make_doc("内容ABC"),
            _make_doc("内容ABC"),
            _make_doc("其他内容"),
        ]
        result = await svc.retrieve_document("query")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        svc, _, retriever = _make_service()
        retriever.ainvoke.side_effect = RuntimeError("retrieve failed")
        result = await svc.retrieve_document("query")
        assert result == []

    @pytest.mark.asyncio
    async def test_initializes_retriever_if_none(self):
        svc, vs, retriever = _make_service()
        retriever.ainvoke.return_value = []
        await svc.retrieve_document("q")
        vs.get_retriever.assert_awaited()

    @pytest.mark.asyncio
    async def test_thinking_callback_for_hyde(self):
        calls = []

        async def cb(payload):
            calls.append(payload)

        svc, _, retriever = _make_service(thinking_callback=cb)
        retriever.ainvoke.return_value = []
        # Patch the hypothetical doc generation to return a known value
        svc.hyde_prompt_template = MagicMock()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="hypo")
        svc.hyde_prompt_template.__or__.return_value.__or__.return_value = mock_chain
        await svc.retrieve_document("q")
        stages = [c["stage"] for c in calls]
        assert "hyde" in stages
        assert "retrieval" in stages


# ---------------------------------------------------------------------------
# reorder_documents
# ---------------------------------------------------------------------------


class TestReorderDocuments:
    @pytest.mark.asyncio
    async def test_returns_reordered_documents(self):
        svc, _, _ = _make_service()
        fake_result = {
            "success": True,
            "documents": [
                {"document": "B", "similarity": 0.9},
                {"document": "A", "similarity": 0.5},
            ],
            "error": "",
        }
        with patch("app.rag.rag_service.reorder_service") as rs:
            rs.reorder_documents = AsyncMock(return_value=fake_result)
            result = await svc.reorder_documents("q", ["A", "B"])
        assert result == ["B", "A"]

    @pytest.mark.asyncio
    async def test_returns_original_on_failure(self):
        svc, _, _ = _make_service()
        with patch("app.rag.rag_service.reorder_service") as rs:
            rs.reorder_documents = AsyncMock(return_value={"success": False, "documents": [], "error": "boom"})
            docs = ["A", "B"]
            result = await svc.reorder_documents("q", docs)
        assert result == docs

    @pytest.mark.asyncio
    async def test_thinking_callback_fires(self):
        calls = []

        async def cb(payload):
            calls.append(payload)

        svc, _, _ = _make_service(thinking_callback=cb)
        fake_result = {
            "success": True,
            "documents": [{"document": "A", "similarity": 0.7}],
            "error": "",
        }
        with patch("app.rag.rag_service.reorder_service") as rs:
            rs.reorder_documents = AsyncMock(return_value=fake_result)
            await svc.reorder_documents("q", ["A"])
        assert any(c["stage"] == "reorder" for c in calls)


# ---------------------------------------------------------------------------
# get_documents_and_summary / rag_summary
# ---------------------------------------------------------------------------


class TestGetDocumentsAndSummary:
    @pytest.mark.asyncio
    async def test_empty_user_id_returns_standard_message(self):
        svc, _, _ = _make_service(user_id=None)
        result = await svc.get_documents_and_summary("q")
        assert result["documents"] == []
        assert "抱歉" in result["summary"]

    @pytest.mark.asyncio
    async def test_no_documents_returns_standard_message(self):
        svc, _, retriever = _make_service()
        retriever.ainvoke.return_value = []
        result = await svc.get_documents_and_summary("q")
        assert result["documents"] == []
        assert "抱歉" in result["summary"]

    @pytest.mark.asyncio
    async def test_summarization_single_doc(self):
        svc, _, retriever = _make_service()
        retriever.ainvoke.return_value = [_make_doc("doc content")]
        # Make reorder succeed
        with patch("app.rag.rag_service.reorder_service") as rs:
            rs.reorder_documents = AsyncMock(return_value={
                "success": True,
                "documents": [{"document": "doc content", "similarity": 0.9}],
                "error": "",
            })
            # Mock the chain
            svc.chain = MagicMock()
            svc.chain.ainvoke = AsyncMock(return_value="summary text")
            result = await svc.get_documents_and_summary("q")
        assert result["documents"] == ["doc content"]
        assert result["summary"] == "summary text"

    @pytest.mark.asyncio
    async def test_summarization_multi_doc(self):
        svc, _, retriever = _make_service()
        retriever.ainvoke.return_value = [
            _make_doc("doc1"),
            _make_doc("doc2"),
        ]
        with patch("app.rag.rag_service.reorder_service") as rs:
            rs.reorder_documents = AsyncMock(return_value={
                "success": True,
                "documents": [
                    {"document": "doc1", "similarity": 0.9},
                    {"document": "doc2", "similarity": 0.5},
                ],
                "error": "",
            })
            svc.chain = MagicMock()
            # First call: summarize doc1 -> "s1", second: doc2 -> "s2", third: combined -> "final"
            svc.chain.ainvoke = AsyncMock(side_effect=["s1", "s2", "final summary"])
            result = await svc.get_documents_and_summary("q")
        assert result["summary"] == "final summary"

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_message(self):
        import asyncio

        svc, _, retriever = _make_service()
        retriever.ainvoke.return_value = [_make_doc("doc")]
        with patch("app.rag.rag_service.reorder_service") as rs:
            rs.reorder_documents = AsyncMock(return_value={
                "success": True,
                "documents": [{"document": "doc", "similarity": 0.9}],
                "error": "",
            })
            svc.chain = MagicMock()
            svc.chain.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
            result = await svc.get_documents_and_summary("q")
        assert "超时" in result["summary"]
        assert result["documents"] == ["doc"]

    @pytest.mark.asyncio
    async def test_exception_returns_empty_or_error_message(self):
        # An exception inside retrieve_document is caught and returns [],
        # which then makes get_documents_and_summary return the empty-message.
        svc, _, retriever = _make_service()
        retriever.ainvoke.side_effect = RuntimeError("boom")
        result = await svc.get_documents_and_summary("q")
        assert "抱歉" in result["summary"]


class TestRagSummary:
    @pytest.mark.asyncio
    async def test_delegates_to_get_documents_and_summary(self):
        svc, _, _ = _make_service()
        svc.get_documents_and_summary = AsyncMock(return_value={"documents": [], "summary": "X"})
        assert await svc.rag_summary("q") == "X"

    @pytest.mark.asyncio
    async def test_missing_summary_returns_default(self):
        svc, _, _ = _make_service()
        svc.get_documents_and_summary = AsyncMock(return_value={"documents": []})
        assert await svc.rag_summary("q") == "抱歉，处理您的请求时出现了错误。"
