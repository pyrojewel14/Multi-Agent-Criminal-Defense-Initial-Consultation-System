"""Unit tests for ``app.rag.rag_service.RagService``.

All external dependencies (vector store, retriever, reranker, chat model) are
mocked. Tests focus on the orchestration logic: HyDE generation, document
retrieval, reordering, summarization, and the empty-result / error paths.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.observability.tracing import (
    BoundedTraceStore,
    SessionBudget,
    SessionBudgetExceeded,
    trace_span,
)
from app.rag import rag_service as rag_module
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
        assert svc.retrieval_failed is False

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
        assert svc.retrieval_failed is True

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
# retrieve_documents – retrieval-only path
# ---------------------------------------------------------------------------


class TestRetrieveDocuments:
    @pytest.mark.asyncio
    async def test_failed_retrieval_log_keeps_error_type_without_sensitive_message(self, monkeypatch):
        svc, _, _ = _make_service()
        secret = "张三的案情提示词13800138000110105199001011234"
        svc.retrieve_document = AsyncMock(side_effect=RuntimeError(secret))
        messages = []

        class RecordingLogger:
            def error(self, template, *args):
                messages.append(template % args)

        monkeypatch.setattr(rag_module, "_logger", RecordingLogger())
        result = await svc.retrieve_documents("synthetic query")

        assert result == []
        assert svc.retrieval_failed is True
        assert messages
        assert all(secret not in message and "张三" not in message for message in messages)
        assert any("RuntimeError" in message for message in messages)

    @pytest.mark.asyncio
    async def test_failed_retrieval_log_rejects_untrusted_exception_class_name(self, monkeypatch):
        svc, _, _ = _make_service()
        secret = "13800138000"
        LeakError = type(f"LeakError_{secret}", (Exception,), {})
        svc.retrieve_document = AsyncMock(side_effect=LeakError("hidden detail"))
        messages = []

        class RecordingLogger:
            def error(self, template, *args):
                messages.append(template % args)

        monkeypatch.setattr(rag_module, "_logger", RecordingLogger())
        assert await svc.retrieve_documents("synthetic query") == []
        assert secret not in str(messages)
        assert "OtherError" in str(messages)

    @pytest.mark.asyncio
    async def test_ranked_trace_is_bounded_to_top_five(self, monkeypatch):
        store = BoundedTraceStore(max_events=30)
        monkeypatch.setattr(rag_module, "trace_store", store)
        svc, _, _ = _make_service()
        contents = [f"doc-{index}" for index in range(7)]
        svc.retrieve_document = AsyncMock(return_value=[_make_doc(content) for content in contents])
        svc.reorder_documents = AsyncMock(return_value=contents)

        with trace_span(store, event_type="node", name="law_ref", session_id="bounded-ranking"):
            result = await svc.retrieve_documents("query")

        assert result == contents
        ranked = [event for event in store.events() if event.event_type == "rag_ranked_result"]
        assert [event.attempt for event in ranked] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_ranked_results_trace_has_order_and_fingerprint_without_content(self, monkeypatch):
        store = BoundedTraceStore(max_events=20)
        monkeypatch.setattr(rag_module, "trace_store", store)
        svc, _, _ = _make_service()
        svc.retrieve_document = AsyncMock(return_value=[_make_doc("案情甲"), _make_doc("案情乙")])
        svc.reorder_documents = AsyncMock(return_value=["案情乙", "案情甲"])

        with trace_span(store, event_type="node", name="law_ref", session_id="ranking-session"):
            result = await svc.retrieve_documents("query")

        assert result == ["案情乙", "案情甲"]
        ranked = [event.to_dict() for event in store.events() if event.event_type == "rag_ranked_result"]
        assert [event["attempt"] for event in ranked] == [1, 2]
        assert all(event["metadata"]["content"]["sha256"].startswith("sha256:") for event in ranked)
        assert all(event["metadata"]["origin"]["sha256"].startswith("sha256:") for event in ranked)
        assert "案情乙" not in str(ranked)
        assert "案情甲" not in str(ranked)
        assert "src.txt" not in str(ranked)

    @pytest.mark.asyncio
    async def test_budget_exhaustion_is_not_swallowed_as_empty_retrieval(self, monkeypatch):
        """预算异常若被 RAG 的宽泛 except 吞掉，会继续形成无界重试。"""
        store = BoundedTraceStore(max_events=10)
        monkeypatch.setattr(rag_module, "trace_store", store, raising=False)
        monkeypatch.setattr(
            rag_module,
            "session_budget",
            SessionBudget(max_calls=1, max_tokens=100),
            raising=False,
        )
        svc, _, _ = _make_service()

        with trace_span(
            store,
            event_type="node",
            name="law_ref",
            request_id="rag-budget-request",
            session_id="rag-budget-session",
        ):
            with pytest.raises(SessionBudgetExceeded):
                await svc.retrieve_documents("案件查询")

        rag_event = next(event for event in store.events() if event.event_type == "rag")
        assert rag_event.outcome == "error"

    @pytest.mark.asyncio
    async def test_returns_ranked_documents_without_summary_llm_calls(self):
        """检索路径只返回重排序文档，不应调用摘要链。"""
        svc, _, retriever = _make_service()
        retriever.ainvoke.return_value = [
            _make_doc("doc1"),
            _make_doc("doc2"),
            _make_doc("doc3"),
        ]

        with patch("app.rag.rag_service.reorder_service") as rs:
            rs.reorder_documents = AsyncMock(
                return_value={
                    "success": True,
                    "documents": [
                        {"document": "doc3", "similarity": 0.9},
                        {"document": "doc1", "similarity": 0.8},
                        {"document": "doc2", "similarity": 0.7},
                    ],
                    "error": "",
                }
            )
            svc.chain = MagicMock()
            svc.chain.ainvoke = AsyncMock()

            result = await svc.retrieve_documents("q")

        assert result == ["doc3", "doc1", "doc2"]
        svc.chain.ainvoke.assert_not_awaited()


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
            _make_doc("doc3"),
        ]
        with patch("app.rag.rag_service.reorder_service") as rs:
            rs.reorder_documents = AsyncMock(return_value={
                "success": True,
                "documents": [
                    {"document": "doc1", "similarity": 0.9},
                    {"document": "doc2", "similarity": 0.5},
                    {"document": "doc3", "similarity": 0.4},
                ],
                "error": "",
            })
            svc.chain = MagicMock()
            # 三次单文档摘要后，再由显式摘要入口请求一次汇总。
            svc.chain.ainvoke = AsyncMock(side_effect=["s1", "s2", "s3", "final summary"])
            result = await svc.get_documents_and_summary("q")
        assert result["summary"] == "final summary"
        assert svc.chain.ainvoke.await_count == 4

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
