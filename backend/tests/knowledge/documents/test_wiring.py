"""Document assembly uses isolated collections and lazy, injected models."""

from io import BytesIO

import chromadb
import pytest
from fastapi import UploadFile
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.knowledge import wiring
from app.knowledge.retrieval.config import RerankerConfig
from app.knowledge.retrieval.rerankers import CrossEncoderReranker
from app.knowledge.storage.chroma import ChromaSource, ChromaStore


@pytest.fixture
def source(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return ChromaSource("uploaded-documents", client, "uploaded-documents")


class Embedding:
    def __init__(self):
        self.calls = []

    async def aembed_documents(self, texts):
        self.calls.append(("documents", tuple(texts)))
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text):
        self.calls.append(("query", text))
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_new_upload_collection_requires_explicit_initialization_and_shares_lazy_embedding(source, tmp_path):
    resources = wiring.KnowledgeResources()
    model, created = Embedding(), []

    def factory():
        created.append(True)
        return model

    with pytest.raises(chromadb.errors.NotFoundError):
        wiring.build_document_services(source=source, md5_directory=tmp_path / "hashes", resources=resources)
    assert source.client.list_collections() == []
    service = wiring.build_document_services(
        source=source,
        create_missing=True,
        md5_directory=tmp_path / "hashes",
        resources=resources,
        embedding_factory=factory,
        env={"LAW_FULL_RERANK_ENABLED": "false"},
    )
    resources.readiness()
    assert created == []
    assert (await service.search("anything", user_id=None)).candidates == ()
    assert created == []
    await service.handle_add_vector_single(UploadFile(file=BytesIO(b"owned content"), filename="case.txt"), "alice")
    result = await service.search("content", user_id="alice")
    assert result.candidates[0].content == "owned content"
    assert result.candidates[0].source_id == "uploaded-documents"
    assert created == [True]
    assert model.calls == [("documents", ("owned content",)), ("query", "content")]
    assert (await service.search("content", user_id="bob", include_public=False)).candidates == ()
    resources.close()


def test_default_layout_uses_configured_collection_and_parent_md5_directory(source, tmp_path, monkeypatch):
    from app.infrastructure.config import rag

    monkeypatch.setattr(
        rag,
        "chroma_config",
        {
            "collection_name": source.collection_name,
            "persist_directory": str(tmp_path / "upload-index"),
            "md5_hex_store": str(tmp_path / "hash-layout" / "md5_hex_store.txt"),
            "splitter_type": "legal_article",
            "chunk_size": 800,
            "chunk_overlap": 50,
            "k": 5,
        },
    )
    seen = []

    def client_factory(*, path, **kwargs):
        seen.append(path)
        return source.client

    monkeypatch.setattr(chromadb, "PersistentClient", client_factory)
    service = wiring.build_document_services(create_missing=True, env={"LAW_FULL_RERANK_ENABLED": "false"})
    assert seen == [str(tmp_path / "upload-index")]
    assert service.repository.store.source.collection_name == source.collection_name
    assert service.repository.fingerprints.base_dir == str(tmp_path / "hash-layout")
    assert service.processor.splitter.chunk_size == 800
    assert service.top_k == 5


def test_existing_collection_is_never_recreated_and_nonmissing_errors_propagate(source, tmp_path, monkeypatch):
    store = ChromaStore.create(source)
    service = wiring.build_document_services(
        source=source, create_missing=True, md5_directory=tmp_path, env={"LAW_FULL_RERANK_ENABLED": "false"}
    )
    assert service.repository.store._collection.id == store._collection.id

    def fail(*args, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(source.client, "get_collection", fail)
    with pytest.raises(RuntimeError, match="storage unavailable"):
        wiring.build_document_services(source=source, create_missing=True, md5_directory=tmp_path)
    assert len(source.client.list_collections()) == 1


def test_document_cross_encoder_is_explicit_and_does_not_replace_law_qwen(source, tmp_path):
    resources = wiring.KnowledgeResources()
    config = RerankerConfig()
    law_scorer = resources.scorer(config)
    service = wiring.build_document_services(
        source=source,
        create_missing=True,
        md5_directory=tmp_path,
        resources=resources,
        reranker_type="cross_encoder",
        env={},
    )
    assert isinstance(service.pipeline.reranker, CrossEncoderReranker)
    assert resources.scorer(config) is law_scorer
    assert service.pipeline.reranker is not law_scorer
    assert service.pipeline.reranker.readiness()["status"] == "not_loaded"
    resources.close()


def test_readiness_and_close_do_not_create_embedding(source, tmp_path):
    resources = wiring.KnowledgeResources()

    def forbidden():
        raise AssertionError("model construction forbidden")

    wiring.build_document_services(
        source=source, create_missing=True, md5_directory=tmp_path, resources=resources, embedding_factory=forbidden
    )
    resources.readiness()
    resources.close()
    resources.close()


@pytest.mark.asyncio
async def test_async_embedding_injection_does_not_fall_back_to_another_splitter_model(source, tmp_path, monkeypatch):
    from app.infrastructure.config import rag

    monkeypatch.setitem(rag.chroma_config, "splitter_type", "recursive")
    monkeypatch.setitem(rag.chroma_config, "chunk_size", 10)
    monkeypatch.setitem(rag.chroma_config, "chunk_overlap", 0)
    model, created = Embedding(), []

    def forbidden():
        created.append(True)
        return model

    resources = wiring.KnowledgeResources()
    service = wiring.build_document_services(
        source=source,
        create_missing=True,
        md5_directory=tmp_path,
        resources=resources,
        embedding_factory=forbidden,
        embed_documents=model.aembed_documents,
        embed_query=model.aembed_query,
        env={"LAW_FULL_RERANK_ENABLED": "false"},
    )
    await service.handle_add_vector_single(
        UploadFile(file=BytesIO(b"one chunk. another chunk."), filename="case.txt"), "alice"
    )
    assert created == []
    assert service.repository.store.get_by_metadata(where={"user_id": "alice"})
    resources.close()


@pytest.mark.asyncio
async def test_summary_wiring_builds_prompt_chat_parser_chain():
    from app.knowledge.retrieval.models import RetrievalCandidate, SearchResult

    inputs = []

    async def chat(prompt):
        inputs.append(prompt.to_string())
        return AIMessage(content="assembled summary")

    service = wiring.get_document_summary_service(chat_model=RunnableLambda(chat))
    result = SearchResult((RetrievalCandidate("one", "uploads", "selected content", {}),), {}, False)
    assert await service.rag_summary("question", result) == "assembled summary"
    assert len(inputs) == 1
    assert "selected content" in inputs[0] and "question" in inputs[0]


@pytest.mark.asyncio
async def test_default_embedding_uses_assembly_snapshot_and_closes_only_initialized_client(
    source, tmp_path, monkeypatch
):
    import langchain_ollama

    created, closed = [], []
    model = Embedding()
    model.close = lambda: closed.append(True)

    def create(**kwargs):
        created.append(kwargs)
        return model

    monkeypatch.setattr(langchain_ollama, "OllamaEmbeddings", create)
    resources = wiring.KnowledgeResources()
    service = wiring.build_document_services(
        source=source,
        create_missing=True,
        md5_directory=tmp_path,
        resources=resources,
        env={
            "LAW_FULL_RERANK_ENABLED": "false",
            "EMBED_MODEL_TYPE": "OLLAMA",
            "TEXT_EMBEDDING_MODEL_NAME": "configured-embedding",
            "OLLAMA_BASE_URL": "http://127.0.0.1:12345",
        },
    )
    monkeypatch.setenv("TEXT_EMBEDDING_MODEL_NAME", "changed-after-assembly")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9999")
    assert created == []
    await service.handle_add_vector_single(UploadFile(file=BytesIO(b"snapshot content"), filename="case.txt"), "alice")
    await service.search("content", user_id="alice")
    assert len(created) == 1
    assert created[0]["model"] == "configured-embedding"
    assert created[0]["base_url"] == "http://127.0.0.1:12345"
    resources.close()
    resources.close()
    assert closed == [True]


@pytest.mark.asyncio
async def test_aliyun_client_uses_snapshot_and_is_closed(source, tmp_path, monkeypatch):
    from app.infrastructure.llm import factory

    created, closed = [], []
    model = Embedding()
    model.close = lambda: closed.append(True)

    def create(**kwargs):
        created.append(kwargs)
        return model

    monkeypatch.setattr(factory, "DashScopeEmbeddingsWrapper", create)
    resources = wiring.KnowledgeResources()
    service = wiring.build_document_services(
        source=source,
        create_missing=True,
        md5_directory=tmp_path,
        resources=resources,
        env={
            "LAW_FULL_RERANK_ENABLED": "false",
            "EMBED_MODEL_TYPE": "ALIYUN",
            "ALIYUN_EMBED_MODEL_NAME": "configured-model",
            "DASHSCOPE_API_KEY": "fake-test-key",
            "ALIYUN_BASE_URL": "https://configured.example/v1",
        },
    )
    monkeypatch.setenv("ALIYUN_EMBED_MODEL_NAME", "changed-after-assembly")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://ambient.example/v1")
    await service.handle_add_vector_single(UploadFile(file=BytesIO(b"snapshot content"), filename="case.txt"), "alice")
    assert created == [
        {"model_name": "configured-model", "api_key": "fake-test-key", "base_url": "https://configured.example/v1"}
    ]
    resources.close()
    resources.close()
    assert closed == [True]


@pytest.mark.asyncio
async def test_default_management_entry_initializes_once_and_reopens_existing_collection(tmp_path, monkeypatch):
    resources = wiring.KnowledgeResources()
    monkeypatch.setattr(wiring, "knowledge_resources", resources)
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "fresh-upload-store"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "fresh-upload-store")
    monkeypatch.setenv("CHROMA_MD5_STORE", str(tmp_path / "hashes" / "md5_hex_store.txt"))
    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "false")
    model = Embedding()
    monkeypatch.setattr(wiring, "_create_document_embedding", lambda values: model)
    service = wiring.get_document_service()
    assert wiring.get_document_service() is service
    await service.handle_add_vector_single(UploadFile(file=BytesIO(b"fresh content"), filename="case.txt"), "alice")
    collection_id = service.repository.store._collection.id
    resources.close()
    reopened = wiring.get_document_service()
    assert reopened.repository.store._collection.id == collection_id
    assert (await reopened.handle_get_document_detail("alice", "case.txt"))["content"] == "fresh content"
    resources.close()


def test_sync_close_releases_both_real_ollama_http_clients_without_network():
    resources = wiring.KnowledgeResources()
    proxy = resources.document_embeddings(
        values={"EMBED_MODEL_TYPE": "OLLAMA", "OLLAMA_BASE_URL": "http://127.0.0.1:1"}
    )
    model = proxy._get_model()
    assert not model._client._client.is_closed
    assert not model._async_client._client.is_closed
    try:
        resources.close()
        assert model._client._client.is_closed
        assert model._async_client._client.is_closed
    finally:
        # Also keep the RED run from leaking the actual clients.
        import asyncio

        model._client.close()
        asyncio.run(model._async_client.close())


@pytest.mark.asyncio
async def test_sync_close_in_running_loop_requires_awaited_async_close_without_losing_clients():
    resources = wiring.KnowledgeResources()
    proxy = resources.document_embeddings(
        values={"EMBED_MODEL_TYPE": "OLLAMA", "OLLAMA_BASE_URL": "http://127.0.0.1:1"}
    )
    model = proxy._get_model()
    try:
        with pytest.raises(RuntimeError, match="aclose"):
            resources.close()
        assert not model._client._client.is_closed
        assert not model._async_client._client.is_closed
        await resources.aclose()
        await resources.aclose()
        assert model._client._client.is_closed
        assert model._async_client._client.is_closed
    finally:
        model._client.close()
        await model._async_client.close()


@pytest.mark.parametrize("endpoint_setting", ["DASHSCOPE_BASE_URL", "ALIYUN_BASE_URL"])
def test_aliyun_endpoint_snapshot_ignores_later_environment(endpoint_setting, monkeypatch):
    snapshot = {
        "EMBED_MODEL_TYPE": "ALIYUN",
        "DASHSCOPE_API_KEY": "fake-snapshot-key",
        endpoint_setting: "https://configured.example/v1",
    }
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://ambient.example/v1")
    model = wiring._create_document_embedding(snapshot)
    try:
        assert str(model.client.base_url) == "https://configured.example/v1/"
    finally:
        model.close()


def test_aliyun_missing_snapshot_key_cannot_borrow_ambient_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-ambient-key")
    with pytest.raises(ValueError, match="API Key"):
        wiring._create_document_embedding({"EMBED_MODEL_TYPE": "ALIYUN"})
