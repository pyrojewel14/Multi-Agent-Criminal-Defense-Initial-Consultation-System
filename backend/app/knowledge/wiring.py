"""显式绑定法条、上传数据源和共享模型资源，不在导入时加载模型。"""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from app.knowledge.law.corpus import FULL_CORPUS_PATH, load_criminal_law_data
from app.knowledge.law.index import verify_full_index
from app.knowledge.law.repository import LawRepository
from app.knowledge.law.service import LawSearchService
from app.knowledge.retrieval.bm25 import BM25Retriever
from app.knowledge.retrieval.config import RerankerConfig, RetrievalConfig
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from app.knowledge.retrieval.rerankers import CrossEncoderReranker, QwenReranker
from app.knowledge.retrieval.vector import ChromaVectorRetriever
from app.knowledge.storage.chroma import ChromaSource, ChromaStore, validate_embedding


class _RejectedLawScorer:
    """将不受支持的法律评分配置留在可选阶段拒绝，不创建模型资源。"""

    probability_scores = True

    async def score(self, query, documents, *, timeout_s):
        raise ValueError("full reranker requires configured Qwen3-Reranker-0.6B")


def _create_document_embedding(values):
    """复用既有客户端协议，按组装时的配置创建，不临时改写进程环境。"""
    backend = values.get("EMBED_MODEL_TYPE", "OLLAMA").upper()
    if backend == "OLLAMA":
        from langchain_ollama import OllamaEmbeddings

        from app.infrastructure.llm.factory import _ollama_client_kwargs
        base_url = values.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=values.get("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b"),
                                base_url=base_url, client_kwargs=_ollama_client_kwargs(base_url))
    if backend == "ALIYUN":
        api_key = values.get("DASHSCOPE_API_KEY") or values.get("ALIYUN_ACCESS_KEY_SECRET")
        if not api_key:
            raise ValueError("组装配置未设置 API Key")
        from app.infrastructure.llm.factory import DashScopeEmbeddingsWrapper
        return DashScopeEmbeddingsWrapper(model_name=values.get("ALIYUN_EMBED_MODEL_NAME", "text-embedding-v4"),
            api_key=api_key, base_url=values.get("DASHSCOPE_BASE_URL") or values.get("ALIYUN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    raise ValueError("unsupported document embedding backend")


class _LazyDocumentEmbeddings:
    """上传与查询共享同一个延迟创建的 Embedding，切片器可使用同步接口。"""

    def __init__(self, factory):
        self._factory = factory
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        """只有实际 Embedding 调用才能创建客户端，不执行模型下载。"""
        with self._lock:
            if self._model is None:
                self._model = self._factory()
            return self._model

    async def aembed_documents(self, texts):
        """在首次创建后复用模型的异步文档接口。"""
        model = await asyncio.to_thread(self._get_model)
        return await model.aembed_documents(texts)

    async def aembed_query(self, text):
        """查询和上传使用同一模型，避免向量空间不一致。"""
        model = await asyncio.to_thread(self._get_model)
        return await model.aembed_query(text)

    def embed_query(self, text):
        """供工作线程中的可选语义切片使用。"""
        return self._get_model().embed_query(text)

    def _close_actions(self):
        """识别已建模型的关闭协议，Ollama 的两个客户端分别持有连接池。"""
        with self._lock:
            model = self._model
        if model is None:
            return model, (), ()
        targets = [model]
        if not callable(getattr(model, "close", None)) and not callable(getattr(model, "aclose", None)):
            # 当前 OllamaEmbeddings 未暴露关闭协议，ollama 客户端的 close 分别为同步和异步。
            targets = [getattr(model, "_client", None), getattr(model, "_async_client", None)]
        sync_actions, async_actions = [], []
        for target in targets:
            aclose, close = getattr(target, "aclose", None), getattr(target, "close", None)
            if callable(aclose):
                async_actions.append(aclose)
            elif callable(close):
                (async_actions if inspect.iscoroutinefunction(close) else sync_actions).append(close)
        return model, tuple(sync_actions), tuple(async_actions)

    def requires_async_close(self):
        """只查看现有资源，不通过关闭检查创建模型。"""
        return bool(self._close_actions()[2])

    def close(self):
        """同步上下文关闭所有客户端；运行中的事件循环必须等待 aclose。"""
        model, sync_actions, async_actions = self._close_actions()
        if async_actions:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.aclose())
                return
            raise RuntimeError("异步 Embedding 资源需要 await aclose()")
        for action in sync_actions:
            action()
        with self._lock:
            if self._model is model:
                self._model = None

    async def aclose(self):
        """等待已创建的同步与异步连接池释放，失败时保留模型供再次清理。"""
        model, sync_actions, async_actions = self._close_actions()
        for action in sync_actions:
            await asyncio.to_thread(action)
        for action in async_actions:
            await action()
        with self._lock:
            if self._model is model:
                self._model = None


class KnowledgeResources:
    """持有有界缓存及唯一配置评分实例，所有请求状态留在服务局部。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._scorers: dict[RerankerConfig, QwenReranker] = {}
        self._cross_encoders: dict[RerankerConfig, CrossEncoderReranker] = {}
        self._document_embeddings: dict[Any, _LazyDocumentEmbeddings] = {}
        self._document_lock = threading.Lock()
        self._document_service = None
        self._law_config_error: str | None = None
        self._bm25: OrderedDict[tuple[Any, ...], BM25Retriever] = OrderedDict()

    def scorer(self, config: RerankerConfig) -> QwenReranker:
        """只创建轻量评分器；权重和设备等待首次真实评分。"""
        with self._lock:
            if config not in self._scorers:
                self._scorers[config] = QwenReranker(config)
            return self._scorers[config]

    def reject_law_scorer(self, config: RerankerConfig) -> _RejectedLawScorer:
        """记录配置错误供 readiness 展示，拒绝工作只在可选评分阶段发生。"""
        with self._lock:
            self._law_config_error = config.model_name
        return _RejectedLawScorer()

    def document_scorer(self, config: RerankerConfig, *, kind: str = "causal_lm"):
        """文档默认复用唯一 Qwen，Cross-Encoder 只接受显式构造选项。"""
        if kind == "causal_lm":
            return self.scorer(config)
        if kind != "cross_encoder":
            raise ValueError("unknown document reranker type")
        with self._lock:
            if config not in self._cross_encoders:
                self._cross_encoders[config] = CrossEncoderReranker(config)
            return self._cross_encoders[config]

    def document_embeddings(self, factory=None, *, values: Mapping[str, str]):
        """按模型配置快照复用代理；显式工厂自行负责配置与实例创建。"""
        settings = {name: values[name] for name in (
            "EMBED_MODEL_TYPE", "TEXT_EMBEDDING_MODEL_NAME", "OLLAMA_BASE_URL", "ALIYUN_EMBED_MODEL_NAME",
            "DASHSCOPE_API_KEY", "ALIYUN_ACCESS_KEY_SECRET", "DASHSCOPE_BASE_URL", "ALIYUN_BASE_URL") if name in values}
        key = factory if factory is not None else ("configured", tuple(sorted(settings.items())))
        with self._lock:
            if key not in self._document_embeddings:
                self._document_embeddings[key] = _LazyDocumentEmbeddings(
                    factory if factory is not None else partial(_create_document_embedding, settings))
            return self._document_embeddings[key]

    def bm25(self, store: ChromaStore, *, identity: tuple[Any, ...]) -> BM25Retriever:
        """按绑定库的独立身份复用缓存，不在请求中切换已有实例的数据源。"""
        with self._lock:
            if identity not in self._bm25:
                self._bm25[identity] = BM25Retriever(store)
            self._bm25.move_to_end(identity)
            while len(self._bm25) > 8:
                self._bm25.popitem(last=False)
            return self._bm25[identity]

    def readiness(self) -> dict[str, Any]:
        """仅读取当前资源状态，不能借健康检查创建模型或访问外部服务。"""
        with self._lock:
            scorers = tuple((scorer, "causal_lm") for scorer in self._scorers.values())
            scorers += tuple((scorer, "cross_encoder") for scorer in self._cross_encoders.values())
            law_error = self._law_config_error
        if law_error:
            return {"status": "error", "available": False, "model": law_error, "error": "ValueError", "type": "causal_lm"}
        if not scorers:
            return {"status": "not_loaded", "available": False, "model": "Qwen/Qwen3-Reranker-0.6B", "error": None, "type": "causal_lm"}
        states = [{**scorer.readiness(), "type": kind} for scorer, kind in scorers]
        state = next((state for state in states if not state["available"]), states[-1])
        return state

    def close(self) -> None:
        """同步关闭评分器及文档资源；事件循环中有异步客户端时拒绝假关闭。"""
        with self._lock:
            embeddings = tuple(self._document_embeddings.values())
        if any(embedding.requires_async_close() for embedding in embeddings):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.aclose())
                return
            raise RuntimeError("KnowledgeResources 的异步资源需要 await aclose()")
        self._close_scorers()
        for embedding in embeddings:
            embedding.close()
        self._clear_documents()

    def _close_scorers(self):
        """关闭当前评分资源并清空召回缓存，不抢先丢弃待关闭的客户端。"""
        with self._lock:
            scorers = tuple(self._scorers.values()) + tuple(self._cross_encoders.values())
            self._scorers.clear()
            self._cross_encoders.clear()
            self._law_config_error = None
            self._bm25.clear()
        for scorer in scorers:
            scorer.close()

    def _clear_documents(self):
        """客户端关闭成功后才移除引用，保持失败清理可重试。"""
        with self._lock:
            self._document_embeddings.clear()
        with self._document_lock:
            self._document_service = None

    async def aclose(self) -> None:
        """在原事件循环等待文档客户端释放，readiness 与关闭均不加载模型。"""
        self._close_scorers()
        with self._lock:
            embeddings = tuple(self._document_embeddings.values())
        results = await asyncio.gather(*(embedding.aclose() for embedding in embeddings), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        self._clear_documents()


knowledge_resources = KnowledgeResources()


def build_document_services(*, store: ChromaStore | None = None, source: ChromaSource | None = None,
                            create_missing: bool = False, md5_directory=None, processor=None,
                            embed_documents=None, embed_query=None, embedding_factory=None,
                            pipeline: RetrievalPipeline | None = None, reranker=None,
                            reranker_type: str = "causal_lm", resources: KnowledgeResources | None = None,
                            env: Mapping[str, str] | None = None):
    """按上传配置快照组装文档服务；只有显式选项允许创建缺失集合。

    env 作用于本次组装和默认 Embedding；显式回调、工厂及 Pipeline 的配置由调用方负责。
    """
    from app.infrastructure.config.paths import get_abstract_path
    from app.infrastructure.config.rag import chroma_config
    from app.knowledge.documents.fingerprints import FingerprintStore
    from app.knowledge.documents.processor import DocumentProcessor
    from app.knowledge.documents.repository import DocumentRepository
    from app.knowledge.documents.service import DocumentService
    from app.knowledge.documents.splitters.legal import LegalArticleSplitter
    from app.knowledge.documents.splitters.text import AsyncTextSplitter

    values = dict(os.environ if env is None else env)
    settings = dict(chroma_config)
    for key, variable in (("persist_directory", "CHROMA_PERSIST_DIRECTORY"),
                          ("collection_name", "CHROMA_COLLECTION_NAME"), ("md5_hex_store", "CHROMA_MD5_STORE")):
        if values.get(variable):
            settings[key] = values[variable]
    owned = resources if resources is not None else knowledge_resources
    if store is None:
        import chromadb
        if source is None:
            from chromadb.config import Settings
            client = chromadb.PersistentClient(path=get_abstract_path(settings["persist_directory"]),
                                               settings=Settings(anonymized_telemetry=False))
            source = ChromaSource("uploaded-documents", client, settings["collection_name"])
        try:
            store = ChromaStore.open_existing(source)
        except chromadb.errors.NotFoundError:
            if not create_missing:
                raise
            store = ChromaStore.create(source)
    md5_directory = md5_directory if md5_directory is not None else os.path.dirname(get_abstract_path(settings["md5_hex_store"]))
    if (embed_documents is None) != (embed_query is None):
        raise ValueError("document and query embedding callbacks must be injected together")
    injected_embeddings = embed_documents is not None
    embedding = None if injected_embeddings else owned.document_embeddings(embedding_factory, values=values)
    embed_documents = embed_documents if embed_documents is not None else embedding.aembed_documents
    embed_query = embed_query if embed_query is not None else embedding.aembed_query
    if processor is None:
        if settings.get("splitter_type", "recursive") == "legal_article":
            splitter = LegalArticleSplitter(chunk_size=settings.get("chunk_size", 2000),
                chunk_overlap=settings.get("chunk_overlap", 200),
                preserve_structure=settings.get("preserve_legal_structure", True),
                enable_metadata_extraction=settings.get("enable_metadata_extraction", True))
        else:
            # 仅有异步回调时不能隐式另建同步模型；自定义语义切片器由 processor 注入。
            splitter = AsyncTextSplitter(chunk_size=settings.get("chunk_size", 2000),
                chunk_overlap=settings.get("chunk_overlap", 200), separators=settings.get("separators"),
                embedding_model=None if injected_embeddings else embedding)
        processor = DocumentProcessor(splitter)
    config = RetrievalConfig.from_env(values)
    if pipeline is None:
        scorer = reranker if reranker is not None else (
            owned.document_scorer(RerankerConfig.from_env(values), kind=reranker_type) if config.rerank else None)
        expander = None
        if config.hyde:
            from app.knowledge.retrieval.hyde import HyDEExpander, ollama_generate
            model_name = values.get("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
            expander = HyDEExpander(partial(ollama_generate,
                base_url=values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), model_name=model_name), model_name=model_name)
        pipeline = RetrievalPipeline(ChromaVectorRetriever(store, embed_query, None),
            bm25=owned.bm25(store, identity=(id(store.source.client), store.source.collection_name, store.source.source_id)),
            expander=expander, reranker=scorer, config=config)
    return DocumentService(DocumentRepository(store, FingerprintStore(md5_directory)), processor,
                           embed_documents, pipeline, recall_k=config.recall_k, top_k=settings.get("k", 5))


def get_document_service():
    """上传库管理入口显式允许首次初始化，不影响法律索引的只读预检。"""
    with knowledge_resources._document_lock:
        if knowledge_resources._document_service is None:
            knowledge_resources._document_service = build_document_services(create_missing=True)
        return knowledge_resources._document_service


def get_document_summary_service(*, chat_model=None, thinking_callback=None):
    """组装既有摘要提示词与输出解析链，检索结果由调用方显式传入。"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    from app.infrastructure.config.prompts import prompt_loader
    from app.knowledge.documents.summary import DocumentSummaryService
    if chat_model is None:
        from app.infrastructure.llm.factory import get_chat_model
        chat_model = get_chat_model()
    chain = PromptTemplate.from_template(prompt_loader.load("rag_summary_prompt")) | chat_model | StrOutputParser()
    return DocumentSummaryService(chain, thinking_callback=thinking_callback)


def build_law_search_service(*, law_data: dict[str, Any] | None = None,
                             profile: str | None = None, pipeline: RetrievalPipeline | None = None,
                             source_id: str | None = None, resources: KnowledgeResources | None = None,
                             env: Mapping[str, str] | None = None) -> LawSearchService:
    """选择资产并绑定匹配数据源；snapshot 的 RAG 必须由调用方显式注入。"""
    values = dict(os.environ if env is None else env)
    profile = profile if profile is not None else values.get("LAW_KNOWLEDGE_PROFILE", "full")
    repository = LawRepository(law_data if law_data is not None else load_criminal_law_data(profile=profile, env=values), source_id=source_id)
    if pipeline is not None:
        return LawSearchService(repository, pipeline)
    config = RetrievalConfig.from_env(values)
    scorer_config = RerankerConfig.from_env(values)
    owned = resources if resources is not None else knowledge_resources
    unsupported = profile == "full" and scorer_config.model_name != "Qwen/Qwen3-Reranker-0.6B"
    scorer = (owned.reject_law_scorer(scorer_config) if unsupported else owned.scorer(scorer_config)) if config.rerank else None
    expander = None
    if config.hyde:
        from app.knowledge.retrieval.hyde import HyDEExpander, ollama_generate
        model_name = values.get("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
        expander = HyDEExpander(partial(ollama_generate,
                                      base_url=values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                                      model_name=model_name), model_name=model_name)
    template = RetrievalPipeline(None, expander=expander, reranker=scorer, config=config)
    if profile != "full":
        return LawSearchService(repository, template)

    async def preflight() -> RetrievalPipeline:
        """每请求核验实际 digest、manifest、完整正文及向量，然后打开唯一公共库。"""
        import chromadb
        import httpx

        directory, digest = values.get("LAW_FULL_INDEX_DIRECTORY"), values.get("LAW_FULL_EMBEDDING_DIGEST")
        if not directory or not digest or values.get("EMBED_MODEL_TYPE", "OLLAMA") != "OLLAMA":
            raise ValueError("全量 RAG 需要独立目录、Ollama embedding 与明确 digest")
        model = values.get("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
        collection = values.get("LAW_FULL_INDEX_COLLECTION", "criminal_law_full")
        corpus_path = Path(values.get("LAW_FULL_CORPUS_PATH", str(FULL_CORPUS_PATH)))
        base_url = values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            response = await client.get(base_url + "/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
        if not any(item.get("name") == model and item.get("digest") == digest for item in models):
            raise ValueError("实际 embedding 模型 digest 不一致")
        manifest = await asyncio.to_thread(verify_full_index, corpus_path, Path(directory), collection, model, digest)
        if manifest["corpus_sha256"] != repository.law_data.get("metadata", {}).get("corpus_sha256"):
            raise ValueError("controlled corpus does not match the bound index")
        client = await asyncio.to_thread(chromadb.PersistentClient, path=str(Path(directory).resolve()))
        store = await asyncio.to_thread(ChromaStore.open_existing, ChromaSource(repository.source_id, client, collection))

        async def embed_query(query: str) -> list[float]:
            """直接使用 Ollama 原始接口，禁止隐式截断原查询。"""
            async with httpx.AsyncClient(trust_env=False, timeout=60) as http:
                response = await http.post(base_url + "/api/embed", json={"model": model, "input": [query], "truncate": False})
                response.raise_for_status()
                vectors = response.json()["embeddings"]
            if len(vectors) != 1:
                raise ValueError("query embedding count invalid")
            return validate_embedding(vectors[0], expected_dimension=manifest["embedding_dimension"])

        vector = ChromaVectorRetriever(store, embed_query, manifest["embedding_dimension"])
        bm25 = owned.bm25(store, identity=(str(Path(directory).resolve()), collection, repository.source_id, model, digest))
        return RetrievalPipeline(vector, bm25=bm25, expander=expander, reranker=scorer, config=config)

    return LawSearchService(repository, template, preflight=preflight, required_index=True,
                            rerank_protocol="qwen3-yes-no-token-ids-v1" if scorer is not None and not unsupported else None,
                            integrity_metadata={"embedding_model": values.get("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b"),
                                                "embedding_model_digest": values.get("LAW_FULL_EMBEDDING_DIGEST")})
