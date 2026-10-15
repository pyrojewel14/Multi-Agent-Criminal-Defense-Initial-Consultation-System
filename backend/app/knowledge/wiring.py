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
        """在重排阶段报告配置错误，由 Pipeline 保留融合顺序。"""
        raise ValueError("full reranker requires configured Qwen3-Reranker-0.6B")


def _create_document_embedding(values):
    """复用既有客户端协议，按组装时的配置创建，不临时改写进程环境。"""
    backend = values.get("EMBED_MODEL_TYPE", "OLLAMA").upper()
    if backend == "OLLAMA":
        from langchain_ollama import OllamaEmbeddings

        from app.infrastructure.llm.factory import _ollama_client_kwargs

        base_url = values.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(
            model=values.get("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b"),
            base_url=base_url,
            client_kwargs=_ollama_client_kwargs(base_url),
        )
    if backend == "ALIYUN":
        api_key = values.get("DASHSCOPE_API_KEY") or values.get("ALIYUN_ACCESS_KEY_SECRET")
        if not api_key:
            raise ValueError("组装配置未设置 API Key")
        from app.infrastructure.llm.factory import DashScopeEmbeddingsWrapper

        return DashScopeEmbeddingsWrapper(
            model_name=values.get("ALIYUN_EMBED_MODEL_NAME", "text-embedding-v4"),
            api_key=api_key,
            base_url=values.get("DASHSCOPE_BASE_URL")
            or values.get("ALIYUN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    raise ValueError("unsupported document embedding backend")


class _LazyDocumentEmbeddings:
    """上传与查询共享同一个延迟创建的 Embedding，切片器可使用同步接口。"""

    def __init__(self, factory):
        self._factory = factory
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        """首次调用时执行注入工厂，并在线程锁内缓存返回的客户端。"""
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
    """按配置复用评分器、Embedding 和有界 BM25 缓存。

    实例在进程内共享；用户身份、候选池和核验结果由每次请求单独持有。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._scorers: dict[tuple[str, RerankerConfig], QwenReranker | CrossEncoderReranker] = {}
        self._document_embeddings: dict[Any, _LazyDocumentEmbeddings] = {}
        self._document_lock = threading.Lock()
        self._document_service = None
        self._law_config_error: str | None = None
        self._bm25: OrderedDict[tuple[Any, ...], BM25Retriever] = OrderedDict()

    def scorer(
        self, config: RerankerConfig, *, kind: str = "causal_lm"
    ) -> QwenReranker | CrossEncoderReranker:
        """按类型和配置复用轻量评分器；未知类型报错，权重等待首次真实评分。"""
        if kind == "causal_lm":
            scorer_type = QwenReranker
        elif kind == "cross_encoder":
            scorer_type = CrossEncoderReranker
        else:
            raise ValueError("unknown reranker type")
        key = (kind, config)
        with self._lock:
            if key not in self._scorers:
                self._scorers[key] = scorer_type(config)
            return self._scorers[key]

    def reject_law_scorer(self, config: RerankerConfig) -> _RejectedLawScorer:
        """记录配置错误供 readiness 展示，拒绝工作只在可选评分阶段发生。"""
        with self._lock:
            self._law_config_error = config.model_name
        return _RejectedLawScorer()

    def document_embeddings(self, factory=None, *, values: Mapping[str, str]):
        """按模型配置快照复用代理；显式工厂自行负责配置与实例创建。"""
        settings = {
            name: values[name]
            for name in (
                "EMBED_MODEL_TYPE",
                "TEXT_EMBEDDING_MODEL_NAME",
                "OLLAMA_BASE_URL",
                "ALIYUN_EMBED_MODEL_NAME",
                "DASHSCOPE_API_KEY",
                "ALIYUN_ACCESS_KEY_SECRET",
                "DASHSCOPE_BASE_URL",
                "ALIYUN_BASE_URL",
            )
            if name in values
        }
        key = factory if factory is not None else ("configured", tuple(sorted(settings.items())))

        # 默认按配置区分客户端；显式工厂按自身身份复用，配置责任留给调用方。
        with self._lock:
            if key not in self._document_embeddings:
                self._document_embeddings[key] = _LazyDocumentEmbeddings(
                    factory if factory is not None else partial(_create_document_embedding, settings)
                )
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
            # 稳定按类型分组，保持 Qwen 在前、Cross-Encoder 在后的原有检查顺序。
            entries = sorted(self._scorers.items(), key=lambda item: item[0][0])
            scorers = tuple((scorer, kind) for (kind, _), scorer in entries)
            law_error = self._law_config_error
        if law_error:
            return {
                "status": "error",
                "available": False,
                "model": law_error,
                "error": "ValueError",
                "type": "causal_lm",
            }
        if not scorers:
            return {
                "status": "not_loaded",
                "available": False,
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "error": None,
                "type": "causal_lm",
            }
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
        """在锁内取出并清空评分器和缓存，再在锁外关闭评分器。"""
        with self._lock:
            # 与原来两个缓存的遍历顺序一致，类型内保留创建顺序。
            scorers = tuple(scorer for _, scorer in sorted(self._scorers.items(), key=lambda item: item[0][0]))
            self._scorers.clear()
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


def build_document_services(
    *,
    store: ChromaStore | None = None,
    source: ChromaSource | None = None,
    create_missing: bool = False,
    md5_directory=None,
    processor=None,
    embed_documents=None,
    embed_query=None,
    embedding_factory=None,
    pipeline: RetrievalPipeline | None = None,
    reranker=None,
    reranker_type: str = "causal_lm",
    resources: KnowledgeResources | None = None,
    env: Mapping[str, str] | None = None,
):
    """组装上传文档的存储、切片、Embedding 和检索服务。

    env 作用于本次组装和默认 Embedding；显式回调、工厂及 Pipeline 的配置由调用方负责。
    embed_documents 与 embed_query 必须成对注入，确保上传和查询使用同一向量空间。
    仅 create_missing=True 时允许创建缺失的 Chroma collection；其他存储异常直接向上传播。
    """
    from app.infrastructure.config.paths import get_abstract_path
    from app.infrastructure.config.rag import chroma_config
    from app.knowledge.documents.fingerprints import FingerprintStore
    from app.knowledge.documents.processor import DocumentProcessor
    from app.knowledge.documents.repository import DocumentRepository
    from app.knowledge.documents.service import DocumentService
    from app.knowledge.documents.splitters.legal import LegalArticleSplitter
    from app.knowledge.documents.splitters.text import AsyncTextSplitter

    # 复制配置后应用本次覆盖，避免修改全局环境或共享的上传库配置。
    values = dict(os.environ if env is None else env)
    settings = dict(chroma_config)
    for key, variable in (
        ("persist_directory", "CHROMA_PERSIST_DIRECTORY"),
        ("collection_name", "CHROMA_COLLECTION_NAME"),
        ("md5_hex_store", "CHROMA_MD5_STORE"),
    ):
        if values.get(variable):
            settings[key] = values[variable]
    owned = resources if resources is not None else knowledge_resources

    # 优先使用调用方绑定的存储；默认入口只连接上传库，不借用法条索引。
    if store is None:
        import chromadb

        if source is None:
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=get_abstract_path(settings["persist_directory"]), settings=Settings(anonymized_telemetry=False)
            )
            source = ChromaSource("uploaded-documents", client, settings["collection_name"])
        try:
            store = ChromaStore.open_existing(source)
        except chromadb.errors.NotFoundError:
            if not create_missing:
                raise
            store = ChromaStore.create(source)
    md5_directory = (
        md5_directory if md5_directory is not None else os.path.dirname(get_abstract_path(settings["md5_hex_store"]))
    )

    # 未注入回调时只取得延迟代理，组装服务不会立即请求 Embedding。
    if (embed_documents is None) != (embed_query is None):
        raise ValueError("document and query embedding callbacks must be injected together")
    injected_embeddings = embed_documents is not None
    embedding = None if injected_embeddings else owned.document_embeddings(embedding_factory, values=values)
    embed_documents = embed_documents if embed_documents is not None else embedding.aembed_documents
    embed_query = embed_query if embed_query is not None else embedding.aembed_query

    if processor is None:
        if settings.get("splitter_type", "recursive") == "legal_article":
            splitter = LegalArticleSplitter(
                chunk_size=settings.get("chunk_size", 2000),
                chunk_overlap=settings.get("chunk_overlap", 200),
                preserve_structure=settings.get("preserve_legal_structure", True),
                enable_metadata_extraction=settings.get("enable_metadata_extraction", True),
            )
        else:
            # 仅有异步回调时不能隐式另建同步模型；自定义语义切片器由 processor 注入。
            splitter = AsyncTextSplitter(
                chunk_size=settings.get("chunk_size", 2000),
                chunk_overlap=settings.get("chunk_overlap", 200),
                separators=settings.get("separators"),
                embedding_model=None if injected_embeddings else embedding,
            )
        processor = DocumentProcessor(splitter)

    # 自定义 Pipeline 直接复用；默认流程才组装召回、HyDE 和重排组件。
    config = RetrievalConfig.from_env(values)
    if pipeline is None:
        scorer = (
            reranker
            if reranker is not None
            else (owned.scorer(RerankerConfig.from_env(values), kind=reranker_type) if config.rerank else None)
        )
        expander = None
        if config.hyde:
            from app.knowledge.retrieval.hyde import HyDEExpander, ollama_generate

            model_name = values.get("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
            expander = HyDEExpander(
                partial(
                    ollama_generate,
                    base_url=values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    model_name=model_name,
                ),
                model_name=model_name,
            )
        pipeline = RetrievalPipeline(
            ChromaVectorRetriever(store, embed_query, None),
            bm25=owned.bm25(
                store, identity=(id(store.source.client), store.source.collection_name, store.source.source_id)
            ),
            expander=expander,
            reranker=scorer,
            config=config,
        )

    return DocumentService(
        DocumentRepository(store, FingerprintStore(md5_directory)),
        processor,
        embed_documents,
        pipeline,
        recall_k=config.recall_k,
        top_k=settings.get("k", 5),
    )


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


def build_law_search_service(
    *,
    law_data: dict[str, Any] | None = None,
    profile: str | None = None,
    pipeline: RetrievalPipeline | None = None,
    source_id: str | None = None,
    resources: KnowledgeResources | None = None,
    env: Mapping[str, str] | None = None,
) -> LawSearchService:
    """选择法条资产，并组装绑定同一来源的法律检索服务。

    law_data 优先于按 profile 加载的资产；env 显式传入时替代进程环境。
    注入 pipeline 时直接使用，不附加默认的全量索引预检。
    默认 full 流程在搜索时核验已有索引；snapshot 的 RAG 需由调用方注入。
    """
    values = dict(os.environ if env is None else env)
    profile = profile if profile is not None else values.get("LAW_KNOWLEDGE_PROFILE", "full")
    repository = LawRepository(
        law_data if law_data is not None else load_criminal_law_data(profile=profile, env=values), source_id=source_id
    )

    # 显式注入的流程由调用方负责绑定数据源，避免混入默认索引或模型配置。
    if pipeline is not None:
        return LawSearchService(repository, pipeline)

    config = RetrievalConfig.from_env(values)
    scorer_config = RerankerConfig.from_env(values)
    owned = resources if resources is not None else knowledge_resources

    # full 只接受既定 Qwen 评分协议；不支持的配置留到可选重排阶段报告降级。
    unsupported = profile == "full" and scorer_config.model_name != "Qwen/Qwen3-Reranker-0.6B"
    scorer = (
        (owned.reject_law_scorer(scorer_config) if unsupported else owned.scorer(scorer_config))
        if config.rerank
        else None
    )

    expander = None
    if config.hyde:
        from app.knowledge.retrieval.hyde import HyDEExpander, ollama_generate

        model_name = values.get("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
        expander = HyDEExpander(
            partial(
                ollama_generate, base_url=values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), model_name=model_name
            ),
            model_name=model_name,
        )

    # 此时尚未打开向量库；full 的实际召回组件在预检通过后才绑定。
    template = RetrievalPipeline(None, expander=expander, reranker=scorer, config=config)
    if profile != "full":
        return LawSearchService(repository, template)

    async def preflight() -> RetrievalPipeline:
        """在非空且有用户身份的搜索中核验模型与索引，再绑定公共召回库。

        任一校验失败均向 LawSearchService 抛出异常，由服务保留依赖失败状态。
        """
        import chromadb
        import httpx

        directory, digest = values.get("LAW_FULL_INDEX_DIRECTORY"), values.get("LAW_FULL_EMBEDDING_DIGEST")
        if not directory or not digest or values.get("EMBED_MODEL_TYPE", "OLLAMA") != "OLLAMA":
            raise ValueError("全量 RAG 需要独立目录、Ollama embedding 与明确 digest")
        model = values.get("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
        collection = values.get("LAW_FULL_INDEX_COLLECTION", "criminal_law_full")
        corpus_path = Path(values.get("LAW_FULL_CORPUS_PATH", str(FULL_CORPUS_PATH)))
        base_url = values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

        # 同名模型可能更换权重，必须核对服务实际返回的 digest。
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            response = await client.get(base_url + "/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
        if not any(item.get("name") == model and item.get("digest") == digest for item in models):
            raise ValueError("实际 embedding 模型 digest 不一致")

        # 完整性核验包含同步磁盘读取和向量检查，放到工作线程避免阻塞事件循环。
        manifest = await asyncio.to_thread(verify_full_index, corpus_path, Path(directory), collection, model, digest)
        if manifest["corpus_sha256"] != repository.law_data.get("metadata", {}).get("corpus_sha256"):
            raise ValueError("controlled corpus does not match the bound index")

        client = await asyncio.to_thread(chromadb.PersistentClient, path=str(Path(directory).resolve()))
        store = await asyncio.to_thread(
            ChromaStore.open_existing, ChromaSource(repository.source_id, client, collection)
        )

        async def embed_query(query: str) -> list[float]:
            """请求单条查询向量，禁用隐式截断并校验数量、维度与有限数值。"""
            async with httpx.AsyncClient(trust_env=False, timeout=60) as http:
                response = await http.post(
                    base_url + "/api/embed", json={"model": model, "input": [query], "truncate": False}
                )
                response.raise_for_status()
                vectors = response.json()["embeddings"]
            if len(vectors) != 1:
                raise ValueError("query embedding count invalid")
            return validate_embedding(vectors[0], expected_dimension=manifest["embedding_dimension"])

        vector = ChromaVectorRetriever(store, embed_query, manifest["embedding_dimension"])
        # 缓存身份包含索引位置、来源和模型版本，防止不同向量空间共用统计实例。
        bm25 = owned.bm25(
            store, identity=(str(Path(directory).resolve()), collection, repository.source_id, model, digest)
        )
        return RetrievalPipeline(vector, bm25=bm25, expander=expander, reranker=scorer, config=config)

    return LawSearchService(
        repository,
        template,
        preflight=preflight,
        required_index=True,
        rerank_protocol="qwen3-yes-no-token-ids-v1" if scorer is not None and not unsupported else None,
        integrity_metadata={
            "embedding_model": values.get("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b"),
            "embedding_model_digest": values.get("LAW_FULL_EMBEDDING_DIGEST"),
        },
    )
