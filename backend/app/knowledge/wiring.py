"""显式组装法律、上传文档与借用的模型依赖，不持有模型资源。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import partial
from typing import Any

from app.infrastructure.llm.config import ChatConfig, EmbeddingConfig, RerankerConfig
from app.infrastructure.llm.manager import ModelManager, get_model_manager
from app.knowledge.law.binding import create_full_law_preflight
from app.knowledge.law.corpus import load_criminal_law_data
from app.knowledge.law.repository import LawRepository
from app.knowledge.law.service import LawSearchService
from app.knowledge.resources import KnowledgeResources, knowledge_resources
from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from app.knowledge.retrieval.vector import ChromaVectorRetriever
from app.knowledge.storage.chroma import ChromaSource, ChromaStore


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
    model_manager: ModelManager | None = None,
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
    manager = model_manager if model_manager is not None else get_model_manager()

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
    embedding = (
        None
        if injected_embeddings
        else manager.get_embedding(EmbeddingConfig.from_env(values), factory=embedding_factory)
    )
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
            else (manager.get_reranker(RerankerConfig.from_env(values), kind=reranker_type) if config.rerank else None)
        )
        expander = None
        if config.hyde:
            from app.knowledge.retrieval.hyde import HyDEExpander, generate_hyde

            model_name = values.get("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
            expander = HyDEExpander(
                partial(
                    generate_hyde,
                    client=manager.get_ollama(values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
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
    return knowledge_resources.document_service(lambda: build_document_services(create_missing=True))


def get_document_summary_service(*, chat_model=None, model_manager=None, thinking_callback=None):
    """组装既有摘要提示词与输出解析链，检索结果由调用方显式传入。"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    from app.infrastructure.config.prompts import prompt_loader
    from app.knowledge.documents.summary import DocumentSummaryService

    if chat_model is None:
        manager = model_manager if model_manager is not None else get_model_manager()
        chat_model = manager.get_chat(ChatConfig.from_env(dict(os.environ), streaming=True, top_p=0.7))
    chain = PromptTemplate.from_template(prompt_loader.load("rag_summary_prompt")) | chat_model | StrOutputParser()
    return DocumentSummaryService(chain, thinking_callback=thinking_callback)


def build_law_search_service(
    *,
    law_data: dict[str, Any] | None = None,
    profile: str | None = None,
    pipeline: RetrievalPipeline | None = None,
    source_id: str | None = None,
    resources: KnowledgeResources | None = None,
    model_manager: ModelManager | None = None,
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
    manager = model_manager if model_manager is not None else get_model_manager()

    # full 只接受既定 Qwen 评分协议；不支持的配置留到可选重排阶段报告降级。
    unsupported = profile == "full" and scorer_config.model_name != "Qwen/Qwen3-Reranker-0.6B"
    scorer = (
        (owned.reject_law_scorer(scorer_config) if unsupported else manager.get_reranker(scorer_config))
        if config.rerank
        else None
    )

    expander = None
    if config.hyde:
        from app.knowledge.retrieval.hyde import HyDEExpander, generate_hyde

        model_name = values.get("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
        expander = HyDEExpander(
            partial(
                generate_hyde,
                client=manager.get_ollama(values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
                model_name=model_name,
            ),
            model_name=model_name,
        )

    # 此时尚未打开向量库；full 的实际召回组件在预检通过后才绑定。
    template = RetrievalPipeline(None, expander=expander, reranker=scorer, config=config)
    if profile != "full":
        return LawSearchService(repository, template)

    preflight = create_full_law_preflight(repository, values, template, model_manager=manager, bm25=owned.bm25)

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
