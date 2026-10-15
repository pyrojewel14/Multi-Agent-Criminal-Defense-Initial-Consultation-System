"""法律评分配置与每请求全量索引完整性绑定，不持有模型生命周期。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from pathlib import Path

from app.infrastructure.llm.config import RerankerConfig
from app.knowledge.law.corpus import FULL_CORPUS_PATH
from app.knowledge.law.index import verify_full_index
from app.knowledge.law.repository import LawRepository
from app.knowledge.retrieval.pipeline import RetrievalPipeline
from app.knowledge.retrieval.vector import ChromaVectorRetriever
from app.knowledge.storage.chroma import ChromaSource, ChromaStore, validate_embedding


class RejectedLawScorer:
    """在可选重排阶段拒绝不受支持的法律评分配置，保留融合顺序。"""

    probability_scores = True

    async def score(self, query, documents, *, timeout_s):
        """不创建模型，由 Pipeline 将配置错误标记为重排降级。"""
        raise ValueError("full reranker requires configured Qwen3-Reranker-0.6B")


class LawRerankerState:
    """独立记录法律配置错误，不改变通用模型资源的可用性。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._model_name = None

    def reject(self, config: RerankerConfig) -> RejectedLawScorer:
        """保留配置错误直到领域资源清理，并返回轻量拒绝评分器。"""
        with self._lock:
            self._model_name = config.model_name
        return RejectedLawScorer()

    def readiness(self):
        """只返回已记录的配置状态，不访问模型或外部服务。"""
        with self._lock:
            model = self._model_name
        if model is None:
            return None
        return {"status": "error", "available": False, "model": model, "error": "ValueError", "type": "causal_lm"}

    def clear(self):
        """清除已记录的法律配置状态。"""
        with self._lock:
            self._model_name = None


def create_full_law_preflight(
    repository: LawRepository, values: Mapping[str, str], template: RetrievalPipeline, *, model_manager, bm25
):
    """绑定显式依赖，每次执行重新核验实际模型、完整索引与受控正文。"""
    values = dict(values)

    async def preflight() -> RetrievalPipeline:
        """在非空且有用户身份的搜索中核验模型与索引，再绑定公共召回库。

        任一校验失败均向 LawSearchService 抛出异常，由服务保留依赖失败状态。
        """
        import chromadb

        directory, digest = values.get("LAW_FULL_INDEX_DIRECTORY"), values.get("LAW_FULL_EMBEDDING_DIGEST")
        if not directory or not digest or values.get("EMBED_MODEL_TYPE", "OLLAMA") != "OLLAMA":
            raise ValueError("全量 RAG 需要独立目录、Ollama embedding 与明确 digest")
        model = values.get("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
        collection = values.get("LAW_FULL_INDEX_COLLECTION", "criminal_law_full")
        corpus_path = Path(values.get("LAW_FULL_CORPUS_PATH", str(FULL_CORPUS_PATH)))
        base_url = values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

        # 同名模型可能更换权重，必须核对服务实际返回的 digest。
        ollama = model_manager.get_ollama(base_url)
        models = (await ollama.tags(timeout_s=10)).get("models", [])
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
            response = await ollama.embed(model, [query], truncate=False, timeout_s=60)
            vectors = response["embeddings"]
            if len(vectors) != 1:
                raise ValueError("query embedding count invalid")
            return validate_embedding(vectors[0], expected_dimension=manifest["embedding_dimension"])

        vector = ChromaVectorRetriever(store, embed_query, manifest["embedding_dimension"])
        # 缓存身份包含索引位置、来源和模型版本，防止不同向量空间共用统计实例。
        keyword = bm25(
            store, identity=(str(Path(directory).resolve()), collection, repository.source_id, model, digest)
        )
        return RetrievalPipeline(
            vector, bm25=keyword, expander=template.expander, reranker=template.reranker, config=template.config
        )

    return preflight
