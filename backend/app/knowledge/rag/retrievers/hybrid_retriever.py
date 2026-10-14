import asyncio
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from chromadb.api.types import Where
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.infrastructure.config.rag import chroma_config
from app.knowledge.rag.tokenization import TOKENIZER_VERSION, chinese_tokens

from .empty_retriever import EmptyRetriever

_BM25_CACHE_SIZE = 8


class HybridRetriever:
    """混合检索器，按授权快照复用有界的 BM25 统计索引。"""

    def __init__(self, vectors_store: Chroma):
        self.vectors_store = vectors_store
        self._bm25_cache: OrderedDict[tuple[str | None, bool], tuple[str, Any]] = OrderedDict()
        self._bm25_cache_lock = threading.Lock()

    async def get_bm25_retriever(self, user_id: str | None = None, include_public: bool = False):
        """读取当前授权快照，并在线程池中复用或重建 BM25 索引。

        Args:
            user_id: 用户 ID；为空时仅在 include_public=True 时读取公共文档。
            include_public: 是否包含公共文档。

        Returns:
            独立的 BM25Retriever 实例；未授权或当前范围无文档时返回 None。
        """
        if not user_id and not include_public:
            return None

        return await asyncio.to_thread(self._get_bm25_retriever_sync, user_id, include_public)

    def _get_bm25_retriever_sync(self, user_id: str | None, include_public: bool):
        """每次核对授权快照指纹，最多保留八个范围的统计索引。"""
        scope = (user_id or None, include_public)
        where_filter: Where
        if user_id and include_public:
            where_filter = {"$or": [{"user_id": user_id}, {"is_public": True}]}
        elif user_id:
            where_filter = {"user_id": user_id}
        else:
            where_filter = {"is_public": True}

        all_docs_result = self.vectors_store.get(include=["documents", "metadatas"], where=where_filter)

        if not all_docs_result["documents"]:
            with self._bm25_cache_lock:
                self._bm25_cache.pop(scope, None)
            return None

        documents = []
        seen_content = set()
        for i, doc_content in enumerate(all_docs_result["documents"]):
            if doc_content in seen_content:
                continue
            seen_content.add(doc_content)
            metadata = all_docs_result["metadatas"][i] if i < len(all_docs_result["metadatas"]) else {}
            documents.append(Document(page_content=doc_content, metadata=metadata))

        snapshot = json.dumps(
            {
                "tokenizer_version": TOKENIZER_VERSION,
                "documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
        bm25_k = min(chroma_config["k"], 2)
        with self._bm25_cache_lock:
            cached = self._bm25_cache.get(scope)
            if cached is not None and cached[0] == fingerprint:
                vectorizer = cached[1]
            else:
                # 全部为标点等无 token 文本时，BM25 无法计算 IDF，交回向量检索。
                if not any(chinese_tokens(doc.page_content) for doc in documents):
                    self._bm25_cache.pop(scope, None)
                    return None
                vectorizer = BM25Retriever.from_documents(
                    documents=documents, k=bm25_k, preprocess_func=chinese_tokens
                ).vectorizer
                self._bm25_cache[scope] = (fingerprint, vectorizer)
            self._bm25_cache.move_to_end(scope)
            while len(self._bm25_cache) > _BM25_CACHE_SIZE:
                self._bm25_cache.popitem(last=False)

        # 只共享只读统计，文档容器与 k 每次独立，避免调用方修改污染后续检索。
        return BM25Retriever(
            vectorizer=vectorizer, docs=documents, k=bm25_k, preprocess_func=chinese_tokens
        )

    async def _get_all_documents(self) -> list[Document]:
        """获取向量库中的所有文档。

        Returns:
            文档列表。
        """
        all_docs = await asyncio.to_thread(self.vectors_store.get, include=["documents", "metadatas"])
        documents = []
        for i, doc in enumerate(all_docs["documents"]):
            metadata = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
            documents.append(Document(page_content=doc, metadata=metadata))
        return documents

    async def get_retriever(
        self,
        query: str | None = None,
        user_id: str | None = None,
        include_public: bool = False,
    ) -> BaseRetriever:
        """获取混合检索器（BM25 + 向量检索）。

        Args:
            query: 查询语句，用于动态调整权重。
            user_id: 用户 ID，用于过滤用户的文档，为空时只返回公共文档（若 include_public=True）。
            include_public: 是否包含公共文档。

        Returns:
            EnsembleRetriever 实例或单独的向量检索器。
        """
        if not user_id and not include_public:
            return EmptyRetriever()

        query_length = len(query) if query else 0

        if user_id and include_public:
            filter_dict = {"$or": [{"user_id": user_id}, {"is_public": True}]}
        elif user_id:
            filter_dict = {"user_id": user_id}
        else:
            filter_dict = {"is_public": True}

        if query_length >= 200:
            vector_retriever = self.vectors_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": chroma_config["k"], "filter": filter_dict},
            )
            return vector_retriever

        vec_k = chroma_config["k"]
        vector_retriever = self.vectors_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": vec_k, "filter": filter_dict},
        )

        bm25_retriever = await self.get_bm25_retriever(user_id, include_public)

        if bm25_retriever:
            weights = self.get_dynamic_weights(query)
            ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=weights)
            return ensemble_retriever
        else:
            return vector_retriever

    @staticmethod
    def get_dynamic_weights(query: str | None = None):
        """根据查询动态调整权重。

        法律文本 BM25 易产生噪音（法条间共用大量法律术语），
        对长查询（如 HyDE 生成的假设性文档）提高向量权重以抑制噪音。

        Args:
            query: 查询语句。

        Returns:
            权重列表 [向量检索权重, BM25 检索权重]。
        """
        if not query:
            return [0.8, 0.2]

        query_length = len(query)

        if query_length > 200:
            return [0.9, 0.1]
        elif query_length > 50:
            return [0.8, 0.2]
        elif query_length < 20:
            return [0.3, 0.7]
        else:
            return [0.6, 0.4]
