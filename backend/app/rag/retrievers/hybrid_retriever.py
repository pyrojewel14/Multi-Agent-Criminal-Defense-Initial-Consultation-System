import asyncio
from chromadb.api.types import Where
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from app.utils.config import chroma_config

from .empty_retriever import EmptyRetriever


class HybridRetriever:
    """混合检索器（BM25 + 向量检索）"""

    def __init__(self, vectors_store: Chroma):
        self.vectors_store = vectors_store

    async def get_bm25_retriever(self, user_id: str | None = None, include_public: bool = False):
        """获取 BM25 检索器。

        Args:
            user_id: 用户 ID，必须提供，否则返回 None。
            include_public: 是否包含公共文档。

        Returns:
            BM25Retriever 实例。
        """
        if not user_id and not include_public:
            return None

        where_filter: Where
        if user_id and include_public:
            where_filter = {'$or': [{'user_id': user_id}, {'is_public': True}]}
        elif user_id:
            where_filter = {'user_id': user_id}
        else:
            where_filter = {'is_public': True}

        all_docs_result = await asyncio.to_thread(
            self.vectors_store.get,
            include=['documents', 'metadatas'],
            where=where_filter
        )

        if not all_docs_result['documents']:
            return None

        documents = []
        seen_content = set()
        for i, doc_content in enumerate(all_docs_result['documents']):
            if doc_content in seen_content:
                continue
            seen_content.add(doc_content)
            metadata = all_docs_result['metadatas'][i] if i < len(all_docs_result['metadatas']) else {}
            documents.append(Document(page_content=doc_content, metadata=metadata))

        if documents:
            bm25_k = min(chroma_config['k'], 2)
            bm25_retriever = BM25Retriever.from_documents(
                documents=documents,
                k=bm25_k
            )
            return bm25_retriever
        else:
            return None

    async def _get_all_documents(self) -> list[Document]:
        """获取向量库中的所有文档。

        Returns:
            文档列表。
        """
        all_docs = await asyncio.to_thread(
            self.vectors_store.get,
            include=['documents', 'metadatas']
        )
        documents = []
        for i, doc in enumerate(all_docs['documents']):
            metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
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
            filter_dict = {'$or': [{'user_id': user_id}, {'is_public': True}]}
        elif user_id:
            filter_dict = {'user_id': user_id}
        else:
            filter_dict = {'is_public': True}

        if query_length >= 200:
            vector_retriever = self.vectors_store.as_retriever(
                search_type='similarity',
                search_kwargs={'k': chroma_config['k'], 'filter': filter_dict},
            )
            return vector_retriever

        vec_k = chroma_config['k']
        vector_retriever = self.vectors_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': vec_k, 'filter': filter_dict},
        )

        bm25_retriever = await self.get_bm25_retriever(user_id, include_public)

        if bm25_retriever:
            weights = self.get_dynamic_weights(query)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=weights
            )
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
