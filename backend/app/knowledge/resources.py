"""复用领域检索缓存和文档服务，不持有或关闭模型资源。"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from app.knowledge.law.binding import LawRerankerState
from app.knowledge.retrieval.bm25 import BM25Retriever


class KnowledgeResources:
    """按来源复用有界 BM25 缓存，原子获取文档服务及法律配置状态。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._document_lock = threading.Lock()
        self._document_service = None
        self._bm25: OrderedDict[tuple[Any, ...], BM25Retriever] = OrderedDict()
        self._law_reranker = LawRerankerState()

    def bm25(self, store, *, identity: tuple[Any, ...]) -> BM25Retriever:
        """缓存独立绑定的检索器，最多保留八项，不能切换已有实例的数据源。"""
        with self._lock:
            if identity not in self._bm25:
                self._bm25[identity] = BM25Retriever(store)
            self._bm25.move_to_end(identity)
            while len(self._bm25) > 8:
                self._bm25.popitem(last=False)
            return self._bm25[identity]

    def document_service(self, factory):
        """首次获取时在锁内组装服务，其他调用复用同一实例。"""
        with self._document_lock:
            if self._document_service is None:
                self._document_service = factory()
            return self._document_service

    def reject_law_scorer(self, config):
        """将法律协议拒绝状态交给领域对象，不构造模型资源。"""
        return self._law_reranker.reject(config)

    def law_reranker_readiness(self):
        """返回法律配置错误快照，由应用与模型状态独立聚合。"""
        return self._law_reranker.readiness()

    def close(self):
        """清理本对象持有的领域缓存，不关闭借用的 ModelManager。"""
        with self._lock:
            self._bm25.clear()
        with self._document_lock:
            self._document_service = None
        self._law_reranker.clear()

    async def aclose(self):
        """保留异步清理入口，仅执行轻量领域缓存清理。"""
        self.close()


knowledge_resources = KnowledgeResources()
