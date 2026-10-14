"""按当前授权快照构建并有界缓存唯一的 BM25 统计索引。"""

from __future__ import annotations

import asyncio
import math
import threading
from collections import OrderedDict
from dataclasses import replace

from rank_bm25 import BM25Okapi

from app.knowledge.storage.chroma import ChromaStore, CorpusSnapshot

from . import tokenization
from .models import ChannelResult, RetrievalRequest, StageStatus
from .vector import candidate_from_row, scope_where


class BM25Retriever:
    """只复用统计索引；每次请求都重新读取绑定数据源的授权快照。"""

    def __init__(self, store: ChromaStore, *, cache_size: int = 8):
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer")
        self.store = store
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple, BM25Okapi | None] = OrderedDict()
        self._lock = threading.Lock()

    async def retrieve(self, request: RetrievalRequest) -> ChannelResult:
        """先检查来源与授权，再在线程中读取快照、切词和评分；依赖异常向上传播。"""
        if request.scope.source_id != self.store.source.source_id:
            raise ValueError("request source does not match the bound store")
        if not request.scope.user_id and not request.scope.include_public:
            return ChannelResult("bm25", (), {}, StageStatus("skipped_unauthorized"))
        if not request.query.strip():
            return ChannelResult("bm25", (), {}, StageStatus("skipped_empty"))
        where = scope_where(request.scope)
        return await asyncio.to_thread(self._retrieve, request, where)

    def _index(self, request: RetrievalRequest, snapshot: CorpusSnapshot) -> BM25Okapi | None:
        """将数据源、授权范围、语料指纹和分词版本共同作为缓存身份。"""
        scope = request.scope
        cache_key = (snapshot.source_id, scope.user_id, scope.include_public,
                     snapshot.fingerprint, tokenization.TOKENIZER_VERSION)
        with self._lock:
            if cache_key not in self._cache:
                tokens = [tokenization.chinese_tokens(row.content) for row in snapshot.rows]
                self._cache[cache_key] = BM25Okapi(tokens) if any(tokens) else None
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return self._cache[cache_key]

    def _retrieve(self, request: RetrievalRequest, where: dict) -> ChannelResult:
        """先读取最新快照，读取失败时不得将缓存当作当前语料返回。"""
        snapshot = self.store.read_snapshot(where=where)
        if snapshot.source_id != request.scope.source_id:
            raise ValueError("snapshot source does not match request source")
        index = self._index(request, snapshot)
        query_tokens = tokenization.chinese_tokens(request.query)
        if index is None or not query_tokens:
            return ChannelResult("bm25", (), {}, StageStatus("executed"))
        scores = index.get_scores(query_tokens)
        matches = [(row, float(score)) for row, score in zip(snapshot.rows, scores)
                   if math.isfinite(float(score)) and score > 0]
        matches.sort(key=lambda pair: (-pair[1], pair[0].row_id))
        candidates = []
        raw_scores = {}
        for rank, (row, score) in enumerate(matches[:request.recall_k], 1):
            candidate = candidate_from_row(row, source_id=snapshot.source_id, channel="bm25")
            candidates.append(replace(candidate, recall_ranks={"bm25": rank},
                                      recall_scores={"bm25": score}))
            raw_scores[candidate.candidate_id] = score
        return ChannelResult("bm25", tuple(candidates), raw_scores, StageStatus("executed"))
