"""由存储与异步 Embedding 参数化的唯一向量召回实现。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

from app.knowledge.storage.chroma import ChromaStore, StoredRow, validate_embedding

from .models import CandidateOrigin, ChannelResult, RetrievalCandidate, RetrievalRequest, RetrievalScope, StageStatus


def scope_where(scope: RetrievalScope) -> dict[str, Any]:
    """为向量和 BM25 快照生成相同授权过滤；无权限时拒绝全库读取。"""
    if scope.user_id and scope.include_public:
        return {"$or": [{"user_id": scope.user_id}, {"is_public": True}]}
    if scope.user_id:
        return {"user_id": scope.user_id}
    if scope.include_public:
        return {"is_public": True}
    raise PermissionError("retrieval scope has no authorized documents")


def candidate_identity(source_id: str, row_id: str, content: str) -> str:
    """以稳定来源、存储行 ID 与正文指纹生成身份，不输出本地路径。"""
    payload = [source_id, row_id, hashlib.sha256(content.encode("utf-8")).hexdigest()]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def candidate_from_row(row: StoredRow, *, source_id: str, channel: str) -> RetrievalCandidate:
    """统一各通道候选身份；存储行 ID 用于融合，原始 metadata 保留在来源中。"""
    cid = candidate_identity(source_id, row.row_id, row.content)
    return RetrievalCandidate(
        cid, source_id, row.content, {**row.metadata, "row_id": row.row_id},
        origins=(CandidateOrigin(source_id, cid, channel, row.metadata),),
    )


class ChromaVectorRetriever:
    """只处理一个显式数据源；依赖失败和取消向调用方传播。"""

    def __init__(self, store: ChromaStore, embed_query: Callable[[str], Awaitable[Sequence[float]]],
                 expected_dimension: int | None):
        if expected_dimension is not None and (type(expected_dimension) is not int or expected_dimension <= 0):
            raise ValueError("expected embedding dimension must be a positive integer or None")
        self.store = store
        self.embed_query = embed_query
        self.expected_dimension = expected_dimension

    async def retrieve(self, request: RetrievalRequest) -> ChannelResult:
        """先校验来源和授权，再嵌入原查询；同步存储查询交给线程池。"""
        if request.scope.source_id != self.store.source.source_id:
            raise ValueError("request source does not match the bound store")
        if not request.scope.user_id and not request.scope.include_public:
            return ChannelResult("vector", (), {}, StageStatus("skipped_unauthorized"))
        if not request.query.strip():
            return ChannelResult("vector", (), {}, StageStatus("skipped_empty"))
        where = scope_where(request.scope)
        vector = validate_embedding(await self.embed_query(request.query), expected_dimension=self.expected_dimension)
        rows = await asyncio.to_thread(self.store.query_by_vector, vector, k=request.recall_k, where=where)
        candidates = []
        scores = {}
        for rank, row in enumerate(rows, 1):
            if row.distance is None:
                raise ValueError("vector recall requires a native Chroma distance")
            candidate = candidate_from_row(row, source_id=self.store.source.source_id, channel="vector")
            candidates.append(replace(
                candidate,
                recall_ranks={"vector": rank}, recall_scores={"vector": row.distance},
            ))
            scores[candidate.candidate_id] = row.distance
        return ChannelResult("vector", tuple(candidates), scores, StageStatus("executed"))
