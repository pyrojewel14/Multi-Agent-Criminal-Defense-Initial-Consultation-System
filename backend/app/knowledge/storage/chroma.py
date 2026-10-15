"""显式绑定 Chroma 数据源，保留正文、metadata 和原始向量距离。"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    """复制并冻结存储快照，防止调用方修改 metadata 污染缓存依据。"""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    """将冻结容器还原为 Chroma 和稳定 JSON 可接受的普通容器。"""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def validate_embedding(vector: Sequence[float], *, expected_dimension: int | None = None) -> list[float]:
    """校验非空、有限且可存为 float32 的向量，不替调用方选择模型。"""
    if isinstance(vector, (str, bytes)):
        raise ValueError("embedding vector invalid")
    try:
        values = list(vector)
    except TypeError as exc:
        raise ValueError("embedding vector invalid") from exc
    if not values or (expected_dimension is not None and len(values) != expected_dimension):
        raise ValueError("embedding dimension invalid")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("embedding vector invalid")
        number = float(value)
        try:
            stored = struct.unpack("f", struct.pack("f", number))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError("embedding float32 overflow") from exc
        if not math.isfinite(number) or not math.isfinite(stored):
            raise ValueError("embedding vector must be finite")
        result.append(number)
    return result


@dataclass(frozen=True)
class ChromaSource:
    """由组装层指定的稳定来源标识、client 与 Collection 名称。"""

    source_id: str
    client: Any
    collection_name: str


@dataclass(frozen=True)
class StoredRow:
    """与检索契约独立的存储行；distance 保留 Chroma 原始距离。"""

    row_id: str
    content: str
    metadata: Mapping[str, Any]
    distance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class CorpusSnapshot:
    """授权语料的不可变快照与正文、metadata 指纹。"""

    source_id: str
    rows: tuple[StoredRow, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))


def _where(where: Mapping[str, Any]) -> dict[str, Any]:
    """空过滤条件不得被解释为全库访问；具体表达式由 Chroma 校验。"""
    if not isinstance(where, Mapping) or not where:
        raise ValueError("where must be a nonempty metadata filter")
    return _plain(where)


def _rows(
    ids: Sequence[str],
    documents: Sequence[str] | None,
    metadatas: Sequence[Any] | None,
    distances: Sequence[float] | None = None,
) -> tuple[StoredRow, ...]:
    """严格按存储 ID 对齐字段，拒绝不完整响应而非静默丢行。"""
    count = len(ids)
    if documents is None or metadatas is None or len(documents) != count or len(metadatas) != count:
        raise ValueError("Chroma documents/metadata response is not aligned")
    if distances is not None and len(distances) != count:
        raise ValueError("Chroma distance response is not aligned")
    if any(not isinstance(content, str) for content in documents):
        raise ValueError("Chroma document content is missing")
    return tuple(
        StoredRow(row_id, documents[i], metadatas[i] or {}, None if distances is None else distances[i])
        for i, row_id in enumerate(ids)
    )


class ChromaStore:
    """同步的原生 Chroma 读写边界，在线调用方须在线程池执行 I/O。"""

    def __init__(self, source: ChromaSource, collection: Any):
        self.source = source
        self._collection = collection

    @classmethod
    def open_existing(cls, source: ChromaSource) -> ChromaStore:
        """只打开指定 Collection，缺失时保留 Chroma 异常且不自动建库。"""
        collection = source.client.get_collection(name=source.collection_name, embedding_function=None)
        return cls(source, collection)

    @classmethod
    def create(cls, source: ChromaSource, *, metadata: Mapping[str, Any] | None = None) -> ChromaStore:
        """仅供显式构建使用，不绑定默认 Embedding 或覆盖现有 Collection。"""
        collection = source.client.create_collection(
            name=source.collection_name,
            embedding_function=None,
            metadata=None if metadata is None else _plain(metadata),
        )
        return cls(source, collection)

    def query_by_vector(self, vector: Sequence[float], *, k: int, where: Mapping[str, Any]) -> tuple[StoredRow, ...]:
        """使用显式查询向量召回，返回顺序与原始距离均由 Chroma 保留。"""
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        result = self._collection.query(
            query_embeddings=[validate_embedding(vector)],
            n_results=k,
            where=_where(where),
            include=["documents", "metadatas", "distances"],
        )
        return _rows(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])

    def get_by_metadata(self, *, where: Mapping[str, Any]) -> tuple[StoredRow, ...]:
        """按显式 metadata 范围读取，不执行 Embedding。"""
        result = self._collection.get(where=_where(where), include=["documents", "metadatas"])
        return _rows(result["ids"], result["documents"], result["metadatas"])

    def read_snapshot(self, *, where: Mapping[str, Any]) -> CorpusSnapshot:
        """按行 ID 排序并计算指纹，读取顺序变化不触发无效缓存重建。"""
        rows = tuple(sorted(self.get_by_metadata(where=where), key=lambda row: row.row_id))
        payload = {
            "source_id": self.source.source_id,
            "rows": [{"row_id": row.row_id, "content": row.content, "metadata": _plain(row.metadata)} for row in rows],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return CorpusSnapshot(self.source.source_id, rows, hashlib.sha256(raw.encode("utf-8")).hexdigest())

    def upsert_rows(self, rows: Sequence[StoredRow], *, embeddings: Sequence[Sequence[float]]) -> None:
        """写入调用方已计算的等维向量，校验整批后再执行存储副作用。"""
        rows, embeddings = tuple(rows), tuple(embeddings)
        if len(rows) != len(embeddings):
            raise ValueError("embedding count invalid")
        if not rows:
            return
        vectors = [validate_embedding(vector) for vector in embeddings]
        if any(len(vector) != len(vectors[0]) for vector in vectors):
            raise ValueError("embedding dimensions must match")
        self._collection.upsert(
            ids=[row.row_id for row in rows],
            documents=[row.content for row in rows],
            metadatas=[_plain(row.metadata) or None for row in rows],
            embeddings=vectors,
        )

    def delete_rows(self, *, where: Mapping[str, Any]) -> None:
        """仅删除显式指定的 metadata 范围，禁止空条件全库删除。"""
        self._collection.delete(where=_where(where))

    def clear_all_rows(self) -> None:
        """仅供管理员显式清空绑定集合，不重建集合或调用 Embedding。"""
        ids = self._collection.get(include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
