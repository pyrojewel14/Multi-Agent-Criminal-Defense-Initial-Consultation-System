"""共享检索记录，保存候选身份、阶段状态及不可变的请求快照。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .errors import safe_error_type


def _freeze(value: Any) -> Any:
    """复制并冻结嵌套映射、序列和集合，避免调用方修改已提交的记录。"""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _positive_count(name: str, value: int) -> None:
    """计数只接受正整数，避免将布尔值误作候选数量。"""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class RetrievalScope:
    """由服务端绑定的数据源身份及访问范围。"""

    source_id: str
    user_id: str | None
    include_public: bool


@dataclass(frozen=True)
class RetrievalRequest:
    """保留原查询与本次搜索的数量参数。"""

    query: str
    scope: RetrievalScope
    recall_k: int
    top_k: int

    def __post_init__(self) -> None:
        _positive_count("recall_k", self.recall_k)
        _positive_count("top_k", self.top_k)


@dataclass(frozen=True)
class CandidateOrigin:
    """保留一个原始通道来源，不随融合代表的选择改写。"""

    source_id: str
    candidate_id: str
    channel: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class RetrievalCandidate:
    """未经领域核验的候选正文及各阶段排名记录。"""

    candidate_id: str
    source_id: str
    content: str
    metadata: Mapping[str, Any]
    rank: int | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    recall_ranks: Mapping[str, int] = field(default_factory=dict)
    recall_scores: Mapping[str, float] = field(default_factory=dict)
    origins: tuple[CandidateOrigin, ...] = ()

    def __post_init__(self) -> None:
        for name in ("metadata", "recall_ranks", "recall_scores"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "origins", tuple(self.origins))


_STAGE_STATES = frozenset({
    "executed", "disabled", "unconfigured", "skipped_empty", "skipped_unauthorized",
    "skipped_exact", "failed", "timeout", "busy", "budget_exhausted",
})


@dataclass(frozen=True)
class StageStatus:
    """区分跳过、降级和失败，不以空候选推断执行结果。"""

    state: str
    error_type: str | None = None
    degraded: bool = False

    def __post_init__(self) -> None:
        if self.state not in _STAGE_STATES:
            raise ValueError(f"unknown retrieval stage state: {self.state}")
        object.__setattr__(self, "error_type", safe_error_type(self.error_type))


@dataclass(frozen=True)
class ChannelResult:
    """按召回顺序保存通道候选，原始分数始终按候选 ID 对齐。"""

    name: str
    candidates: tuple[RetrievalCandidate, ...]
    scores: Mapping[str, float]
    status: StageStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "scores", _freeze(self.scores))


@dataclass(frozen=True)
class RecallPool:
    """保存尚未最终截断的通道池及必需依赖失败状态。"""

    channels: Mapping[str, ChannelResult]
    retrieval_status: Mapping[str, Any]
    dependency_failed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", _freeze(self.channels))
        object.__setattr__(self, "retrieval_status", _freeze(self.retrieval_status))


@dataclass(frozen=True)
class SearchResult:
    """保存最终候选；空结果同样携带完整执行与依赖状态。"""

    candidates: tuple[RetrievalCandidate, ...]
    retrieval_status: Mapping[str, Any]
    dependency_failed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "retrieval_status", _freeze(self.retrieval_status))
