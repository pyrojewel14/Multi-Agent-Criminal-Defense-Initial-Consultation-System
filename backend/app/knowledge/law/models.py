"""分离法律来源资格与检索分数，在工具边界显式生成普通 JSON 容器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.knowledge.retrieval.models import RecallPool, SearchResult, _freeze


class LawDataSource(str, Enum):
    """维持既有覆盖 schema 允许的法条来源类别。"""

    RAG_UNVERIFIED = "rag_unverified"
    LLM_EXTRACTED = "llm_extracted"
    RAG_VERIFIED = "rag_verified"
    JSON_KEYWORD = "json_keyword"


class LawKnowledgeDataError(RuntimeError):
    """法条资产缺失、损坏或不符合来源及标注契约。"""


def plain(value: Any) -> Any:
    """递归还原冻结容器，避免 asdict 深拷贝只读映射失败。"""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class LawVerification:
    """原始候选的核验结果，不随融合代表或相关性评分改写。"""

    article_id: str
    source: str
    coverage_eligible: bool
    required_elements: tuple[Any, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_elements", _freeze(self.required_elements))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


def verification_dict(verification: LawVerification) -> dict[str, Any]:
    """转换来源核验，供工具保留融合前的失败来源证据。"""
    return {
        "article_id": verification.article_id,
        "source": verification.source,
        "coverage_eligible": verification.coverage_eligible,
        "required_elements": plain(verification.required_elements),
        "provenance": plain(verification.provenance),
    }


@dataclass(frozen=True)
class VerifiedLawPool:
    """独立核验映射与尚未排序的完整召回池。"""

    pool: RecallPool
    verifications: Mapping[str, LawVerification]

    def __post_init__(self) -> None:
        object.__setattr__(self, "verifications", _freeze(self.verifications))


@dataclass(frozen=True)
class LawSearchResult:
    """保存最终检索结果，以及核验后保留的全部原始候选资格。"""

    search: SearchResult
    verifications: Mapping[str, LawVerification]

    def __post_init__(self) -> None:
        object.__setattr__(self, "verifications", _freeze(self.verifications))

    def as_laws(self) -> list[dict[str, Any]]:
        """生成 Agent 所需普通容器，保留独立分数与来源核验。"""
        result = []
        for candidate in self.search.candidates:
            verification = self.verifications[candidate.candidate_id]
            origins = []
            # 融合代表之外的原始来源也保留核验记录，避免合并后掩盖不可信来源。
            for origin in candidate.origins:
                item = {
                    "source_id": origin.source_id,
                    "candidate_id": origin.candidate_id,
                    "channel": origin.channel,
                    "metadata": plain(origin.metadata),
                }
                original = self.verifications.get(origin.candidate_id)
                if original is not None:
                    item["verification"] = verification_dict(original)
                origins.append(item)
            result.append(
                {
                    **plain(candidate.metadata),
                    "candidate_id": candidate.candidate_id,
                    "source_id": candidate.source_id,
                    "content": candidate.content,
                    "data_source": verification.source,
                    "coverage_eligible": verification.coverage_eligible,
                    "retrieval_method": candidate.metadata.get("retrieval_method")
                    or self.search.retrieval_status.get("method"),
                    "required_elements": plain(verification.required_elements),
                    "rank": candidate.rank,
                    "fusion_score": candidate.fusion_score,
                    "rerank_score": candidate.rerank_score,
                    "recall_ranks": plain(candidate.recall_ranks),
                    "recall_scores": plain(candidate.recall_scores),
                    "origins": origins,
                    "provenance": plain(verification.provenance),
                    "retrieval_status": plain(
                        self.search.retrieval_status or candidate.metadata.get("retrieval_status", {})
                    ),
                }
            )
        return result
