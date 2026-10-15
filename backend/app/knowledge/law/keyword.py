"""将受控法条关键词规则转换为统一召回通道。"""

from typing import Any

from app.knowledge.retrieval.models import CandidateOrigin, ChannelResult, RetrievalCandidate, StageStatus
from app.knowledge.retrieval.vector import candidate_identity

from .identifiers import _normalize_article_number
from .repository import LawRepository, is_article_in_force
from .verification import is_lawref_eligible


def flatten_fact_terms(value: Any) -> list[str]:
    """展开结构化事实供旧调用方构造查询，不添加案件事实。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [term for item in value.values() for term in flatten_fact_terms(item)]
    if isinstance(value, (list, tuple, set)):
        return [term for item in value for term in flatten_fact_terms(item)]
    return [str(value)]


def facts_query(facts: dict[str, Any]) -> str:
    """沿用行为与后果字段生成原查询。"""
    return " ".join(
        flatten_fact_terms(facts.get("behavior_sequence", [])) + flatten_fact_terms(facts.get("consequence", ""))
    )


def facts_terms(facts: dict[str, Any]) -> list[str]:
    """保留结构化行为及后果的原始词项供受控关键词适配使用。"""
    return flatten_fact_terms(facts.get("behavior_sequence", [])) + flatten_fact_terms(facts.get("consequence", ""))


class LawKeywordRetriever:
    """只召回受控资产；业务关键词与通用 BM25 分开。"""

    def __init__(self, repository: LawRepository):
        self.repository = repository

    def candidate(self, article: dict[str, Any], *, exact: bool = False, score: float = 0) -> RetrievalCandidate:
        """受控正文使用独立来源身份，失败 RAG 不会被同条关键词洗白。"""
        number = _normalize_article_number(article["article_number"])
        cid = candidate_identity(self.repository.source_id, f"keyword:{number}", article["content"])
        metadata = {**article, "retrieval_method": "article_id_json" if exact else "keyword", "relevance_score": score}
        return RetrievalCandidate(
            cid,
            self.repository.source_id,
            article["content"],
            metadata,
            origins=(CandidateOrigin(self.repository.source_id, cid, "exact" if exact else "keyword", metadata),),
        )

    def recall(self, query: str, k: int, *, terms: list[str] | None = None) -> ChannelResult:
        """沿用罪名、标签、关键词和正文匹配分召回，最终排序由 Pipeline 完成。"""
        terms = (
            [term.strip().lower() for term in terms if term.strip()]
            if terms is not None
            else [term.lower() for term in query.split() if term]
        )
        matched = []
        for number, article in self.repository.article_index.items():
            if not is_article_in_force(article) or (
                not is_lawref_eligible(article) and not article.get("text_provenance")
            ):
                continue
            title = (article.get("title") or article.get("display_title", "")).lower()
            tags = " ".join(article.get("charge_tags", [])).lower()
            keywords = " ".join(article.get("common_keywords", [])).lower()
            content = article.get("content", "").lower()
            score = sum(
                (100 if _normalize_article_number(term) == number else 0)
                + (3 if term in title else 0)
                + (2 if term in tags else 0)
                + (1 if term in keywords else 0)
                + (0.5 if term in content else 0)
                for term in terms
            )
            if score > 0:
                matched.append((article, score))
        matched.sort(key=lambda pair: -pair[1])
        candidates = tuple(self.candidate(article, score=score) for article, score in matched[:k])
        return ChannelResult(
            "keyword",
            candidates,
            {c.candidate_id: c.metadata["relevance_score"] for c in candidates},
            StageStatus("executed" if terms else "skipped_empty"),
        )
