"""持有受控法条资产并提供规范条号索引及精确读取。"""

import copy
from typing import Any, Dict

from .identifiers import _normalize_article_number


def _build_article_index(law_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """构建归一化法条编号到验证快照条目的索引。"""
    index = {}
    for chapter in law_data.get("chapters", []):
        for article in chapter.get("articles", []):
            normalized = _normalize_article_number(article.get("article_number", ""))
            if normalized:
                index[normalized] = {**article, "chapter": chapter.get("chapter", "")}
                if law_data.get("metadata", {}).get("schema_version") == "full-text-v1":
                    index[normalized]["corpus_sha256"] = law_data["metadata"].get("corpus_sha256", "")
                    index[normalized]["corpus_version"] = law_data["metadata"]["dataset_version"]
    return index


def is_article_in_force(article: Dict[str, Any]) -> bool:
    """排除快照中明确标记失效或正文已删去的条文。"""
    return article.get("status") in (None, "active") and article.get("content", "").strip() not in (
        "（删去）",
        "(删去)",
    )


class LawRepository:
    """复制受控资产，按语料版本绑定稳定来源身份。"""

    def __init__(self, law_data: Dict[str, Any], *, source_id: str | None = None):
        self.law_data = copy.deepcopy(law_data)
        self.article_index = _build_article_index(self.law_data)
        metadata = self.law_data.get("metadata", {})
        self.source_id = (
            source_id
            or f"law:{metadata.get('dataset_id', 'criminal-law')}:{metadata.get('dataset_version', 'snapshot')}"
        )

    def get(self, article_id: str) -> Dict[str, Any] | None:
        """只返回仍有效的受控条文，不存在和删去条文返回空。"""
        article = self.article_index.get(_normalize_article_number(article_id))
        return copy.deepcopy(article) if article is not None and is_article_in_force(article) else None
