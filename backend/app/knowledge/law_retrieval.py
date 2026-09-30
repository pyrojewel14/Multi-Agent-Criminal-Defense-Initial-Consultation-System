"""法条候选召回、RAG 访问和快照核验。"""

import re
from typing import Any, Dict, List

from app.infrastructure.logging import get_logger
from app.infrastructure.observability.tracing import SessionBudgetExceeded
from app.knowledge.law_knowledge import _normalize_article_number, is_article_in_force, is_lawref_eligible
from app.knowledge.schemas import LawDataSource

_logger = get_logger("LawRetrieval")


class LawSearchResults(list):
    """携带依赖失败标记的法条检索结果，保持与普通列表兼容。"""

    def __init__(
        self,
        values: List[Dict[str, Any]] | None = None,
        *,
        dependency_failed: bool = False,
    ):
        super().__init__(values or [])
        self.dependency_failed = dependency_failed


def _extract_article_number_from_text(text: str | None) -> str:
    """从法条文本内容中提取法条编号。

    Args:
        text: 法条文本内容，允许为空

    Returns:
        提取到的法条编号，未找到则返回空字符串
    """
    if not text:
        return ""
    # 匹配 "第X条" 格式（X 为中文数字或阿拉伯数字）
    match = re.search(r"第[一二三四五六七八九十百千零\d]+条(?:之[一二三四五六七八九十百千零\d]+)?", text)
    return match.group(0) if match else ""


def _verify_and_enrich_with_json(
    rag_results: List[Dict[str, Any]], article_index: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """用 JSON 知识库验证并增强 RAG 检索结果。

    从 RAG 结果中提取法条编号，在 JSON 知识库中精确匹配，
    仅用项目维护的标注替换 RAG 文档中未经核验的元数据。

    Args:
        rag_results: RAG 检索结果列表
        article_index: 法条编号到 JSON 条目的索引

    Returns:
        增强后的法条列表
    """
    enriched = []
    for rag_law in rag_results:
        article_number = rag_law.get("article_number", "")
        normalized = _normalize_article_number(article_number)
        json_match = article_index.get(normalized) if normalized else None

        if json_match and not is_article_in_force(json_match):
            continue

        full_match = bool(json_match and json_match.get("text_provenance"))
        if json_match is not None and full_match and (
            rag_law.get("corpus_sha256") != json_match.get("corpus_sha256")
            or rag_law.get("corpus_version") != json_match.get("corpus_version")
            or rag_law.get("content") != f"{json_match['article_number']} {json_match['content']}"
        ):
            enriched.append({**rag_law, "elements": [], "required_elements": [], "data_source": "rag_unverified"})
            continue
        if json_match is not None and full_match and not is_lawref_eligible(json_match):
            enriched.append({**json_match, "elements": [], "required_elements": [], "data_source": "text_only", "coverage_eligible": False, "retrieval_method": rag_law.get("retrieval_method", "rag")})
            continue
        if json_match and is_lawref_eligible(json_match):
            # 项目维护标注通过安全门槛后，才可增强 RAG 结果。
            enriched.append(
                {
                    **rag_law,
                    "title": json_match.get("title", rag_law.get("title", "")),
                    "content": json_match.get("content", rag_law.get("content", "")),
                    "required_elements": json_match.get("elements", []),
                    "elements": json_match.get("elements", []),
                    "base_sentence": json_match.get("base_sentence", ""),
                    "charge_tags": json_match.get("charge_tags", []),
                    "common_keywords": json_match.get("common_keywords", []),
                    "chapter": json_match.get("chapter", rag_law.get("chapter", "")),
                    "data_source": "rag_verified",
                    "annotation_usage": json_match.get("annotation_usage", "project_regression"),
                    "annotation_source": json_match.get("annotation_source", ""),
                }
            )
        else:
            # 未命中或标注未通过安全门槛时，只保留未验证的原始结果。
            enriched.append({**rag_law, "data_source": "rag_unverified"})

    verified_count = sum(1 for law in enriched if law.get("data_source") == "rag_verified")
    _logger.info(
        "【_verify_and_enrich_with_json】RAG 结果验证完成，%d/%d 条通过 JSON 精确匹配",
        verified_count,
        len(enriched),
    )
    return enriched


def _merge_and_deduplicate(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并两组法条结果并按法条编号去重。

    primary 中的结果优先，secondary 中仅补充 primary 未覆盖的法条。

    Args:
        primary: 主结果列表（优先级高）
        secondary: 补充结果列表（优先级低）

    Returns:
        合并去重后的法条列表
    """
    seen_numbers: Dict[str, int] = {}
    result = []

    for law in primary:
        num = _normalize_article_number(law.get("article_number", ""))
        key = num or law.get("title", "")
        if key and key not in seen_numbers:
            seen_numbers[key] = len(result)
            result.append(law)
        elif key:
            # 已存在，跳过
            pass
        else:
            # 无编号也无标题，直接加入
            result.append(law)

    for law in secondary:
        num = _normalize_article_number(law.get("article_number", ""))
        key = num or law.get("title", "")
        if key and key not in seen_numbers:
            seen_numbers[key] = len(result)
            result.append(law)

    return result


def _flatten_fact_terms(value: Any) -> List[str]:
    """将 FactDigger 的嵌套事实值展开为可检索文本。"""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        terms: List[str] = []
        for nested_value in value.values():
            terms.extend(_flatten_fact_terms(nested_value))
        return terms
    if isinstance(value, (list, tuple, set)):
        terms = []
        for item in value:
            terms.extend(_flatten_fact_terms(item))
        return terms
    return [str(value)]


async def search_laws_by_keyword(facts_structured: Dict[str, Any], law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """通过关键词搜索匹配的刑法条文。

    Args:
        facts_structured: 结构化的事实数据。
        law_data: 刑法条文数据。

    Returns:
        匹配的条文列表。
    """
    behavior_sequence = facts_structured.get("behavior_sequence", [])
    consequence = facts_structured.get("consequence", "")

    search_terms = _flatten_fact_terms(behavior_sequence)
    search_terms.extend(_flatten_fact_terms(consequence))

    matched_laws = []

    for chapter in law_data.get("chapters", []):
        for article in chapter.get("articles", []):
            full_article = bool(article.get("text_provenance"))
            eligible = is_lawref_eligible(article)
            if not is_article_in_force(article) or (not eligible and not full_article):
                continue
            charge_name = (article.get("title") or article.get("display_title", "")).lower()
            charge_tags = " ".join(article.get("charge_tags", [])).lower()
            content = article.get("content", "").lower()
            common_keywords = " ".join(article.get("common_keywords", [])).lower()

            relevance_score = 0
            matched_tags = []

            for term in search_terms:
                term_lower = term.lower()
                if _normalize_article_number(term) == _normalize_article_number(article.get("article_number", "")):
                    relevance_score += 100
                if term_lower in charge_name:
                    relevance_score += 3
                    matched_tags.append(f"罪名匹配: {term}")
                if term_lower in charge_tags:
                    relevance_score += 2
                    matched_tags.append(f"标签匹配: {term}")
                if term_lower in common_keywords:
                    relevance_score += 1
                    matched_tags.append(f"关键词匹配: {term}")
                if term_lower in content:
                    relevance_score += 0.5

            if relevance_score > 0:
                matched_laws.append(
                    {
                        "article_number": article.get("article_number", ""),
                        "title": article.get("title", ""),
                        "content": article.get("content", ""),
                        "required_elements": article.get("elements", []),
                        "elements": article.get("elements", []),
                        "base_sentence": article.get("base_sentence", ""),
                        "charge_tags": article.get("charge_tags", []),
                        "common_keywords": article.get("common_keywords", []),
                        "chapter": chapter.get("chapter", ""),
                        "relevance_score": relevance_score,
                        "matched_tags": matched_tags,
                        "data_source": LawDataSource.JSON_KEYWORD.value if eligible else "text_only",
                        "coverage_eligible": eligible,
                        "annotation_usage": article.get("annotation_usage", "project_regression"),
                        "annotation_source": article.get("annotation_source", ""),
                        "text_provenance": article.get("text_provenance", {}),
                        "annotations": article.get("annotations", {}),
                        "display_title": article.get("display_title", ""),
                    }
                )

    matched_laws.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    _logger.info("【search_laws_by_keyword】关键词匹配找到 %d 条相关法条", len(matched_laws))

    return matched_laws


async def search_laws_by_rag(facts_structured: Dict[str, Any], user_id: str | None) -> List[Dict[str, Any]]:
    """通过 RAG 向量检索搜索匹配的刑法条文。

    返回原始 RAG 文档内容，元数据由后续 _verify_and_enrich_with_json 通过
    JSON 知识库精确匹配来填充，避免逐条 LLM 调用的开销和幻觉风险。
    旧状态缺少 user_id 时跳过 RAG，避免用 session_id 冒充用户身份。

    Args:
        facts_structured: 结构化的事实数据。
        user_id: 当前认证用户 ID；为空时跳过 RAG。

    Returns:
        匹配的条文列表，元数据待 JSON 知识库验证增强。
    """
    if not user_id:
        _logger.warning("【search_laws_by_rag】user_id 为空，跳过 RAG 检索")
        return []

    try:
        import os
        if os.getenv("LAW_KNOWLEDGE_PROFILE", "snapshot") == "full":
            from app.knowledge.full_law_index import search_full_index
            return LawSearchResults(await search_full_index(facts_structured))
        from app.knowledge.rag.rag_service import RagService

        behavior_sequence = facts_structured.get("behavior_sequence", [])
        consequence = facts_structured.get("consequence", "")

        query_parts = _flatten_fact_terms(behavior_sequence)
        query_parts.extend(_flatten_fact_terms(consequence))

        query = " ".join(query_parts)
        if not query.strip():
            query = "刑事犯罪"

        rag_service = RagService(user_id=user_id, include_public=True)
        await rag_service.initialize_retriever(query)

        documents = await rag_service.retrieve_documents(query)

        matched_laws = []
        for doc in documents:
            if isinstance(doc, str):
                doc_content = doc
            elif hasattr(doc, "page_content"):
                doc_content = doc.page_content
            else:
                continue

            # 从文档内容中尝试提取法条编号（供后续 JSON 精确匹配使用）
            article_number = _extract_article_number_from_text(doc_content)

            matched_laws.append(
                {
                    "article_number": article_number,
                    "title": "",
                    "content": doc_content[:500],
                    "elements": [],
                    "base_sentence": "",
                    "charge_tags": [],
                    "common_keywords": [],
                    "chapter": "",
                    "relevance_score": 1.0,
                    "matched_tags": ["RAG 向量检索"],
                    "data_source": LawDataSource.RAG_UNVERIFIED.value,
                }
            )

        _logger.info("【search_laws_by_rag】RAG 检索找到 %d 条相关法条", len(matched_laws))
        return LawSearchResults(
            matched_laws,
            dependency_failed=getattr(rag_service, "retrieval_failed", False) is True,
        )

    except SessionBudgetExceeded:
        raise
    except Exception as e:
        _logger.error(
            "【search_laws_by_rag】RAG 检索失败: error_type=%s",
            type(e).__name__,
        )
        return LawSearchResults(dependency_failed=True)
