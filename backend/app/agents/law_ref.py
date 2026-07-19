import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from app.security.sensitive_filter import mask_pii
from app.utils.llm_gateway import llm_gateway
from app.utils.logger import get_logger
from app.utils.prompt_loader import prompt_loader

if TYPE_CHECKING:
    from app.state.consultation_state import ConsultationState

_logger = get_logger("Agent.LawRef")

LAW_KNOWLEDGE_PATH = Path(__file__).parent.parent.parent / "data" / "law_knowledge" / "criminal_law_chapters.json"

DEFAULT_LAW_EXTRACT_PROMPT = """你是一名刑事法律专家，根据用户描述的案件事实和匹配的刑法条文，提取结构化的法律信息。

请分析以下信息并输出 JSON 格式的结构化法律分析：

输出格式：
{
    "charges": [
        {
            "charge_name": "罪名名称",
            "article_number": "法条编号",
            "elements_matched": ["匹配的构成要件"],
            "elements_missing": ["缺失的构成要件"],
            "base_sentence": "基准刑",
            "probability": "high/medium/low"
        }
    ],
    "procedural_notes": ["程序性注意事项"]
}

重要：
1. 只分析匹配度较高的罪名
2. 如果无法确定，明确说明
3. 所有法律引用必须准确
"""


def _load_law_extract_prompt() -> str:
    """加载法条提取提示词"""
    try:
        return prompt_loader.load("lawref_prompt")
    except KeyError:
        return DEFAULT_LAW_EXTRACT_PROMPT


@lru_cache(maxsize=1)
def load_criminal_law_data() -> Dict[str, Any]:
    """加载刑事法律条文数据（带缓存）。

    Returns:
        包含章节和条文数据的字典。如果文件不存在，返回空结构。
        注意：JSON 文件结构为数组，每个元素包含 chapter 和 article 信息。
    """
    try:
        if LAW_KNOWLEDGE_PATH.exists():
            with open(LAW_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if isinstance(raw_data, list):
                chapters_dict: Dict[str, List[Dict]] = {}
                for article in raw_data:
                    chapter_name = article.get("chapter", "未分类")
                    if chapter_name not in chapters_dict:
                        chapters_dict[chapter_name] = []
                    chapters_dict[chapter_name].append(
                        {
                            "article_number": article.get("article_number", ""),
                            "title": article.get("title", ""),
                            "content": article.get("content", ""),
                            "elements": article.get("elements", []),
                            "base_sentence": article.get("base_sentence", ""),
                            "charge_tags": article.get("charge_tags", []),
                            "common_keywords": article.get("common_keywords", []),
                        }
                    )

                chapters_list = [
                    {"chapter": chapter_name, "articles": articles} for chapter_name, articles in chapters_dict.items()
                ]

                data = {"chapters": chapters_list}
            else:
                data = raw_data

            _logger.info("【load_criminal_law_data】成功加载法条数据，共 %d 章", len(data.get("chapters", [])))
            return data
        else:
            _logger.warning("【load_criminal_law_data】法条数据文件不存在: %s", LAW_KNOWLEDGE_PATH)
            return {"chapters": []}
    except Exception as e:
        _logger.error("【load_criminal_law_data】加载法条数据失败: %s", str(e))
        return {"chapters": []}


# 中文数字到阿拉伯数字的映射
_CN_NUM_MAP = {
    "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}
_CN_DIGITS = set("零一二三四五六七八九十百千")


def _cn_to_arabic(cn: str) -> str:
    """将中文数字转换为阿拉伯数字。

    支持格式如 "二百三十四" -> "234"、"一十" -> "10"。

    Args:
        cn: 中文数字字符串

    Returns:
        阿拉伯数字字符串
    """
    if not cn or not all(c in _CN_DIGITS for c in cn):
        return cn

    result = 0
    current = 0
    for char in cn:
        if char in _CN_NUM_MAP and char != "十":
            current = int(_CN_NUM_MAP[char])
        elif char == "十":
            if current == 0:
                current = 1
            result += current * 10
            current = 0
        elif char == "百":
            result += current * 100
            current = 0
        elif char == "千":
            result += current * 1000
            current = 0
    result += current
    return str(result)


def _normalize_article_number(article_number: str) -> str:
    """归一化法条编号，用于跨格式匹配。

    将 "第二百三十四条" 和 "第234条" 统一为 "第234条" 格式。

    Args:
        article_number: 原始法条编号

    Returns:
        归一化后的法条编号
    """
    if not article_number:
        return ""

    match = re.match(r"第(.+?)条(之[一二三四五六七八九十百千零\d]+)?", article_number)
    if not match:
        return article_number.strip()

    num_part = match.group(1)
    subarticle_suffix = match.group(2) or ""
    # 如果已经是纯数字，直接返回
    if num_part.isdigit():
        return f"第{num_part}条{subarticle_suffix}"

    # 尝试中文数字转换
    arabic = _cn_to_arabic(num_part)
    if arabic != num_part:  # 转换成功
        return f"第{arabic}条{subarticle_suffix}"

    return article_number.strip()


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


def _build_article_index(law_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """构建法条编号到 JSON 知识库条目的索引。

    Args:
        law_data: 刑法条文数据

    Returns:
        归一化法条编号到条目字典的映射
    """
    index: Dict[str, Dict[str, Any]] = {}
    for chapter in law_data.get("chapters", []):
        chapter_name = chapter.get("chapter", "")
        for article in chapter.get("articles", []):
            normalized = _normalize_article_number(article.get("article_number", ""))
            if normalized:
                index[normalized] = {**article, "chapter": chapter_name}
    return index


def _verify_and_enrich_with_json(
    rag_results: List[Dict[str, Any]], article_index: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """用 JSON 知识库验证并增强 RAG 检索结果。

    从 RAG 结果中提取法条编号，在 JSON 知识库中精确匹配，
    用 JSON 中的可靠元数据替换 RAG 文档中不可靠的元数据。

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

        if json_match:
            # 用 JSON 知识库的可靠数据增强 RAG 结果
            enriched.append({
                **rag_law,
                "title": json_match.get("title", rag_law.get("title", "")),
                "content": json_match.get("content", rag_law.get("content", "")),
                "elements": json_match.get("elements", []),
                "base_sentence": json_match.get("base_sentence", ""),
                "charge_tags": json_match.get("charge_tags", []),
                "common_keywords": json_match.get("common_keywords", []),
                "chapter": json_match.get("chapter", rag_law.get("chapter", "")),
                "data_source": "rag_verified",
            })
        else:
            # RAG 结果在 JSON 中找不到，保留原始内容但标记为未验证
            enriched.append({**rag_law, "data_source": "rag_unverified"})

    verified_count = sum(1 for law in enriched if law.get("data_source") == "rag_verified")
    _logger.info(
        "【_verify_and_enrich_with_json】RAG 结果验证完成，%d/%d 条通过 JSON 精确匹配",
        verified_count, len(enriched),
    )
    return enriched


def _merge_and_deduplicate(
    primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
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


def _is_unverified_rag_result(law: Dict[str, Any]) -> bool:
    """判断是否为未经验证的 RAG 检索结果（缺少可靠元数据）。

    仅未通过 JSON 知识库验证的 RAG 结果被视为不可靠。
    已通过 JSON 验证的 RAG 结果（data_source == "rag_verified"）视为可靠。

    Args:
        law: 法条字典

    Returns:
        True 表示是未验证的 RAG 结果，False 表示是可靠结果
    """
    return law.get("data_source") == "rag_unverified"


def _build_element_to_law_mapping(laws: List[Dict[str, Any]], elements_key: str = "elements") -> Dict[str, Dict[str, str]]:
    """构建构成要件到法条的映射

    Args:
        laws: 法条列表
        elements_key: elements 字段名（structured_laws 用 "elements_matched"，matched_laws 用 "elements"）

    Returns:
        要件到法条信息的映射
    """
    mapping: Dict[str, Dict[str, str]] = {}
    for law in laws:
        charge_name = law.get("charge_name", law.get("title", ""))
        elements = law.get(elements_key, [])
        for element in elements:
            if element not in mapping:
                mapping[element] = {
                    "charge_name": charge_name,
                    "article_number": law.get("article_number", ""),
                    "base_sentence": law.get("base_sentence", ""),
                }
    return mapping


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
            charge_name = article.get("title", "").lower()
            charge_tags = " ".join(article.get("charge_tags", [])).lower()
            content = article.get("content", "").lower()
            common_keywords = " ".join(article.get("common_keywords", [])).lower()

            relevance_score = 0
            matched_tags = []

            for term in search_terms:
                term_lower = term.lower()
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
                        "elements": article.get("elements", []),
                        "base_sentence": article.get("base_sentence", ""),
                        "charge_tags": article.get("charge_tags", []),
                        "common_keywords": article.get("common_keywords", []),
                        "chapter": chapter.get("chapter", ""),
                        "relevance_score": relevance_score,
                        "matched_tags": matched_tags,
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
        from app.rag.rag_service import RagService

        behavior_sequence = facts_structured.get("behavior_sequence", [])
        consequence = facts_structured.get("consequence", "")

        query_parts = _flatten_fact_terms(behavior_sequence)
        query_parts.extend(_flatten_fact_terms(consequence))

        query = " ".join(query_parts)
        if not query.strip():
            query = "刑事犯罪"

        rag_service = RagService(user_id=user_id, include_public=True)
        await rag_service.initialize_retriever(query)

        result = await rag_service.get_documents_and_summary(query)
        documents = result.get("documents", [])

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
                    "data_source": "rag",
                }
            )

        _logger.info("【search_laws_by_rag】RAG 检索找到 %d 条相关法条", len(matched_laws))
        return matched_laws

    except Exception as e:
        _logger.error("【search_laws_by_rag】RAG 检索失败: %s", str(e))
        return []


async def extract_structured_laws(
    matched_laws: List[Dict[str, Any]], facts_structured: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """使用 LLM 从匹配的法条中提取结构化信息。

    Args:
        matched_laws: 匹配的条文列表。
        facts_structured: 结构化的事实数据。

    Returns:
        结构化的法律信息列表。
    """
    if not matched_laws:
        return []

    laws_context = []
    for i, law in enumerate(matched_laws[:5], 1):
        law_parts = [
            f"【法条 {i}】",
            f"- 条款: {law.get('article_number', '')}",
            f"- 罪名: {law.get('title', '')}",
            f"- 内容: {law.get('content', '')[:200]}...",
            f"- 构成要件: {', '.join(law.get('elements', []))}",
            f"- 基准刑: {law.get('base_sentence', '')}",
            f"- 标签: {', '.join(law.get('charge_tags', []))}",
        ]
        laws_context.append("\n".join(law_parts))

    context_text = "\n".join(laws_context)

    # P0-1: 对传给 LLM 的案件事实进行 PII 脱敏
    facts_text = mask_pii(
        f"行为描述: {', '.join(str(x) for x in facts_structured.get('behavior_sequence', []))}\n"
        f"后果: {facts_structured.get('consequence', '未知')}"
    )

    system_prompt = _load_law_extract_prompt()

    user_message_parts = [
        "案件事实：",
        facts_text,
        "",
        "匹配的刑法条文：",
        context_text,
        "",
        "请提取结构化的法律分析。",
    ]
    user_message = "\n".join(user_message_parts)

    try:
        response = await llm_gateway.generate(system_prompt, user_message, is_legal=True)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            structured_result = json.loads(json_match.group())
            _logger.info("【extract_structured_laws】LLM 结构化提取成功")
            return structured_result.get("charges", [])
        else:
            _logger.warning("【extract_structured_laws】LLM 响应中未找到 JSON")
            return []

    except Exception as e:
        _logger.error("【extract_structured_laws】LLM 调用失败: %s", str(e))
        return []


def _build_applied_laws_from_structured(
    structured_laws: List[Dict[str, Any]], matched_laws: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """从 LLM 结构化结果构建 applied_laws。

    LLM 结构化提取是对所有 matched_laws 的综合分析，因此 data_source
    取对应法条编号在 matched_laws 中的来源；若无法匹配则默认为可靠来源。

    Args:
        structured_laws: LLM 结构化提取的法律信息列表
        matched_laws: 原始匹配的法条列表（用于回填 data_source）

    Returns:
        构建好的 applied_laws 列表
    """
    # 构建法条编号到 data_source 的映射
    source_map: Dict[str, str] = {}
    for law in matched_laws:
        num = _normalize_article_number(law.get("article_number", ""))
        if num:
            source_map[num] = law.get("data_source", "json_keyword")

    applied_laws = []
    for law in structured_laws:
        charge_name = law.get("charge_name", "")
        article_number = law.get("article_number", "")
        normalized = _normalize_article_number(article_number)
        data_source = source_map.get(normalized, "llm_extracted")

        applied_laws.append(
            {
                "charge_name": charge_name,
                "article_number": article_number,
                "elements": law.get("elements_matched", []),
                "elements_missing": law.get("elements_missing", []),
                "base_sentence": law.get("base_sentence", ""),
                "probability": law.get("probability", "medium"),
                "data_source": data_source,
            }
        )
    return applied_laws


def _build_applied_laws_from_matched(matched_laws: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从匹配结果构建 applied_laws（LLM 提取失败时的回退路径）。

    Args:
        matched_laws: 匹配的法条列表

    Returns:
        构建好的 applied_laws 列表
    """
    applied_laws = []
    for law in matched_laws:
        charge_name = law.get("title", "")
        elements = law.get("elements", [])
        applied_laws.append(
            {
                "charge_name": charge_name,
                "article_number": law.get("article_number", ""),
                "elements": elements,
                "base_sentence": law.get("base_sentence", ""),
                "charge_tags": law.get("charge_tags", []),
                "data_source": law.get("data_source", "json_keyword"),
            }
        )
    return applied_laws


async def law_ref_node(state: "ConsultationState") -> "ConsultationState":
    """LawRef Agent 节点函数 - 法条检索 Agent。

    采用两阶段检索策略：
    1. RAG 语义检索（召回层）：通过向量检索召回语义相关的法条
    2. JSON 知识库精确匹配（验证+增强层）：用 JSON 知识库验证并补充可靠元数据
    3. LLM 结构化信息提取

    Args:
        state: 当前 ConsultationState。

    Returns:
        更新后的 ConsultationState。
    """
    _logger.info("【law_ref_node】LawRef 节点开始执行")

    session_id = state.get("session_id", "unknown")
    user_id = state.get("user_id")
    facts_structured = state.get("facts_structured", {})

    if not facts_structured:
        _logger.warning("【law_ref_node】facts_structured 为空，跳过法条检索")
        state["applied_laws"] = []
        state["current_agent"] = "LawRef"
        return state

    law_data = load_criminal_law_data()

    # 阶段1：RAG 语义检索（召回层）
    _logger.info("【law_ref_node】阶段1：RAG 语义检索")
    rag_results = await search_laws_by_rag(facts_structured, user_id)
    _logger.info(
        "【law_ref_node】RAG 检索返回 %d 条结果: %s",
        len(rag_results),
        [f"article_number={r.get('article_number', '')}, title={r.get('title', '')}" for r in rag_results],
    )

    # 阶段2：JSON 知识库精确匹配（验证+增强层）
    if rag_results and law_data.get("chapters"):
        _logger.info("【law_ref_node】阶段2：JSON 知识库验证增强")
        article_index = _build_article_index(law_data)
        _logger.debug(
            "【law_ref_node】JSON 索引构建完成，共 %d 条法条: %s",
            len(article_index),
            list(article_index.keys())[:10],
        )
        rag_results = _verify_and_enrich_with_json(rag_results, article_index)
        _logger.info(
            "【law_ref_node】JSON 验证增强后: %s",
            [f"article_number={r.get('article_number', '')}, data_source={r.get('data_source', '')}, title={r.get('title', '')}" for r in rag_results],
        )

    # JSON 关键词匹配作为补充（覆盖 RAG 可能遗漏的精确匹配场景）
    keyword_laws = []
    if law_data.get("chapters"):
        _logger.info("【law_ref_node】JSON 关键词匹配补充")
        keyword_laws = await search_laws_by_keyword(facts_structured, law_data)
        _logger.info(
            "【law_ref_node】关键词匹配返回 %d 条结果: %s",
            len(keyword_laws),
            [f"article_number={k.get('article_number', '')}, title={k.get('title', '')}" for k in keyword_laws],
        )

    # 合并去重：RAG 验证结果优先，关键词匹配补充
    matched_laws = _merge_and_deduplicate(rag_results, keyword_laws)
    _logger.info(
        "【law_ref_node】合并去重后共 %d 条: %s",
        len(matched_laws),
        [f"article_number={m.get('article_number', '')}, data_source={m.get('data_source', '')}" for m in matched_laws],
    )

    # 阶段3：LLM 结构化信息提取
    structured_laws = await extract_structured_laws(matched_laws, facts_structured)

    if structured_laws:
        element_to_law_mapping = _build_element_to_law_mapping(structured_laws, "elements_matched")
        applied_laws = _build_applied_laws_from_structured(structured_laws, matched_laws)
    else:
        element_to_law_mapping = _build_element_to_law_mapping(matched_laws[:5], "elements")
        applied_laws = _build_applied_laws_from_matched(matched_laws[:5])

    # 统计各来源数量
    rag_verified = sum(1 for law in matched_laws if law.get("data_source") == "rag_verified")
    rag_unverified = sum(1 for law in matched_laws if law.get("data_source") == "rag_unverified")
    json_keyword = sum(1 for law in matched_laws if law.get("data_source") != "rag_verified" and law.get("data_source") != "rag_unverified")
    has_verified = rag_verified > 0

    state["applied_laws"] = applied_laws
    state["element_to_law_mapping"] = element_to_law_mapping
    state["current_agent"] = "LawRef"
    state["rag_only"] = not has_verified and len(applied_laws) > 0

    if not has_verified and len(applied_laws) > 0:
        _logger.warning(
            "【law_ref_node】无 JSON 知识库验证结果，仅使用 RAG 未验证结果，覆盖度可能受影响"
        )

    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append(
        {
            "agent": "LawRef",
            "action": "law_search",
            "matched_count": len(applied_laws),
            "rag_verified": rag_verified,
            "rag_unverified": rag_unverified,
            "json_keyword": json_keyword,
            "session_id": session_id,
        }
    )

    _logger.info(
        "【law_ref_node】法条检索完成，找到 %d 个匹配罪名 (RAG验证: %d, RAG未验证: %d, JSON关键词: %d)",
        len(applied_laws),
        rag_verified,
        rag_unverified,
        json_keyword,
    )

    return state
