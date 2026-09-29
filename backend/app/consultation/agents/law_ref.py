from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import ValidationError

from app.consultation.schemas.artifacts import (
    ArtifactSource,
    LawArtifact,
    parse_json_object,
    record_artifact_result,
    validate_artifact,
)
from app.consultation.schemas.law import LawDataSource, LawSourceSchema
from app.errors.exceptions import LLMServiceException, LLMTimeoutException
from app.infrastructure.config.prompts import prompt_loader
from app.infrastructure.llm.gateway import llm_gateway
from app.infrastructure.logging import get_logger
from app.knowledge.law_knowledge import (
    LawKnowledgeDataError,
    _element_name,
    _normalize_article_number,
    load_criminal_law_data,
)
from app.security.sensitive_filter import mask_pii

if TYPE_CHECKING:
    from app.consultation.state import ConsultationState

_logger = get_logger("Agent.LawRef")


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
2. article_number 必须来自输入候选，禁止生成未检索命中的法条
3. elements_matched 与 elements_missing 必须共同覆盖输入候选的完整构成要件，且不得重叠
4. 如果无法确定，将对应要件放入 elements_missing
5. 所有法律引用必须准确
"""


def _load_law_extract_prompt() -> str:
    """加载法条提取提示词"""
    try:
        return prompt_loader.load("lawref_prompt")
    except KeyError:
        return DEFAULT_LAW_EXTRACT_PROMPT


def _validated_data_source(value: Any) -> str:
    """将候选来源校验为 schema 枚举，非法值按 LLM 未验证来源处理。"""
    try:
        return LawSourceSchema(data_source=value).data_source.value
    except ValidationError:
        return LawDataSource.LLM_EXTRACTED.value


def _build_element_to_law_mapping(
    laws: List[Dict[str, Any]], elements_key: str = "elements"
) -> Dict[str, Dict[str, str]]:
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
            element_name = _element_name(element)
            if element_name and element_name not in mapping:
                mapping[element_name] = {
                    "charge_name": charge_name,
                    "article_number": law.get("article_number", ""),
                    "base_sentence": law.get("base_sentence", ""),
                }
    return mapping


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
        required_elements = law.get("required_elements") or law.get("elements") or []
        element_names = [_element_name(element) for element in required_elements]
        law_parts = [
            f"【法条 {i}】",
            f"- 条款: {law.get('article_number', '')}",
            f"- 罪名: {law.get('title', '')}",
            f"- 内容: {law.get('content', '')[:200]}...",
            f"- 构成要件: {', '.join(name for name in element_names if name)}",
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
        payload = parse_json_object(response)
        artifact, result = validate_artifact(
            LawArtifact,
            payload,
            source=ArtifactSource.CONTENT_JSON,
        )
        if artifact is not None:
            _logger.info("【extract_structured_laws】LLM 结构化提取成功")
            return [charge.model_dump(mode="json") for charge in artifact.charges]
        _logger.warning(
            "【extract_structured_laws】LLM 结构化提取失败: error_count=%d",
            len(result.validation_errors),
        )
        return []
    except (LLMServiceException, LLMTimeoutException):
        raise
    except Exception as e:
        _logger.error(
            "【extract_structured_laws】LLM 调用失败: error_type=%s",
            type(e).__name__,
        )
        return []


def _build_applied_laws_from_structured(
    structured_laws: List[Dict[str, Any]], matched_laws: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """从 LLM 结构化结果构建 applied_laws。

    仅法条编号与 matched_laws 连接成功时继承来源和权威构成要件；
    未命中或来源非法的模型候选统一标记为 llm_extracted。

    Args:
        structured_laws: LLM 结构化提取的法律信息列表
        matched_laws: 原始匹配的法条列表（用于回填 data_source）

    Returns:
        构建好的 applied_laws 列表
    """
    # 编号连接同时携带来源和权威要件，避免模型自行声明 coverage 分母。
    matched_map: Dict[str, Dict[str, Any]] = {}
    for law in matched_laws:
        num = _normalize_article_number(law.get("article_number", ""))
        if num:
            matched_map[num] = law

    applied_laws = []
    for law in structured_laws:
        charge_name = law.get("charge_name", "")
        article_number = law.get("article_number", "")
        normalized = _normalize_article_number(article_number)
        matched_law = matched_map.get(normalized)
        data_source = (
            _validated_data_source(matched_law.get("data_source")) if matched_law else LawDataSource.LLM_EXTRACTED.value
        )
        if matched_law and data_source != LawDataSource.LLM_EXTRACTED.value:
            required_elements = list(matched_law.get("required_elements") or matched_law.get("elements") or [])
        else:
            required_elements = []

        claimed_matched = {str(item) for item in law.get("elements_matched", [])}
        elements_matched = [
            _element_name(element) for element in required_elements if _element_name(element) in claimed_matched
        ]
        elements_missing = [
            _element_name(element) for element in required_elements if _element_name(element) not in claimed_matched
        ]

        applied_laws.append(
            {
                "charge_name": charge_name,
                "article_number": article_number,
                "required_elements": required_elements,
                "elements": required_elements,
                "elements_matched": elements_matched,
                "elements_missing": elements_missing,
                "base_sentence": law.get("base_sentence", ""),
                "probability": law.get("probability", "medium"),
                "data_source": data_source,
            }
        )
    return applied_laws


async def law_ref_node(state: "ConsultationState") -> "ConsultationState":
    """在受控 workflow 节点内运行有限法律检索循环并适配下游契约。"""
    from app.consultation.agents.legal_research import LawResearchResult, run_legal_research

    session_id = state.get("session_id", "unknown")
    facts_structured = state.get("facts_structured", {})
    state["current_agent"] = "LawRef"
    if not facts_structured:
        state["applied_laws"] = []
        state["element_to_law_mapping"] = {}
        state["law_search_status"] = "missing_facts"
        state["rag_only"] = False
        state["law_research"] = LawResearchResult(termination_reason="missing_facts").audit_summary()
        return state

    try:
        law_data = load_criminal_law_data()
        if not law_data.get("chapters"):
            raise LawKnowledgeDataError("法条验证快照不含章节")
        research = await run_legal_research(facts_structured, state.get("user_id"), law_data)
    except (LawKnowledgeDataError, ValueError) as exc:
        reason = "knowledge_unavailable" if isinstance(exc, LawKnowledgeDataError) else "configuration_error"
        state["applied_laws"] = []
        state["element_to_law_mapping"] = {}
        state["law_search_status"] = "dependency_failure"
        state["rag_only"] = False
        state["law_research"] = LawResearchResult(termination_reason=reason).audit_summary()
        _, failure_result = validate_artifact(
            LawArtifact,
            {"charges": [], "procedural_notes": []},
            source=ArtifactSource.CONTENT_JSON,
        )
        record_artifact_result(state, "law", failure_result.model_copy(update={"degraded_reason": reason}))  # pyright: ignore[reportArgumentType]
        state.setdefault("conversation_history", []).append(
            {
                "agent": "LawRef",
                "action": "law_search",
                "matched_count": 0,
                "search_status": "dependency_failure",
                "termination_reason": reason,
                "session_id": session_id,
            }
        )
        return state

    state["law_research"] = research.audit_summary()
    if research.termination_reason == "final_answer" and research.candidate_laws:
        structured_laws = []
        for law in research.candidate_laws:
            article_id = _normalize_article_number(law.get("article_number", ""))
            structured_laws.append(
                {
                    "charge_name": law.get("title", ""),
                    "article_number": law.get("article_number", ""),
                    "elements_matched": research.matched_elements.get(article_id, []),
                    "elements_missing": research.missing_elements.get(article_id, []),
                    "base_sentence": law.get("base_sentence", ""),
                    "probability": research.confidence,
                }
            )
        artifact, artifact_result = validate_artifact(
            LawArtifact,
            {"charges": structured_laws, "procedural_notes": []},
            source=ArtifactSource.CONTENT_JSON,
        )
        if artifact is not None:
            validated = [charge.model_dump(mode="json") for charge in artifact.charges]
            state["applied_laws"] = _build_applied_laws_from_structured(validated, research.candidate_laws)
            state["element_to_law_mapping"] = _build_element_to_law_mapping(validated, "elements_matched")
            state["law_search_status"] = "success"
            state["rag_only"] = False
        else:
            state["applied_laws"] = []
            state["element_to_law_mapping"] = {}
            state["law_search_status"] = "dependency_failure"
    else:
        state["applied_laws"] = []
        state["element_to_law_mapping"] = {}
        state["rag_only"] = False
        searched_empty = any(
            step.tool_name == "search_laws" and step.tool_status == "empty" for step in research.trajectory
        )
        state["law_search_status"] = (
            "no_law_match"
            if searched_empty and all(step.tool_status in {"empty", "invalid_final"} for step in research.trajectory)
            else "dependency_failure"
        )
        _, artifact_result = validate_artifact(
            LawArtifact,
            {"charges": [], "procedural_notes": []},
            source=ArtifactSource.CONTENT_JSON,
        )
        artifact_result = artifact_result.model_copy(update={"degraded_reason": research.termination_reason})
    record_artifact_result(state, "law", artifact_result)  # pyright: ignore[reportArgumentType]
    state.setdefault("conversation_history", []).append(
        {
            "agent": "LawRef",
            "action": "law_search",
            "matched_count": len(state["applied_laws"]),
            "search_status": state["law_search_status"],
            "termination_reason": research.termination_reason,
            "session_id": session_id,
        }
    )
    return state
