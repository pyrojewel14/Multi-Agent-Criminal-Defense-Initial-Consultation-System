import json
from typing import TYPE_CHECKING, Any, Dict, List

from app.security.disclaimer import disclaimer
from app.security.sensitive_filter import detect_high_risk, mask_pii, sanitize_input
from app.tools.fact_tools import extract_case_facts
from app.utils.llm_gateway import llm_gateway
from app.utils.logger import get_logger
from app.utils.prompt_loader import prompt_loader

if TYPE_CHECKING:
    from app.state.consultation_state import ConsultationState

_logger = get_logger("Agent.FactDigger")

COVERAGE_THRESHOLD = 0.8

# 默认提示词（外部文件加载失败时使用）
DEFAULT_EXTRACT_CASE_FACTS_PROMPT = """从咨询者描述中提取刑事案件关键事实要素。

请提取以下字段：
- incident_time: 事件发生时间
- incident_location: 事件发生地点（已脱敏）
- parties: 当事人列表（包含 role, name, relationship）
- behavior_sequence: 行为时间序列（包含 time, actor, action, method, target）
- consequence: 后果描述
- evidence_mentioned: 提到的证据线索
- arrest_status: 当前羁押状态
- surrender: 是否自首
- victim_forgiveness: 被害人是否谅解
- prior_record: 是否有前科劣迹

如果某字段信息不明确，设为 null。"""

DEFAULT_FOLLOW_UP_PROMPT = """你是一个刑事案件事实挖掘专家。
根据以下缺失的构成要件，生成追问问题。
要求：
1. 追问必须是开放性的，不能诱导用户做出特定回答
2. 禁止假设性提问，不假设用户做过或没做过某事
3. 保持客观性，不评判用户行为的对错
4. 问题要简洁、清晰

请生成 JSON 格式的问题列表。"""

DEFAULT_SUMMARY_PROMPT = """你是一个刑事案件事实整理专家。
根据以下结构化事实和原始描述，生成一份清晰、准确的案件事实摘要。
要求：
1. 按时间顺序整理事件经过
2. 突出关键情节和重要细节
3. 保持客观、中性的表述
4. 使用简洁的语言

请生成 Markdown 格式的事实摘要。"""

DEFAULT_FIRST_PROMPT = """请按以下时间线结构描述事件经过：

1. 【事前】- 事件发生前的背景
   - 什么情况下发生的？
   - 当事人之间的关系是什么？

2. 【事发】- 事件发生时的经过
   - 具体发生了什么？
   - 在什么时间、地点发生的？
   - 有哪些人参与？

3. 【事中】- 事件进行中的关键情节
   - 是否有暴力、威胁等行为？
   - 是否有财产损失或人身伤害？
   - 当事人各方分别做了什么？

4. 【事后】- 事件发生后的发展
   - 事后各方有何反应？
   - 是否有人报警？
   - 公安或其他部门是否介入？"""


def _load_extract_case_facts_prompt() -> str:
    """加载事实提取提示词"""
    try:
        return prompt_loader.load("extract_case_facts_prompt")
    except KeyError:
        return DEFAULT_EXTRACT_CASE_FACTS_PROMPT


def _load_first_prompt() -> str:
    """加载首次引导提示词"""
    try:
        return prompt_loader.load("fact_digger_first_prompt")
    except KeyError:
        return DEFAULT_FIRST_PROMPT


def _load_follow_up_prompt() -> str:
    """加载追问提示词"""
    try:
        return prompt_loader.load("fact_digger_prompt")
    except KeyError:
        return DEFAULT_FOLLOW_UP_PROMPT


def _load_summary_prompt() -> str:
    """加载摘要提示词"""
    try:
        return prompt_loader.load("fact_digger_summary_prompt")
    except KeyError:
        return DEFAULT_SUMMARY_PROMPT


async def _analyze_coverage(facts_structured: Dict[str, Any], applied_laws: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析构成要件覆盖度。

    注意：跳过 RAG 检索结果，只使用 JSON 知识库的结果计算覆盖度。

    Args:
        facts_structured: 结构化事实数据
        applied_laws: 适用的法律条款列表

    Returns:
        覆盖度分析结果字典
    """
    if not applied_laws:
        return {
            "total_elements": 0,
            "covered_elements": 0,
            "coverage_rate": 0.0,
            "missing_elements": [],
            "weak_elements": [],
            "source": "no_laws",
        }

    # 延迟导入避免循环依赖
    from app.agents.law_ref import _is_unverified_rag_result

    # 过滤掉未验证的 RAG 结果，只用 JSON 知识库验证过的结果计算覆盖度
    json_laws = [law for law in applied_laws if not _is_unverified_rag_result(law)]
    rag_laws = [law for law in applied_laws if _is_unverified_rag_result(law)]

    if not json_laws and rag_laws:
        _logger.warning(
            "【_analyze_coverage】无可用的 JSON 知识库结果，%d 个 RAG 结果被跳过",
            len(rag_laws),
        )
        return {
            "total_elements": 0,
            "covered_elements": 0,
            "coverage_rate": 0.0,
            "missing_elements": [],
            "weak_elements": [],
            "source": "rag_only",
            "rag_count": len(rag_laws),
        }

    covered_elements = []
    missing_elements = []
    weak_elements = []

    for law in json_laws:
        required_elements = law.get("elements", [])

        for element in required_elements:
            if isinstance(element, dict):
                element_name = element.get("name", "")
                element_key = element.get("key", element_name)
            else:
                element_name = str(element)
                element_key = str(element)

            fact_value = _get_fact_value(facts_structured, element_key)

            if fact_value is not None and fact_value != "":
                covered_elements.append(element_name)

                if isinstance(fact_value, list) and len(fact_value) == 0:
                    weak_elements.append(element_name)
                elif isinstance(fact_value, bool) and not fact_value:
                    weak_elements.append(element_name)
            else:
                missing_elements.append(element_name)

    total_elements = len(covered_elements) + len(missing_elements)
    coverage_rate = len(covered_elements) / total_elements if total_elements > 0 else 0.0

    return {
        "total_elements": total_elements,
        "covered_elements": len(covered_elements),
        "coverage_rate": coverage_rate,
        "missing_elements": missing_elements,
        "weak_elements": weak_elements,
        "source": "json_knowledge",
        "json_law_count": len(json_laws),
        "rag_count": len(rag_laws),
    }


def _get_fact_value(facts_structured: Dict[str, Any], key: str) -> Any:
    """从结构化事实中获取指定键的值。

    Args:
        facts_structured: 结构化事实数据
        key: 要获取的键名

    Returns:
        键对应的值，如果不存在返回 None
    """
    key_mapping = {
        "time": "incident_time",
        "location": "incident_location",
        "parties": "parties",
        "behavior": "behavior_sequence",
        "consequence": "consequence",
        "evidence": "evidence_mentioned",
        "arrest": "arrest_status",
        "surrender": "surrender",
        "forgiveness": "victim_forgiveness",
        "record": "prior_record",
    }

    mapped_key = key_mapping.get(key, key)
    return facts_structured.get(mapped_key)


async def _generate_follow_up_questions(missing_elements: List[str], facts_structured: Dict[str, Any]) -> List[str]:
    """根据缺失的构成要件生成追问问题。

    Args:
        missing_elements: 缺失的构成要件列表
        facts_structured: 当前结构化事实数据

    Returns:
        追问问题列表
    """
    if not missing_elements:
        return []

    system_prompt = _load_follow_up_prompt()

    # P0-1: 对结构化事实中的 PII 进行脱敏
    facts_text = mask_pii(json.dumps(facts_structured, ensure_ascii=False))

    user_message_parts = [
        "当前已收集的事实：",
        facts_text,
        "",
        "缺失的构成要件：",
        str(missing_elements),
        "",
        "请针对每个缺失的构成要件生成一个追问问题，返回格式：",
        '{"questions": ["问题1", "问题2", ...]}',
    ]
    user_message = "\n".join(user_message_parts)

    try:
        response = await llm_gateway.generate(system_prompt=system_prompt, user_message=user_message, temperature=0.1)

        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            return data.get("questions", [])
    except Exception as e:
        _logger.error("【_generate_follow_up_questions】生成追问失败: %s", e)

    return []


async def _generate_fact_summary(facts_structured: Dict[str, Any], facts_raw: List[str]) -> str:
    """生成事实摘要。

    Args:
        facts_structured: 结构化事实数据
        facts_raw: 原始用户输入

    Returns:
        生成的事实摘要
    """
    system_prompt = _load_summary_prompt()

    # P0-1: 对结构化事实中的 PII 进行脱敏
    facts_text = mask_pii(json.dumps(facts_structured, ensure_ascii=False))

    user_message_parts = [
        "结构化事实：",
        facts_text,
        "",
        "原始用户描述：",
        "\n".join(facts_raw),
        "",
        "请生成案件事实摘要。",
    ]
    user_message = "\n".join(user_message_parts)

    try:
        summary = await llm_gateway.generate(system_prompt=system_prompt, user_message=user_message, temperature=0.1)
        return summary
    except Exception as e:
        _logger.error("【_generate_fact_summary】生成事实摘要失败: %s", e)
        return ""


async def _extract_structured_facts(facts_raw: List[str]) -> Dict[str, Any]:
    """使用 LLM Function Calling 提取结构化事实。

    Args:
        facts_raw: 原始用户输入列表

    Returns:
        结构化事实字典
    """
    combined_facts = "\n".join([f"- {fact}" for fact in facts_raw])
    user_message = f"请从以下描述中提取案件事实要素：\n\n{combined_facts}"

    try:
        # 使用 LLMGateway 的 generate_with_tools 方法
        _logger.debug("【_extract_structured_facts】开始调用 extract_case_facts 工具")
        result = await llm_gateway.generate_with_tools(
            system_prompt=_load_extract_case_facts_prompt(),
            user_message=user_message,
            tools=[extract_case_facts],
            temperature=0.1,
        )

        # 优先使用 tool_calls 返回
        if result.get("has_tool_call"):
            for tool_call in result["tool_calls"]:
                if tool_call.get("name") == "extract_case_facts":
                    extracted_args = tool_call.get("args", {})
                    _logger.info(
                        "【_extract_structured_facts】extract_case_facts 调用成功，返回字段数: %d",
                        len(extracted_args),
                    )
                    _logger.debug("【_extract_structured_facts】提取结果: %s", extracted_args)
                    return extracted_args

            _logger.warning("【_extract_structured_facts】存在 tool_calls 但未找到 extract_case_facts")

        # 回退：尝试从 content 中解析 JSON
        content = result.get("content", "")
        if content:
            import re

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                _logger.info("【_extract_structured_facts】从 content 回退解析 JSON 成功")
                return json.loads(json_match.group())

        _logger.warning("【_extract_structured_facts】tool_calls 和 content 均无有效结果")
        return {}
    except Exception as e:
        _logger.error("【_extract_structured_facts】Function Calling 提取失败: %s", e)
        return {}


def _handle_first_interaction(state: "ConsultationState") -> "ConsultationState":
    """处理首次交互（无 facts_raw 且无 user_input 的情况）。

    Args:
        state: 当前 ConsultationState

    Returns:
        更新后的 ConsultationState
    """
    conversation_history = state.get("conversation_history", [])
    response_text = disclaimer.inject(_load_first_prompt())

    conversation_history.append({"agent": "FactDigger", "role": "assistant", "content": response_text})

    state["current_agent"] = "FactDigger"
    state["conversation_history"] = conversation_history
    state["final_output"] = response_text
    state["facts_coverage_rate"] = 0.0

    _logger.info("【fact_digger_node】发送首次引导语")
    return state


def _handle_high_risk_input(
    state: "ConsultationState", user_input: str, conversation_history: List[Dict[str, Any]]
) -> "ConsultationState":
    """处理高风险输入。

    Args:
        state: 当前 ConsultationState
        user_input: 用户输入内容
        conversation_history: 对话历史

    Returns:
        更新后的 ConsultationState
    """
    is_high_risk, risk_type = detect_high_risk(user_input)
    if is_high_risk:
        _logger.warning("【fact_digger_node】检测到高风险语句，风险类型: %s", risk_type)
        state["alert_triggered"] = True
        state["risk_assessment"] = {"risk_type": risk_type, "risk_level": "high"}
        state["current_agent"] = "HumanAlert"
        state["conversation_history"] = conversation_history
        return state
    return state


async def _handle_insufficient_coverage(
    state: "ConsultationState",
    coverage_analysis: Dict[str, Any],
    facts_structured: Dict[str, Any],
    pending_questions: List[str],
    conversation_history: List[Dict[str, Any]],
) -> "ConsultationState":
    """处理覆盖度不足的情况。

    Args:
        state: 当前 ConsultationState
        coverage_analysis: 覆盖度分析结果
        facts_structured: 结构化事实数据
        pending_questions: 待追问问题列表
        conversation_history: 对话历史

    Returns:
        更新后的 ConsultationState
    """
    missing_elements = coverage_analysis.get("missing_elements", [])
    new_questions = await _generate_follow_up_questions(missing_elements, facts_structured)

    pending_questions.extend(new_questions)

    response_text = "为了更准确地分析案件，请您补充以下信息：\n\n"
    for i, question in enumerate(new_questions, 1):
        response_text += f"{i}. {question}\n"

    response_text = disclaimer.inject(response_text)

    conversation_history.append({"agent": "FactDigger", "role": "assistant", "content": response_text})

    state["pending_questions"] = pending_questions
    state["final_output"] = response_text
    state["current_agent"] = "FactDigger"
    state["conversation_history"] = conversation_history

    _logger.info("【fact_digger_node】生成追问 %d 个，覆盖度不足", len(new_questions))
    return state


async def _handle_sufficient_coverage(
    state: "ConsultationState",
    facts_structured: Dict[str, Any],
    facts_raw: List[str],
    conversation_history: List[Dict[str, Any]],
) -> "ConsultationState":
    """处理覆盖度达标的情况。

    Args:
        state: 当前 ConsultationState
        facts_structured: 结构化事实数据
        facts_raw: 原始用户输入列表
        conversation_history: 对话历史

    Returns:
        更新后的 ConsultationState
    """
    summary = await _generate_fact_summary(facts_structured, facts_raw)

    response_text_parts = [
        "## 案件事实摘要",
        "",
        "请确认以下内容是否准确：",
        "",
        summary,
        "",
        "如有不准确之处，请修正。",
    ]
    response_text = "\n".join(response_text_parts)
    response_text = disclaimer.inject(response_text)

    conversation_history.append({"agent": "FactDigger", "role": "assistant", "content": response_text})

    state["final_output"] = response_text
    state["current_agent"] = "RiskAssessor"
    state["conversation_history"] = conversation_history

    _logger.info("【fact_digger_node】生成事实摘要，覆盖度已达标，跳转 RiskAssessor")
    return state


async def fact_digger_node(state: "ConsultationState") -> "ConsultationState":
    """FactDigger Agent 节点函数

    处理事实收集、结构化提取和追问逻辑
    与 LawRef 双向交互：当 applied_laws 有数据时分析覆盖度

    Args:
        state: 当前 ConsultationState

    Returns:
        更新后的 ConsultationState
    """
    _logger.info("【fact_digger_node】FactDigger 节点开始执行")

    facts_raw = state.get("facts_raw", [])
    facts_structured = state.get("facts_structured", {})
    applied_laws = state.get("applied_laws", [])
    pending_questions = state.get("pending_questions", [])
    conversation_history = state.get("conversation_history", [])

    user_input = state.get("current_input", "")

    # 首次交互：无原始事实且无用户输入
    if not facts_raw and not user_input:
        return _handle_first_interaction(state)

    # 处理用户输入
    if user_input:
        # P0-2: 高风险语句检测
        state = _handle_high_risk_input(state, user_input, conversation_history)
        if state.get("current_agent") == "HumanAlert":
            return state

        # P0-1: PII 脱敏后存储
        sanitized_input = sanitize_input(user_input)
        facts_raw.append(sanitized_input)
        _logger.debug("【fact_digger_node】追加用户输入到 facts_raw，当前共 %d 条", len(facts_raw))

    facts_structured = await _extract_structured_facts(facts_raw)
    _logger.debug("【fact_digger_node】提取结构化事实完成")

    state["facts_structured"] = facts_structured
    state["facts_raw"] = facts_raw

    if not applied_laws:
        _logger.warning(
            "【fact_digger_node】applied_laws 为空，设置 coverage_rate=0，等待 LawRef 返回"
        )
        state["facts_coverage_rate"] = 0.0
        state["current_agent"] = "FactDigger"
        state["conversation_history"] = conversation_history
        return state

    # 分析覆盖度
    coverage_analysis = await _analyze_coverage(facts_structured, applied_laws)
    coverage_rate = coverage_analysis.get("coverage_rate", 0.0)

    # 更新循环计数
    loop_count = state.get("fact_law_loop_count", 0) + 1
    state["fact_law_loop_count"] = loop_count
    _logger.debug(
        "【fact_digger_node】当前循环次数: %d/%d",
        loop_count,
        10,  # 与 workflow.py 中的 max_loops 保持一致
    )

    state["facts_coverage_rate"] = coverage_rate

    _logger.info(
        "【fact_digger_node】构成要件覆盖度: %.2f (%d/%d)",
        coverage_rate,
        coverage_analysis.get("covered_elements", 0),
        coverage_analysis.get("total_elements", 0),
    )

    if coverage_rate < COVERAGE_THRESHOLD:
        return await _handle_insufficient_coverage(state, coverage_analysis, facts_structured, pending_questions, conversation_history)
    else:
        return await _handle_sufficient_coverage(state, facts_structured, facts_raw, conversation_history)
