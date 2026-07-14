import json
from typing import TYPE_CHECKING, Any, Dict, List

from app.security.disclaimer import disclaimer
from app.security.sensitive_filter import mask_pii
from app.utils.llm_gateway import llm_gateway
from app.utils.logger import get_logger
from app.utils.prompt_loader import prompt_loader

if TYPE_CHECKING:
    from app.state.consultation_state import ConsultationState

_logger = get_logger("Agent.RiskAssessor")

DEFAULT_RISK_ASSESSOR_PROMPT = """你是一位专业的刑事辩护律师，擅长进行案件风险评估。你的职责是基于提供的案件事实和适用法律，进行全面客观的风险分析。

风险评估应包含以下维度：
1. 量刑预测
2. 强制措施风险
3. 证据风险点
4. 程序风险

输出格式为JSON，包含以下字段：
- predicted_sentence_range: 量刑区间预测（如"3年以下有期徒刑"、"5-10年有期徒刑"等）
- mitigating_factors: 从轻或减轻处罚的情节列表
- aggravating_factors: 从重处罚的情节列表
- compulsory_measure_risk: 强制措施风险评估对象，包含：
    - detention_status: 当事人当前羁押状态
    - bail_possibility: 取保候审可能性评估
    - measure_change_space: 变更强制措施的空间评估
    - prolonged_detention_risk: 超期羁押风险评估
- evidence_risk_points: 证据风险点列表，包含：
    - gap: 证据链缺口描述
    - exclusion_possibility: 非法证据排除可能性评估
- procedure_risks: 程序风险列表，包含：
    - type: 风险类型（如"诉讼时效"、"管辖权"等）
    - description: 风险描述
    - severity: 风险严重程度（low/medium/high）

确保评估客观、依据充分、符合中国刑事诉讼法规定。"""


def _load_prompt() -> str:
    """加载风险评估提示词"""
    try:
        return prompt_loader.load("risk_assessor_prompt")
    except KeyError:
        _logger.warning("未找到 risk_assessor_prompt，使用默认提示词")
        return DEFAULT_RISK_ASSESSOR_PROMPT


async def risk_assessor_node(state: "ConsultationState") -> "ConsultationState":
    """RiskAssessor Agent 节点函数

    基于结构化事实和法条检索结果生成风险评估

    Args:
        state: 当前 ConsultationState

    Returns:
        更新后的 ConsultationState
    """
    _logger.info("【risk_assessor_node】开始风险评估，consultation_id: %s", state.get("consultation_id"))

    facts_structured = state.get("facts_structured", {})
    applied_laws = state.get("applied_laws", [])

    risk_assessment = await _generate_risk_assessment(facts_structured, applied_laws)

    state["risk_assessment"] = risk_assessment
    state["current_agent"] = "ServicePlanner"

    _add_to_conversation_history(state=state, agent="RiskAssessor", assessment=risk_assessment)

    _logger.info("【risk_assessor_node】风险评估完成，下一跳: ServicePlanner")
    return state


async def _generate_risk_assessment(facts_structured: Dict[str, Any], applied_laws: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成综合风险评估

    Args:
        facts_structured: 结构化案件事实
        applied_laws: 适用的法律法规列表

    Returns:
        风险评估结果字典
    """
    system_prompt = _load_prompt()

    # P0-1: 对传给 LLM 的案件事实进行 PII 脱敏
    facts_text = mask_pii(json.dumps(facts_structured, ensure_ascii=False))

    user_message_parts = [
        "案件事实：",
        facts_text,
        "",
        "适用法律：",
        str(applied_laws),
        "",
        "请基于上述信息进行全面的风险评估。",
    ]
    user_message = "\n".join(user_message_parts)

    response = await llm_gateway.generate(
        system_prompt=system_prompt, user_message=user_message, temperature=0.1, is_legal=True
    )

    _logger.debug("【_generate_risk_assessment】LLM生成风险评估完成")

    try:
        assessment = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        _logger.warning("【_generate_risk_assessment】JSON解析失败，使用默认格式")
        assessment = _parse_fallback_assessment(response)

    if not isinstance(assessment, dict):
        _logger.warning("【_generate_risk_assessment】响应顶层不是对象，使用默认格式")
        assessment = _parse_fallback_assessment(response)

    return assessment


def _parse_fallback_assessment(raw_response: str) -> Dict[str, Any]:
    """解析非JSON格式的评估响应作为后备方案

    Args:
        raw_response: LLM返回的原始文本

    Returns:
        结构化的风险评估字典
    """
    return {
        "predicted_sentence_range": "待评估",
        "mitigating_factors": [],
        "aggravating_factors": [],
        "compulsory_measure_risk": {
            "detention_status": "待确认",
            "bail_possibility": "待评估",
            "measure_change_space": "待评估",
            "prolonged_detention_risk": "待评估",
        },
        "evidence_risk_points": [],
        "procedure_risks": [],
    }


def _add_to_conversation_history(state: "ConsultationState", agent: str, assessment: Dict[str, Any]) -> None:
    """更新对话历史记录

    Args:
        state: 当前 ConsultationState
        agent: 当前agent名称
        assessment: 风险评估结果
    """
    history_entry = {
        "agent": agent,
        "action": "risk_assessment_completed",
        "assessment_summary": {
            "predicted_sentence_range": assessment.get("predicted_sentence_range", "待评估"),
            "key_risks": _extract_key_risks(assessment),
        },
    }

    conversation_history = state.get("conversation_history", [])
    conversation_history.append(history_entry)
    state["conversation_history"] = conversation_history

    _logger.debug("【_add_to_conversation_history】已添加对话历史记录")


def _extract_key_risks(assessment: Dict[str, Any]) -> List[str]:
    """提取关键风险摘要

    Args:
        assessment: 风险评估结果

    Returns:
        关键风险列表
    """
    key_risks = []

    compulsory_measure = assessment.get("compulsory_measure_risk", {})
    if compulsory_measure.get("prolonged_detention_risk") == "high":
        key_risks.append("超期羁押风险")

    evidence_risks = assessment.get("evidence_risk_points", [])
    if evidence_risks:
        key_risks.append(f"存在{len(evidence_risks)}个证据风险点")

    procedure_risks = assessment.get("procedure_risks", [])
    for risk in procedure_risks:
        if risk.get("severity") == "high":
            key_risks.append(f"高风险: {risk.get('type', '未知')}")

    return key_risks


def format_risk_assessment_report(risk_assessment: Dict[str, Any]) -> str:
    """格式化风险评估报告

    Args:
        risk_assessment: 风险评估结果

    Returns:
        格式化的报告文本
    """
    report_lines = [
        "【风险评估报告】",
        "",
        "一、量刑预测",
        f"  量刑区间: {risk_assessment.get('predicted_sentence_range', '待评估')}",
        "",
        "  从轻/减轻情节:",
        _format_factor_list(risk_assessment.get("mitigating_factors", [])),
        "",
        "  从重情节:",
        _format_factor_list(risk_assessment.get("aggravating_factors", [])),
        "",
        "二、强制措施风险",
    ]

    compulsory_measure = risk_assessment.get("compulsory_measure_risk", {})
    report_lines.extend(
        [
            f"  羁押状态: {compulsory_measure.get('detention_status', '待确认')}",
            f"  取保候审可能性: {compulsory_measure.get('bail_possibility', '待评估')}",
            f"  变更强制措施空间: {compulsory_measure.get('measure_change_space', '待评估')}",
            f"  超期羁押风险: {compulsory_measure.get('prolonged_detention_risk', '待评估')}",
        ]
    )

    report_lines.extend(
        [
            "",
            "三、证据风险点",
        ]
    )
    evidence_risks = risk_assessment.get("evidence_risk_points", [])
    if evidence_risks:
        for i, risk in enumerate(evidence_risks, 1):
            report_lines.extend(
                [
                    f"  {i}. {risk.get('gap', '证据缺口')}",
                    f"     非法证据排除可能性: {risk.get('exclusion_possibility', '待评估')}",
                ]
            )
    else:
        report_lines.append("  未发现明显证据风险点")

    report_lines.extend(
        [
            "",
            "四、程序风险",
        ]
    )
    procedure_risks = risk_assessment.get("procedure_risks", [])
    if procedure_risks:
        for risk in procedure_risks:
            report_lines.append(
                f"  [{risk.get('severity', 'unknown').upper()}] {risk.get('type', '未知')}: "
                f"{risk.get('description', '')}"
            )
    else:
        report_lines.append("  未发现明显程序风险")

    report_content = "\n".join(report_lines)
    return disclaimer.inject(report_content)


def _format_factor_list(factors: List[str]) -> str:
    """格式化情节列表

    Args:
        factors: 情节列表

    Returns:
        格式化后的文本
    """
    if not factors:
        return "    无"
    return "    " + "\n    ".join(f"- {factor}" for factor in factors)
