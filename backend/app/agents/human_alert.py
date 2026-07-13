from typing import TYPE_CHECKING

from app.security.disclaimer import disclaimer
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.state.consultation_state import ConsultationState

_logger = get_logger("Agent.HumanAlert")

ALERT_MESSAGE = "为保护您的权益，此部分内容建议直接与律师单独沟通。"


async def human_alert_node(state: "ConsultationState") -> "ConsultationState":
    """HumanAlert Agent 节点函数。

    处理高风险检测触发的人工介入。当系统检测到高风险内容时，
    此节点负责：
    1. 向用户发送安抚语
    2. 终止自动流程
    3. 通知律师进行人工处理

    Args:
        state: 当前 ConsultationState。

    Returns:
        更新后的 ConsultationState，流程已暂停等待人工介入。
    """
    _logger.info("=" * 60)
    _logger.warning("【human_alert_node】HumanAlert 节点触发 - 检测到高风险内容")
    _logger.warning("【human_alert_node】会话 ID: %s", state.get("session_id", "N/A"))
    _logger.warning("【human_alert_node】用户 ID: %s", state.get("user_id", "N/A"))
    _logger.warning("【human_alert_node】用户类型: %s", state.get("user_type", "N/A"))

    risk_assessment = state.get("risk_assessment")
    if risk_assessment:
        risk_type = risk_assessment.get("risk_type", "未知")
        risk_level = risk_assessment.get("risk_level", "未知")
        _logger.warning("【human_alert_node】风险类型: %s", risk_type)
        _logger.warning("【human_alert_node】风险等级: %s", risk_level)

        if risk_assessment.get("details"):
            _logger.warning("【human_alert_node】风险详情: %s", risk_assessment["details"])

    _logger.warning("【human_alert_node】自动流程已暂停，等待律师人工介入")
    _logger.info("=" * 60)

    response_content = ALERT_MESSAGE

    response_with_disclaimer = disclaimer.inject(response_content)

    updated_conversation_history = state.get("conversation_history", [])
    updated_conversation_history.append({
        "agent": "HumanAlert",
        "content": response_with_disclaimer,
        "alert_triggered": True,
        "risk_assessment": risk_assessment
    })

    # 保留告警标记，供 HTTP / WebSocket 调用方识别本次流程已转人工处理。
    state["alert_triggered"] = True
    state["current_agent"] = "HumanAlert"
    state["conversation_history"] = updated_conversation_history
    state["final_output"] = response_with_disclaimer
    state["lawyer_review_needed"] = True

    _logger.debug("【human_alert_node】状态更新完成 - alert_triggered 保持为 True")
    _logger.debug("【human_alert_node】状态更新完成 - current_agent 设置为 HumanAlert")
    _logger.debug("【human_alert_node】状态更新完成 - lawyer_review_needed 设置为 True")
    _logger.debug("【human_alert_node】HumanAlert 节点处理完成，自动流程已终止")

    return state
