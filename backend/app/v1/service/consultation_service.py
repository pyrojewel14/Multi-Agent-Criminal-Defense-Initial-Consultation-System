import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.redis_config import get_redis_cache_json, set_redis_cache
from app.models.user import Consultation, ConsultationMessage, ConsultationStatus
from app.orchestrator.workflow import orchestrator
from app.security.disclaimer import disclaimer
from app.state.consultation_state import ConsultationState
from app.utils.logger import get_logger
from app.v1.router.consultation.constants import HIGH_RISK_ALERT_MESSAGE, SESSION_TTL

_logger = get_logger("Service.Consultation")


@dataclass
class ProcessMessageResult:
    """消息处理结果"""
    response_content: str
    next_agent: str
    alert_triggered: bool
    result_state: Optional[ConsultationState]
    is_workflow_finished: bool = False
    error: Optional[str] = None


async def get_session_state(session_id: str) -> Optional[ConsultationState]:
    """统一的状态获取（checkpointer → Redis 回退）

    Args:
        session_id: 会话ID

    Returns:
        会话状态，如果不存在返回 None
    """
    # 优先从 checkpointer 获取最新状态
    snapshot = await orchestrator.get_snapshot(session_id)
    if snapshot and snapshot.values:
        return snapshot.values

    # 回退到 Redis
    state = await get_redis_cache_json(f"session:{session_id}")
    if state:
        return ConsultationState(**state) if isinstance(state, dict) else state

    # 最后尝试内存缓存
    return orchestrator.get_session_context(session_id)


async def persist_state(session_id: str, state: ConsultationState) -> None:
    """统一的状态持久化（内存缓存 + Redis）

    注意：checkpointer 由 LangGraph 自动管理，此方法仅同步内存和 Redis。

    Args:
        session_id: 会话ID
        state: 会话状态
    """
    orchestrator.update_session_context(session_id, state)
    await set_redis_cache(f"session:{session_id}", dict(state), expire=SESSION_TTL)


async def handle_high_risk_alert(
    session_id: str,
    result: ConsultationState,
    current_agent: str,
) -> None:
    """处理高风险警报，持久化状态。

    Args:
        session_id: 会话ID
        result: 更新后的状态（alert_triggered=True）
        current_agent: 检测到风险的 Agent 名称
    """
    _logger.warning(
        "【_handle_high_risk_alert】高风险警报: session_id=%s, agent=%s",
        session_id,
        current_agent,
    )

    if "conversation_history" not in result:
        result["conversation_history"] = []
    result["conversation_history"].append({
        "agent": current_agent,
        "action": "high_risk_alert",
        "content": HIGH_RISK_ALERT_MESSAGE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    await persist_state(session_id, result)


async def generate_welcome_message(_user_type: str = "suspect") -> str:
    """生成欢迎语和隐私条款告知

    Args:
        user_type: 用户类型

    Returns:
        欢迎语文本
    """
    welcome = """您好，欢迎使用刑事辩护初期咨询系统。

我是您的智能法律咨询助手，可以帮助您了解相关法律问题和权利义务。

在开始之前，请您仔细阅读以下重要提示：

【权利义务告知】

1. 本系统提供的仅为初步法律咨询参考，不构成正式法律意见
2. 律师-当事人关系将在您与律师正式签订委托合同后建立
3. 为保护您的权益，在咨询过程中请您如实陈述案件情况
4. 您有权随时终止咨询并寻求当面法律服务
5. 我们会严格保护您的个人信息和案件隐私

请回复"同意"或"确认"表示您已阅读并理解上述告知内容。"""

    return disclaimer.inject(welcome)


async def create_consultation_record(
    _session_id: str, user_id: str, user_type: str, db: AsyncSession
) -> str:
    """创建咨询数据库记录

    Args:
        session_id: 会话ID
        user_id: 用户ID
        user_type: 用户类型
        db: 数据库会话

    Returns:
        咨询记录ID
    """
    consultation = Consultation(
        client_id=user_id,
        user_type=user_type,
        consent_given=False,
        status=ConsultationStatus.PENDING,
    )
    db.add(consultation)
    await db.commit()
    await db.refresh(consultation)
    _logger.info("【_create_consultation_record】咨询记录创建: consultation_id=%s", consultation.id)
    return consultation.id


async def save_message_to_db(
    consultation_id: str,
    session_id: str,
    content: str,
    sender_type: str,
    sender_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    db: Optional[AsyncSession] = None,
) -> str:
    """保存消息到数据库

    Args:
        consultation_id: 咨询记录ID
        session_id: 会话ID
        content: 消息内容
        sender_type: 发送者类型
        sender_id: 发送者ID
        agent_name: Agent名称
        db: 数据库会话

    Returns:
        消息ID
    """
    if not db:
        return str(uuid.uuid4())

    message = ConsultationMessage(
        consultation_id=consultation_id,
        sender_type=sender_type,
        sender_id=sender_id,
        content=content,
        agent_name=agent_name,
        message_type="text",
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message.id


async def start_session(state: ConsultationState) -> ConsultationState:
    """启动新会话的工作流，执行到第一个中断点。

    Args:
        state: 初始会话状态

    Returns:
        工作流执行到中断点时的状态
    """
    result = await orchestrator.start_workflow(state)
    await persist_state(state["session_id"], result)
    return result


async def process_message(
    session_id: str,
    content: str,
    state: ConsultationState,
    current_agent: str,
) -> ProcessMessageResult:
    """统一的消息处理逻辑（HTTP 和 WebSocket 共用）

    通过 resume_workflow 恢复工作流，由 LangGraph 条件边自动路由到下一节点。

    Args:
        session_id: 会话ID
        content: 用户消息内容
        state: 当前会话状态
        current_agent: 当前 Agent 名称（仅用于日志和兼容）

    Returns:
        处理结果对象
    """
    try:
        # 构建状态更新：将用户消息追加到 facts_raw
        facts_raw = list(state.get("facts_raw", []))
        facts_raw.append(content)

        state_updates: Dict[str, Any] = {
            "facts_raw": facts_raw,
            "current_input": content,
        }

        # 通过 resume_workflow 恢复，LangGraph 自动路由
        result = await orchestrator.resume_workflow(session_id, state_updates)

        response_content = result.get("final_output", "")
        next_agent = result.get("current_agent", current_agent)
        alert_triggered = result.get("alert_triggered", False)
        is_finished = await orchestrator.is_workflow_finished(session_id)

        if alert_triggered:
            await handle_high_risk_alert(session_id, result, current_agent)
            return ProcessMessageResult(
                response_content=HIGH_RISK_ALERT_MESSAGE,
                next_agent=next_agent,
                alert_triggered=True,
                result_state=result,
                is_workflow_finished=is_finished,
            )

        if "conversation_history" not in result:
            result["conversation_history"] = []
        result["conversation_history"].append({
            "agent": current_agent,
            "user_message": content,
            "agent_response": response_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        await persist_state(session_id, result)

        return ProcessMessageResult(
            response_content=response_content,
            next_agent=next_agent,
            alert_triggered=False,
            result_state=result,
            is_workflow_finished=is_finished,
        )

    except Exception as e:
        _logger.error("【process_message】消息处理异常: session_id=%s, error=%s", session_id, str(e))
        return ProcessMessageResult(
            response_content="",
            next_agent=current_agent,
            alert_triggered=False,
            result_state=None,
            error=str(e),
        )


async def process_consent(
    session_id: str,
    consent_given: bool,
    identity_info: Optional[dict] = None,
) -> ConsultationState:
    """处理用户同意确认，通过 resume_workflow 恢复工作流。

    Args:
        session_id: 会话ID
        consent_given: 是否同意
        identity_info: 身份信息

    Returns:
        更新后的状态
    """
    state_updates: Dict[str, Any] = {"consent_given": consent_given}
    if identity_info:
        state_updates["identity_info"] = identity_info

    if consent_given:
        result = await orchestrator.resume_workflow(session_id, state_updates)
    else:
        # 用户不同意，更新状态但不恢复工作流（流程将走向 END）
        result = await orchestrator.resume_workflow(session_id, state_updates)

    await persist_state(session_id, result)
    return result


async def process_lawyer_review(
    session_id: str,
    decision: str,
    feedback: Optional[str] = None,
    final_output: Optional[str] = None,
) -> ConsultationState:
    """处理律师审核反馈，通过工作流自动路由。

    Args:
        session_id: 会话ID
        decision: 律师决策 (approved/revise_facts/revise_risk)
        feedback: 律师反馈
        final_output: 最终输出（approved 时可能附带）

    Returns:
        更新后的状态
    """
    state_updates: Dict[str, Any] = {
        "lawyer_decision": decision,
        "lawyer_feedback": feedback,
        "awaiting_lawyer_review": False,
    }
    if decision == "approved" and final_output:
        state_updates["final_output"] = final_output

    result = await orchestrator.resume_workflow(session_id, state_updates)
    await persist_state(session_id, result)
    return result
