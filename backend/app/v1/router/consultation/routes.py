import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.db_config import get_db
from app.models.user import Consultation
from app.orchestrator.workflow import orchestrator
from app.security.rbac import get_current_user, require_lawyer
from app.state.consultation_state import validate_consultation_state
from app.utils.logger import get_logger
from app.v1.router.consultation.constants import HIGH_RISK_ALERT_MESSAGE
from app.v1.schemas.consultation_schemas import (
    ConfirmConsentRequest,
    ConfirmConsentResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    LawyerReviewRequest,
    LawyerReviewResponse,
    ReportDraftResponse,
    SendMessageRequest,
    SendMessageResponse,
    SessionCloseRequest,
    SessionCloseResponse,
    SessionListItem,
    SessionListResponse,
    SessionStateResponse,
)
from app.v1.service import consultation_service

_logger = get_logger("Router.Consultation")

router = APIRouter(prefix="/sessions", tags=["consultation"])


def _can_access_session(state: dict, current_user: dict) -> bool:
    """判断用户是否为会话所有者、已分配律师或管理员。"""
    if current_user["role"] == "admin":
        return True
    if current_user["role"] == "client":
        return state.get("user_id") == current_user["user_id"]
    if current_user["role"] == "lawyer":
        return state.get("lawyer_id") == current_user["user_id"]
    return False


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """创建新会话

    创建新的咨询会话，启动工作流（执行到第一个中断点），返回欢迎语。

    Args:
        request: 创建会话请求参数
        current_user: 当前认证用户
        db: 数据库会话

    Returns:
        会话创建响应，包含session_id和欢迎语
    """
    _logger.info(
        "【create_session】创建会话请求: user_id=%s, user_type=%s",
        current_user["user_id"],
        request.user_type,
    )

    session_id = str(uuid.uuid4())
    user_id = current_user["user_id"]
    user_type = request.user_type if request.user_type else "suspect"

    consultation_id = await consultation_service.create_consultation_record(session_id, user_id, user_type, db)

    welcome_message = await consultation_service.generate_welcome_message(user_type)

    initial_state = validate_consultation_state({
        "consultation_id": consultation_id,
        "user_id": user_id,
        "session_id": session_id,
        "user_type": user_type,
        "consent_given": False,
        "facts_raw": [],
        "facts_structured": {},
        "applied_laws": [],
        "current_agent": "Receptionist",
        "pending_questions": [],
        "alert_triggered": False,
        "conversation_history": [],
        "user_role": current_user.get("role", "client"),
    })

    # 启动工作流，执行到第一个中断点（receptionist 执行后中断，等待同意）
    result = await consultation_service.start_session(initial_state)
    current_agent = result.get("current_agent", "Receptionist")

    _logger.info("【create_session】会话创建成功: session_id=%s, consultation_id=%s", session_id, consultation_id)

    return CreateSessionResponse(
        session_id=session_id,
        consultation_id=consultation_id,
        welcome_message=welcome_message,
        current_agent=current_agent,
        created_at=datetime.now(timezone.utc),
    )


@router.post("/{session_id}/message", response_model=SendMessageResponse)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """发送消息

    处理用户消息，通过 resume_workflow 恢复工作流，由条件边自动路由到下一节点。

    Args:
        session_id: 会话ID
        request: 发送消息请求参数
        current_user: 当前认证用户
        db: 数据库会话

    Returns:
        Agent回复响应
    """
    _logger.info(
        "【send_message】发送消息: session_id=%s, user_id=%s, content=%s",
        session_id,
        current_user["user_id"],
        request.content[:50],
    )

    state = await consultation_service.get_session_state(session_id)

    if not state:
        _logger.warning("【send_message】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    if state.get("user_id") != current_user["user_id"]:
        _logger.warning(
            "【send_message】无权访问会话: session_id=%s, user_id=%s",
            session_id,
            current_user["user_id"],
        )
        raise HTTPException(status_code=403, detail="无权访问此会话")

    if not state.get("consent_given"):
        _logger.warning("【send_message】用户未同意隐私条款: session_id=%s", session_id)
        raise HTTPException(status_code=403, detail="请先确认隐私条款")

    current_agent = state.get("current_agent", "Receptionist")

    # 通过 resume_workflow 恢复，LangGraph 条件边自动路由
    result = await consultation_service.process_message(session_id, request.content, state, current_agent)

    if result.error:
        _logger.error("【send_message】消息处理异常: session_id=%s, error=%s", session_id, result.error)
        raise HTTPException(status_code=500, detail="消息处理失败，请稍后重试")

    if result.result_state is None:
        _logger.error("【send_message】result_state 为空: session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="消息处理失败，请稍后重试")

    if result.alert_triggered:
        raise HTTPException(
            status_code=403,
            detail=HIGH_RISK_ALERT_MESSAGE,
        )

    await consultation_service.save_message_to_db(
        consultation_id=result.result_state.get("consultation_id", ""),
        session_id=session_id,
        content=request.content,
        sender_type="user",
        sender_id=current_user["user_id"],
        db=db,
    )

    if result.response_content:
        await consultation_service.save_message_to_db(
            consultation_id=result.result_state.get("consultation_id", ""),
            session_id=session_id,
            content=result.response_content,
            sender_type="agent",
            agent_name=current_agent,
            db=db,
        )

    message_id = str(uuid.uuid4())

    _logger.info(
        "【send_message】消息处理完成: session_id=%s, agent=%s, next_agent=%s",
        session_id,
        current_agent,
        result.next_agent,
    )

    return SendMessageResponse(
        session_id=session_id,
        message_id=message_id,
        agent_name=current_agent,
        response_content=result.response_content,
        is_complete=result.is_workflow_finished,
        pending_questions=result.result_state.get("pending_questions"),
        alert_triggered=result.alert_triggered,
        created_at=datetime.now(timezone.utc),
    )


@router.post("/{session_id}/confirm-consent", response_model=ConfirmConsentResponse)
async def confirm_consent(
    session_id: str,
    request: ConfirmConsentRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """确认隐私同意

    通过 resume_workflow 恢复工作流，check_consent 条件边自动路由。

    Args:
        session_id: 会话ID
        request: 确认同意请求参数
        current_user: 当前认证用户
        db: 数据库会话

    Returns:
        确认结果响应
    """
    _logger.info(
        "【confirm_consent】确认隐私同意: session_id=%s, consent_accepted=%s, user_id=%s",
        session_id,
        request.consent_given,
        current_user["user_id"],
    )

    state = await consultation_service.get_session_state(session_id)

    if not state:
        _logger.warning("【confirm_consent】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    if state.get("user_id") != current_user["user_id"]:
        _logger.warning(
            "【confirm_consent】无权访问会话: session_id=%s, user_id=%s",
            session_id,
            current_user["user_id"],
        )
        raise HTTPException(status_code=403, detail="无权访问此会话")

    consultation_id = state.get("consultation_id")

    if consultation_id:
        consultation_result = await db.execute(
            select(Consultation).where(Consultation.id == consultation_id)
        )
        consultation = consultation_result.scalar_one_or_none()
        if consultation:
            consultation.consent_given = request.consent_given
            await db.commit()

    # 通过 resume_workflow 恢复，条件边 check_consent 自动路由
    result = await consultation_service.process_consent(
        session_id=session_id,
        consent_given=request.consent_given,
        identity_info=request.identity_info if request.consent_given else None,
    )

    if request.consent_given:
        next_prompt = "感谢您的同意。请告诉我您的身份类型（嫌疑人、受害者或家属）和案件发生的大概城市。"
        conversation_started = False
    else:
        next_prompt = "如需继续使用咨询服务，请回复'同意'确认您已阅读并理解权利义务告知。"
        conversation_started = False

    _logger.info(
        "【confirm_consent】隐私同意确认完成: session_id=%s, consent_given=%s",
        session_id,
        request.consent_given,
    )

    return ConfirmConsentResponse(
        session_id=session_id,
        success=True,
        current_agent=result.get("current_agent", "Receptionist"),
        next_prompt=next_prompt,
        conversation_started=conversation_started,
    )


@router.get("/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """获取会话状态

    返回当前会话状态，供律师端恢复。

    Args:
        session_id: 会话ID
        current_user: 当前认证用户

    Returns:
        会话状态响应
    """
    _logger.info(
        "【get_session_state】获取会话状态: session_id=%s, user_id=%s, role=%s",
        session_id,
        current_user["user_id"],
        current_user["role"],
    )

    state = await consultation_service.get_session_state(session_id)

    if not state:
        _logger.warning("【get_session_state】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    user_id = state.get("user_id", "")
    if not _can_access_session(state, current_user):
        _logger.warning(
            "【get_session_state】无权访问会话: session_id=%s, user_id=%s",
            session_id,
            current_user["user_id"],
        )
        raise HTTPException(status_code=403, detail="无权访问此会话")

    _logger.info(
        "【get_session_state】会话状态获取成功: session_id=%s, current_agent=%s",
        session_id,
        state.get("current_agent"),
    )

    is_approved = state.get("lawyer_decision") == "approved"
    workflow_finished = await orchestrator.is_workflow_finished(session_id)
    status = "completed" if is_approved and workflow_finished else "active"

    return SessionStateResponse(
        session_id=session_id,
        consultation_id=state.get("consultation_id"),
        user_id=user_id,
        user_type=state.get("user_type") or "suspect",
        consent_given=state.get("consent_given", False),
        current_agent=state.get("current_agent", "Receptionist"),
        conversation_history=state.get("conversation_history", []),
        facts_raw=state.get("facts_raw", []),
        facts_structured=state.get("facts_structured"),
        applied_laws=state.get("applied_laws", []),
        pending_questions=state.get("pending_questions", []),
        alert_triggered=state.get("alert_triggered", False),
        risk_assessment=state.get("risk_assessment"),
        final_output=state.get("final_output"),
        lawyer_id=state.get("lawyer_id"),
        status=status,
    )


@router.get("/{session_id}/report-draft", response_model=ReportDraftResponse)
async def get_report_draft(
    session_id: str,
    current_user: dict = Depends(require_lawyer),
):
    """获取已分配会话的工作流报告草案。"""
    state = await consultation_service.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    if not _can_access_session(state, current_user):
        raise HTTPException(status_code=403, detail="无权访问此会话的报告草案")

    report_draft = state.get("report_draft")
    if not isinstance(report_draft, str) or not report_draft.strip():
        raise HTTPException(status_code=404, detail="报告草案尚未生成")

    return ReportDraftResponse(
        session_id=session_id,
        consultation_id=state.get("consultation_id"),
        report_draft=report_draft,
        service_plan=state.get("service_plan"),
        awaiting_lawyer_review=bool(state.get("awaiting_lawyer_review")),
    )


@router.put("/{session_id}/review", response_model=LawyerReviewResponse)
async def lawyer_review(
    session_id: str,
    request: LawyerReviewRequest,
    current_user: dict = Depends(require_lawyer),
):
    """律师审核反馈

    通过 resume_workflow 恢复工作流，lawyer_decision 条件边自动路由：
    - approved: 批准报告，流程结束
    - revise_facts: 返回 FactDigger 重新收集
    - revise_risk: 返回 RiskAssessor 重新评估

    Args:
        session_id: 会话ID
        request: 律师审核请求参数
        current_user: 当前认证用户（律师或管理员）

    Returns:
        律师审核响应
    """
    _logger.info(
        "【lawyer_review】律师审核请求: session_id=%s, decision=%s, user_id=%s, role=%s",
        session_id,
        request.decision,
        current_user["user_id"],
        current_user["role"],
    )

    state = await consultation_service.get_session_state(session_id)

    if not state:
        _logger.warning("【lawyer_review】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    if not state.get("awaiting_lawyer_review"):
        _logger.warning(
            "【lawyer_review】会话未等待审核: session_id=%s, awaiting=%s",
            session_id,
            state.get("awaiting_lawyer_review"),
        )
        raise HTTPException(status_code=400, detail="此会话未在等待律师审核")

    if not _can_access_session(state, current_user):
        _logger.warning(
            "【lawyer_review】无权审核会话: session_id=%s, user_id=%s",
            session_id,
            current_user["user_id"],
        )
        raise HTTPException(status_code=403, detail="无权审核此会话")

    if request.decision not in ["approved", "revise_facts", "revise_risk"]:
        _logger.warning("【lawyer_review】无效的审核决定: decision=%s", request.decision)
        raise HTTPException(status_code=400, detail="无效的审核决定")

    try:
        # 通过 resume_workflow 恢复，lawyer_decision 条件边自动路由
        result = await consultation_service.process_lawyer_review(
            session_id=session_id,
            decision=request.decision,
            feedback=request.feedback,
            final_output=request.final_output if request.decision == "approved" else None,
        )

        is_finished = await orchestrator.is_workflow_finished(session_id)
        next_agent = None if is_finished else result.get("current_agent")

        _logger.info(
            "【lawyer_review】律师审核处理完成: session_id=%s, decision=%s, next_agent=%s",
            session_id,
            request.decision,
            next_agent,
        )

        return LawyerReviewResponse(
            session_id=session_id,
            decision=request.decision,
            feedback=request.feedback,
            next_agent=next_agent,
            processed_at=datetime.now(timezone.utc),
        )

    except Exception as e:
        _logger.error("【lawyer_review】律师审核处理异常: session_id=%s, error=%s", session_id, str(e))
        raise HTTPException(status_code=500, detail="审核处理失败，请稍后重试")


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    current_user: dict = Depends(get_current_user),
):
    """获取会话列表

    根据用户角色返回不同的会话列表：
    - admin: 所有会话
    - lawyer: 分配给自己的会话
    - client: 自己的会话

    Args:
        current_user: 当前认证用户

    Returns:
        会话列表响应
    """
    _logger.info(
        "【list_sessions】会话列表请求: user_id=%s, role=%s",
        current_user["user_id"],
        current_user["role"],
    )

    active_sessions = orchestrator.get_active_sessions()
    sessions = []

    for sess_id, state in active_sessions.items():
        user_id = state.get("user_id", "")

        if current_user["role"] == "client" and user_id != current_user["user_id"]:
            continue

        if current_user["role"] == "lawyer":
            lawyer_id = state.get("lawyer_id")
            if lawyer_id != current_user["user_id"]:
                continue

        risk_assessment = state.get("risk_assessment") or {}
        sessions.append(SessionListItem(
            session_id=sess_id,
            consultation_id=state.get("consultation_id", ""),
            user_id=user_id,
            user_type=state.get("user_type") or "suspect",
            current_agent=state.get("current_agent", "Receptionist"),
            status="active" if state.get("awaiting_lawyer_review") else "in_progress",
            consent_given=state.get("consent_given", False),
            alert_triggered=state.get("alert_triggered", False),
            awaiting_lawyer_review=bool(state.get("awaiting_lawyer_review")),
            risk_level=risk_assessment.get("risk_level"),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))

    _logger.info("【list_sessions】会话列表返回: total=%d", len(sessions))

    return SessionListResponse(
        sessions=sessions,
        total=len(sessions),
    )


@router.post("/{session_id}/close", response_model=SessionCloseResponse)
async def close_session(
    session_id: str,
    request: SessionCloseRequest | None = None,
    current_user: dict = Depends(get_current_user),
):
    """关闭会话

    关闭指定会话，清理会话状态。

    Args:
        session_id: 会话ID
        request: 关闭会话请求参数
        current_user: 当前认证用户

    Returns:
        关闭会话响应
    """
    _logger.info(
        "【close_session】关闭会话请求: session_id=%s, user_id=%s, role=%s",
        session_id,
        current_user["user_id"],
        current_user["role"],
    )

    state = await consultation_service.get_session_state(session_id)

    if not state:
        _logger.warning("【close_session】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    if not _can_access_session(state, current_user):
        _logger.warning(
            "【close_session】无权关闭会话: session_id=%s, user_id=%s",
            session_id,
            current_user["user_id"],
        )
        raise HTTPException(status_code=403, detail="无权关闭此会话")

    reason = request.reason if request else None

    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append({
        "agent": "system",
        "action": "session_closed",
        "reason": reason,
        "closed_by": current_user["user_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    state["current_agent"] = "END"
    await consultation_service.persist_state(session_id, state)

    _logger.info(
        "【close_session】会话关闭成功: session_id=%s, reason=%s",
        session_id,
        reason,
    )

    return SessionCloseResponse(
        session_id=session_id,
        success=True,
        message=f"会话已成功关闭{'，原因：' + reason if reason else ''}",
        closed_at=datetime.now(timezone.utc),
    )
