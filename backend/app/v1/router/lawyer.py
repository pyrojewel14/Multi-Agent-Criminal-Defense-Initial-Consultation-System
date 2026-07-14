from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.success_response import success_response
from app.db.db_config import get_db
from app.models.user import Consultation, ConsultationMessage, ConsultationStatus, User
from app.security.rbac import require_lawyer
from app.utils.logger import get_logger
from app.v1.schemas.lawyer_schemas import (
    ApproveReportRequest,
    ApproveReportResponse,
    InterventionResponse,
    LawyerSessionDetail,
    LawyerSessionItem,
    LawyerSessionListResponse,
    MessageHistoryItem,
    RejectSessionRequest,
    RejectSessionResponse,
    RiskAlertItem,
)

_logger = get_logger("Router.Lawyer")

lawyer_session_router = APIRouter(prefix="/lawyer", tags=["lawyer"])


@lawyer_session_router.get("/sessions", response_model=LawyerSessionListResponse)
async def get_sessions(
    status: Optional[str] = Query(None, description="按状态筛选: pending, in_progress, awaiting_review, completed"),
    risk_level: Optional[str] = Query(None, description="按风险等级筛选: low, medium, high, critical"),
    needs_review: Optional[bool] = Query(None, description="筛选需要审核的会话"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: dict = Depends(require_lawyer),
    db: AsyncSession = Depends(get_db),
):
    """获取分配给当前律师的会话列表。

    支持分页、状态筛选、风险等级筛选和审核需求筛选。

    Args:
        status: 按状态筛选（pending, in_progress, awaiting_review, completed）
        risk_level: 按风险等级筛选（low, medium, high, critical）
        needs_review: 筛选需要律师审核的会话
        page: 页码，从1开始
        page_size: 每页记录数，最大100
        current_user: 当前认证用户（律师或管理员）
        db: 数据库会话

    Returns:
        会话列表响应，包含会话项、总数量、页码和每页大小
    """
    lawyer_id = current_user["user_id"]
    _logger.info(
        "【get_sessions】律师获取会话列表: lawyer_id=%s, status=%s, risk_level=%s, needs_review=%s, page=%s, page_size=%s",
        lawyer_id,
        status,
        risk_level,
        needs_review,
        page,
        page_size,
    )

    query = select(Consultation).where(Consultation.assigned_lawyer_id == lawyer_id)
    count_query = select(func.count()).select_from(Consultation).where(Consultation.assigned_lawyer_id == lawyer_id)

    if status:
        try:
            status_enum = ConsultationStatus(status)
            query = query.where(Consultation.status == status_enum)
            count_query = count_query.where(Consultation.status == status_enum)
        except ValueError:
            _logger.warning("【get_sessions】无效的状态值: %s", status)
            raise HTTPException(status_code=400, detail=f"无效的状态值: {status}")

    if needs_review is not None:
        query = query.where(Consultation.lawyer_review_needed == needs_review)
        count_query = count_query.where(Consultation.lawyer_review_needed == needs_review)

    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    offset = (page - 1) * page_size
    query = query.order_by(desc(Consultation.updated_at)).offset(offset).limit(page_size)

    result = await db.execute(query)
    consultations = result.scalars().all()

    session_items = []
    for consultation in consultations:
        client_result = await db.execute(select(User).where(User.id == consultation.client_id))
        client = client_result.scalar_one_or_none()

        session_item = LawyerSessionItem(
            id=consultation.id,
            client_id=consultation.client_id,
            client_username=client.username if client else None,
            client_real_name=client.real_name if client else None,
            user_type=consultation.user_type,
            status=consultation.status.value,
            risk_level=getattr(consultation, "risk_level", None),
            alert_triggered=getattr(consultation, "alert_triggered", False),
            lawyer_review_needed=getattr(consultation, "lawyer_review_needed", False),
            created_at=consultation.created_at,
            updated_at=consultation.updated_at,
        )
        session_items.append(session_item)

    _logger.info(
        "【get_sessions】会话列表查询成功: lawyer_id=%s, total=%s, returned=%s", lawyer_id, total, len(session_items)
    )

    return success_response(
        data=LawyerSessionListResponse(
            sessions=session_items,
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@lawyer_session_router.get("/sessions/{session_id}", response_model=LawyerSessionDetail)
async def get_session_detail(
    session_id: str,
    current_user: dict = Depends(require_lawyer),
    db: AsyncSession = Depends(get_db),
):
    """获取会话详情。

    返回会话的完整信息，包括客户信息、结构化数据、对话历史等。
    只有分配给当前律师的会话才能被访问。

    Args:
        session_id: 会话ID
        current_user: 当前认证用户（律师或管理员）
        db: 数据库会话

    Returns:
        会话详情响应

    Raises:
        404: 会话不存在
        403: 无权访问此会话
    """
    lawyer_id = current_user["user_id"]
    _logger.info("【get_session_detail】获取会话详情: session_id=%s, lawyer_id=%s", session_id, lawyer_id)

    result = await db.execute(select(Consultation).where(Consultation.id == session_id))
    consultation = result.scalar_one_or_none()

    if not consultation:
        _logger.warning("【get_session_detail】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在")

    if consultation.assigned_lawyer_id != lawyer_id:
        _logger.warning(
            "【get_session_detail】无权访问: session_id=%s, lawyer_id=%s, assigned_lawyer=%s",
            session_id,
            lawyer_id,
            consultation.assigned_lawyer_id,
        )
        raise HTTPException(status_code=403, detail="无权访问此会话")

    client_result = await db.execute(select(User).where(User.id == consultation.client_id))
    client = client_result.scalar_one_or_none()

    messages_result = await db.execute(
        select(ConsultationMessage)
        .where(ConsultationMessage.consultation_id == session_id)
        .order_by(ConsultationMessage.created_at)
    )
    messages = messages_result.scalars().all()

    conversation_history = [
        MessageHistoryItem(
            id=msg.id,
            sender_type=msg.sender_type,
            sender_id=msg.sender_id,
            content=msg.content,
            agent_name=msg.agent_name,
            message_type=msg.message_type,
            created_at=msg.created_at,
        )
        for msg in messages
    ]

    facts_raw = None
    if hasattr(consultation, "facts_raw") and consultation.facts_raw:
        import json

        try:
            facts_raw = (
                json.loads(consultation.facts_raw)
                if isinstance(consultation.facts_raw, str)
                else consultation.facts_raw
            )
        except json.JSONDecodeError:
            facts_raw = None

    facts_structured = None
    if consultation.facts_structured:
        import json

        try:
            facts_structured = (
                json.loads(consultation.facts_structured)
                if isinstance(consultation.facts_structured, str)
                else consultation.facts_structured
            )
        except json.JSONDecodeError:
            facts_structured = None

    applied_laws = None
    if consultation.applied_laws:
        import json

        try:
            applied_laws = (
                json.loads(consultation.applied_laws)
                if isinstance(consultation.applied_laws, str)
                else consultation.applied_laws
            )
        except json.JSONDecodeError:
            applied_laws = None

    session_detail = LawyerSessionDetail(
        id=consultation.id,
        client_id=consultation.client_id,
        client_username=client.username if client else None,
        client_real_name=client.real_name if client else None,
        user_type=consultation.user_type,
        consent_given=consultation.consent_given,
        status=consultation.status.value,
        risk_level=getattr(consultation, "risk_level", None),
        alert_triggered=getattr(consultation, "alert_triggered", False),
        lawyer_review_needed=getattr(consultation, "lawyer_review_needed", False),
        facts_raw=facts_raw,
        facts_structured=facts_structured,
        applied_laws=applied_laws,
        risk_assessment=getattr(consultation, "risk_assessment", None),
        report_draft=getattr(consultation, "report_draft", None),
        service_plan=getattr(consultation, "service_plan", None),
        final_output=consultation.final_output,
        conversation_history=conversation_history,
        created_at=consultation.created_at,
        updated_at=consultation.updated_at,
        completed_at=consultation.completed_at,
    )

    _logger.info("【get_session_detail】会话详情获取成功: session_id=%s", session_id)
    return success_response(data=session_detail)


@lawyer_session_router.put("/sessions/{session_id}/report", response_model=ApproveReportResponse)
async def approve_report(
    session_id: str,
    request: ApproveReportRequest,
    current_user: dict = Depends(require_lawyer),
    db: AsyncSession = Depends(get_db),
):
    """审核并批准报告。

    律师审核报告草案，确认最终输出内容，并将会话状态设置为已完成。
    只有分配给当前律师的会话才能被操作。

    Args:
        session_id: 会话ID
        request: 审核报告请求，包含最终报告内容和可选反馈
        current_user: 当前认证用户（律师或管理员）
        db: 数据库会话

    Returns:
        审核成功响应

    Raises:
        404: 会话不存在
        403: 无权操作此会话
        400: 会话状态不允许审核
    """
    lawyer_id = current_user["user_id"]
    _logger.info(
        "【approve_report】律师审核报告: session_id=%s, lawyer_id=%s, has_feedback=%s",
        session_id,
        lawyer_id,
        request.feedback is not None,
    )

    result = await db.execute(select(Consultation).where(Consultation.id == session_id))
    consultation = result.scalar_one_or_none()

    if not consultation:
        _logger.warning("【approve_report】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在")

    if consultation.assigned_lawyer_id != lawyer_id:
        _logger.warning("【approve_report】无权操作: session_id=%s, lawyer_id=%s", session_id, lawyer_id)
        raise HTTPException(status_code=403, detail="无权操作此会话")

    if consultation.status == ConsultationStatus.COMPLETED:
        _logger.warning("【approve_report】会话已完成: session_id=%s", session_id)
        raise HTTPException(status_code=400, detail="会话已完成，无法重复审核")

    consultation.final_output = request.final_output
    consultation.status = ConsultationStatus.COMPLETED
    consultation.completed_at = datetime.utcnow()
    consultation.lawyer_review_needed = False

    if request.feedback:
        _logger.info("【approve_report】律师反馈: %s", request.feedback)

    await db.commit()

    _logger.info("【approve_report】报告审核成功: session_id=%s", session_id)

    return success_response(
        data=ApproveReportResponse(
            success=True,
            message="报告已审核通过",
            session_id=session_id,
            approved_at=datetime.utcnow(),
        )
    )


@lawyer_session_router.post("/sessions/{session_id}/reject", response_model=RejectSessionResponse)
async def reject_session(
    session_id: str,
    request: RejectSessionRequest,
    current_user: dict = Depends(require_lawyer),
    db: AsyncSession = Depends(get_db),
):
    """退回会话重做。

    律师将会话退回给指定节点重新处理，可指定退回原因和修改要求。
    目标节点可以是 fact_digger（事实收集）或 risk_assessor（风险评估）。

    Args:
        session_id: 会话ID
        request: 退回请求，包含目标节点、退回原因和修改要求
        current_user: 当前认证用户（律师或管理员）
        db: 数据库会话

    Returns:
        退回成功响应

    Raises:
        404: 会话不存在
        403: 无权操作此会话
        400: 无效的目标节点
    """
    lawyer_id = current_user["user_id"]
    _logger.info(
        "【reject_session】律师退回会话: session_id=%s, lawyer_id=%s, target_node=%s, reason=%s",
        session_id,
        lawyer_id,
        request.target_node,
        request.reason,
    )

    valid_nodes = ["fact_digger", "risk_assessor"]
    if request.target_node not in valid_nodes:
        _logger.warning("【reject_session】无效的目标节点: %s", request.target_node)
        raise HTTPException(status_code=400, detail=f"无效的目标节点，可选值: {', '.join(valid_nodes)}")

    result = await db.execute(select(Consultation).where(Consultation.id == session_id))
    consultation = result.scalar_one_or_none()

    if not consultation:
        _logger.warning("【reject_session】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在")

    if consultation.assigned_lawyer_id != lawyer_id:
        _logger.warning("【reject_session】无权操作: session_id=%s, lawyer_id=%s", session_id, lawyer_id)
        raise HTTPException(status_code=403, detail="无权操作此会话")

    consultation.status = ConsultationStatus.IN_PROGRESS
    consultation.lawyer_review_needed = False

    _logger.info(
        "【reject_session】会话已退回: session_id=%s, target_node=%s, feedback=%s",
        session_id,
        request.target_node,
        request.feedback,
    )

    await db.commit()

    return success_response(
        data=RejectSessionResponse(
            success=True,
            message=f"会话已退回至 {request.target_node} 重新处理",
            session_id=session_id,
            target_node=request.target_node,
            rejected_at=datetime.utcnow(),
        )
    )


@lawyer_session_router.post("/sessions/{session_id}/intervene", response_model=InterventionResponse)
async def intervene_session(
    session_id: str,
    current_user: dict = Depends(require_lawyer),
    db: AsyncSession = Depends(get_db),
):
    """人工接管会话。

    律师主动接管当前会话，终止自动流程，开始人工对话。
    适用于需要律师直接参与的紧急情况或复杂案件。

    Args:
        session_id: 会话ID
        current_user: 当前认证用户（律师或管理员）
        db: 数据库会话

    Returns:
        接管成功响应

    Raises:
        404: 会话不存在
        403: 无权操作此会话
    """
    lawyer_id = current_user["user_id"]
    _logger.info("【intervene_session】律师接管会话: session_id=%s, lawyer_id=%s", session_id, lawyer_id)

    result = await db.execute(select(Consultation).where(Consultation.id == session_id))
    consultation = result.scalar_one_or_none()

    if not consultation:
        _logger.warning("【intervene_session】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在")

    if consultation.assigned_lawyer_id != lawyer_id:
        _logger.warning("【intervene_session】无权操作: session_id=%s, lawyer_id=%s", session_id, lawyer_id)
        raise HTTPException(status_code=403, detail="无权操作此会话")

    consultation.status = ConsultationStatus.IN_PROGRESS

    message = ConsultationMessage(
        consultation_id=session_id,
        sender_type="lawyer",
        sender_id=lawyer_id,
        content="律师已人工接管此会话，正在处理中...",
        message_type="system",
    )
    db.add(message)

    await db.commit()

    _logger.info("【intervene_session】会话接管成功: session_id=%s, lawyer_id=%s", session_id, lawyer_id)

    return success_response(
        data=InterventionResponse(
            success=True,
            message="已成功接管会话",
            session_id=session_id,
            intervened_at=datetime.utcnow(),
            current_agent="Lawyer",
        )
    )


@lawyer_session_router.get("/alerts", response_model=List[RiskAlertItem])
async def get_alerts(
    is_read: Optional[bool] = Query(None, description="按已读状态筛选"),
    risk_level: Optional[str] = Query(None, description="按风险等级筛选: low, medium, high, critical"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: dict = Depends(require_lawyer),
    db: AsyncSession = Depends(get_db),
):
    """获取高风险告警列表。

    返回分配给当前律师的高风险会话告警，支持按已读状态和风险等级筛选。

    Args:
        is_read: 按已读状态筛选
        risk_level: 按风险等级筛选（low, medium, high, critical）
        page: 页码，从1开始
        page_size: 每页记录数，最大100
        current_user: 当前认证用户（律师或管理员）
        db: 数据库会话

    Returns:
        高风险告警列表
    """
    lawyer_id = current_user["user_id"]
    _logger.info(
        "【get_alerts】获取高风险告警: lawyer_id=%s, is_read=%s, risk_level=%s, page=%s, page_size=%s",
        lawyer_id,
        is_read,
        risk_level,
        page,
        page_size,
    )

    query = select(Consultation).where(
        Consultation.assigned_lawyer_id == lawyer_id,
        Consultation.alert_triggered == True,
    )

    if risk_level:
        query = query.where(Consultation.risk_level == risk_level)

    offset = (page - 1) * page_size
    query = query.order_by(desc(Consultation.updated_at)).offset(offset).limit(page_size)

    result = await db.execute(query)
    consultations = result.scalars().all()

    alerts = []
    for consultation in consultations:
        client_result = await db.execute(select(User).where(User.id == consultation.client_id))
        client = client_result.scalar_one_or_none()

        risk_assessment = getattr(consultation, "risk_assessment", None) or {}
        if isinstance(risk_assessment, str):
            import json

            try:
                risk_assessment = json.loads(risk_assessment)
            except json.JSONDecodeError:
                risk_assessment = {}

        alert_item = RiskAlertItem(
            id=f"alert_{consultation.id}",
            session_id=consultation.id,
            client_id=consultation.client_id,
            client_real_name=client.real_name if client else None,
            risk_type=risk_assessment.get("risk_type", "未知"),
            risk_level=risk_assessment.get("risk_level", consultation.risk_level or "medium"),
            details=risk_assessment.get("details"),
            is_read=getattr(consultation, "alert_read", False),
            created_at=consultation.created_at,
        )
        alerts.append(alert_item)

    _logger.info("【get_alerts】告警列表获取成功: lawyer_id=%s, count=%s", lawyer_id, len(alerts))

    return success_response(data=alerts)


@lawyer_session_router.put("/alerts/{alert_id}/read")
async def mark_alert_read(
    alert_id: str,
    current_user: dict = Depends(require_lawyer),
    db: AsyncSession = Depends(get_db),
):
    """标记告警为已读。

    Args:
        alert_id: 告警ID（格式: alert_{session_id}）
        current_user: 当前认证用户（律师或管理员）
        db: 数据库会话

    Returns:
        操作成功消息
    """
    lawyer_id = current_user["user_id"]
    _logger.info("【mark_alert_read】标记告警已读: alert_id=%s, lawyer_id=%s", alert_id, lawyer_id)

    if not alert_id.startswith("alert_"):
        raise HTTPException(status_code=400, detail="无效的告警ID格式")

    session_id = alert_id.replace("alert_", "")

    result = await db.execute(select(Consultation).where(Consultation.id == session_id))
    consultation = result.scalar_one_or_none()

    if not consultation:
        _logger.warning("【mark_alert_read】会话不存在: session_id=%s", session_id)
        raise HTTPException(status_code=404, detail="会话不存在")

    if consultation.assigned_lawyer_id != lawyer_id:
        _logger.warning("【mark_alert_read】无权操作: session_id=%s, lawyer_id=%s", session_id, lawyer_id)
        raise HTTPException(status_code=403, detail="无权操作此会话")

    consultation.alert_read = True
    await db.commit()

    _logger.info("【mark_alert_read】告警已标记为已读: alert_id=%s", alert_id)

    return success_response(message="告警已标记为已读")
