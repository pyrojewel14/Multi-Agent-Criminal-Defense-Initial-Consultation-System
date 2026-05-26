from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import rate_limit
from app.core.success_response import success_response
from app.db.db_config import get_db
from app.models.user import Consultation, ConsultationMessage, ConsultationStatus, User, UserRole
from app.security.rbac import get_current_user, require_admin, require_lawyer
from app.utils.logger import get_logger
from app.v1.schemas.consultation_schemas import (
    AssignLawyerRequest,
    ConsultationListResponse,
    ConsultationResponse,
    MessageResponse,
    UpdateStatusRequest,
)

_logger = get_logger("Router.Consultation")

consultation_router = APIRouter(prefix="/consultations", tags=["consultations"])


async def _build_consultation_response(consultation: Consultation, db: AsyncSession) -> ConsultationResponse:
    """构建咨询响应对象。

    Args:
        consultation: 咨询记录。
        db: 数据库会话。

    Returns:
        咨询响应对象。
    """
    client_result = await db.execute(select(User).where(User.id == consultation.client_id))
    client = client_result.scalar_one_or_none()

    lawyer_name = None
    if consultation.assigned_lawyer_id:
        lawyer_result = await db.execute(select(User).where(User.id == consultation.assigned_lawyer_id))
        lawyer = lawyer_result.scalar_one_or_none()
        lawyer_name = lawyer.real_name if lawyer else None

    return ConsultationResponse(
        id=consultation.id,
        client_id=consultation.client_id,
        client_username=client.username if client else None,
        client_real_name=client.real_name if client else None,
        assigned_lawyer_id=consultation.assigned_lawyer_id,
        assigned_lawyer_name=lawyer_name,
        user_type=consultation.user_type,
        consent_given=consultation.consent_given,
        status=consultation.status.value if consultation.status else None,
        facts_structured=consultation.facts_structured,
        applied_laws=consultation.applied_laws,
        final_output=consultation.final_output,
        created_at=consultation.created_at,
        updated_at=consultation.updated_at,
        completed_at=consultation.completed_at,
    )


@consultation_router.get("/list", response_model=ConsultationListResponse)
async def list_consultations(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取咨询列表，根据用户角色返回不同范围的咨询记录。

    Args:
        page: 页码。
        page_size: 每页记录数。
        status: 按状态过滤。
        current_user: 当前认证用户。
        db: 数据库会话。

    Returns:
        咨询列表响应。
    """
    _logger.info(
        "【list_consultations】咨询列表查询: user_id=%s, role=%s, page=%s, page_size=%s, status=%s",
        current_user["user_id"],
        current_user["role"],
        page,
        page_size,
        status,
    )

    offset = (page - 1) * page_size

    if current_user["role"] == "admin":
        query = select(Consultation)
        count_query = select(Consultation)
        _logger.debug("【list_consultations】超管查询: 获取所有咨询")
    elif current_user["role"] == "lawyer":
        query = select(Consultation).where(Consultation.assigned_lawyer_id == current_user["user_id"])
        count_query = select(Consultation).where(Consultation.assigned_lawyer_id == current_user["user_id"])
        _logger.debug("【list_consultations】律师查询: user_id=%s", current_user["user_id"])
    else:
        query = select(Consultation).where(Consultation.client_id == current_user["user_id"])
        count_query = select(Consultation).where(Consultation.client_id == current_user["user_id"])
        _logger.debug("【list_consultations】客户查询: user_id=%s", current_user["user_id"])

    if status:
        query = query.where(Consultation.status == ConsultationStatus(status))
        count_query = count_query.where(Consultation.status == ConsultationStatus(status))

    total_result = await db.execute(select(func.count()).select_from(count_query.subquery()))
    total = total_result.scalar_one()

    query = query.offset(offset).limit(page_size).order_by(Consultation.created_at.desc())
    result = await db.execute(query)
    consultations = result.scalars().all()

    response_list = []
    for c in consultations:
        response_list.append(await _build_consultation_response(c, db))

    _logger.info("【list_consultations】咨询列表查询成功: total=%s, 返回数量=%s", total, len(response_list))

    return success_response(
        data=ConsultationListResponse(
            consultations=response_list,
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@consultation_router.get("/{consultation_id}", response_model=ConsultationResponse)
async def get_consultation(
    consultation_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取咨询详情，根据用户角色进行权限验证。

    Args:
        consultation_id: 咨询 ID。
        current_user: 当前认证用户。
        db: 数据库会话。

    Returns:
        咨询详情响应。
    """
    result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
    consultation = result.scalar_one_or_none()

    if not consultation:
        raise HTTPException(status_code=404, detail="咨询记录不存在")

    if current_user["role"] == "client" and consultation.client_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="无权访问此咨询记录")

    if current_user["role"] == "lawyer" and consultation.assigned_lawyer_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="无权访问此咨询记录")

    return success_response(data=await _build_consultation_response(consultation, db))


@consultation_router.get("/{consultation_id}/messages")
async def get_consultation_messages(
    consultation_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取咨询的所有消息记录。

    Args:
        consultation_id: 咨询 ID。
        current_user: 当前认证用户。
        db: 数据库会话。

    Returns:
        消息列表响应。
    """
    consultation_result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
    consultation = consultation_result.scalar_one_or_none()

    if not consultation:
        raise HTTPException(status_code=404, detail="咨询记录不存在")

    if current_user["role"] == "client" and consultation.client_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="无权访问此咨询记录")

    if current_user["role"] == "lawyer" and consultation.assigned_lawyer_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="无权访问此咨询记录")

    messages_result = await db.execute(
        select(ConsultationMessage)
        .where(ConsultationMessage.consultation_id == consultation_id)
        .order_by(ConsultationMessage.created_at)
    )
    messages = messages_result.scalars().all()

    return success_response(data=[MessageResponse.model_validate(m) for m in messages])


@consultation_router.post("/assign")
async def assign_lawyer(
    request: AssignLawyerRequest,
    _: None = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """超管将咨询分配给律师。

    Args:
        request: 分配律师请求参数。
        db: 数据库会话。

    Returns:
        分配成功消息。
    """
    _logger.info(
        "【assign_lawyer】分配律师请求: consultation_id=%s, lawyer_id=%s", request.consultation_id, request.lawyer_id
    )

    consultation_id = request.consultation_id
    lawyer_id = request.lawyer_id

    lawyer_result = await db.execute(select(User).where(User.id == lawyer_id, User.role == UserRole.LAWYER))
    lawyer = lawyer_result.scalar_one_or_none()

    if not lawyer:
        _logger.warning("【assign_lawyer】分配失败: 律师不存在 lawyer_id=%s", lawyer_id)
        raise HTTPException(status_code=404, detail="律师不存在")

    consultation_result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
    consultation = consultation_result.scalar_one_or_none()

    if not consultation:
        _logger.warning("【assign_lawyer】分配失败: 咨询记录不存在 consultation_id=%s", consultation_id)
        raise HTTPException(status_code=404, detail="咨询记录不存在")

    consultation.assigned_lawyer_id = lawyer_id
    await db.commit()

    _logger.info(
        "【assign_lawyer】咨询分配律师成功: consultation_id=%s, lawyer_id=%s, lawyer_name=%s",
        consultation_id,
        lawyer_id,
        lawyer.real_name,
    )

    return success_response(message=f"已将咨询分配给律师 {lawyer.real_name}")


@consultation_router.put("/{consultation_id}/status")
async def update_consultation_status(
    consultation_id: str,
    request: UpdateStatusRequest,
    current_user: dict = Depends(require_lawyer),
    db: AsyncSession = Depends(get_db),
):
    """更新咨询状态。律师和超管可以更新状态。

    Args:
        consultation_id: 咨询 ID。
        request: 更新状态请求参数。
        current_user: 当前认证用户（律师或管理员）。
        db: 数据库会话。

    Returns:
        更新成功消息。
    """
    _logger.info(
        "【update_consultation_status】更新咨询状态请求: consultation_id=%s, new_status=%s, user_id=%s, role=%s",
        consultation_id,
        request.status,
        current_user["user_id"],
        current_user["role"],
    )

    result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
    consultation = result.scalar_one_or_none()

    if not consultation:
        _logger.warning("【update_consultation_status】咨询记录不存在 consultation_id=%s", consultation_id)
        raise HTTPException(status_code=404, detail="咨询记录不存在")

    if current_user["role"] == "lawyer" and consultation.assigned_lawyer_id != current_user["user_id"]:
        _logger.warning(
            "【update_consultation_status】权限不足: 律师只能修改分配给自己的咨询 user_id=%s", current_user["user_id"]
        )
        raise HTTPException(status_code=403, detail="只能修改分配给自己的咨询状态")

    try:
        new_status = ConsultationStatus(request.status)
    except ValueError:
        _logger.warning("【update_consultation_status】无效的状态值: %s", request.status)
        raise HTTPException(status_code=400, detail="无效的状态值")

    old_status = consultation.status.value if consultation.status else None
    consultation.status = new_status

    if new_status == ConsultationStatus.COMPLETED:
        consultation.completed_at = datetime.utcnow()

    await db.commit()

    _logger.info(
        "【update_consultation_status】咨询状态更新成功: consultation_id=%s, old_status=%s, new_status=%s",
        consultation_id,
        old_status,
        request.status,
    )

    return success_response(message=f"咨询状态已更新为 {request.status}")
