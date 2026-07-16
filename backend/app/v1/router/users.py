from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.success_response import success_response
from app.db.db_config import get_db
from app.models.user import User, UserRole
from app.security.rbac import require_admin
from app.utils.logger import get_logger
from app.v1.schemas.auth_schemas import (
    UserListResponse,
    UserResponse,
    UserRoleUpdateRequest,
    UserUpdateRequest,
)

_logger = get_logger("Router.Users")

user_router = APIRouter(prefix="/users", tags=["users"])


@user_router.get("/", response_model=UserListResponse)
async def list_users(
    skip: int = 0,
    limit: int = 20,
    role: Optional[str] = None,
    _: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """获取用户列表接口（仅管理员可访问）。

    Args:
        skip: 跳过记录数。
        limit: 返回记录数。
        role: 按角色过滤。
        db: 数据库会话。

    Returns:
        用户列表。
    """
    query = select(User)
    if role:
        query = query.where(User.role == UserRole(role))

    count_query = select(User)
    if role:
        count_query = count_query.where(User.role == UserRole(role))

    total_result = await db.execute(count_query)
    total = len(total_result.scalars().all())

    query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
    result = await db.execute(query)
    users = result.scalars().all()

    return success_response(
        data=UserListResponse(
            users=[UserResponse.model_validate(u) for u in users],
            total=total,
        )
    )


@user_router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    _: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """获取指定用户信息接口（仅管理员可访问）。

    Args:
        user_id: 用户 ID。
        db: 数据库会话。

    Returns:
        用户信息。
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    return success_response(data=UserResponse.model_validate(user))


@user_router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    _: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """更新用户信息接口（仅管理员可访问）。

    Args:
        user_id: 用户 ID。
        request: 更新请求参数。
        db: 数据库会话。

    Returns:
        更新后的用户信息。
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if request.email:
        existing_email = await db.execute(select(User).where(User.email == request.email, User.id != user_id))
        if existing_email.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="邮箱已被使用")

    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)

    await db.commit()
    await db.refresh(user)

    _logger.info("【update_user】用户信息更新: user_id=%s", user_id)

    return success_response(data=UserResponse.model_validate(user))


@user_router.put("/{user_id}/role", response_model=UserResponse)
async def update_user_role(
    user_id: str,
    request: UserRoleUpdateRequest,
    _: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """更新用户角色接口（仅管理员可访问）。

    Args:
        user_id: 用户 ID。
        request: 角色更新请求参数。
        db: 数据库会话。

    Returns:
        更新后的用户信息。
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if request.role not in ["admin", "lawyer", "client"]:
        raise HTTPException(status_code=400, detail="无效的角色")

    user.role = UserRole(request.role)
    await db.commit()
    await db.refresh(user)

    _logger.info("【update_user_role】用户角色更新: user_id=%s, new_role=%s", user_id, request.role)

    return success_response(data=UserResponse.model_validate(user))


@user_router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    _: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """删除用户接口（仅管理员可访问）。

    Args:
        user_id: 用户 ID。
        db: 数据库会话。

    Returns:
        删除成功消息。
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    await db.delete(user)
    await db.commit()

    _logger.info("【delete_user】用户删除: user_id=%s", user_id)

    return success_response(message="用户已删除")
