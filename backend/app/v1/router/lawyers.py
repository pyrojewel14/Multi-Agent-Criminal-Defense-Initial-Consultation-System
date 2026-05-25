from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.success_response import success_response
from app.db.db_config import get_db
from app.models.user import User, UserRole
from app.security.jwt import hash_password
from app.security.rbac import require_admin
from app.utils.logger import get_logger
from app.v1.schemas.auth_schemas import (
    LawyerListResponse,
    UserRegisterRequest,
    UserResponse,
    UserUpdateRequest,
)

_logger = get_logger("Router.Lawyers")

lawyer_management_router = APIRouter(prefix="/lawyers", tags=["lawyers"])


@lawyer_management_router.get("/", response_model=LawyerListResponse)
async def list_lawyers(
    _: None = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """获取律师列表接口（仅管理员可访问）。

    Args:
        db: 数据库会话。

    Returns:
        律师列表。
    """
    result = await db.execute(select(User).where(User.role == UserRole.LAWYER).order_by(User.created_at.desc()))
    lawyers = result.scalars().all()

    return success_response(
        data=LawyerListResponse(
            lawyers=[UserResponse.model_validate(l) for l in lawyers],
            total=len(lawyers),
        )
    )


@lawyer_management_router.post("/", response_model=UserResponse)
async def create_lawyer(
    request: UserRegisterRequest,
    _: None = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """创建律师账号接口（仅管理员可访问）。

    Args:
        request: 律师注册请求参数。
        db: 数据库会话。

    Returns:
        创建的律师账号信息。
    """
    existing = await db.execute(select(User).where(User.username == request.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="用户名已存在")

    if request.email:
        existing_email = await db.execute(select(User).where(User.email == request.email))
        if existing_email.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="邮箱已被使用")

    user = User(
        username=request.username,
        password_hash=hash_password(request.password),
        email=request.email,
        phone=request.phone,
        real_name=request.real_name,
        role=UserRole.LAWYER,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    _logger.info("【create_lawyer】律师账号创建: user_id=%s", user.id)

    return success_response(data=UserResponse.model_validate(user), message="律师账号创建成功")


@lawyer_management_router.put("/{lawyer_id}", response_model=UserResponse)
async def update_lawyer(
    lawyer_id: str,
    request: UserUpdateRequest,
    _: None = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """更新律师信息接口（仅管理员可访问）。

    Args:
        lawyer_id: 律师用户 ID。
        request: 更新请求参数。
        db: 数据库会话。

    Returns:
        更新后的律师信息。
    """
    result = await db.execute(select(User).where(User.id == lawyer_id, User.role == UserRole.LAWYER))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="律师不存在")

    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)

    await db.commit()
    await db.refresh(user)

    _logger.info("【update_lawyer】律师信息更新: lawyer_id=%s", lawyer_id)

    return success_response(data=UserResponse.model_validate(user))
