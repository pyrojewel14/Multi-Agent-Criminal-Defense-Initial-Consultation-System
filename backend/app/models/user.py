from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.auth import Role, user_roles
from app.models.base import Base


class UserRole(str, PyEnum):
    ADMIN = "admin"
    LAWYER = "lawyer"
    CLIENT = "client"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(100), unique=True, nullable=True, index=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    real_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False, default=UserRole.CLIENT)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    refresh_token: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    refresh_token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    roles: Mapped[list[Role]] = relationship("Role", secondary=user_roles, back_populates="users")

    consultations_as_client: Mapped[list[Consultation]] = relationship(
        "Consultation", foreign_keys="Consultation.client_id", back_populates="client"
    )
    consultations_as_lawyer: Mapped[list[Consultation]] = relationship(
        "Consultation",
        foreign_keys="Consultation.assigned_lawyer_id",
        back_populates="assigned_lawyer",
    )


# 在类定义完成后引入 Consultation，保持跨模块类型提示可直接解析。
from app.models.consultation import Consultation  # noqa: E402,F401
