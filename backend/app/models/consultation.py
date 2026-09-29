from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ConsultationStatus(str, PyEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Consultation(Base):
    __tablename__ = "consultations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, unique=True, index=True)
    client_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    assigned_lawyer_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )

    user_type: Mapped[str] = mapped_column(String(20), nullable=False)
    consent_given: Mapped[bool] = mapped_column(Boolean, default=False)

    status: Mapped[ConsultationStatus] = mapped_column(Enum(ConsultationStatus), default=ConsultationStatus.PENDING)

    facts_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    facts_structured: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    applied_laws: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    final_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    risk_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    risk_assessment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    alert_triggered: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    alert_read: Mapped[bool] = mapped_column(Boolean, default=False)
    lawyer_review_needed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    report_draft: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    service_plan: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    client: Mapped[User] = relationship("User", foreign_keys=[client_id], back_populates="consultations_as_client")
    assigned_lawyer: Mapped[Optional[User]] = relationship(
        "User",
        foreign_keys=[assigned_lawyer_id],
        back_populates="consultations_as_lawyer",
    )
    messages: Mapped[list[ConsultationMessage]] = relationship(
        "ConsultationMessage",
        back_populates="consultation",
        cascade="all, delete-orphan",
    )


class ConsultationMessage(Base):
    __tablename__ = "consultation_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    consultation_id: Mapped[str] = mapped_column(String(36), ForeignKey("consultations.id"), nullable=False, index=True)

    sender_type: Mapped[str] = mapped_column(String(20), nullable=False)
    sender_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)

    agent_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    message_type: Mapped[str] = mapped_column(String(20), default="text")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    consultation: Mapped[Consultation] = relationship("Consultation", back_populates="messages")


# 在类定义完成后引入 User，保持跨模块类型提示可直接解析。
from app.models.user import User  # noqa: E402,F401
