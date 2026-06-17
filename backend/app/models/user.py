import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, String, Table, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", String(36), ForeignKey("users.id"), primary_key=True),
    Column("role_id", String(36), ForeignKey("roles.id"), primary_key=True),
)

role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", String(36), ForeignKey("roles.id"), primary_key=True),
    Column("permission_id", String(36), ForeignKey("permissions.id"), primary_key=True),
)


class UserRole(str, PyEnum):
    ADMIN = "admin"
    LAWYER = "lawyer"
    CLIENT = "client"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, nullable=True, index=True)
    phone = Column(String(20), nullable=True)
    real_name = Column(String(50), nullable=True)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.CLIENT)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    refresh_token = Column(String(500), nullable=True)
    refresh_token_expires_at = Column(DateTime, nullable=True)

    roles = relationship("Role", secondary=user_roles, back_populates="users")

    consultations_as_client = relationship(
        "Consultation", foreign_keys="Consultation.client_id", back_populates="client"
    )
    consultations_as_lawyer = relationship(
        "Consultation",
        foreign_keys="Consultation.assigned_lawyer_id",
        back_populates="assigned_lawyer",
    )


class Role(Base):
    __tablename__ = "roles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")


class Permission(Base):
    __tablename__ = "permissions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    resource = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    description = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")


class ConsultationStatus(str, PyEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    assigned_lawyer_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)

    user_type = Column(String(20), nullable=False)
    consent_given = Column(Boolean, default=False)

    status = Column(Enum(ConsultationStatus), default=ConsultationStatus.PENDING)

    facts_raw = Column(Text, nullable=True)
    facts_structured = Column(Text, nullable=True)
    applied_laws = Column(Text, nullable=True)
    final_output = Column(Text, nullable=True)

    risk_level = Column(String(20), nullable=True, index=True)
    risk_assessment = Column(Text, nullable=True)
    alert_triggered = Column(Boolean, default=False, index=True)
    alert_read = Column(Boolean, default=False)
    lawyer_review_needed = Column(Boolean, default=False, index=True)
    report_draft = Column(Text, nullable=True)
    service_plan = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    client = relationship("User", foreign_keys=[client_id], back_populates="consultations_as_client")
    assigned_lawyer = relationship(
        "User",
        foreign_keys=[assigned_lawyer_id],
        back_populates="consultations_as_lawyer",
    )
    messages = relationship(
        "ConsultationMessage",
        back_populates="consultation",
        cascade="all, delete-orphan",
    )


class ConsultationMessage(Base):
    __tablename__ = "consultation_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    consultation_id = Column(String(36), ForeignKey("consultations.id"), nullable=False, index=True)

    sender_type = Column(String(20), nullable=False)
    sender_id = Column(String(36), nullable=True)

    content = Column(Text, nullable=False)

    agent_name = Column(String(50), nullable=True)
    message_type = Column(String(20), default="text")

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    consultation = relationship("Consultation", back_populates="messages")
