"""Typing contracts for SQLAlchemy user and consultation models."""

from typing import get_origin, get_type_hints

from sqlalchemy.orm import DeclarativeBase, Mapped

from app.models.user import (
    Base,
    Consultation,
    ConsultationMessage,
    Permission,
    Role,
    User,
)


def test_models_use_sqlalchemy_2_typed_declarative_mapping():
    """Every mapped attribute should expose a ``Mapped[...]`` annotation."""
    assert issubclass(Base, DeclarativeBase)

    expected_attributes = {
        User: {"id", "username", "password_hash", "role", "roles"},
        Role: {"id", "name", "users", "permissions"},
        Permission: {"id", "code", "roles"},
        Consultation: {"id", "client_id", "client", "messages"},
        ConsultationMessage: {"id", "consultation_id", "consultation"},
    }

    for model, attribute_names in expected_attributes.items():
        annotations = get_type_hints(model, include_extras=True)
        for attribute_name in attribute_names:
            assert get_origin(annotations[attribute_name]) is Mapped
