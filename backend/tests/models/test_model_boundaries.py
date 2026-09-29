"""验证模型拆分后仍共享原有 SQLAlchemy metadata 和关系。"""

from importlib import import_module

from sqlalchemy.orm import configure_mappers


def test_model_groups_share_one_metadata() -> None:
    base = import_module("app.models.base").Base
    user = import_module("app.models.user").User
    role = import_module("app.models.auth").Role
    consultation = import_module("app.models.consultation").Consultation

    assert user.metadata is base.metadata
    assert role.metadata is base.metadata
    assert consultation.metadata is base.metadata
    assert set(base.metadata.tables) == {
        "users",
        "roles",
        "permissions",
        "user_roles",
        "role_permissions",
        "consultations",
        "consultation_messages",
    }
    configure_mappers()
