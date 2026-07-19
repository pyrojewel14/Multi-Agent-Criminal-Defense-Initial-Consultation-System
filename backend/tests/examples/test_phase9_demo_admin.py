import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.user import Base, User, UserRole
from app.security.jwt import verify_password
from examples.phase9_demo_admin import (
    build_parser,
    load_demo_password,
    require_local_demo_confirmation,
    upsert_demo_admin,
)


@pytest.fixture
async def demo_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


def test_requires_explicit_local_demo_confirmation():
    with pytest.raises(ValueError, match="--confirm-local-demo"):
        require_local_demo_confirmation(False)


def test_refuses_production_environment_even_with_confirmation():
    with pytest.raises(ValueError, match="production"):
        require_local_demo_confirmation(True, environment="production")


def test_reads_password_from_named_environment_variable(monkeypatch):
    monkeypatch.setenv("PHASE9_ADMIN_PASSWORD", "LocalDemo-Admin-2026")

    assert load_demo_password("PHASE9_ADMIN_PASSWORD") == "LocalDemo-Admin-2026"


def test_missing_named_password_environment_variable_fails_closed(monkeypatch):
    monkeypatch.delenv("PHASE9_MISSING_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="PHASE9_MISSING_PASSWORD"):
        load_demo_password("PHASE9_MISSING_PASSWORD")


def test_rejects_short_demo_password(monkeypatch):
    monkeypatch.setenv("PHASE9_ADMIN_PASSWORD", "short")

    with pytest.raises(ValueError, match="至少 8 位"):
        load_demo_password("PHASE9_ADMIN_PASSWORD")


def test_non_interactive_password_prompt_explains_environment_fallback(monkeypatch):
    def _raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("examples.phase9_demo_admin.getpass.getpass", _raise_eof)

    with pytest.raises(ValueError, match="--password-env"):
        load_demo_password(None)


@pytest.mark.parametrize("unsafe_option", ["--password", "--pass"])
def test_parser_rejects_password_like_abbreviations(unsafe_option):
    """未声明的密码参数缩写必须被 argparse 拒绝。"""
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--confirm-local-demo", unsafe_option, "MY_SECRET_ENV"])

    assert exc_info.value.code == 2


def test_parser_accepts_explicit_password_environment_option():
    args = build_parser().parse_args(
        ["--confirm-local-demo", "--password-env", "PHASE9_ADMIN_PASSWORD"]
    )

    assert args.password_env == "PHASE9_ADMIN_PASSWORD"


@pytest.mark.asyncio
async def test_creates_admin_with_password_hash_only(demo_db):
    user, created = await upsert_demo_admin(
        demo_db,
        username="phase9_admin",
        password="LocalDemo-Admin-2026",
        real_name="Phase 9 演示管理员",
    )

    assert created is True
    assert user.role == UserRole.ADMIN
    assert user.password_hash != "LocalDemo-Admin-2026"
    assert verify_password("LocalDemo-Admin-2026", user.password_hash)


@pytest.mark.asyncio
async def test_refuses_to_promote_existing_non_admin_user(demo_db):
    demo_db.add(
        User(
            username="existing_client",
            password_hash="irrelevant",
            role=UserRole.CLIENT,
        )
    )
    await demo_db.commit()

    with pytest.raises(ValueError, match="不是 admin"):
        await upsert_demo_admin(
            demo_db,
            username="existing_client",
            password="LocalDemo-Admin-2026",
            real_name="不应覆盖",
        )
