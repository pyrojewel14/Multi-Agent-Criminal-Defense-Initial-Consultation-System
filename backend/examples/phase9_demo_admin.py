"""为本地 Phase 9 演示初始化受控的管理员账号。"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.db_config import AsyncSessionLocal, close_db, init_db
from app.models.user import User, UserRole
from app.security.jwt import hash_password


def require_local_demo_confirmation(
    confirmed: bool,
    *,
    environment: str | None = None,
) -> None:
    """确认命令只在显式授权的非生产本地演示中运行。"""
    if not confirmed:
        raise ValueError("必须提供 --confirm-local-demo 才能修改本地演示账号")
    current_environment = (environment or os.getenv("APP_ENV", "development")).strip().lower()
    if current_environment in {"prod", "production"}:
        raise ValueError("production 环境禁止运行本地演示账号初始化")


def load_demo_password(
    password_env: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """从指定环境变量或无回显交互输入读取演示密码。"""
    values = environ if environ is not None else os.environ
    if password_env:
        password = values.get(password_env, "")
        if not password:
            raise ValueError(f"环境变量 {password_env} 未设置或为空")
    else:
        try:
            password = getpass.getpass("Phase 9 本地管理员密码: ")
        except EOFError as exc:
            raise ValueError("当前终端不能安全读取密码，请使用 --password-env 指定临时环境变量") from exc

    if len(password) < 8:
        raise ValueError("本地演示管理员密码至少 8 位")
    return password


async def upsert_demo_admin(
    db: AsyncSession,
    *,
    username: str,
    password: str,
    real_name: str,
) -> tuple[User, bool]:
    """创建本地 admin；同名 admin 存在时仅显式重置其密码。"""
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    created = user is None

    if user is not None and user.role != UserRole.ADMIN:
        raise ValueError(f"用户 {username} 已存在且不是 admin，拒绝提升角色")

    if user is None:
        user = User(
            username=username,
            password_hash=hash_password(password),
            real_name=real_name,
            role=UserRole.ADMIN,
            is_active=True,
        )
        db.add(user)
    else:
        user.password_hash = hash_password(password)
        user.real_name = real_name
        user.is_active = True

    await db.commit()
    await db.refresh(user)
    return user, created


def build_parser() -> argparse.ArgumentParser:
    """构建不接收命令行明文密码的参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--confirm-local-demo", action="store_true", help="确认只修改当前本地演示数据库")
    parser.add_argument("--username", default="phase9_admin", help="管理员用户名")
    parser.add_argument("--real-name", default="Phase 9 演示管理员", help="管理员显示名称")
    parser.add_argument(
        "--password-env",
        metavar="ENV_NAME",
        help="从指定环境变量读取密码；省略时使用无回显交互输入",
    )
    return parser


async def run(args: argparse.Namespace) -> User:
    """执行数据库初始化并返回创建或更新后的管理员。"""
    require_local_demo_confirmation(args.confirm_local_demo)
    password = load_demo_password(args.password_env)
    await init_db()
    async with AsyncSessionLocal() as db:
        user, created = await upsert_demo_admin(
            db,
            username=args.username,
            password=password,
            real_name=args.real_name,
        )
    action = "created" if created else "updated"
    print(f"phase9_demo_admin {action}: username={user.username} role={user.role.value} user_id={user.id}")
    return user


def main(argv: list[str] | None = None) -> int:
    """运行 CLI，并把安全校验错误转换为非零退出码。"""
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run(args))
    except ValueError as exc:
        print(f"phase9_demo_admin refused: {exc}", file=sys.stderr)
        return 2
    finally:
        asyncio.run(close_db())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
