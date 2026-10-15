"""Create the first local administrator without exposing an admin registration API.

Run from backend/: .venv/bin/python scripts/create_local_admin.py --username local-admin

Only intended for a developer-controlled SQLite environment. Password is entered
interactively and never placed in CLI arguments, environment variables, or logs.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

# 直接执行 backend/scripts/*.py 时，Python 的模块搜索路径不一定包含 backend/。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select

from app.infrastructure.database.db import AsyncSessionLocal, close_db, init_db
from app.models import User, UserRole
from app.security.jwt import hash_password


async def create_first_admin(username: str, password: str) -> None:
    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            existing_admins = await db.scalar(select(func.count()).select_from(User).where(User.role == UserRole.ADMIN))
            if existing_admins:
                raise ValueError("数据库中已有管理员；此初始化命令不会新增第二个管理员")
            existing_name = await db.scalar(select(User.id).where(User.username == username))
            if existing_name:
                raise ValueError("用户名已被占用，请选择新用户名")
            db.add(User(username=username, password_hash=hash_password(password), role=UserRole.ADMIN))
            await db.commit()
    finally:
        await close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description="为本地演示数据库创建首个管理员账号")
    parser.add_argument("--username", required=True, help="3 至 50 位登录用户名")
    args = parser.parse_args()
    username = args.username.strip()
    if not 3 <= len(username) <= 50:
        parser.error("用户名必须为 3 至 50 个字符")
    if not sys.stdin.isatty():
        parser.error("必须在交互式终端输入密码，不能以参数或管道传递")
    password = getpass.getpass("管理员密码（至少 6 位）：")
    if len(password) < 6 or len(password) > 100 or len(password.encode("utf-8")) > 72:
        parser.error("密码需为 6 至 100 字符，且不超过 bcrypt 的 72 字节限制")
    confirmed = getpass.getpass("重复管理员密码：")
    if password != confirmed:
        parser.error("两次输入的密码不一致")
    try:
        asyncio.run(create_first_admin(username, password))
    except ValueError as exc:
        parser.error(str(exc))
    print("管理员已创建：", username)
    print("现在可使用前端登录，在“律师账号”页创建律师并分配咨询。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
