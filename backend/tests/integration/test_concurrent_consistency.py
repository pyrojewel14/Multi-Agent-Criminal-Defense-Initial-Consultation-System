"""
Redis-SQLite 并发数据一致性测试脚本

测试场景：
1. 多用户并发创建会话 - 验证 Redis/SQLite 双写一致性
2. 同一会话并发消息 - 验证状态更新竞态条件
3. 缓存失效与回源一致性 - 验证 Redis TTL 过期后的数据恢复
4. 限流器并发竞争 - 验证 Redis 限流原子性
5. 会话关闭与状态同步 - 验证删除/关闭操作的一致性
"""

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 使用 session 级别事件循环，避免模块级单例（Redis 连接池等）跨测试失效
pytestmark = pytest.mark.asyncio(loop_scope="session")

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.db_config import AsyncSessionLocal, async_engine, init_db  # noqa: E402
from app.db.redis_config import (  # noqa: E402
    close_redis,
    connect_redis,
    get_redis_cache_json,
    init_redis,
    set_redis_cache,
)
from app.models.user import (  # noqa: E402
    Base,
    Consultation,
    ConsultationMessage,
    ConsultationStatus,
    User,
    UserRole,
)
from app.security.jwt import hash_password


# ============================================================
# 测试结果收集器
# ============================================================
class TestReport:
    """测试报告收集器"""

    __test__ = False  # 防止 pytest 将此类误收集为测试类

    def __init__(self):
        self.results: List[dict] = []
        self.start_time = time.time()

    def add_result(self, test_name: str, passed: bool, details: str, metrics: dict = None):
        self.results.append(
            {
                "test_name": test_name,
                "passed": passed,
                "details": details,
                "metrics": metrics or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        elapsed = time.time() - self.start_time

        lines = [
            "=" * 80,
            "Redis-SQLite 并发数据一致性测试报告",
            "=" * 80,
            f"测试时间: {datetime.now(timezone.utc).isoformat()}",
            f"总耗时: {elapsed:.2f} 秒",
            f"总用例: {total}  |  通过: {passed}  |  失败: {failed}",
            "-" * 80,
        ]

        for i, r in enumerate(self.results, 1):
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"[{status}] {i}. {r['test_name']}")
            lines.append(f"       {r['details']}")
            if r["metrics"]:
                for k, v in r["metrics"].items():
                    lines.append(f"       {k}: {v}")
            lines.append("")

        # 失败用例汇总
        failed_tests = [r for r in self.results if not r["passed"]]
        if failed_tests:
            lines.append("-" * 80)
            lines.append("失败用例详情:")
            for r in failed_tests:
                lines.append(f"  * {r['test_name']}: {r['details']}")

        lines.append("=" * 80)
        return "\n".join(lines)


report = TestReport()


# ============================================================
# pytest session 级别初始化 / 清理
# ============================================================
@pytest.fixture(scope="session", autouse=True)
async def _setup_test_env():
    """session 级别初始化数据库和 Redis，所有测试共享同一事件循环。"""
    await init_db()
    try:
        await init_redis()
    except Exception:
        pass  # Redis 不可用时部分测试会跳过
    yield
    # 清理
    try:
        await close_redis()
    except Exception:
        pass
    await async_engine.dispose()


# ============================================================
# 辅助函数
# ============================================================
async def create_test_user(db: AsyncSession, username: str, role: UserRole = UserRole.CLIENT) -> User:
    """创建测试用户"""
    existing = await db.execute(select(User).where(User.username == username))
    user = existing.scalar_one_or_none()
    if user:
        return user

    user = User(
        username=username,
        password_hash=hash_password("test123"),
        email=f"{username}@test.com",
        role=role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def cleanup_test_data():
    """清理测试数据"""
    redis_client = await connect_redis()
    # 清理测试 session 缓存
    keys = await redis_client.keys("session:test_*")
    if keys:
        await redis_client.delete(*keys)
    # 清理测试限流 key
    keys = await redis_client.keys("rate_limit:test_*")
    if keys:
        await redis_client.delete(*keys)


# ============================================================
# 测试 1: 多用户并发创建会话 - Redis/SQLite 双写一致性
# ============================================================
async def test_concurrent_session_creation():
    """模拟多用户同时创建会话，验证 Redis 缓存与 SQLite 记录的一致性"""
    test_name = "多用户并发创建会话 - 双写一致性"
    num_users = 20

    async with AsyncSessionLocal() as db:
        # 创建测试用户
        users = []
        for i in range(num_users):
            user = await create_test_user(db, f"concurrent_user_{i}")
            users.append(user)

    async def create_single_session(user: User, index: int) -> dict:
        """单个用户创建会话的模拟"""
        session_id = f"test_session_{index}_{uuid.uuid4().hex[:8]}"
        consultation_id = str(uuid.uuid4())

        # 模拟 consultation.py 中 create_session 的逻辑
        # 1. 写入 SQLite
        async with AsyncSessionLocal() as db:
            consultation = Consultation(
                client_id=user.id,
                user_type="suspect",
                consent_given=False,
                status=ConsultationStatus.PENDING,
            )
            db.add(consultation)
            await db.commit()
            await db.refresh(consultation)
            db_consultation_id = consultation.id

        # 2. 写入 Redis
        state = {
            "consultation_id": db_consultation_id,
            "user_id": user.id,
            "session_id": session_id,
            "user_type": "suspect",
            "consent_given": False,
            "facts_raw": [],
            "facts_structured": {},
            "applied_laws": [],
            "current_agent": "Receptionist",
            "pending_questions": [],
            "alert_triggered": False,
            "conversation_history": [],
        }
        redis_ok = await set_redis_cache(f"session:{session_id}", state, expire=7200)

        return {
            "session_id": session_id,
            "consultation_id": db_consultation_id,
            "user_id": user.id,
            "redis_ok": redis_ok,
        }

    # 并发执行
    start = time.time()
    tasks = [create_single_session(users[i], i) for i in range(num_users)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start

    # 验证一致性
    inconsistencies = []
    successful = 0

    for r in results:
        if isinstance(r, Exception):
            inconsistencies.append(f"异常: {r}")
            continue
        successful += 1

        # 验证 Redis 中存在
        redis_state = await get_redis_cache_json(f"session:{r['session_id']}")
        if not redis_state:
            inconsistencies.append(f"session:{r['session_id']} Redis 缓存缺失")
            continue

        # 验证 Redis 与 SQLite 的 consultation_id 一致
        if redis_state.get("consultation_id") != r["consultation_id"]:
            inconsistencies.append(
                f"session:{r['session_id']} consultation_id 不一致: "
                f"Redis={redis_state.get('consultation_id')}, SQLite={r['consultation_id']}"
            )

        # 验证 SQLite 中存在
        async with AsyncSessionLocal() as db:
            db_result = await db.execute(select(Consultation).where(Consultation.id == r["consultation_id"]))
            consultation = db_result.scalar_one_or_none()
            if not consultation:
                inconsistencies.append(f"consultation_id={r['consultation_id']} SQLite 记录缺失")
            elif consultation.client_id != r["user_id"]:
                inconsistencies.append(
                    f"consultation_id={r['consultation_id']} client_id 不一致: "
                    f"SQLite={consultation.client_id}, expected={r['user_id']}"
                )

    passed = len(inconsistencies) == 0
    detail = f"并发创建 {num_users} 个会话, 成功 {successful}, 不一致 {len(inconsistencies)}"
    if inconsistencies:
        detail += f" | 问题: {inconsistencies[:3]}"

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "并发数": num_users,
            "耗时": f"{elapsed:.3f}s",
            "成功率": f"{successful}/{num_users}",
            "不一致数": len(inconsistencies),
        },
    )


# ============================================================
# 测试 2: 同一会话并发消息 - 状态更新竞态条件
# ============================================================
async def test_concurrent_message_on_same_session():
    """模拟同一会话的并发消息发送，检测竞态条件导致的状态覆盖"""
    test_name = "同一会话并发消息 - 竞态条件检测"

    # 准备：创建一个会话
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db, "race_condition_user")

    session_id = f"test_race_{uuid.uuid4().hex[:8]}"
    consultation_id = str(uuid.uuid4())

    # 创建 SQLite 记录
    async with AsyncSessionLocal() as db:
        consultation = Consultation(
            id=consultation_id,
            client_id=user.id,
            user_type="suspect",
            consent_given=True,
            status=ConsultationStatus.IN_PROGRESS,
        )
        db.add(consultation)
        await db.commit()

    # 初始化 Redis 状态
    initial_state = {
        "consultation_id": consultation_id,
        "user_id": user.id,
        "session_id": session_id,
        "consent_given": True,
        "facts_raw": [],
        "facts_structured": {},
        "current_agent": "FactDigger",
        "conversation_history": [],
    }
    await set_redis_cache(f"session:{session_id}", initial_state, expire=7200)

    num_concurrent = 10

    async def send_concurrent_message(index: int) -> dict:
        """模拟并发消息发送（不调用 LLM，直接模拟状态更新）"""
        # 1. 读取当前状态
        state = await get_redis_cache_json(f"session:{session_id}")
        if not state:
            return {"index": index, "error": "状态读取失败"}

        # 模拟微小延迟（模拟 Agent 处理时间）
        await asyncio.sleep(0.01 * index)

        # 2. 修改状态
        if "facts_raw" not in state:
            state["facts_raw"] = []
        state["facts_raw"].append(f"消息_{index}")
        state["conversation_history"].append(
            {
                "agent": "FactDigger",
                "user_message": f"消息_{index}",
                "agent_response": f"回复_{index}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # 3. 写回 Redis
        await set_redis_cache(f"session:{session_id}", state, expire=7200)

        # 4. 写入 SQLite 消息
        async with AsyncSessionLocal() as db:
            msg = ConsultationMessage(
                consultation_id=consultation_id,
                sender_type="user",
                sender_id=user.id,
                content=f"消息_{index}",
            )
            db.add(msg)
            await db.commit()

        return {"index": index, "facts_count": len(state.get("facts_raw", []))}

    # 并发执行
    start = time.time()
    tasks = [send_concurrent_message(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start

    # 验证最终状态
    final_state = await get_redis_cache_json(f"session:{session_id}")
    redis_facts_count = len(final_state.get("facts_raw", [])) if final_state else 0

    async with AsyncSessionLocal() as db:
        msg_result = await db.execute(
            select(func.count())
            .select_from(ConsultationMessage)
            .where(ConsultationMessage.consultation_id == consultation_id)
        )
        db_msg_count = msg_result.scalar_one()

    # 竞态条件检测：如果 Redis 中 facts_raw 数量 < 并发数，说明存在丢失更新
    data_loss = num_concurrent - redis_facts_count
    passed = data_loss == 0 and db_msg_count == num_concurrent

    detail = (
        f"并发 {num_concurrent} 条消息 | "
        f"Redis facts_raw: {redis_facts_count}/{num_concurrent} | "
        f"SQLite 消息数: {db_msg_count}/{num_concurrent} | "
        f"数据丢失: {data_loss} 条"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "并发数": num_concurrent,
            "耗时": f"{elapsed:.3f}s",
            "Redis事实数": redis_facts_count,
            "SQLite消息数": db_msg_count,
            "丢失更新数": data_loss,
            "存在竞态": data_loss > 0,
        },
    )


# ============================================================
# 测试 3: 缓存失效与回源一致性
# ============================================================
async def test_cache_expiry_consistency():
    """测试 Redis TTL 过期后的数据恢复一致性"""
    test_name = "缓存失效与回源一致性"

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db, "cache_expiry_user")

    session_id = f"test_expiry_{uuid.uuid4().hex[:8]}"
    consultation_id = str(uuid.uuid4())

    # 创建 SQLite 记录
    async with AsyncSessionLocal() as db:
        consultation = Consultation(
            id=consultation_id,
            client_id=user.id,
            user_type="suspect",
            consent_given=True,
            status=ConsultationStatus.IN_PROGRESS,
            facts_structured=json.dumps({"incident_time": "2024-01-01"}),
        )
        db.add(consultation)
        await db.commit()

    # 写入 Redis，设置极短 TTL
    state = {
        "consultation_id": consultation_id,
        "user_id": user.id,
        "session_id": session_id,
        "consent_given": True,
        "facts_raw": ["测试事实1"],
        "current_agent": "FactDigger",
    }
    await set_redis_cache(f"session:{session_id}", state, expire=3)  # 3 秒过期

    # 验证 Redis 有数据
    redis_before = await get_redis_cache_json(f"session:{session_id}")
    redis_before_ok = redis_before is not None

    # 等待过期
    await asyncio.sleep(4)

    # 验证 Redis 已过期
    redis_after = await get_redis_cache_json(f"session:{session_id}")
    redis_expired = redis_after is None

    # 模拟回源：从 SQLite 恢复
    async with AsyncSessionLocal() as db:
        db_result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
        consultation = db_result.scalar_one_or_none()
        sqlite_ok = consultation is not None

    # 关键检测：Redis 过期后，SQLite 数据是否完整可恢复
    # 当前代码中没有从 SQLite 恢复到 Redis 的机制
    can_recover = False
    if consultation:
        # 检查 SQLite 中是否保存了足够的状态信息来恢复
        # consultation 表只有 facts_structured, applied_laws, final_output
        # 缺少: facts_raw, conversation_history, current_agent, pending_questions 等
        has_facts = consultation.facts_structured is not None
        has_status = consultation.status is not None
        # conversation_history 不在 consultation 表中，在 consultation_messages 中
        msg_result = await db.execute(
            select(ConsultationMessage).where(ConsultationMessage.consultation_id == consultation_id)
        )
        messages = msg_result.scalars().all()
        can_recover = has_facts and has_status and len(messages) >= 0
        # 但即使能恢复消息，current_agent, pending_questions 等运行时状态无法恢复

    # 检测：conversation_history 在 SQLite 中是否可重建
    missing_fields_for_recovery = []
    if consultation:
        if not consultation.facts_structured:
            missing_fields_for_recovery.append("facts_structured")
        # current_agent, pending_questions, alert_triggered 等运行时状态
        # 在 consultation 表中没有对应字段
        missing_fields_for_recovery.extend(
            [
                "current_agent",
                "pending_questions",
                "alert_triggered",
                "facts_coverage_rate",
                "conversation_history(运行时)",
            ]
        )

    passed = redis_before_ok and redis_expired and sqlite_ok
    detail = (
        f"Redis 写入: {'OK' if redis_before_ok else 'FAIL'} | "
        f"Redis 过期: {'OK' if redis_expired else 'FAIL'} | "
        f"SQLite 持久: {'OK' if sqlite_ok else 'FAIL'} | "
        f"可恢复: {'部分' if can_recover else '否'} | "
        f"缺失字段: {missing_fields_for_recovery}"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "Redis写入": redis_before_ok,
            "Redis过期": redis_expired,
            "SQLite持久": sqlite_ok,
            "完全可恢复": False,  # 当前架构无法完全恢复
            "缺失恢复字段": len(missing_fields_for_recovery),
        },
    )


# ============================================================
# 测试 4: 限流器并发竞争 - Redis 原子性验证
# ============================================================
async def test_rate_limiter_concurrency():
    """测试 Redis 限流器在并发下的原子性"""
    test_name = "限流器并发竞争 - 原子性验证"

    client_ip = f"test_{uuid.uuid4().hex[:8]}"
    key = f"rate_limit:{client_ip}"
    limit = 10
    window = 60
    num_concurrent = 30  # 超过限制

    redis_client = await connect_redis()

    # 清理
    await redis_client.delete(key)

    async def single_request(index: int) -> dict:
        """模拟单个限流请求"""
        try:
            # 模拟 rate_limit.py 中的逻辑
            current = await redis_client.get(key)
            current = int(current) if current else 0

            if current >= limit:
                return {"index": index, "allowed": False, "current": current}

            if current == 0:
                await redis_client.setex(key, window, 1)
            else:
                await redis_client.incr(key)

            return {"index": index, "allowed": True, "current": current + 1}
        except Exception as e:
            return {"index": index, "error": str(e)}

    # 并发执行
    start = time.time()
    tasks = [single_request(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start

    # 统计
    allowed = sum(1 for r in results if isinstance(r, dict) and r.get("allowed"))
    rejected = sum(1 for r in results if isinstance(r, dict) and not r.get("allowed"))
    errors = sum(1 for r in results if isinstance(r, dict) and "error" in r)

    # 检查最终计数器值
    final_count = await redis_client.get(key)
    final_count = int(final_count) if final_count else 0

    # 原子性问题检测：
    # GET + INCR 不是原子操作，并发下可能超过 limit
    # 正确情况：allowed <= limit，final_count <= limit
    atomic_violation = allowed > limit or final_count > limit

    passed = not atomic_violation and errors == 0
    detail = (
        f"并发 {num_concurrent} 请求 (限制 {limit}) | "
        f"放行: {allowed} | 拒绝: {rejected} | 错误: {errors} | "
        f"最终计数: {final_count} | "
        f"原子性违反: {'是' if atomic_violation else '否'}"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "并发数": num_concurrent,
            "限制": limit,
            "放行数": allowed,
            "拒绝数": rejected,
            "最终计数": final_count,
            "原子性违反": atomic_violation,
            "耗时": f"{elapsed:.3f}s",
        },
    )

    # 清理
    await redis_client.delete(key)


# ============================================================
# 测试 5: 会话状态三层存储一致性 (Orchestrator内存 + Redis + SQLite)
# ============================================================
async def test_three_layer_consistency():
    """测试 Orchestrator 内存 / Redis / SQLite 三层状态一致性"""
    test_name = "三层存储一致性 (内存/Redis/SQLite)"

    from app.orchestrator.workflow import ConsultationOrchestrator

    orchestrator = ConsultationOrchestrator()

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db, "three_layer_user")

    session_id = f"test_3layer_{uuid.uuid4().hex[:8]}"
    consultation_id = str(uuid.uuid4())

    # 创建 SQLite 记录
    async with AsyncSessionLocal() as db:
        consultation = Consultation(
            id=consultation_id,
            client_id=user.id,
            user_type="suspect",
            consent_given=True,
            status=ConsultationStatus.IN_PROGRESS,
        )
        db.add(consultation)
        await db.commit()

    # 初始化三层状态
    state = {
        "consultation_id": consultation_id,
        "user_id": user.id,
        "session_id": session_id,
        "consent_given": True,
        "facts_raw": ["初始事实"],
        "facts_structured": {},
        "current_agent": "FactDigger",
        "conversation_history": [],
    }

    # 写入 Orchestrator 内存
    orchestrator._active_sessions[session_id] = state.copy()
    # 写入 Redis
    await set_redis_cache(f"session:{session_id}", state, expire=7200)

    # 模拟多次状态更新
    num_updates = 5
    for i in range(num_updates):
        # 更新 Orchestrator 内存
        orchestrator._active_sessions[session_id]["facts_raw"].append(f"更新_{i}")
        orchestrator.update_session_context(session_id, orchestrator._active_sessions[session_id])

        # 更新 Redis
        await set_redis_cache(
            f"session:{session_id}",
            dict(orchestrator._active_sessions[session_id]),
            expire=7200,
        )

        # 更新 SQLite（模拟消息保存）
        async with AsyncSessionLocal() as db:
            msg = ConsultationMessage(
                consultation_id=consultation_id,
                sender_type="user",
                content=f"更新_{i}",
            )
            db.add(msg)
            await db.commit()

    # 验证三层一致性
    memory_state = orchestrator.get_session_context(session_id)
    redis_state = await get_redis_cache_json(f"session:{session_id}")

    async with AsyncSessionLocal() as db:
        msg_result = await db.execute(
            select(func.count())
            .select_from(ConsultationMessage)
            .where(ConsultationMessage.consultation_id == consultation_id)
        )
        db_msg_count = msg_result.scalar_one()

    memory_facts = len(memory_state.get("facts_raw", [])) if memory_state else 0
    redis_facts = len(redis_state.get("facts_raw", [])) if redis_state else 0

    memory_redis_match = memory_facts == redis_facts
    redis_sqlite_consistent = redis_facts >= db_msg_count  # Redis 可能包含初始事实

    # 检测：Orchestrator 内存和 Redis 是否同步
    # 注意：当前代码中 update_session_context 使用 dict.update()，
    # 如果并发调用可能导致部分更新
    inconsistencies = []
    if not memory_redis_match:
        inconsistencies.append(f"内存/Redis facts_raw 不一致: 内存={memory_facts}, Redis={redis_facts}")

    # 检测：Redis 中 conversation_history 与 SQLite messages 是否一致
    redis_history = redis_state.get("conversation_history", []) if redis_state else []
    if len(redis_history) != db_msg_count:
        inconsistencies.append(
            f"conversation_history/消息数不一致: Redis历史={len(redis_history)}, SQLite消息={db_msg_count}"
        )

    passed = len(inconsistencies) == 0
    detail = (
        f"更新 {num_updates} 次 | "
        f"内存facts: {memory_facts} | Redis facts: {redis_facts} | "
        f"SQLite消息: {db_msg_count} | "
        f"不一致: {inconsistencies[:2] if inconsistencies else '无'}"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "内存facts数": memory_facts,
            "Redis facts数": redis_facts,
            "SQLite消息数": db_msg_count,
            "内存Redis一致": memory_redis_match,
            "不一致数": len(inconsistencies),
        },
    )


# ============================================================
# 测试 6: 缓存穿透测试 - 查询不存在的 session
# ============================================================
async def test_cache_penetration():
    """测试查询不存在的 session 时的缓存穿透问题"""
    test_name = "缓存穿透检测"

    num_queries = 50
    non_existent_ids = [f"session:nonexist_{uuid.uuid4().hex[:8]}" for _ in range(num_queries)]

    async def query_nonexistent(key: str) -> dict:
        """查询不存在的 key"""
        start = time.time()
        redis_result = await get_redis_cache_json(key)
        elapsed = time.time() - start

        # 如果 Redis 没有，当前代码会尝试从 SQLite 查询
        # 但 consultation.py 中没有从 SQLite 回填 Redis 的逻辑
        return {"key": key, "redis_hit": redis_result is not None, "elapsed": elapsed}

    start = time.time()
    tasks = [query_nonexistent(k) for k in non_existent_ids]
    results = await asyncio.gather(*tasks)
    total_elapsed = time.time() - start

    # 所有查询都应该 miss
    all_miss = all(not r["redis_hit"] for r in results)
    avg_latency = sum(r["elapsed"] for r in results) / len(results)

    # 当前架构风险：没有空值缓存，每次都会穿透到 Redis
    # 如果后续添加了 SQLite 回源逻辑，会穿透到 SQLite
    passed = all_miss
    detail = (
        f"查询 {num_queries} 个不存在的 key | "
        f"全部 miss: {'是' if all_miss else '否'} | "
        f"平均延迟: {avg_latency * 1000:.2f}ms | "
        f"空值缓存: 无 (存在穿透风险)"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "查询数": num_queries,
            "全部Miss": all_miss,
            "平均延迟ms": f"{avg_latency * 1000:.2f}",
            "空值缓存": "无",
            "穿透风险": True,
        },
    )


# ============================================================
# 测试 7: SQLite 写入并发 - 锁竞争检测
# ============================================================
async def test_sqlite_write_concurrency():
    """测试 SQLite 在高并发写入下的锁竞争"""
    test_name = "SQLite 写入并发 - 锁竞争检测"

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db, "sqlite_concurrent_user")

    consultation_id = str(uuid.uuid4())
    num_writes = 30

    # 创建 consultation
    async with AsyncSessionLocal() as db:
        consultation = Consultation(
            id=consultation_id,
            client_id=user.id,
            user_type="suspect",
            consent_given=True,
            status=ConsultationStatus.IN_PROGRESS,
        )
        db.add(consultation)
        await db.commit()

    async def write_message(index: int) -> dict:
        """并发写入消息"""
        start = time.time()
        try:
            async with AsyncSessionLocal() as db:
                msg = ConsultationMessage(
                    consultation_id=consultation_id,
                    sender_type="user",
                    sender_id=user.id,
                    content=f"并发消息_{index}",
                )
                db.add(msg)
                await db.commit()
                elapsed = time.time() - start
                return {"index": index, "success": True, "elapsed": elapsed}
        except Exception as e:
            elapsed = time.time() - start
            return {"index": index, "success": False, "error": str(e), "elapsed": elapsed}

    # 并发执行
    start = time.time()
    tasks = [write_message(i) for i in range(num_writes)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = time.time() - start

    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failures = sum(1 for r in results if isinstance(r, dict) and not r.get("success"))
    exceptions = sum(1 for r in results if isinstance(r, Exception))

    # 检查是否有锁竞争导致的失败
    lock_errors = [
        r for r in results if isinstance(r, dict) and not r.get("success") and "lock" in str(r.get("error", "")).lower()
    ]

    # 验证最终消息数
    async with AsyncSessionLocal() as db:
        count_result = await db.execute(
            select(func.count())
            .select_from(ConsultationMessage)
            .where(ConsultationMessage.consultation_id == consultation_id)
        )
        final_count = count_result.scalar_one()

    data_loss = num_writes - final_count
    passed = data_loss == 0 and failures == 0 and exceptions == 0

    detail = (
        f"并发写入 {num_writes} 条 | "
        f"成功: {successes} | 失败: {failures} | 异常: {exceptions} | "
        f"锁错误: {len(lock_errors)} | "
        f"最终记录: {final_count}/{num_writes} | "
        f"数据丢失: {data_loss}"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "并发数": num_writes,
            "成功数": successes,
            "失败数": failures,
            "锁错误": len(lock_errors),
            "最终记录数": final_count,
            "数据丢失": data_loss,
            "总耗时": f"{total_elapsed:.3f}s",
        },
    )


# ============================================================
# 测试 8: Redis 写入失败时的降级一致性
# ============================================================
async def test_redis_failure_degradation():
    """测试 Redis 不可用时，系统是否能正确降级"""
    test_name = "Redis 故障降级一致性"

    # 模拟 Redis 不可用（通过使用无效连接）
    # 这里我们测试 set_redis_cache 返回 False 的场景
    session_id = f"test_degrade_{uuid.uuid4().hex[:8]}"

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db, "degrade_user")

    consultation_id = str(uuid.uuid4())

    # 1. SQLite 写入成功
    async with AsyncSessionLocal() as db:
        consultation = Consultation(
            id=consultation_id,
            client_id=user.id,
            user_type="suspect",
            consent_given=True,
            status=ConsultationStatus.IN_PROGRESS,
        )
        db.add(consultation)
        await db.commit()

    # 2. 模拟 Redis 写入失败
    # 当前代码中 set_redis_cache 失败返回 False，但不影响主流程
    # consultation.py 中没有检查 set_redis_cache 的返回值
    state = {
        "consultation_id": consultation_id,
        "user_id": user.id,
        "session_id": session_id,
    }

    # 正常写入 Redis
    redis_ok = await set_redis_cache(f"session:{session_id}", state, expire=7200)

    # 验证：如果 Redis 写入失败，SQLite 数据仍然存在
    async with AsyncSessionLocal() as db:
        db_result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
        sqlite_exists = db_result.scalar_one_or_none() is not None

    # 关键风险：当前代码不检查 Redis 写入结果
    # 如果 Redis 写入失败但 SQLite 成功，下次读取时：
    # - Orchestrator 内存可能没有（进程重启后丢失）
    # - Redis 缓存没有
    # - 只有 SQLite 有部分数据
    # 结果：会话"丢失"，用户看到 404

    passed = redis_ok and sqlite_exists
    detail = (
        f"Redis 写入: {'OK' if redis_ok else 'FAIL'} | "
        f"SQLite 写入: {'OK' if sqlite_exists else 'FAIL'} | "
        f"风险: Redis 写入失败不检查返回值，可能导致会话丢失"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "Redis写入": redis_ok,
            "SQLite写入": sqlite_exists,
            "写入结果检查": "无",
            "降级策略": "无",
        },
    )


# ============================================================
# 测试 9: 会话关闭操作的一致性
# ============================================================
async def test_session_close_consistency():
    """测试会话关闭时 Redis/SQLite 状态同步"""
    test_name = "会话关闭一致性"

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db, "close_user")

    session_id = f"test_close_{uuid.uuid4().hex[:8]}"
    consultation_id = str(uuid.uuid4())

    # 创建完整会话
    async with AsyncSessionLocal() as db:
        consultation = Consultation(
            id=consultation_id,
            client_id=user.id,
            user_type="suspect",
            consent_given=True,
            status=ConsultationStatus.IN_PROGRESS,
        )
        db.add(consultation)
        await db.commit()

    state = {
        "consultation_id": consultation_id,
        "user_id": user.id,
        "session_id": session_id,
        "consent_given": True,
        "current_agent": "FactDigger",
        "conversation_history": [{"agent": "test", "content": "test"}],
    }
    await set_redis_cache(f"session:{session_id}", state, expire=7200)

    # 模拟关闭会话（参考 consultation.py close_session 逻辑）
    state["current_agent"] = "END"
    state["conversation_history"].append(
        {
            "agent": "system",
            "action": "session_closed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    await set_redis_cache(f"session:{session_id}", state, expire=7200)

    # 注意：当前代码关闭会话时没有更新 SQLite 的 consultation status
    async with AsyncSessionLocal() as db:
        db_result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
        consultation = db_result.scalar_one_or_none()
        db_status = consultation.status.value if consultation else None

    # Redis 中 current_agent = "END"，但 SQLite status 仍为 "in_progress"
    status_inconsistent = db_status == "in_progress"  # 应该是 completed/cancelled

    passed = not status_inconsistent
    detail = (
        f"Redis current_agent: END | "
        f"SQLite status: {db_status} | "
        f"状态不一致: {'是' if status_inconsistent else '否'} | "
        f"风险: 关闭会话未同步更新 SQLite status"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "Redis状态": "END",
            "SQLite状态": db_status,
            "状态一致": not status_inconsistent,
        },
    )


# ============================================================
# 测试 10: 高并发下 Redis 连接池压力测试
# ============================================================
async def test_redis_connection_pool_stress():
    """测试 Redis 连接池在高并发下的表现"""
    test_name = "Redis 连接池压力测试"

    num_concurrent = 50

    async def redis_operation(index: int) -> dict:
        """执行 Redis 读写操作"""
        key = f"test_stress_{index}"
        start = time.time()
        try:
            await set_redis_cache(key, {"index": index}, expire=60)
            result = await get_redis_cache_json(key)
            elapsed = time.time() - start
            return {
                "index": index,
                "success": result is not None and result.get("index") == index,
                "elapsed": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - start
            return {"index": index, "success": False, "error": str(e), "elapsed": elapsed}

    start = time.time()
    tasks = [redis_operation(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = time.time() - start

    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failures = sum(1 for r in results if isinstance(r, dict) and not r.get("success"))
    exceptions = sum(1 for r in results if isinstance(r, Exception))
    avg_latency = sum(r.get("elapsed", 0) for r in results if isinstance(r, dict)) / max(len(results), 1)

    # 清理
    redis_client = await connect_redis()
    keys = await redis_client.keys("test_stress_*")
    if keys:
        await redis_client.delete(*keys)

    passed = failures == 0 and exceptions == 0
    detail = (
        f"并发 {num_concurrent} 操作 | "
        f"成功: {successes} | 失败: {failures} | 异常: {exceptions} | "
        f"平均延迟: {avg_latency * 1000:.2f}ms | "
        f"总耗时: {total_elapsed:.3f}s"
    )

    report.add_result(
        test_name,
        passed,
        detail,
        {
            "并发数": num_concurrent,
            "成功数": successes,
            "失败数": failures,
            "平均延迟ms": f"{avg_latency * 1000:.2f}",
            "总耗时": f"{total_elapsed:.3f}s",
        },
    )


# ============================================================
# 主函数
# ============================================================
async def main():
    print("=" * 80)
    print("Redis-SQLite 并发数据一致性测试")
    print(f"开始时间: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 初始化
    print("\n[初始化] 正在初始化数据库和 Redis...")
    await init_db()
    try:
        await init_redis()
        redis_available = True
        print("[初始化] Redis 连接成功")
    except Exception as e:
        redis_available = False
        print(f"[初始化] Redis 连接失败: {e}")
        print("[警告] 部分 Redis 测试将跳过")

    # 清理旧数据
    if redis_available:
        await cleanup_test_data()

    # 运行测试
    tests = [
        ("测试1: 多用户并发创建会话", test_concurrent_session_creation),
        ("测试2: 同一会话并发消息竞态", test_concurrent_message_on_same_session),
        ("测试3: 缓存失效与回源一致性", test_cache_expiry_consistency),
        ("测试4: 限流器并发竞争", test_rate_limiter_concurrency),
        ("测试5: 三层存储一致性", test_three_layer_consistency),
        ("测试6: 缓存穿透检测", test_cache_penetration),
        ("测试7: SQLite写入并发锁竞争", test_sqlite_write_concurrency),
        ("测试8: Redis故障降级一致性", test_redis_failure_degradation),
        ("测试9: 会话关闭一致性", test_session_close_consistency),
        ("测试10: Redis连接池压力测试", test_redis_connection_pool_stress),
    ]

    for name, test_func in tests:
        print(f"\n[运行] {name}...")
        try:
            if not redis_available and "Redis" in name:
                print(f"  [跳过] Redis 不可用")
                report.add_result(name, False, "Redis 不可用，测试跳过")
                continue
            await test_func()
            last_result = report.results[-1]
            status = "PASS" if last_result["passed"] else "FAIL"
            print(f"  [{status}] {last_result['details']}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            report.add_result(name, False, f"测试执行异常: {e}")

    # 清理
    if redis_available:
        await cleanup_test_data()
        await close_redis()
    await async_engine.dispose()

    # 输出报告
    print("\n" + report.summary())


if __name__ == "__main__":
    asyncio.run(main())
