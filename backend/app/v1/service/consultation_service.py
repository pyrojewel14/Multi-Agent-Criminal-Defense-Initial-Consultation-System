import asyncio
import hashlib
import json
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.receptionist import get_welcome_message
from app.errors.exceptions import AppException
from app.models.user import Consultation, ConsultationMessage, ConsultationStatus
from app.observability.tracing import trace_span, trace_store
from app.orchestrator.workflow import orchestrator
from app.security.disclaimer import disclaimer
from app.state.consultation_state import ConsultationState, validate_consultation_state
from app.utils.logger import get_logger
from app.v1.router.consultation.constants import HIGH_RISK_ALERT_MESSAGE

_logger = get_logger("Service.Consultation")
LIFECYCLE_REPAIR_DETAIL = "生命周期命令未完成，工作流已停止；请使用相同操作重试修复"
LIFECYCLE_REVIEW_CONFLICT_DETAIL = "当前工作流不在律师审核断点，不能执行批准或退回"


@dataclass
class ProcessMessageResult:
    """消息处理结果。

    ``agent_name`` 是 resume 前 checkpoint 中执行本轮命令的 Agent；
    ``next_agent`` 是 resume 后的新工作流位置，二者不得混用。
    """
    response_content: str
    next_agent: str
    alert_triggered: bool
    result_state: Optional[ConsultationState]
    is_workflow_finished: bool = False
    error: Optional[str] = None
    message_id: Optional[str] = None
    created_at: Optional[datetime] = None
    command_applied: bool = False
    agent_name: Optional[str] = None
    request_id: Optional[str] = None


class IdempotencyConflictError(RuntimeError):
    """同一幂等键被用于不同命令载荷。"""


@dataclass
class _SessionLockEntry:
    """记录单个会话锁及其持有者和等待者数量。"""

    lock: asyncio.Lock
    references: int = 0


@dataclass
class _IdempotencyRecord:
    """保存进程内已完成或已推进命令的重放结果。"""

    fingerprint: str
    result: Any
    expires_at: float


class _SessionCommandCoordinator:
    """在单进程内按会话串行命令，并有界保存幂等结果。"""

    def __init__(self, *, max_results: int = 2048, result_ttl_seconds: int = 3600):
        self._max_results = max_results
        self._result_ttl_seconds = result_ttl_seconds
        self._registry_guard = asyncio.Lock()
        self._locks: dict[str, _SessionLockEntry] = {}
        self._results: OrderedDict[tuple[str, str, str], _IdempotencyRecord] = OrderedDict()

    @asynccontextmanager
    async def serialize(self, session_id: str):
        """仅串行同一 session，并在无持有者或等待者时回收锁。"""
        async with self._registry_guard:
            entry = self._locks.get(session_id)
            if entry is None:
                entry = _SessionLockEntry(lock=asyncio.Lock())
                self._locks[session_id] = entry
            entry.references += 1

        try:
            async with entry.lock:
                yield
        finally:
            async with self._registry_guard:
                entry.references -= 1
                if entry.references == 0 and self._locks.get(session_id) is entry:
                    del self._locks[session_id]

    def replay(
        self,
        session_id: str,
        command_type: str,
        idempotency_key: Optional[str],
        fingerprint: str,
    ) -> Any:
        """返回已完成结果；同 key 不同载荷立即拒绝。"""
        if not idempotency_key:
            return None
        self._discard_expired()
        scope = (session_id, command_type, idempotency_key)
        record = self._results.get(scope)
        if record is None:
            return None
        if record.fingerprint != fingerprint:
            raise IdempotencyConflictError("幂等键已用于不同请求")
        self._results.move_to_end(scope)
        return record.result

    def remember(
        self,
        session_id: str,
        command_type: str,
        idempotency_key: Optional[str],
        fingerprint: str,
        result: Any,
    ) -> None:
        """缓存可安全重放的结果，并按 TTL 与总容量淘汰旧记录。"""
        if not idempotency_key:
            return
        self._discard_expired()
        scope = (session_id, command_type, idempotency_key)
        self._results[scope] = _IdempotencyRecord(
            fingerprint=fingerprint,
            result=result,
            expires_at=monotonic() + self._result_ttl_seconds,
        )
        self._results.move_to_end(scope)
        while len(self._results) > self._max_results:
            self._results.popitem(last=False)

    def metrics(self) -> dict[str, int]:
        """返回不含命令内容的进程内协调器规模。"""
        self._discard_expired()
        return {
            "active_lock_entries": len(self._locks),
            "cached_results": len(self._results),
        }

    def _discard_expired(self) -> None:
        now = monotonic()
        expired = [scope for scope, record in self._results.items() if record.expires_at <= now]
        for scope in expired:
            self._results.pop(scope, None)


_session_commands = _SessionCommandCoordinator()


def _command_fingerprint(payload: dict[str, Any]) -> str:
    """只保存规范化载荷摘要，避免把敏感命令内容放入 registry。"""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_session_command_metrics() -> dict[str, int]:
    """返回单进程锁和幂等结果数量，供可观测性与泄漏回归使用。"""
    return _session_commands.metrics()


def get_command_processed_at(result: object) -> datetime:
    """读取稳定命令时间；兼容未携带元数据的旧调用方。"""
    if isinstance(result, dict):
        processed_at = result.get("command_processed_at")
        if isinstance(processed_at, datetime):
            return processed_at
    return datetime.now(timezone.utc)


class LifecycleConsistencyError(RuntimeError):
    """生命周期 checkpoint 与 SQLite 审计未能完成同一命令。"""

    def __init__(self, *, action: str, failed_stage: str, repair_required: bool = True):
        self.action = action
        self.failed_stage = failed_stage
        self.repair_required = repair_required
        super().__init__(LIFECYCLE_REPAIR_DETAIL)


class LifecycleCommandConflictError(RuntimeError):
    """生命周期操作与当前 LangGraph 执行位置冲突。"""

    def __init__(self, *, action: str):
        self.action = action
        super().__init__(LIFECYCLE_REVIEW_CONFLICT_DETAIL)


async def _get_consultation_for_command(db: AsyncSession, consultation_id: str):
    from sqlalchemy import select

    result = await db.execute(select(Consultation).where(Consultation.id == consultation_id))
    return result.scalar_one_or_none()


async def get_session_state(session_id: str) -> Optional[ConsultationState]:
    """仅从 LangGraph checkpoint 执行权威读取状态。

    Args:
        session_id: 会话ID

    Returns:
        会话状态，如果不存在返回 None
    """
    snapshot = await orchestrator.get_snapshot(session_id)
    if snapshot and snapshot.values:
        return validate_consultation_state(snapshot.values)
    # Redis 和进程缓存只是投影，无法重建待执行节点，因此不得作为回退来源。
    return None


async def persist_state(
    session_id: str,
    state: Optional[ConsultationState] = None,
    *,
    projection_updates: Optional[Dict[str, Any]] = None,
) -> None:
    """同步兼容投影；仅显式字段更新才写入 checkpoint。

    Args:
        session_id: 会话ID
        state: 已由 LangGraph 写入的完整状态，仅用于刷新兼容缓存
        projection_updates: 图执行完成后新增的最小投影字段
    """
    if projection_updates:
        projected = await orchestrator.update_workflow_state(session_id, dict(projection_updates))
        if projected is None:
            raise ValueError(f"会话不存在: {session_id}")
        state = projected
    elif state is None:
        snapshot = await orchestrator.get_snapshot(session_id)
        if snapshot is None:
            raise ValueError(f"会话不存在: {session_id}")
        state = validate_consultation_state(snapshot.values)
    # 仅保留列表和可观测性兼容；执行状态读取不会回退到该缓存。
    orchestrator.update_session_context(session_id, state)


async def execute_lifecycle_command(
    session_id: str,
    action: str,
    *,
    db: AsyncSession,
    actor_id: str,
    final_output: Optional[str] = None,
    feedback: Optional[str] = None,
    target_node: Optional[str] = None,
    reason: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> ConsultationState:
    """按 session 串行执行唯一生命周期命令，并重放成功结果。"""
    fingerprint = _command_fingerprint({
        "action": action,
        "actor_id": actor_id,
        "final_output": final_output,
        "feedback": feedback,
        "target_node": target_node,
        "reason": reason,
    })
    command_type = f"lifecycle:{action}"
    async with _session_commands.serialize(session_id):
        replay = _session_commands.replay(
            session_id,
            command_type,
            idempotency_key,
            fingerprint,
        )
        if replay is not None:
            return replay

        result = await _execute_lifecycle_command_unlocked(
            session_id,
            action,
            db=db,
            actor_id=actor_id,
            final_output=final_output,
            feedback=feedback,
            target_node=target_node,
            reason=reason,
        )
        result["command_processed_at"] = datetime.now(timezone.utc)
        _session_commands.remember(
            session_id,
            command_type,
            idempotency_key,
            fingerprint,
            result,
        )
        return result


async def _execute_lifecycle_command_unlocked(
    session_id: str,
    action: str,
    *,
    db: AsyncSession,
    actor_id: str,
    final_output: Optional[str] = None,
    feedback: Optional[str] = None,
    target_node: Optional[str] = None,
    reason: Optional[str] = None,
) -> ConsultationState:
    """通过唯一应用命令入口执行 approve、reject 或 close。

    checkpoint 先推进，因为它是执行权威；SQLite 随后记录业务审计。若 SQLite
    写入失败，不伪造回滚，而是在当前执行位置写入 repair_required。使用相同
    action 重试时只修复审计投影，不会再次推进工作流。
    """
    if action not in {"approve", "reject", "close"}:
        raise ValueError(f"不支持的生命周期操作: {action}")

    snapshot = await orchestrator.get_snapshot(session_id)
    if not snapshot or not snapshot.values:
        raise ValueError(f"会话不存在: {session_id}")
    before = dict(snapshot.values)
    consultation_id = before.get("consultation_id")
    if not consultation_id:
        raise ValueError(f"会话缺少 consultation_id: {session_id}")

    repair_marker = before.get("consistency_error") or {}
    is_repair = before.get("repair_required") or before.get("workflow_status") == "repair_required"
    if is_repair:
        if repair_marker.get("action") != action:
            raise LifecycleConsistencyError(
                action=action,
                failed_stage="repair_action_mismatch",
            )
        after = before
    elif action in {"approve", "reject"} and tuple(snapshot.next) != ("human_review",):
        raise LifecycleCommandConflictError(action=action)
    elif action == "approve":
        after = await process_lawyer_review(
            session_id, "approved", feedback=feedback, final_output=final_output
        )
        after = await orchestrator.update_workflow_state(
            session_id, {"workflow_status": "completed"}
        ) or after
    elif action == "reject":
        if target_node not in {"fact_digger", "risk_assessor"}:
            raise ValueError("无效的退回目标节点")
        decision = "revise_facts" if target_node == "fact_digger" else "revise_risk"
        after = await process_lawyer_review(session_id, decision, feedback=feedback)
        after = await orchestrator.update_workflow_state(
            session_id, {"workflow_status": "in_progress"}
        ) or after
    else:
        after = await orchestrator.close_workflow(session_id, reason=reason, actor_id=actor_id)

    failed_stage = "sqlite_audit_lookup"
    try:
        consultation = await _get_consultation_for_command(db, consultation_id)
        if consultation is None:
            raise ValueError(f"咨询记录不存在: {consultation_id}")
        if action == "approve":
            consultation.status = ConsultationStatus.COMPLETED
            consultation.final_output = final_output or after.get("final_output")
            consultation.completed_at = datetime.now(timezone.utc)
            consultation.lawyer_review_needed = False
        elif action == "reject":
            consultation.status = ConsultationStatus.IN_PROGRESS
            consultation.lawyer_review_needed = False
        else:
            consultation.status = ConsultationStatus.CANCELLED
            consultation.lawyer_review_needed = False
        failed_stage = "sqlite_audit_commit"
        await db.commit()
    except Exception as exc:
        await db.rollback()
        try:
            await orchestrator.mark_repair_required(
                session_id,
                action=action,
                failed_stage=failed_stage,
            )
        except Exception:
            raise LifecycleConsistencyError(
                action=action,
                failed_stage="checkpoint_repair_marker",
            ) from exc
        raise LifecycleConsistencyError(
            action=action,
            failed_stage=failed_stage,
        ) from exc

    if is_repair:
        final_status = {
            "approve": "completed",
            "reject": "in_progress",
            "close": "closed",
        }[action]
        finalized = await orchestrator.update_workflow_state(
            session_id,
            {
                "workflow_status": final_status,
                "repair_required": False,
                "consistency_error": None,
            },
        )
        if finalized is None:
            raise LifecycleConsistencyError(
                action=action,
                failed_stage="checkpoint_finalize",
            )
        after = finalized
    return after


async def assign_lawyer_to_active_session(
    consultation_id: str, lawyer_id: str, workflow_session_id: Optional[str] = None
) -> Optional[str]:
    """把数据库律师分配同步到对应的活跃工作流状态。"""
    if workflow_session_id:
        snapshot = await orchestrator.get_snapshot(workflow_session_id)
        if snapshot and snapshot.next and snapshot.values.get("consultation_id") == consultation_id:
            updated = await orchestrator.update_workflow_state(
                workflow_session_id, {"lawyer_id": lawyer_id}
            )
            if updated is not None:
                await persist_state(workflow_session_id, updated)
                return workflow_session_id
        return None
    for session_id, state in orchestrator.get_active_sessions().items():
        if state.get("consultation_id") != consultation_id:
            continue
        updated = await orchestrator.update_workflow_state(
            session_id, {"lawyer_id": lawyer_id}
        )
        if updated is not None:
            await persist_state(session_id, updated)
            return session_id
    return None


async def handle_high_risk_alert(
    session_id: str,
    result: ConsultationState,
    current_agent: str,
) -> None:
    """处理高风险警报，持久化状态。

    Args:
        session_id: 会话ID
        result: 更新后的状态（alert_triggered=True）
        current_agent: 检测到风险的 Agent 名称
    """
    _logger.warning(
        "【_handle_high_risk_alert】高风险警报: session_id=%s, agent=%s",
        session_id,
        current_agent,
    )

    if "conversation_history" not in result:
        result["conversation_history"] = []
    result["conversation_history"].append({
        "agent": current_agent,
        "action": "high_risk_alert",
        "content": HIGH_RISK_ALERT_MESSAGE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    await persist_state(
        session_id,
        result,
        projection_updates={"conversation_history": result["conversation_history"]},
    )


async def generate_welcome_message(_user_type: str = "suspect") -> str:
    """返回接待 Agent 使用的确定性欢迎语和隐私条款告知。

    Args:
        user_type: 用户类型

    Returns:
        欢迎语文本
    """
    return get_welcome_message()


async def create_consultation_record(
    _session_id: str, user_id: str, user_type: str, db: AsyncSession
) -> str:
    """创建咨询数据库记录

    Args:
        session_id: 会话ID
        user_id: 用户ID
        user_type: 用户类型
        db: 数据库会话

    Returns:
        咨询记录ID
    """
    consultation = Consultation(
        workflow_session_id=_session_id,
        client_id=user_id,
        user_type=user_type,
        consent_given=False,
        status=ConsultationStatus.PENDING,
    )
    db.add(consultation)
    await db.commit()
    await db.refresh(consultation)
    _logger.info("【_create_consultation_record】咨询记录创建: consultation_id=%s", consultation.id)
    return consultation.id


async def save_message_to_db(
    consultation_id: str,
    session_id: str,
    content: str,
    sender_type: str,
    sender_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    db: Optional[AsyncSession] = None,
) -> str:
    """保存消息到数据库

    Args:
        consultation_id: 咨询记录ID
        session_id: 会话ID
        content: 消息内容
        sender_type: 发送者类型
        sender_id: 发送者ID
        agent_name: Agent名称
        db: 数据库会话

    Returns:
        消息ID
    """
    if not db:
        return str(uuid.uuid4())

    message = ConsultationMessage(
        consultation_id=consultation_id,
        sender_type=sender_type,
        sender_id=sender_id,
        content=content,
        agent_name=agent_name,
        message_type="text",
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message.id


async def _save_message_command_rows(
    *,
    db: AsyncSession,
    consultation_id: str,
    content: str,
    response_content: str,
    sender_id: Optional[str],
    agent_name: str,
) -> None:
    """在同一 SQLite 事务中保存一次消息命令产生的所有消息。"""
    db.add(ConsultationMessage(
        id=str(uuid.uuid4()),
        consultation_id=consultation_id,
        sender_type="user",
        sender_id=sender_id,
        content=content,
        message_type="text",
    ))
    if response_content:
        db.add(ConsultationMessage(
            id=str(uuid.uuid4()),
            consultation_id=consultation_id,
            sender_type="agent",
            content=response_content,
            agent_name=agent_name,
            message_type="text",
        ))
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise


async def start_session(state: ConsultationState) -> ConsultationState:
    """启动新会话的工作流，执行到第一个中断点。

    Args:
        state: 初始会话状态

    Returns:
        工作流执行到中断点时的状态
    """
    session_id = state.get("session_id")
    if not session_id:
        raise ValueError("启动咨询会话需要有效的 session_id")

    result = await orchestrator.start_workflow(state)
    await persist_state(session_id, result)
    return result


async def process_message(
    session_id: str,
    content: str,
    state: ConsultationState,
    current_agent: str,
    *,
    idempotency_key: Optional[str] = None,
    sender_id: Optional[str] = None,
    db: Optional[AsyncSession] = None,
    request_id: Optional[str] = None,
    transport: str = "internal",
) -> ProcessMessageResult:
    """按 session 串行处理 HTTP 和 WebSocket 消息，并重放成功结果。

    ``state`` 和 ``current_agent`` 仅保留调用兼容；执行者必须在持锁后从
    LangGraph checkpoint 重新读取，避免排队请求沿用锁外旧状态。
    """
    fingerprint = _command_fingerprint({
        "content": content,
        "sender_id": sender_id,
    })
    async with _session_commands.serialize(session_id):
        replay = _session_commands.replay(
            session_id,
            "message",
            idempotency_key,
            fingerprint,
        )
        if replay is not None:
            return replay

        snapshot = await orchestrator.get_snapshot(session_id)
        if snapshot is None or not snapshot.values:
            raise ValueError(f"会话不存在: {session_id}")
        authoritative_state = validate_consultation_state(snapshot.values)
        executing_agent = authoritative_state.get("current_agent") or "Receptionist"

        result = await _process_message_unlocked(
            session_id,
            content,
            executing_agent,
            sender_id=sender_id,
            db=db,
            request_id=request_id,
            transport=transport,
        )
        if result.error is None or result.command_applied:
            _session_commands.remember(
                session_id,
                "message",
                idempotency_key,
                fingerprint,
                result,
            )
        return result


async def _process_message_unlocked(
    session_id: str,
    content: str,
    executing_agent: str,
    *,
    sender_id: Optional[str] = None,
    db: Optional[AsyncSession] = None,
    request_id: Optional[str] = None,
    transport: str = "internal",
) -> ProcessMessageResult:
    """在已持有会话锁时推进一次工作流消息。

    通过 resume_workflow 恢复工作流，由 LangGraph 条件边自动路由到下一节点。

    Args:
        session_id: 会话ID
        content: 用户消息内容
        executing_agent: resume 前 checkpoint 中的本轮执行者
        sender_id: 当前用户 ID
        db: HTTP 请求使用的数据库会话

    Returns:
        处理结果对象
    """
    command_applied = False
    correlation_id = request_id or str(uuid.uuid4())
    try:
        # FactDigger 是追加和脱敏 facts_raw 的唯一入口，服务层只传递本轮输入。
        state_updates: Dict[str, Any] = {"current_input": content}

        root_name = {
            "http": "HTTP POST /sessions/{session_id}/message",
            "websocket": "WebSocket message",
        }.get(transport, "application message")
        with trace_span(
            trace_store,
            event_type=transport,
            name=root_name,
            request_id=correlation_id,
            session_id=session_id,
        ):
            result = await orchestrator.resume_workflow(session_id, state_updates)
        command_applied = True

        response_content = result.get("final_output", "")
        # resume 后的 current_agent 表示新的工作流位置，不冒充本轮响应生产者。
        next_agent = result.get("current_agent", executing_agent)
        alert_triggered = result.get("alert_triggered", False)
        is_finished = await orchestrator.is_workflow_finished(session_id)

        if alert_triggered:
            await handle_high_risk_alert(session_id, result, executing_agent)
            return ProcessMessageResult(
                response_content=HIGH_RISK_ALERT_MESSAGE,
                next_agent=next_agent,
                alert_triggered=True,
                result_state=result,
                is_workflow_finished=is_finished,
                message_id=str(uuid.uuid4()),
                created_at=datetime.now(timezone.utc),
                command_applied=True,
                agent_name=executing_agent,
                request_id=correlation_id,
            )

        if "conversation_history" not in result:
            result["conversation_history"] = []
        result["conversation_history"].append({
            "agent": executing_agent,
            "user_message": content,
            "agent_response": response_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        await persist_state(
            session_id,
            result,
            projection_updates={"conversation_history": result["conversation_history"]},
        )

        if db is not None:
            await _save_message_command_rows(
                db=db,
                consultation_id=result.get("consultation_id", ""),
                content=content,
                response_content=response_content,
                sender_id=sender_id,
                agent_name=executing_agent,
            )

        return ProcessMessageResult(
            response_content=response_content,
            next_agent=next_agent,
            alert_triggered=False,
            result_state=result,
            is_workflow_finished=is_finished,
            message_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            command_applied=True,
            agent_name=executing_agent,
            request_id=correlation_id,
        )

    except AppException:
        # 保留可映射到统一 HTTP 错误响应的类型和状态码。
        raise
    except Exception as e:
        _logger.error(
            "【process_message】消息处理异常: session_id=%s, error_type=%s",
            session_id,
            type(e).__name__,
        )
        return ProcessMessageResult(
            response_content="",
            next_agent=executing_agent,
            alert_triggered=False,
            result_state=None,
            error=str(e),
            command_applied=command_applied,
            agent_name=executing_agent,
            request_id=correlation_id,
        )


async def process_consent(
    session_id: str,
    consent_given: bool,
    identity_info: Optional[dict] = None,
) -> ConsultationState:
    """处理用户同意确认，通过 resume_workflow 恢复工作流。

    Args:
        session_id: 会话ID
        consent_given: 是否同意
        identity_info: 身份信息

    Returns:
        更新后的状态
    """
    state_updates: Dict[str, Any] = {"consent_given": consent_given}
    if identity_info:
        state_updates["identity_info"] = identity_info

    if consent_given:
        result = await orchestrator.resume_workflow(session_id, state_updates)
    else:
        # 用户不同意，更新状态但不恢复工作流（流程将走向 END）
        result = await orchestrator.resume_workflow(session_id, state_updates)

    await persist_state(session_id, result)
    return result


async def process_lawyer_review(
    session_id: str,
    decision: str,
    feedback: Optional[str] = None,
    final_output: Optional[str] = None,
) -> ConsultationState:
    """处理律师审核反馈，通过工作流自动路由。

    Args:
        session_id: 会话ID
        decision: 律师决策 (approved/revise_facts/revise_risk)
        feedback: 律师反馈
        final_output: 最终输出（approved 时可能附带）

    Returns:
        更新后的状态
    """
    state_updates: Dict[str, Any] = {
        "lawyer_decision": decision,
        "lawyer_feedback": feedback,
        "awaiting_lawyer_review": False,
    }
    if decision == "approved" and final_output:
        state_updates["final_output"] = final_output

    result = await orchestrator.resume_workflow(session_id, state_updates)
    await persist_state(session_id, result)
    return result
