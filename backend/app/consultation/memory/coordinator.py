"""消息审计与派生记忆协调；跨 SQLite 写入使用待修复回执。"""
import json
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from sqlalchemy import select

from app.consultation.constants import HIGH_RISK_ALERT_MESSAGE
from app.consultation.memory.context import ContextBuilder, MemorySettings, node_context
from app.consultation.memory.summary import SUMMARY_SYSTEM_PROMPT, advance_summary, log_summary_event
from app.consultation.memory.transcript import append_record, memory_rows, stable_id
from app.errors.exceptions import AppException
from app.infrastructure.logging import get_logger
from app.infrastructure.observability.tracing import trace_span, trace_store
from app.models import Consultation, ConsultationMessage

_logger = get_logger("Memory.Coordinator")


async def refresh_memory(service, db, session_id, state):
    """摘要候选保存成功才成为权威；失败不影响已完成的消息命令。"""
    # 原文审计不依赖同意；确认同意后，存档原文才可进入有界派生记忆。
    started = time.perf_counter()
    memory = state.get("memory") or {}
    previous_summary = memory.get("summary") or {}
    cursor = previous_summary.get("through_sequence", 0)
    if state.get("consent_given") is not True:
        log_summary_event("skipped", batch_count=0, cursor_before=cursor, cursor_after=cursor,
                          version=previous_summary.get("version", 0),
                          started=started, error_code="consent_required")
        return
    rows = await memory_rows(db, state["consultation_id"], cursor)
    settings = MemorySettings.from_env()
    async def summarize(old, batch):
        from app.infrastructure.llm.gateway import LLMCallPolicy, LLMGateway
        from app.security.sensitive_filter import mask_pii
        # 可选调用独立使用一次 attempt 和有限总超时，避免主调用重试预算放大。
        gateway = LLMGateway(policy=LLMCallPolicy(max_attempts=1, total_timeout_seconds=settings.summary_timeout,
                                                  attempt_timeout_seconds=settings.summary_timeout))
        message = mask_pii(json.dumps({"previous_summary": old, "new_messages": batch}, ensure_ascii=False))
        ContextBuilder(replace(settings, context_token_budget=settings.summary_input_budget)).build(
            SUMMARY_SYSTEM_PROMPT, message)
        with node_context("summary", None):
            return await gateway.generate(
                SUMMARY_SYSTEM_PROMPT, message,
                output_limit=max(1, settings.summary_output_budget // 3), reasoning=False)
    candidate = await advance_summary(memory, rows, summarize, settings)
    await service.persist_state(session_id, state, projection_updates={"memory": candidate,
        "conversation_history": state.get("conversation_history", [])[-settings.recent_messages:],
        "facts_raw": state.get("facts_raw", [])[-settings.recent_messages:]})
    summary = candidate["summary"]
    if summary["through_sequence"] > cursor:
        log_summary_event("saved", batch_count=sum(row.get("record_kind") == "external" and
                          cursor < row["sequence"] <= summary["through_sequence"] for row in rows),
                          cursor_before=cursor, cursor_after=summary["through_sequence"],
                          version=summary["version"], started=started)


async def refresh_optional_memory(service, db, session_id, state):
    """统一隔离原文提交后的可选投影失败；不记录正文或异常详情。"""
    started = time.perf_counter()
    try:
        await refresh_memory(service, db, session_id, state)
    except Exception:
        summary = (state.get("memory") or {}).get("summary") or {}
        cursor = summary.get("through_sequence", 0)
        # 失败可能发生在读取、候选生成或保存阶段，外层无法可靠得知尝试批次。
        log_summary_event("projection_failed", batch_count=None, cursor_before=cursor, cursor_after=cursor,
                          version=summary.get("version", 0), started=started, error_code="projection_error")


def _result(service, state, metadata, message_id, content):
    state = dict(state)
    if "pending_questions" in metadata:
        state["pending_questions"] = metadata["pending_questions"]
    return service.ProcessMessageResult(response_content=content, next_agent=metadata["next_agent"],
        alert_triggered=metadata["alert_triggered"], result_state=state,
        is_workflow_finished=metadata["is_finished"], message_id=message_id,
        created_at=datetime.fromisoformat(metadata["created_at"]), command_applied=True,
        agent_name=metadata["agent_name"], request_id=metadata["request_id"])


async def process_external(service, session_id, content, state, *, db, idempotency_key, sender_id, request_id, transport, fingerprint,
                           state_updates=None, output_override=None, command_name="message"):
    """必须在同会话锁内调用；输入先提交，执行后的审计修复不重跑图。"""
    key = command_name + ":" + (idempotency_key or request_id or str(uuid.uuid4()))
    command_id = stable_id(session_id, key, "command")
    input_id = stable_id(session_id, key, "user")
    reply_id = stable_id(session_id, key, "reply")
    request_id = request_id or str(uuid.uuid4())
    agent = state.get("current_agent", "Receptionist")
    applied = False
    replayed_error = False
    invoke_started = False
    try:
        pending = state.get("message_audit")
        if pending and pending["command_id"] != command_id:
            raise ValueError("存在未完成消息，请以相同幂等键修复")
        existing = await db.scalar(select(ConsultationMessage).where(ConsultationMessage.id == input_id))
        if existing is not None:
            original = json.loads(existing.record_metadata or "{}")
            if original.get("fingerprint") != fingerprint:
                raise service.IdempotencyConflictError("幂等键已用于不同请求")
            reply = await db.scalar(select(ConsultationMessage).where(ConsultationMessage.id == reply_id))
            if reply is not None:
                metadata = json.loads(reply.record_metadata)
                if metadata.get("terminal_error"):
                    from app.infrastructure.llm.gateway import ContextBudgetException
                    if pending:
                        await service.persist_state(session_id, projection_updates={"message_audit": None,
                            "current_input": None, "current_message_id": None})
                    replayed_error = True
                    raise ContextBudgetException(detail="context_budget_exceeded")
                # 原文与图已成功，摘要/清回执失败只重试投影，不重跑图。
                if pending:
                    await refresh_optional_memory(service, db, session_id, state)
                    await service.persist_state(session_id, projection_updates={"message_audit": None})
                return _result(service, await service.get_session_state(session_id), metadata, reply.id, reply.content)
            if not pending:
                raise ValueError("原文已记录但执行结果未知，需人工核验 checkpoint")
        else:
            if state.get("repair_required"):
                raise ValueError("必须先修复生命周期命令")
            await append_record(db, state["consultation_id"], input_id, content, "user", command_id=command_id,
                sender_id=sender_id, metadata={"fingerprint": fingerprint, "transport": transport, "request_id": request_id})
            await db.commit()
            pending = {"command_id": command_id, "input_id": input_id, "reply_id": reply_id,
                       "fingerprint": fingerprint, "status": "prepared", "agent_name": agent,
                       "request_id": request_id}
            await service.persist_state(session_id, projection_updates={"message_audit": pending})
        if pending["fingerprint"] != fingerprint:
            raise service.IdempotencyConflictError("幂等键已用于不同请求")
        if pending["status"] == "running":
            raise ValueError("消息运行中断且结果未知，需人工核验 checkpoint")
        if pending["status"] != "applied":
            pending = dict(pending, status="running")
            await service.persist_state(session_id, projection_updates={"message_audit": pending})
            with trace_span(trace_store, event_type=transport, name="external message", session_id=session_id, request_id=request_id):
                invoke_started = True
                if command_name == "consent" and bool(state.get("consent_given")) == bool((state_updates or {}).get("consent_given")):
                    # 重复确认仅更新投影，不能重新摄取旧事实或越过当前 pending。
                    result = await service.orchestrator.update_workflow_state(session_id, state_updates)
                else:
                    result = await service.orchestrator.resume_workflow(session_id,
                        state_updates if state_updates is not None else {"current_input": content, "current_message_id": input_id})
            applied = True
            metadata = {"agent_name": pending["agent_name"], "next_agent": result.get("current_agent", agent),
                "alert_triggered": bool(result.get("alert_triggered")),
                "is_finished": await service.orchestrator.is_workflow_finished(session_id),
                "created_at": datetime.now(timezone.utc).isoformat(), "request_id": pending["request_id"]}
            metadata["pending_questions"] = result.get("pending_questions", [])
            response = output_override if output_override is not None else (HIGH_RISK_ALERT_MESSAGE if metadata["alert_triggered"] else result.get("final_output", ""))
            pending = dict(pending, status="applied", response=response, metadata=metadata)
            await service.persist_state(session_id, result, projection_updates={"message_audit": pending})
            state = result
        else:
            applied = True
            metadata = pending.get("metadata") or {
                "agent_name": pending["agent_name"], "next_agent": state.get("current_agent", agent),
                "alert_triggered": bool(state.get("alert_triggered")),
                "is_finished": await service.orchestrator.is_workflow_finished(session_id),
                "created_at": pending["completed_at"], "request_id": pending["request_id"]}
            response = pending.get("response", state.get("final_output", ""))
        await append_record(db, state["consultation_id"], reply_id, response, "agent", command_id=command_id,
                            agent_name=metadata["agent_name"], metadata=metadata)
        if command_name == "consent":
            consultation = await db.scalar(select(Consultation).where(Consultation.id == state["consultation_id"]))
            if consultation is not None:
                consultation.consent_given = bool(state.get("consent_given"))
        await db.commit()
        # 派生摘要失败不能撤销成功消息，也不能让用户再次推进已完成图。
        await refresh_optional_memory(service, db, session_id, state)
        await service.persist_state(session_id, projection_updates={"message_audit": None})
        return _result(service, await service.get_session_state(session_id), metadata, reply_id, response)
    except service.IdempotencyConflictError:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        if replayed_error:
            raise
        snapshot = await service.orchestrator.get_snapshot(session_id)
        latest = dict(snapshot.values) if snapshot else state
        marker = latest.get("message_audit")
        if marker and marker["command_id"] == command_id and marker["status"] == "running" and invoke_started and not applied:
            # 已进入暂停节点的回执在节点 checkpoint 中为 applied；只有失败节点可继续恢复。
            marker = dict(marker, status="failed")
            await service.persist_state(session_id, projection_updates={"message_audit": marker})
        if isinstance(exc, AppException):
            rejected_input = exc.detail == "context_budget_exceeded" and tuple(snapshot.next) == ("fact_intake",)
            try:
                await append_record(db, state["consultation_id"], str(uuid.uuid4()), json.dumps({"event": "model_error", "error_code": exc.code.value}), "system",
                    command_id=command_id, record_kind="internal", metadata={"event": "model_error", "error_code": exc.code.value})
                await append_record(db, state["consultation_id"], reply_id if rejected_input else str(uuid.uuid4()), exc.message,
                    "system", command_id=command_id, metadata={"event": "error_response", "error_code": exc.code.value,
                        "status_code": exc.status_code, "request_id": request_id, "terminal_error": rejected_input})
                await db.commit()
                if rejected_input:
                    await service.persist_state(session_id, projection_updates={"message_audit": None,
                        "current_input": None, "current_message_id": None})
            except Exception:
                await db.rollback()
            raise
        try:
            recorded_input = await db.scalar(select(ConsultationMessage.id).where(ConsultationMessage.id == input_id))
            if recorded_input:
                await append_record(db, state["consultation_id"], stable_id(session_id, key, "failed_event"),
                    '{"event":"message_failed"}', "system", command_id=command_id, record_kind="internal",
                    metadata={"event": "message_failed", "error_type": type(exc).__name__, "applied": applied})
                await append_record(db, state["consultation_id"], stable_id(session_id, key, "error_response"),
                    "消息处理失败，请稍后重试", "system", command_id=command_id,
                    metadata={"event": "error_response", "status_code": 500, "request_id": request_id})
                await db.commit()
        except Exception:
            await db.rollback()
        _logger.error("外部消息待修复: error_type=%s, applied=%s", type(exc).__name__, applied)
        return service.ProcessMessageResult(response_content="", next_agent=agent, alert_triggered=False,
            result_state=None, error="消息审计或执行未完成，请使用相同幂等键重试", command_applied=applied,
            agent_name=agent, request_id=request_id)
