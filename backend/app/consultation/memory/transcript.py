"""外部原文审计；稳定消息 ID 与单案顺序不承担工作流执行权威。"""
import json
import uuid

from sqlalchemy import func, select

from app.consultation.memory.context import MemorySettings
from app.models import ConsultationMessage


def stable_id(session_id, key, role):
    """相同命令载荷由调用层核验；本层仅生成跨请求稳定实体 ID。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"consultation:{session_id}:{key}:{role}"))


async def append_record(db, consultation_id, message_id, content, sender_type, *, command_id=None,
                        record_kind="external", metadata=None, sender_id=None, agent_name=None):
    """追加完整原文；既有同 ID 行必须完全匹配，不静默覆盖审计。"""
    existing = await db.scalar(select(ConsultationMessage).where(ConsultationMessage.id == message_id))
    encoded = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, default=str)
    if existing is not None:
        if existing.content != content or existing.consultation_id != consultation_id or existing.sender_type != sender_type:
            raise ValueError("原始消息 ID 载荷冲突")
        return existing
    last = await db.scalar(select(func.max(ConsultationMessage.sequence)).where(ConsultationMessage.consultation_id == consultation_id))
    row = ConsultationMessage(id=message_id, consultation_id=consultation_id, sequence=(last or 0) + 1,
        command_id=command_id, record_kind=record_kind, record_metadata=encoded, content=content,
        sender_type=sender_type, sender_id=sender_id, agent_name=agent_name, message_type="event" if record_kind == "internal" else "text")
    db.add(row)
    await db.flush()
    return row


def as_context_row(row):
    return {"id": row.id, "sequence": row.sequence, "sender_type": row.sender_type,
            "command_id": row.command_id,
            "record_kind": row.record_kind, "content": row.content}


async def memory_rows(db, consultation_id, cursor):
    """分页读取新增待摘要段和有界近期消息，禁止每轮读取完整档案。"""
    settings = MemorySettings.from_env()
    scope = select(ConsultationMessage).where(ConsultationMessage.consultation_id == consultation_id,
        ConsultationMessage.record_kind == "external", ConsultationMessage.sequence.is_not(None))
    recent = list((await db.scalars(scope.order_by(ConsultationMessage.sequence.desc()).limit(settings.recent_messages + 2))).all())
    recent.reverse()
    boundary = recent[0].sequence if recent else 0
    old = list((await db.scalars(scope.where(ConsultationMessage.sequence > cursor,
        ConsultationMessage.sequence < boundary).order_by(ConsultationMessage.sequence).limit(33))).all())
    if len(old) == 33:
        last_command = old[-1].command_id
        # 页末可能只读到用户行；保守移除该命令，下一轮从原游标继续读取完整组。
        old = [row for row in old if row.command_id != last_command] if last_command else old[:-1]
    return [as_context_row(row) for row in old + recent]
