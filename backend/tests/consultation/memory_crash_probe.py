"""隔离子进程崩溃探针：真实图和两套 SQLite，仅模型与法律节点用确定性替身。"""

import asyncio
import json
import os
import sys

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.consultation import service, workflow
from app.consultation.agents import fact_digger
from app.consultation.memory.transcript import append_record, stable_id
from app.consultation.schemas.artifacts import ArtifactSource
from app.infrastructure.llm.gateway import LLMGateway
from app.models import Base, ConsultationMessage

SID = "synthetic-memory-crash"
TARGET = "合成目标陈述"
KEY = "target"
CRASH_EXIT = 86
seen = []
summary_batches = []


async def reception(state):
    return dict(state, current_agent="Receptionist", final_output="合成欢迎")


async def extract(rows):
    seen.append(rows)
    return {"incident_time": None, "incident_location": rows[-1], "parties": [],
            "behavior_sequence": [{"action": rows[-1]}], "consequence": None,
            "evidence_mentioned": None, "arrest_status": None, "surrender": None,
            "victim_forgiveness": None, "prior_record": None}, ArtifactSource.TOOL_CALL


async def law(state):
    return state


async def coverage(state):
    return dict(state, current_agent="FactDigger", facts_coverage_rate=0.0,
                final_output="合成回复", pending_questions=["补充"])


async def summary(self, system_prompt, user_message, **kwargs):
    summary_batches.append(json.loads(user_message)["new_messages"])
    return "合成已覆盖陈述，仍待核实。"


def crash():
    # 不运行 finally、连接清理或 lifespan；区别于正常 SIGTERM 退出。
    os._exit(CRASH_EXIT)


def inject(window, graph, db, saver):
    original_put = saver.aput

    async def put(config, checkpoint, *args, **kwargs):
        result = await original_put(config, checkpoint, *args, **kwargs)
        values = checkpoint["channel_values"]
        receipt = values.get("message_audit") or {}
        case = (values.get("memory") or {}).get("case") or {}
        items = case.get("fields", {}).get("behavior_sequence", {}).get("items", [])
        if window == "intake_checkpoint" and receipt.get("status") == "running" and any(
            item["value"]["action"] == TARGET for item in items
        ):
            crash()
        return result

    saver.aput = put
    original_commit = db.commit
    commits = 0

    async def commit():
        nonlocal commits
        commits += 1
        if (window == "input_failure" and commits == 1) or (window == "reply_failure" and commits == 2):
            raise RuntimeError("synthetic recoverable commit failure")
        if window == "input_uncommitted" and commits == 1:
            crash()
        if window == "reply_uncommitted" and commits == 2:
            crash()
        await original_commit()
        if window == "input_committed" and commits == 1:
            crash()
        if window == "reply_committed" and commits == 2:
            crash()

    db.commit = commit
    original_persist = service.persist_state

    async def persist(session_id, state=None, *, projection_updates=None):
        updates = projection_updates or {}
        if window == "summary_unsaved" and "memory" in updates:
            crash()
        await original_persist(session_id, state, projection_updates=projection_updates)
        marker = updates.get("message_audit") or {}
        if window == "prepared" and marker.get("status") == "prepared":
            crash()
        if window == "running" and marker.get("status") == "running":
            crash()
        if window == "applied" and marker.get("metadata"):
            crash()
        if window == "summary_saved" and "memory" in updates:
            crash()
        if window == "receipt_cleared" and "message_audit" in updates and updates["message_audit"] is None:
            crash()

    service.persist_state = persist
    original_resume = graph.resume_workflow

    async def resume(*args, **kwargs):
        result = await original_resume(*args, **kwargs)
        if window == "graph_applied":
            crash()
        return result

    graph.resume_workflow = resume


async def snapshot_data(graph, db):
    snapshot = await graph.get_snapshot(SID)
    state = snapshot.values
    rows = (await db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
    return {"pending": list(snapshot.next), "receipt": state.get("message_audit"),
            "case": (state.get("memory") or {}).get("case"),
            "summary": (state.get("memory") or {}).get("summary"),
            "recent": (state.get("memory") or {}).get("recent"),
            "rows": [{"id": row.id, "sequence": row.sequence, "content": row.content,
                      "sender_type": row.sender_type, "record_kind": row.record_kind,
                      "command_id": row.command_id,
                      "metadata": json.loads(row.record_metadata or "{}")} for row in rows]}


async def main():
    mode, window, checkpoint_path, audit_path = sys.argv[1:]
    workflow.receptionist_node = reception
    workflow.law_ref_node = law
    workflow._fact_digger_workflow_node = coverage
    fact_digger._extract_structured_facts = extract
    LLMGateway.generate = summary
    engine = create_async_engine(f"sqlite+aiosqlite:///{audit_path}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with aiosqlite.connect(checkpoint_path) as conn:
        saver = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
        await saver.setup()
        graph = workflow.ConsultationOrchestrator(saver, persistent=True)
        service.orchestrator = graph
        async with factory() as db:
            first_error = None
            if mode in {"crash", "failure"}:
                cid = await service.create_consultation_record(SID, "synthetic-owner", "suspect", db)
                await service.start_session({"session_id": SID, "consultation_id": cid,
                    "user_id": "synthetic-owner", "consent_given": True, "facts_raw": [],
                    "facts_structured": {}, "conversation_history": [], "current_agent": "Receptionist"})
                # 固定两轮种子让目标命令必定产生非空摘要候选。
                for index in range(2):
                    result = await service.process_message(SID, f"合成种子{index}", {}, "FactDigger",
                        db=db, transport="http", sender_id="synthetic-owner", idempotency_key=f"seed-{index}")
                    assert result.error is None
                seen.clear()
                summary_batches.clear()
                inject(window, graph, db, saver)
                failed = await service.process_message(SID, TARGET, {}, "FactDigger", db=db,
                    transport="http", sender_id="synthetic-owner", idempotency_key=KEY)
                if mode == "crash":
                    raise AssertionError(f"crash window not reached: {window}")
                first_error = {"error": failed.error, "applied": failed.command_applied}
            if mode == "recover_legacy_error":
                # 旧错误回复是不可覆盖的原文；恢复时不能为更新指引而篡改历史。
                state = (await graph.get_snapshot(SID)).values
                await append_record(db, state["consultation_id"], stable_id(SID, "message:target", "error_response"),
                    "消息处理失败，请稍后重试", "system",
                    command_id=stable_id(SID, "message:target", "command"),
                    metadata={"event": "error_response", "status_code": 500, "request_id": "legacy-request"})
                await db.commit()
            before = await snapshot_data(graph, db)
            results = []
            for _ in range(2):
                result = await service.process_message(SID, TARGET, {}, "FactDigger", db=db,
                    transport="http", sender_id="synthetic-owner", idempotency_key=KEY)
                results.append({"error": result.error, "reply": result.response_content,
                                "message_id": result.message_id, "applied": result.command_applied})
            after = await snapshot_data(graph, db)
            print("CRASH_RESULT=" + json.dumps({"window": window, "before": before, "after": after,
                "results": results, "first_error": first_error,
                "extractions": seen, "summary_batches": summary_batches}, ensure_ascii=False))
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
