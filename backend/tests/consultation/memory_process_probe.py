"""独立 Python 进程 memory 恢复探针；真实图/SQLite，外部模型与法律节点为确定性替身。"""
import asyncio
import json
import sys

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.consultation import service, workflow
from app.consultation.agents import fact_digger
from app.consultation.schemas.artifacts import ArtifactSource
from app.infrastructure.llm.gateway import LLMGateway
from app.models import Base, ConsultationMessage

SID = "synthetic-memory-process"
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
    return dict(state, current_agent="FactDigger", facts_coverage_rate=0.0, final_output="合成回复", pending_questions=["补充"])

async def summary(self, system_prompt, user_message, **kwargs):
    summary_batches.append(json.loads(user_message)["new_messages"])
    return "此前合成对话摘要，用户陈述尚待核实。"

async def main():
    mode, checkpoint_path, audit_path = sys.argv[1:]
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
        assert graph.get_active_sessions() == {}
        async with factory() as db:
            if mode == "write":
                cid = await service.create_consultation_record(SID, "synthetic-owner", "suspect", db)
                state = await service.start_session({"session_id": SID, "consultation_id": cid,
                    "user_id": "synthetic-owner", "consent_given": True, "facts_raw": [], "facts_structured": {},
                    "conversation_history": [], "current_agent": "Receptionist"})
                await service.record_external_exchange(SID, db=db, key="welcome", output="合成欢迎")
                for index in range(12):
                    result = await service.process_message(SID, f"合成陈述{index}", state, "FactDigger",
                        db=db, transport="http", sender_id="synthetic-owner", idempotency_key=f"round-{index}")
                    assert result.error is None
                    state = result.result_state
                assert all(len(rows) == 1 for rows in seen)
                snapshot = await graph.get_snapshot(SID)
                assert len(snapshot.values["facts_raw"]) == 4
                assert len(snapshot.values["memory"]["recent"]) == 4
                assert snapshot.values["memory"]["summary"]["through_sequence"] == 21
                assert snapshot.values["memory"]["summary"]["version"] == 11
            else:
                snapshot = await graph.get_snapshot(SID)
                assert snapshot.next == ("fact_intake",)
                state = snapshot.values
                assert state["memory"]["summary"]["through_sequence"] == 21
                assert state["memory"]["case"]["fields"]["incident_location"]["status"] == "conflicted"
                assert len(state["memory"]["case"]["fields"]["behavior_sequence"]["items"]) == 12
                replay = await service.process_message(SID, "合成陈述11", state, "FactDigger", db=db,
                    transport="websocket", sender_id="synthetic-owner", idempotency_key="round-11")
                assert replay.error is None
                assert seen == []
                assert await db.scalar(select(func.count()).select_from(ConsultationMessage)) == 25
                result = await service.process_message(SID, "合成新增一轮", state, "FactDigger", db=db,
                    transport="http", sender_id="synthetic-owner", idempotency_key="round-12")
                assert result.error is None
                assert seen == [["合成新增一轮"]]
                assert len(summary_batches) == 1
                assert [row["sequence"] for row in summary_batches[0]] == [22, 23]
                assert result.result_state["memory"]["summary"]["version"] == 12
            count = await db.scalar(select(func.count()).select_from(ConsultationMessage))
            print(json.dumps({"mode": mode, "raw_rows": count, "extraction_calls": len(seen),
                "summary_calls": len(summary_batches), "pending": list((await graph.get_snapshot(SID)).next)}))
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
