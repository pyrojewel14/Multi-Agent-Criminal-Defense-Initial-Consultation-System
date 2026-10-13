"""隔离 HTTP worker 探针：真实路由、service、图和双 SQLite，模型及法律节点确定化。"""
import asyncio
import json
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
import uvicorn
from fastapi import FastAPI
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1.routers.consultation import routes
from app.consultation import service, workflow
from app.consultation.agents import fact_digger
from app.consultation.memory import coordinator
from app.consultation.schemas.artifacts import ArtifactSource
from app.infrastructure.database.db import get_db
from app.infrastructure.llm.gateway import LLMGateway
from app.models import Base, ConsultationMessage
from app.security.rbac import get_current_user


async def serve(audit_path, checkpoint_path, control_path, worker, port):
    control = Path(control_path)
    events = []
    gates = {}

    def storage_error(error, store):
        events.append({"event": "storage_error", "store": store, "error_type": type(error).__name__,
                       "sqlite_errorcode": getattr(error, "sqlite_errorcode", None),
                       "sqlite_errorname": getattr(error, "sqlite_errorname", None)})

    class ObservedSaver(AsyncSqliteSaver):
        async def aput(self, *args, **kwargs):
            try:
                return await super().aput(*args, **kwargs)
            except sqlite3.Error as error:
                storage_error(error, "checkpoint")
                raise

        async def aget_tuple(self, *args, **kwargs):
            try:
                return await super().aget_tuple(*args, **kwargs)
            except sqlite3.Error as error:
                storage_error(error, "checkpoint")
                raise

    async def gate(stage):
        name = gates.pop(stage, None)
        if not name:
            return
        marker = control / f"{name}.{worker}.entered"
        marker.write_text("entered")
        deadline = asyncio.get_running_loop().time() + 15
        while not (control / f"{name}.release").exists():
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("probe gate timeout")
            await asyncio.sleep(0.01)

    async def receptionist(state):
        return dict(state, current_agent="Receptionist", final_output="合成欢迎")

    async def extract(rows):
        events.append({"event": "extract", "rows": rows})
        await gate("extract")
        return {"incident_time": None, "incident_location": rows[-1], "parties": [],
                "behavior_sequence": [{"action": rows[-1]}], "consequence": None,
                "evidence_mentioned": [], "arrest_status": None, "surrender": None,
                "victim_forgiveness": None, "prior_record": None}, ArtifactSource.TOOL_CALL

    async def law(state):
        return state

    async def coverage(state):
        return dict(state, current_agent="FactDigger", facts_coverage_rate=0.0,
                    final_output="合成回复", pending_questions=["补充"])

    async def summarize(self, system_prompt, user_message, **kwargs):
        payload = json.loads(user_message)
        events.append({"event": "summary", "payload": payload})
        await gate("summary")
        return "合成增量摘要，陈述待核实。"

    original_external = coordinator.process_external

    async def external(*args, **kwargs):
        # 在真实进程锁内、读取 checkpoint 后暂停；不替代存储或锁。
        await gate("external")
        return await original_external(*args, **kwargs)

    workflow.receptionist_node = receptionist
    workflow.law_ref_node = law
    workflow._fact_digger_workflow_node = coverage
    fact_digger._extract_structured_facts = extract
    LLMGateway.generate = summarize
    coordinator.process_external = external
    engine = create_async_engine(f"sqlite+aiosqlite:///{audit_path}")

    @event.listens_for(engine.sync_engine, "handle_error")
    def observe_raw_error(context):
        storage_error(context.original_exception, "raw")

    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(app):
        if os.getenv("MEMORY_PROBE_FORMAL_LIFESPAN") == "1":
            import main

            async def no_redis():
                return None

            os.environ["LANGGRAPH_CHECKPOINT_DB_PATH"] = str(checkpoint_path)
            main.preflight_law_knowledge = lambda: None
            main.init_redis = no_redis
            main.close_redis = no_redis
            main.AsyncSqliteSaver = ObservedSaver
            graph = workflow.ConsultationOrchestrator()
            main.orchestrator = service.orchestrator = routes.orchestrator = graph
            # 正式 lifespan 保留真实 init_db/checkpointer/所有权防护，仅隔离 Redis 和法律资产。
            try:
                async with main.lifespan(app):
                    yield
            finally:
                await engine.dispose()
            return
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with aiosqlite.connect(checkpoint_path) as connection:
            saver = ObservedSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
            await saver.setup()
            graph = workflow.ConsultationOrchestrator(saver, persistent=True)
            service.orchestrator = graph
            routes.orchestrator = graph
            yield
        await engine.dispose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(routes.router, prefix="/api/v1")

    async def db_dependency():
        async with factory() as db:
            original_flush = db.flush

            async def flush(*args, **kwargs):
                await gate("raw_flush")
                return await original_flush(*args, **kwargs)

            db.flush = flush
            yield db

    async def owner():
        return {"user_id": "synthetic-owner", "role": "client"}

    app.dependency_overrides[get_db] = db_dependency
    app.dependency_overrides[get_current_user] = owner

    @app.get("/probe/ready")
    async def ready():
        return {"pid": os.getpid(), "worker": worker,
                "formal_main_lifespan": os.getenv("MEMORY_PROBE_FORMAL_LIFESPAN") == "1"}

    @app.post("/probe/arm")
    async def arm(payload: dict):
        gates.update(payload)
        return {"armed": gates}

    @app.get("/probe/events")
    async def get_events():
        return events

    @app.get("/probe/snapshot/{sid}")
    async def snapshot(sid: str):
        snap = await service.orchestrator.get_snapshot(sid)
        state = snap.values
        async with factory() as db:
            rows = (await db.scalars(select(ConsultationMessage).where(
                ConsultationMessage.consultation_id == state["consultation_id"]).order_by(
                ConsultationMessage.sequence))).all()
        return {"state": state, "pending": list(snap.next), "raw": [
            {"id": row.id, "sequence": row.sequence, "content": row.content,
             "sender_type": row.sender_type, "record_kind": row.record_kind,
             "command_id": row.command_id, "metadata": json.loads(row.record_metadata or "{}")}
            for row in rows]}

    await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=int(port),
                                       log_level="warning", access_log=False)).serve()


if __name__ == "__main__":
    asyncio.run(serve(*sys.argv[1:]))
