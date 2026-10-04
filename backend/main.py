import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.api.v1.routers.auth import auth_router
from app.api.v1.routers.consultation import router as consultation_router
from app.api.v1.routers.consultation_history import consultation_router as history_router
from app.api.v1.routers.knowledge_router import knowledge_router
from app.api.v1.routers.lawyer import lawyer_session_router
from app.api.v1.routers.lawyers import lawyer_management_router
from app.api.v1.routers.users import user_router
from app.consultation.memory.context import MemorySettings
from app.consultation.workflow import orchestrator
from app.errors.register import register_exception_handlers
from app.infrastructure.database.db import close_db, database_readiness, init_db
from app.infrastructure.database.readiness import SQLiteReadiness
from app.infrastructure.database.redis import close_redis, init_redis
from app.infrastructure.llm.factory import chat_model_factory, embed_model_factory
from app.infrastructure.logging import get_logger
from app.knowledge.law_knowledge import preflight_law_knowledge
from app.knowledge.rag.reorder_service import reorder_service
from app.security.rbac import attach_user_to_request

load_dotenv()

_logger = get_logger("Main")
checkpoint_readiness = SQLiteReadiness()


# check_and_download_reranker_model()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _logger.info("Starting up application...")
    preflight_law_knowledge()
    checkpoint_path = Path(os.getenv("LANGGRAPH_CHECKPOINT_DB_PATH", "./data/langgraph_checkpoints.db"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(checkpoint_path) as connection:
        # checkpoint 包含咨询原文；只反序列化 LangGraph 内置安全类型。
        checkpointer = AsyncSqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
        await checkpointer.setup()
        orchestrator.configure_checkpointer(checkpointer, persistent=True)
        checkpoint_readiness.start(connection, checkpointer.lock)
        try:
            await init_db()
            await init_redis()
            _logger.info("Database and Redis initialized")
            yield
        finally:
            checkpoint_readiness.close()
            orchestrator.mark_checkpointer_closed()
            _logger.info("Shutting down application...")
            reorder_service.close()
            chat_model_factory.close()
            embed_model_factory.close()
            await close_redis()
            await close_db()
            _logger.info("Cleanup completed")


app = FastAPI(
    title="刑事辩护初期咨询系统",
    description="多 Agent 驱动的法律咨询系统 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(attach_user_to_request)

register_exception_handlers(app)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")
app.include_router(lawyer_management_router, prefix="/api/v1")
app.include_router(lawyer_session_router, prefix="/api/v1")
app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(consultation_router, prefix="/api/v1")
app.include_router(history_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "刑事辩护初期咨询系统 API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check():
    """保留 HTTP 200，必要存储失败以 JSON not_ready 表示。"""
    reranker_state = reorder_service.readiness()
    raw_state, checkpoint_state = await asyncio.gather(database_readiness.check(), checkpoint_readiness.check())
    settings = MemorySettings.from_env()
    stores_ready = raw_state["available"] and checkpoint_state["available"]
    return {
        "status": ("ready" if reranker_state["available"] else "degraded") if stores_ready else "not_ready",
        "api": "ready" if stores_ready else "not_ready",
        "dependencies": {
            "reranker": reranker_state,
            "raw_transcript": raw_state,
            "checkpoint": {
                **checkpoint_state,
                "persistence": orchestrator.checkpoint_persistence,
                "restart_recovery": orchestrator.can_resume_after_restart,
                "restart_recovery_scope": "single_worker_normal_restart",
                "recovery_verified_now": False,
            },
        },
        "memory": {
            "summary": {"enabled": True, "requires_consent": True,
                        "input_budget": settings.summary_input_budget, "output_budget": settings.summary_output_budget,
                        "timeout_seconds": settings.summary_timeout},
            "structured_case": {"enabled": True},
            "context": {"recent_messages": settings.recent_messages, "token_budget": settings.context_token_budget,
                        "model_window": settings.model_window, "output_reserve": settings.output_reserve},
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
