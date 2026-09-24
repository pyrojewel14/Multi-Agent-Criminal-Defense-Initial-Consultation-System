from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.law_ref import preflight_law_knowledge
from app.db.db_config import close_db, init_db
from app.db.redis_config import close_redis, init_redis
from app.errors.register import register_exception_handlers
from app.rag.reorder_service import reorder_service
from app.security.rbac import attach_user_to_request
from app.utils.factory import chat_model_factory, embed_model_factory
from app.utils.logger import get_logger
from app.v1.router.auth import auth_router
from app.v1.router.consultation import router as consultation_router
from app.v1.router.consultation_history import consultation_router as history_router
from app.v1.router.knowledge_router import knowledge_router
from app.v1.router.lawyer import lawyer_session_router
from app.v1.router.lawyers import lawyer_management_router
from app.v1.router.users import user_router
from app.orchestrator.workflow import orchestrator

load_dotenv()

_logger = get_logger("Main")

# check_and_download_reranker_model()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _logger.info("Starting up application...")
    if not orchestrator.can_resume_after_restart:
        _logger.warning(
            "LangGraph checkpoint 当前仅限进程内；注入并显式确认持久化 saver 前不支持重启恢复"
        )
    preflight_law_knowledge()
    await init_db()
    await init_redis()
    _logger.info("Database and Redis initialized")
    try:
        yield
    except Exception as e:
        _logger.error(f"Error occurred: {e}")
    finally:
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
    """报告 API readiness 与可选重排序器的独立状态。"""
    reranker_state = reorder_service.readiness()
    return {
        "status": "ready" if reranker_state["available"] else "degraded",
        "api": "ready",
        "dependencies": {
            "reranker": reranker_state,
            "checkpoint": {
                "persistence": orchestrator.checkpoint_persistence,
                "restart_recovery": orchestrator.can_resume_after_restart,
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
