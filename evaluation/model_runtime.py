"""评测运行期显式持有模型与领域缓存，并恢复临时 Gateway 注入。"""

from contextlib import asynccontextmanager


@asynccontextmanager
async def evaluation_runtime(*, model_manager=None):
    """内部创建的 owner 在退出时关闭；显式注入的 owner 仍归调用方。"""
    from app.infrastructure.llm.gateway import llm_gateway
    from app.infrastructure.llm.manager import ModelManager
    from app.knowledge.resources import KnowledgeResources

    owns_manager = model_manager is None
    manager = ModelManager() if owns_manager else model_manager
    resources = KnowledgeResources()
    original = llm_gateway._model_manager
    llm_gateway._model_manager = manager
    try:
        yield manager, resources
    finally:
        llm_gateway._model_manager = original
        try:
            await resources.aclose()
        finally:
            if owns_manager:
                await manager.aclose()
