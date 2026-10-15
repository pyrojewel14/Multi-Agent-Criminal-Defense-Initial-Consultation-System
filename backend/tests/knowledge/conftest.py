"""知识库测试独立持有模型与领域资源，避免触碰应用默认所有者。"""

import pytest_asyncio

from app.infrastructure.llm import manager as manager_module
from app.knowledge import wiring
from app.knowledge.resources import KnowledgeResources


@pytest_asyncio.fixture(autouse=True)
async def isolated_model_manager(monkeypatch):
    owner = manager_module.ModelManager()
    resources = KnowledgeResources()
    monkeypatch.setattr(manager_module, "_default_manager", owner)
    monkeypatch.setattr(wiring, "knowledge_resources", resources)
    try:
        yield owner
    finally:
        resources.close()
        await owner.aclose()
