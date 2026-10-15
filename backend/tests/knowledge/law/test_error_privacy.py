"""未知异常类名不能随完整性或召回错误进入 Registry observation。"""

import json

import pytest

from app.consultation.agents.legal_research import LegalToolRegistry
from app.knowledge.law.corpus import load_criminal_law_data
from app.knowledge.law.repository import LawRepository
from app.knowledge.law.service import LawSearchService
from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.pipeline import RetrievalPipeline


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary,query", [("vector", "盗窃"), ("preflight", "盗窃"), ("preflight", "第264条")])
async def test_registry_error_categories_do_not_expose_untrusted_exception_names(boundary, query):
    secret = "13800138000"
    error = type(f"LeakError_{secret}", (Exception,), {})(f"private prompt {secret}")

    class Vector:
        async def retrieve(self, request):
            raise error

    async def preflight():
        raise error

    repository = LawRepository(load_criminal_law_data(profile="snapshot"))
    selected = LawSearchService(repository, RetrievalPipeline(Vector(), config=RetrievalConfig(rerank=False)),
        preflight=preflight if boundary == "preflight" else None)
    registry = LegalToolRegistry({}, "alice", repository.law_data, search_service=selected)
    status, observation, _ = await registry.execute("search_laws", {"query": query}, 1)
    assert status == "partial_dependency_failure"
    assert observation["rag_dependency_failed"] is True
    assert observation["candidates"]
    key = "integrity_error" if boundary == "preflight" else "vector_error"
    assert observation["retrieval_status"][key] == "OtherError"
    assert secret not in json.dumps(observation, ensure_ascii=False)
