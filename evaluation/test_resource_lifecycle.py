"""评测持有独立评分资源，并在所有退出路径归还工厂。"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.knowledge import wiring
from app.consultation.agents import legal_research
from evaluation import run_full_eval, run_live_eval
from evaluation.build_live_index import build_index


class FakeScorer:
    def __init__(self):
        self.closed = False

    async def score(self, query, documents, *, timeout_s):
        return [0.5] * len(documents)

    def close(self):
        self.closed = True


class FakeResources:
    def __init__(self):
        self.model = FakeScorer()
        self.close_count = 0

    def scorer(self, config):
        return self.model

    def close(self):
        self.close_count += 1
        self.model.close()


@pytest.fixture
def resource_owners(monkeypatch):
    created = []
    application = FakeResources()

    def create():
        result = FakeResources()
        created.append(result)
        return result

    monkeypatch.setattr(wiring, "KnowledgeResources", create)
    monkeypatch.setattr(wiring, "knowledge_resources", application)
    return created, application


@pytest.mark.parametrize("outcome", ["normal", "error", "cancel"])
def test_snapshot_runtime_reuses_owned_scorer_and_restores_factory_on_exit(tmp_path, monkeypatch, resource_owners, outcome):
    law = run_live_eval.LAW
    index = tmp_path / "snapshot"
    build_index(law, index, "snapshot_collection", "test", "digest", lambda texts: [[1.0] for _ in texts])
    monkeypatch.setenv("TEXT_EMBEDDING_MODEL_NAME", "test")
    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "true")
    monkeypatch.setenv("LAW_FULL_HYDE_ENABLED", "false")
    original = legal_research.build_law_search_service
    created, application = resource_owners

    def run():
        with run_live_eval.snapshot_search_runtime(index_dir=index, preflight_result={
                "index_collection": "snapshot_collection", "embedding_digest": "digest"}):
            first = legal_research.build_law_search_service()
            second = legal_research.build_law_search_service()
            assert first.pipeline.reranker is second.pipeline.reranker
            assert first.pipeline.reranker is not application.model
            assert first.pipeline.reranker.closed is False
            if outcome == "error":
                raise RuntimeError("evaluation failed")
            if outcome == "cancel":
                raise asyncio.CancelledError()

    if outcome == "normal":
        run()
    else:
        with pytest.raises(RuntimeError if outcome == "error" else asyncio.CancelledError):
            run()
    assert legal_research.build_law_search_service is original
    assert len(created) == 1
    assert created[0].close_count == 1 and created[0].model.closed
    assert application.close_count == 0 and not application.model.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["live-rag", "live-lawref"])
@pytest.mark.parametrize("outcome", ["normal", "error", "cancel"])
async def test_full_evaluation_closes_only_owned_resources_and_restores_factory(tmp_path, monkeypatch, resource_owners, mode, outcome):
    from app.knowledge.retrieval.config import RerankerConfig, RetrievalConfig
    from app.knowledge.retrieval.pipeline import RetrievalPipeline

    original_builder = wiring.build_law_search_service
    original_factory = legal_research.build_law_search_service
    created, application = resource_owners
    used = []
    cancellation_ready = asyncio.Event()

    def builder(**kwargs):
        owner = kwargs.get("resources", application)
        used.append(owner)
        kwargs["pipeline"] = RetrievalPipeline(None, reranker=owner.scorer(RerankerConfig()),
                                               config=RetrievalConfig(rerank=False))
        return original_builder(**kwargs)

    monkeypatch.setattr(wiring, "build_law_search_service", builder)

    async def law_node(state):
        legal_research.build_law_search_service(law_data=run_full_eval_corpus())
        if outcome == "cancel":
            cancellation_ready.set()
            await asyncio.Future()
        return {"applied_laws": [], "law_research": {"termination_reason": "final_answer", "trajectory": []}}

    def run_full_eval_corpus():
        from app.knowledge.law.corpus import load_criminal_law_data
        return load_criminal_law_data(profile="full")

    monkeypatch.setattr("app.consultation.agents.law_ref.law_ref_node", law_node)
    if mode == "live-rag" and outcome == "cancel":
        async def cancelled(*args, **kwargs):
            cancellation_ready.set()
            await asyncio.Future()
        monkeypatch.setattr("app.knowledge.law.service.LawSearchService.search", cancelled)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps({"id": identifier, "query": "第264条", "article_id": "第264条",
                                        "contains": "盗窃", "coverage_eligible": True}) + "\n"
                             for identifier in ("theft", "theft-again")))
    output = tmp_path if outcome == "error" else tmp_path / "result.json"

    async def run():
        return await run_full_eval.run_evaluation(mode, output, cases_path=cases)

    if outcome == "normal":
        result = await run()
        assert result["total"] == 2
    elif outcome == "cancel":
        task = asyncio.create_task(run())
        await asyncio.wait_for(cancellation_ready.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(IsADirectoryError):
            await run()
    assert used and all(owner is not application for owner in used)
    assert len(created) == 1 and all(owner is created[0] for owner in used)
    assert created[0].close_count == 1 and created[0].model.closed
    assert application.close_count == 0 and not application.model.closed
    assert legal_research.build_law_search_service is original_factory
