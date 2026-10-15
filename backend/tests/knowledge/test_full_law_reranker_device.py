"""统一重排器组装的设备配置隔离与不可用设备降级。"""

from dataclasses import replace

import pytest
import torch

from app.infrastructure.llm.config import RerankerConfig
from app.infrastructure.llm.manager import ModelManager
from app.infrastructure.llm.rerankers import QwenReranker
from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.models import (
    ChannelResult,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalScope,
    StageStatus,
)
from app.knowledge.retrieval.pipeline import RetrievalPipeline


@pytest.mark.parametrize("requested", ["metal", "mps:0"])
def test_invalid_device_rejected_before_allocating_model(requested):
    with pytest.raises(ValueError, match="device"):
        RerankerConfig(device=requested)


def test_device_setting_changes_cached_scorer_without_loading_model():
    resources = ModelManager()
    config = RerankerConfig(device="cpu")
    try:
        cpu = resources.get_reranker(config)
        mps = resources.get_reranker(replace(config, device="mps"))
        assert cpu is resources.get_reranker(config)
        assert cpu is not mps
        assert cpu.config.device == "cpu" and mps.config.device == "mps"
        assert cpu.readiness()["status"] == mps.readiness()["status"] == "not_loaded"
    finally:
        resources.close()


@pytest.mark.asyncio
async def test_unavailable_mps_preserves_fusion_order_and_records_failure(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    scorer = QwenReranker(RerankerConfig(local_path="unused", device="mps"))

    class Vector:
        async def retrieve(self, request):
            candidates = tuple(
                RetrievalCandidate(str(i), "law", body, {}) for i, body in enumerate(("候选一", "候选二"))
            )
            return ChannelResult("vector", candidates, {}, StageStatus("executed"))

    pipeline = RetrievalPipeline(Vector(), reranker=scorer, config=RetrievalConfig(rerank=True))
    try:
        result = await pipeline.search(RetrievalRequest("query", RetrievalScope("law", "alice", True), 20, 5))
        assert [candidate.content for candidate in result.candidates] == ["候选一", "候选二"]
        assert result.retrieval_status["rerank"] == "failed"
        assert result.retrieval_status["degraded"] is True
        assert result.retrieval_status["rerank_error"] == "RuntimeError"
    finally:
        scorer.close()
