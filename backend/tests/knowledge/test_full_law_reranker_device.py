"""全量重排器的设备选择、配置切换与失败回退。"""

import pytest
import torch

from app.knowledge import full_law_retrieval as retrieval
from app.knowledge.full_law_reranker import QwenFullReranker


@pytest.mark.parametrize(
    "requested,cuda,mps,expected",
    [
        ("auto", True, True, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
        ("mps", True, True, "mps"),
        ("cuda", True, False, "cuda"),
    ],
)
def test_device_selection_respects_explicit_choice(monkeypatch, requested, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert QwenFullReranker.resolve_device(requested) == expected


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_unavailable_explicit_device_is_not_silently_replaced(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        QwenFullReranker.resolve_device(requested)


@pytest.mark.parametrize("requested", ["metal", "mps:0"])
def test_invalid_device_rejected_before_allocating_model(requested):
    with pytest.raises(ValueError, match="device"):
        QwenFullReranker("unused", 512, "instruction", device=requested)


@pytest.mark.asyncio
async def test_device_setting_changes_the_cached_scorer(monkeypatch):
    import app.knowledge.full_law_reranker as module

    class LocalScorer:
        def __init__(self, path, length, instruction, device="auto"):
            self.device = device

        async def score(self, query, documents, timeout):
            return [{"cpu": 0.25, "mps": 0.75, "auto": 0.1}[self.device]] * len(documents)

    monkeypatch.setattr(module, "QwenFullReranker", LocalScorer)
    retrieval._scorer.cache_clear()
    try:
        monkeypatch.setenv("LAW_FULL_RERANK_DEVICE", "cpu")
        assert await retrieval.score_candidates("query", ["document"], 1) == [0.25]
        monkeypatch.setenv("LAW_FULL_RERANK_DEVICE", "mps")
        assert await retrieval.score_candidates("query", ["document"], 1) == [0.75]
    finally:
        retrieval._scorer.cache_clear()


@pytest.mark.asyncio
async def test_unavailable_mps_preserves_fusion_order_and_records_failure(monkeypatch):
    monkeypatch.setenv("LAW_FULL_RERANK_DEVICE", "mps")
    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "true")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    laws = [{"content": "候选一"}, {"content": "候选二"}]
    status = {}
    scorer = QwenFullReranker("unused", 512, "instruction", device="mps")
    monkeypatch.setattr(retrieval, "_scorer", lambda *args: scorer)
    try:
        result = await retrieval.finalize_candidates("query", laws, status, retrieval.RetrievalConfig.from_env())
        assert [law["content"] for law in result] == ["候选一", "候选二"]
        assert status["rerank"] == "failed" and status["degraded"] is True
        assert status["rerank_error"] == "RuntimeError"
    finally:
        scorer.close()
