"""错误诊断只输出固定类别，未知异常类名不能成为数据外泄通道。"""

import pytest

from app.infrastructure.llm.config import RerankerConfig
from app.infrastructure.llm.rerankers import CrossEncoderReranker, QwenReranker
from app.knowledge.retrieval.config import RetrievalConfig
from app.knowledge.retrieval.models import (
    ChannelResult,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalScope,
    StageStatus,
)
from app.knowledge.retrieval.pipeline import RetrievalPipeline

SECRET = "13800138000"


def unknown_error(name=f"LeakError_{SECRET}"):
    return type(name, (Exception,), {})(f"private prompt {SECRET}")


@pytest.mark.parametrize("name", [f"LeakError_{SECRET}", "PrivatePromptError", "MissingCollection", "", " ValueError "])
def test_stage_status_rejects_unknown_error_names_including_plain_identifiers(name):
    status = StageStatus("failed", name, True)
    assert status.error_type == "OtherError"
    assert status.state == "failed" and status.degraded is True


@pytest.mark.parametrize(
    "name",
    [
        "ValueError",
        "RuntimeError",
        "TimeoutError",
        "OSError",
        "FileNotFoundError",
        "TypeError",
        "ConnectError",
        "ReadTimeout",
        "HTTPStatusError",
        "SessionBudgetExceeded",
        "RerankerBusyError",
        "RerankerClosedError",
        "OtherError",
        None,
    ],
)
def test_stage_status_keeps_fixed_diagnostic_categories(name):
    assert StageStatus("failed", name, True).error_type == name


@pytest.mark.parametrize("value", [17, {"message": SECRET}, object()])
def test_stage_status_does_not_serialize_unexpected_error_type_values(value):
    assert StageStatus("failed", value, True).error_type == "OtherError"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["vector", "bm25", "hyde_generate", "hyde_retrieve", "rerank", "external_status"])
async def test_pipeline_sanitizes_each_failed_stage_without_changing_dependency_semantics(phase):
    class Retriever:
        def __init__(self, name):
            self.name = name

        async def retrieve(self, request):
            if phase == self.name or (phase == "hyde_retrieve" and request.query == "hypothesis"):
                raise unknown_error()
            if phase == "external_status" and self.name == "vector":
                return ChannelResult("vector", (), {}, StageStatus("failed", f"LeakError_{SECRET}", True))
            candidate = RetrievalCandidate(self.name, "fixture", "stored body", {})
            return ChannelResult(self.name, (candidate,), {}, StageStatus("executed"))

    class Expander:
        async def expand(self, query, *, timeout_s):
            if phase == "hyde_generate":
                raise unknown_error("PrivatePromptError")
            return "hypothesis"

    class Scorer:
        async def score(self, query, documents, *, timeout_s):
            if phase == "rerank":
                raise unknown_error()
            return [0.5] * len(documents)

    pipeline = RetrievalPipeline(
        Retriever("vector"),
        bm25=Retriever("bm25"),
        expander=Expander(),
        reranker=Scorer(),
        config=RetrievalConfig(hyde=True, rerank=True),
    )
    result = await pipeline.search(RetrievalRequest("query", RetrievalScope("fixture", "alice", True), 20, 5))
    stage = "hyde" if phase.startswith("hyde_") else "vector" if phase == "external_status" else phase
    assert result.retrieval_status[stage] == "failed"
    assert result.retrieval_status[f"{stage}_error"] == "OtherError"
    assert result.retrieval_status["degraded"] is True
    assert result.dependency_failed is (phase in {"vector", "external_status"})
    assert SECRET not in str(result.retrieval_status)
    assert "PrivatePromptError" not in str(result.retrieval_status)


@pytest.mark.parametrize("scorer_type", [QwenReranker, CrossEncoderReranker])
def test_readiness_sanitizes_direct_load_error_strings(scorer_type):
    scorer = scorer_type(RerankerConfig())
    try:
        scorer._set_load_status("error", f"LeakError_{SECRET}")
        assert scorer.readiness()["error"] == "OtherError"
        assert scorer.readiness()["status"] == "error"
    finally:
        scorer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scorer_type", [QwenReranker, CrossEncoderReranker])
async def test_real_model_load_failure_readiness_sanitizes_exception_class(scorer_type, monkeypatch):
    scorer = scorer_type(RerankerConfig())

    def fail_device(requested):
        raise unknown_error()

    monkeypatch.setattr(scorer, "resolve_device", fail_device)
    try:
        with pytest.raises(Exception) as caught:
            await scorer.score("query", ["body"], timeout_s=1)
        assert SECRET in type(caught.value).__name__
        assert scorer.readiness()["error"] == "OtherError"
        assert SECRET not in str(scorer.readiness())
    finally:
        scorer.close()
