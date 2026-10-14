"""Real executor coordination; only model I/O is replaced."""

import asyncio
import threading

import pytest

from app.knowledge.retrieval.config import RerankerConfig
from app.knowledge.retrieval.rerankers import (
    CrossEncoderReranker,
    QwenReranker,
    RerankerBusyError,
    RerankerClosedError,
)


async def eventually(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.005)
    assert predicate()


@pytest.mark.parametrize("scorer_type", [QwenReranker, CrossEncoderReranker])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.asyncio
async def test_timeout_or_cancel_keeps_actual_worker_busy_and_recovers(scorer_type, cancel):
    scorer = scorer_type(RerankerConfig(max_concurrency=8))
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    runs = []
    loop_errors = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: loop_errors.append(context))

    def blocked(query, documents):
        runs.append(query)
        started.set()
        try:
            assert release.wait(3)
            raise RuntimeError("controlled background failure")
        finally:
            finished.set()

    scorer._score_sync = blocked
    try:
        task = asyncio.create_task(scorer.score("first", ["doc"], timeout_s=1 if cancel else 0.01))
        await eventually(started.is_set)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else asyncio.TimeoutError):
            await task
        assert scorer.readiness()["busy"] is True
        for _ in range(5):
            with pytest.raises(RerankerBusyError):
                await scorer.score("must not queue", ["doc"], timeout_s=1)
        assert runs == ["first"]
        release.set()
        await eventually(lambda: finished.is_set() and not scorer.readiness()["busy"])
        await asyncio.sleep(0.02)
        scorer._score_sync = lambda query, documents: [0.75]
        assert await scorer.score("recovery", ["doc"], timeout_s=1) == [0.75]
        assert loop_errors == []
    finally:
        release.set()
        scorer.close()
        loop.set_exception_handler(previous_handler)


@pytest.mark.parametrize("scorer_type", [QwenReranker, CrossEncoderReranker])
def test_readiness_does_not_load_and_close_is_idempotent(scorer_type):
    scorer = scorer_type(RerankerConfig(local_path="missing-model"))
    try:
        assert scorer.readiness() == {
            "status": "not_loaded", "available": False, "model": scorer.config.model_name,
            "error": None, "busy": False, "closed": False,
        }
        assert scorer._model is None
        scorer.close()
        scorer.close()
        assert scorer.readiness()["status"] == "closed"
        assert scorer.readiness()["available"] is False
    finally:
        scorer.close()


@pytest.mark.parametrize("scorer_type", [QwenReranker, CrossEncoderReranker])
@pytest.mark.asyncio
async def test_close_during_work_releases_weights_after_actual_completion(scorer_type):
    scorer = scorer_type(RerankerConfig())
    started, release = threading.Event(), threading.Event()

    def blocked(query, documents):
        scorer._model = object()
        started.set()
        assert release.wait(3)
        return [0.5]

    scorer._score_sync = blocked
    task = asyncio.create_task(scorer.score("query", ["doc"], timeout_s=1))
    try:
        await eventually(started.is_set)
        scorer.close()
        scorer.close()
        assert scorer._model is not None
        with pytest.raises(RerankerClosedError):
            await scorer.score("query", ["doc"], timeout_s=1)
        release.set()
        assert await task == [0.5]
        await eventually(lambda: scorer._model is None)
        assert scorer.readiness()["closed"] is True
    finally:
        release.set()
        scorer.close()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
@pytest.mark.asyncio
async def test_invalid_timeout_does_not_submit_inference(timeout):
    scorer = QwenReranker(RerankerConfig())
    try:
        with pytest.raises(ValueError, match="timeout"):
            await scorer.score("query", ["doc"], timeout_s=timeout)
        assert scorer.readiness()["busy"] is False
    finally:
        scorer.close()


@pytest.mark.asyncio
async def test_empty_documents_do_not_load_and_closed_instance_still_fails():
    scorer = QwenReranker(RerankerConfig())
    assert await scorer.score("query", [], timeout_s=1) == []
    assert scorer.readiness()["status"] == "not_loaded"
    scorer.close()
    with pytest.raises(RerankerClosedError):
        await scorer.score("query", [], timeout_s=1)
