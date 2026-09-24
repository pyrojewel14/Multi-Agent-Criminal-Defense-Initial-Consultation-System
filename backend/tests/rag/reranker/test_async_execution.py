"""重排序器事件循环隔离与并发控制回归测试。"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app.rag.reranker.base import RerankerConfig
from app.rag.reranker.cross_encoder import CrossEncoderReranker


class _Scores:
    """提供 CrossEncoder.predict 返回值所需的最小接口。"""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values

    def __len__(self):
        return len(self._values)


@pytest.mark.asyncio
async def test_slow_model_load_does_not_block_event_loop_heartbeat(tmp_path):
    """缓慢的模型初始化不得延迟事件循环中的轻量协程。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    reranker = CrossEncoderReranker(RerankerConfig(local_path=str(tmp_path), device="cpu"))
    model = MagicMock()
    model.predict.return_value.tolist.return_value = [0.8]

    def slow_model_load(*args, **kwargs):
        time.sleep(0.15)
        return model

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    with patch("app.rag.reranker.cross_encoder.CrossEncoder", side_effect=slow_model_load):
        rerank_task = asyncio.create_task(reranker.rerank("query", ["document"]))
        await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        heartbeat_elapsed = loop.time() - started_at
        result = await rerank_task

    assert heartbeat_elapsed < 0.08
    assert result["success"] is True


@pytest.mark.asyncio
async def test_concurrent_reranks_initialize_model_only_once(tmp_path):
    """并发首次请求只能触发一次模型初始化。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    reranker = CrossEncoderReranker(
        RerankerConfig(local_path=str(tmp_path), device="cpu", max_concurrency=2)
    )
    model = MagicMock()
    model.predict.return_value = _Scores([0.8])
    load_calls = 0
    load_calls_lock = threading.Lock()

    def slow_model_load(*args, **kwargs):
        nonlocal load_calls
        with load_calls_lock:
            load_calls += 1
        time.sleep(0.05)
        return model

    with patch("app.rag.reranker.cross_encoder.CrossEncoder", side_effect=slow_model_load):
        await asyncio.gather(
            reranker.rerank("query-1", ["document-1"]),
            reranker.rerank("query-2", ["document-2"]),
        )

    assert load_calls == 1


@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_model_inference(tmp_path):
    """并发推理数不得超过配置的 semaphore 上限。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    reranker = CrossEncoderReranker(
        RerankerConfig(local_path=str(tmp_path), device="cpu", max_concurrency=2)
    )
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    class SlowModel:
        def predict(self, pairs, batch_size):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.04)
            with active_lock:
                active -= 1
            return _Scores([0.5])

    reranker._model = SlowModel()
    await asyncio.gather(*(reranker.rerank(f"query-{index}", ["document"]) for index in range(5)))

    assert max_active == 2


@pytest.mark.asyncio
async def test_model_load_exception_is_reported_by_readiness(tmp_path):
    """模型初始化异常必须向调用方抛出并保留在 readiness 状态中。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    reranker = CrossEncoderReranker(RerankerConfig(local_path=str(tmp_path), device="cpu"))

    with patch("app.rag.reranker.cross_encoder.CrossEncoder", side_effect=RuntimeError("broken weights")):
        with pytest.raises(RuntimeError, match="broken weights"):
            await reranker._load_model()

    readiness = getattr(reranker, "readiness", lambda: {"status": "missing"})()
    assert readiness["status"] == "error"
    assert "broken weights" in readiness["error"]


@pytest.mark.asyncio
async def test_readiness_exposes_lazy_loading_lifecycle(tmp_path):
    """readiness 应区分尚未加载、加载中和可用状态。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    reranker = CrossEncoderReranker(RerankerConfig(local_path=str(tmp_path), device="cpu"))
    load_started = threading.Event()
    allow_load_to_finish = threading.Event()
    model = MagicMock()

    def controlled_model_load(*args, **kwargs):
        load_started.set()
        allow_load_to_finish.wait(timeout=1)
        return model

    assert reranker.readiness()["status"] == "not_loaded"
    with patch("app.rag.reranker.cross_encoder.CrossEncoder", side_effect=controlled_model_load):
        load_task = asyncio.create_task(reranker._load_model())
        assert await asyncio.to_thread(load_started.wait, 1)
        assert reranker.readiness()["status"] == "loading"
        allow_load_to_finish.set()
        await load_task

    assert reranker.readiness()["status"] == "ready"
    assert reranker.readiness()["available"] is True
