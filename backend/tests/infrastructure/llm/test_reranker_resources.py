"""统一评分器入口保持类型隔离、配置复用与懒加载。"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from app.infrastructure.llm.config import RerankerConfig
from app.infrastructure.llm.manager import ModelManager
from app.infrastructure.llm.rerankers import CrossEncoderReranker, QwenReranker


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, QwenReranker),
        ({"kind": "causal_lm"}, QwenReranker),
        ({"kind": "cross_encoder"}, CrossEncoderReranker),
    ],
)
def test_scorer_selects_type_without_loading_weights(monkeypatch, kwargs, expected):
    def no_load(self):
        pytest.fail("creating or inspecting a scorer must not load weights")

    monkeypatch.setattr(QwenReranker, "_load_model_sync", no_load)
    monkeypatch.setattr(CrossEncoderReranker, "_load_model_sync", no_load)
    resources = ModelManager()
    try:
        scorer = resources.get_reranker(RerankerConfig(local_path="missing-model"), **kwargs)
        assert isinstance(scorer, expected)
        assert scorer.readiness()["status"] == "not_loaded"
        assert resources.reranker_readiness()["status"] == "not_loaded"
    finally:
        resources.close()
    assert scorer.readiness()["closed"] is True


@pytest.mark.parametrize("kind", ["causal_lm", "cross_encoder"])
def test_equal_config_reuses_scorer_but_different_config_does_not(kind):
    resources = ModelManager()
    config = RerankerConfig(local_path="missing-model", device="cpu")
    try:
        scorer = resources.get_reranker(config, kind=kind)
        assert resources.get_reranker(replace(config), kind=kind) is scorer
        assert resources.get_reranker(replace(config, local_path="other-model"), kind=kind) is not scorer
    finally:
        resources.close()


def test_default_and_explicit_qwen_share_cache_but_cross_encoder_is_independent():
    resources = ModelManager()
    config = RerankerConfig(local_path="missing-model")
    try:
        qwen = resources.get_reranker(config)
        cross_encoder = resources.get_reranker(config, kind="cross_encoder")
        assert resources.get_reranker(config, kind="causal_lm") is qwen
        assert resources.get_reranker(config, kind="cross_encoder") is cross_encoder
        assert qwen is not cross_encoder
        assert isinstance(qwen, QwenReranker) and isinstance(cross_encoder, CrossEncoderReranker)
    finally:
        resources.close()
    assert qwen.readiness()["closed"] and cross_encoder.readiness()["closed"]


def test_single_cache_tracks_every_scorer_for_resource_cleanup():
    resources = ModelManager()
    config = RerankerConfig(local_path="missing-model")
    try:
        qwen = resources.get_reranker(config)
        cross_encoder = resources.get_reranker(config, kind="cross_encoder")
        assert set(resources._rerankers.values()) == {qwen, cross_encoder}
    finally:
        resources.close()
    assert resources._rerankers == {}


@pytest.mark.parametrize(
    "qwen_ready,cross_ready,expected_kind",
    [
        (False, False, "causal_lm"),
        (False, True, "causal_lm"),
        (True, False, "cross_encoder"),
        (True, True, "cross_encoder"),
    ],
)
def test_readiness_preserves_type_priority_when_cross_encoder_is_created_first(qwen_ready, cross_ready, expected_kind):
    resources = ModelManager()
    config = RerankerConfig(local_path="missing-model")
    try:
        cross_encoder = resources.get_reranker(config, kind="cross_encoder")
        qwen = resources.get_reranker(config)
        if qwen_ready:
            qwen._set_load_status("ready")
        if cross_ready:
            cross_encoder._set_load_status("ready")
        assert resources.reranker_readiness()["type"] == expected_kind
    finally:
        resources.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_close", [False, True])
async def test_both_close_paths_clear_cache_and_close_once_outside_lock_in_original_order(monkeypatch, async_close):
    resources = ModelManager()
    config = RerankerConfig(local_path="missing-model")
    cross_encoder = resources.get_reranker(config, kind="cross_encoder")
    qwen = resources.get_reranker(config)
    closed = []

    def tracked_close(kind, original):
        def close():
            assert resources._lock.acquire(blocking=False)
            resources._lock.release()
            closed.append(kind)
            original()

        return close

    monkeypatch.setattr(qwen, "close", tracked_close("causal_lm", qwen.close))
    monkeypatch.setattr(cross_encoder, "close", tracked_close("cross_encoder", cross_encoder.close))
    try:
        for _ in range(2):
            if async_close:
                await resources.aclose()
            else:
                resources.close()
        assert closed == ["causal_lm", "cross_encoder"]
        assert qwen.readiness()["closed"] and cross_encoder.readiness()["closed"]
    finally:
        resources.close()


@pytest.mark.parametrize("kind", ["unknown", "", "CAUSAL_LM"])
def test_unknown_scorer_kind_is_rejected_without_creating_resources(monkeypatch, kind):
    def no_create(config):
        pytest.fail("invalid kind must not create a scorer")

    monkeypatch.setattr("app.infrastructure.llm.rerankers.QwenReranker", no_create)
    monkeypatch.setattr("app.infrastructure.llm.rerankers.CrossEncoderReranker", no_create)
    resources = ModelManager()
    try:
        with pytest.raises(ValueError, match="^unknown reranker type$"):
            resources.get_reranker(RerankerConfig(), kind=kind)
        assert resources.reranker_readiness()["status"] == "not_loaded"
    finally:
        resources.close()


def test_concurrent_requests_create_only_one_instance_per_kind(monkeypatch):
    created = []

    def constructor(kind, cls):
        def create(config):
            time.sleep(0.01)
            created.append(kind)
            return cls(config)

        return create

    monkeypatch.setattr("app.infrastructure.llm.rerankers.QwenReranker", constructor("causal_lm", QwenReranker))
    monkeypatch.setattr(
        "app.infrastructure.llm.rerankers.CrossEncoderReranker", constructor("cross_encoder", CrossEncoderReranker)
    )
    resources = ModelManager()
    config = RerankerConfig(local_path="missing-model")
    barrier = threading.Barrier(8)

    def get_scorer(kind):
        barrier.wait(timeout=5)
        return kind, resources.get_reranker(config, kind=kind)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(get_scorer, ["causal_lm", "cross_encoder"] * 4))
        for kind in ("causal_lm", "cross_encoder"):
            assert len({id(scorer) for selected, scorer in results if selected == kind}) == 1
            assert created.count(kind) == 1
        assert results[0][1] is not results[1][1]
    finally:
        resources.close()
