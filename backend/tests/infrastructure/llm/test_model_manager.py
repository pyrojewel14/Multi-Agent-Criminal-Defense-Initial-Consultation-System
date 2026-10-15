"""显式配置和唯一资源所有者的行为边界。"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from app.infrastructure.llm.config import ChatConfig, EmbeddingConfig, RerankerConfig
from app.infrastructure.llm.manager import ModelManager, get_model_manager, set_model_manager


def test_explicit_config_defaults_snapshot_and_secrets(monkeypatch):
    values = {"LLM_TYPE": "OLLAMA", "OLLAMA_MODEL_NAME": "one", "OLLAMA_BASE_URL": "http://one"}
    chat = ChatConfig.from_env(values, temperature=0, top_p=0)
    values["OLLAMA_MODEL_NAME"] = "two"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ambient")
    assert (chat.model_name, chat.base_url, chat.temperature, chat.top_p) == ("one", "http://one", 0, 0)
    assert ChatConfig.from_env({"ALIYUN_ACCESS_KEY_SECRET": "secret"}).model_name == "qwen3-max"
    assert ChatConfig.from_env({"LLM_TYPE": "OLLAMA"}).model_name == "qwen3.5:0.8b"
    assert EmbeddingConfig.from_env({}).base_url == "http://localhost:11434"
    cloud = EmbeddingConfig.from_env(
        {
            "EMBED_MODEL_TYPE": "ALIYUN",
            "DASHSCOPE_API_KEY": "secret",
            "ALIYUN_ACCESS_KEY_SECRET": "other",
            "DASHSCOPE_BASE_URL": "http://first",
            "ALIYUN_BASE_URL": "http://second",
        }
    )
    assert (cloud.api_key, cloud.base_url) == ("secret", "http://first")
    assert "secret" not in repr(cloud)
    with pytest.raises(FrozenInstanceError):
        chat.model_name = "changed"


def test_chat_cache_separates_full_identity_and_concurrent_creation():
    count = []

    class Factory:
        def create(self, config):
            count.append(config)
            return object()

    manager = ModelManager(chat_factory=Factory())
    config = ChatConfig.from_env({"LLM_TYPE": "OLLAMA"})
    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(lambda _: manager.get_chat(config), range(20)))
    assert all(model is models[0] for model in models)
    assert len(count) == 1
    for changed in (
        replace(config, base_url="http://other"),
        replace(config, model_name="other"),
        replace(config, api_key="different"),
        replace(config, streaming=True),
        replace(config, temperature=0),
        replace(config, top_p=0),
    ):
        assert manager.get_chat(changed) is not models[0]
    manager.close()
    assert manager.closed
    with pytest.raises(RuntimeError):
        manager.get_chat(config)


@pytest.mark.asyncio
async def test_embedding_is_lazy_shared_by_explicit_factory_identity_and_snapshot():
    created = []

    class Embed:
        def embed_query(self, text):
            return [1.0]

        async def aembed_query(self, text):
            return [2.0]

        async def aembed_documents(self, texts):
            return [[3.0] for _ in texts]

        def close(self):
            created.append("closed")

    def factory():
        created.append("created")
        return Embed()

    manager = ModelManager()
    config = EmbeddingConfig.from_env({})
    proxy = manager.get_embedding(config, factory=factory)
    assert manager.get_embedding(replace(config, model_name="other"), factory=factory) is proxy
    assert created == []
    assert "secret" not in repr(manager.readiness())
    assert created == []
    assert proxy.embed_query("one") == [1.0]
    assert await proxy.aembed_query("two") == [2.0]
    assert await proxy.aembed_documents(["three"]) == [[3.0]]
    assert created == ["created"]
    await manager.aclose()
    assert created == ["created", "closed"]
    with pytest.raises(RuntimeError):
        proxy.embed_query("late")


@pytest.mark.asyncio
async def test_close_dual_clients_deduplicates_shallow_copies_and_retries_failed_refs():
    calls = []

    class Sync:
        def close(self):
            calls.append("sync")

    class Async:
        failed = True

        async def close(self):
            calls.append("async")
            if self.failed:
                self.failed = False
                raise ValueError("controlled")

    sync, asynchronous = Sync(), Async()

    class Factory:
        def create(self, config):
            return SimpleNamespace(_client=sync, _async_client=asynchronous)

    manager = ModelManager(chat_factory=Factory())
    config = ChatConfig.from_env({"LLM_TYPE": "OLLAMA"})
    manager.get_chat(config)
    manager.get_chat(replace(config, temperature=0))
    with pytest.raises(RuntimeError, match="aclose"):
        manager.close()
    assert calls == []
    assert manager.closed
    with pytest.raises(ValueError):
        await manager.aclose()
    assert calls == ["sync", "async"]
    await manager.aclose()
    assert calls == ["sync", "async", "async"]
    await manager.aclose()
    assert calls == ["sync", "async", "async"]


@pytest.mark.asyncio
async def test_close_handles_awaitable_return_and_continues_after_failure():
    closed = []

    class Bad:
        def close(self):
            raise ValueError("controlled")

    class Awaitable:
        def close(self):
            async def release():
                closed.append("awaited")

            return release()

    class Factory:
        def create(self, config):
            return Bad() if config.temperature is None else Awaitable()

    manager = ModelManager(chat_factory=Factory())
    config = ChatConfig.from_env({"LLM_TYPE": "OLLAMA"})
    manager.get_chat(config)
    manager.get_chat(replace(config, temperature=0))
    with pytest.raises(ValueError):
        await manager.aclose()
    assert closed == ["awaited"]


def test_creation_close_race_does_not_leave_an_unowned_model():
    started, release = threading.Event(), threading.Event()
    closed = []

    class Model:
        def close(self):
            closed.append(True)

    class Factory:
        def create(self, config):
            started.set()
            assert release.wait(3)
            return Model()

    manager = ModelManager(chat_factory=Factory())
    config = ChatConfig.from_env({"LLM_TYPE": "OLLAMA"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        creation = pool.submit(manager.get_chat, config)
        assert started.wait(3)
        shutdown = pool.submit(manager.close)
        release.set()
        creation.result()
        shutdown.result()
    assert closed == [True]
    with pytest.raises(RuntimeError):
        manager.get_chat(config)


@pytest.mark.asyncio
async def test_reranker_identity_readiness_and_close_are_lightweight():
    manager = ModelManager()
    config = RerankerConfig(local_path="missing")
    cross = manager.get_reranker(config, kind="cross_encoder")
    qwen = manager.get_reranker(config)
    assert manager.get_reranker(replace(config)) is qwen
    assert cross is not qwen
    assert manager.reranker_readiness()["type"] == "causal_lm"
    assert manager.reranker_readiness()["status"] == "not_loaded"
    with pytest.raises(ValueError):
        manager.get_reranker(config, kind="unknown")
    await manager.aclose()
    assert cross.readiness()["closed"] and qwen.readiness()["closed"]


def test_default_provider_replaces_explicitly_without_reopening():
    old = get_model_manager()
    replacement = ModelManager()
    try:
        set_model_manager(replacement)
        replacement.close()
        assert get_model_manager() is replacement
        assert get_model_manager().closed
    finally:
        set_model_manager(old)


def test_missing_cloud_credentials_remain_lazy_and_cannot_leak_ambient_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ambient-secret")
    chat = ChatConfig.from_env({"LLM_TYPE": "ALIYUN"})
    embedding = EmbeddingConfig.from_env({"EMBED_MODEL_TYPE": "ALIYUN"})
    assert chat.api_key is None and embedding.api_key is None
    manager = ModelManager()
    proxy = manager.get_embedding(embedding)
    assert manager.readiness()["embedding_instances"] == 0
    with pytest.raises(ValueError, match="API Key"):
        proxy.embed_query("public")
    with pytest.raises(ValueError, match="API Key"):
        manager.get_chat(chat)
    manager.close()


@pytest.mark.asyncio
async def test_real_sdk_models_release_shared_sync_and_async_pools():
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    chat = ChatOllama(model="public", base_url="http://localhost:11434", client_kwargs={"trust_env": False})
    embedding = OllamaEmbeddings(model="public", base_url="http://localhost:11434", client_kwargs={"trust_env": False})

    class Factory:
        def create(self, config):
            return chat.model_copy()

    manager = ModelManager(chat_factory=Factory())
    config = ChatConfig.from_env({"LLM_TYPE": "OLLAMA"})
    manager.get_chat(config)
    manager.get_chat(replace(config, temperature=0))
    proxy = manager.get_embedding(EmbeddingConfig.from_env({}), factory=lambda: embedding)
    proxy._get_model()
    await manager.aclose()
    assert chat._client._client.is_closed and chat._async_client._client.is_closed
    assert embedding._client._client.is_closed and embedding._async_client._client.is_closed


@pytest.mark.asyncio
async def test_cancelled_close_keeps_inflight_ownership_for_retry():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    class Model:
        async def aclose(self):
            calls.append("start")
            entered.set()
            await release.wait()
            calls.append("done")

    class Factory:
        def create(self, config):
            return Model()

    manager = ModelManager(chat_factory=Factory())
    manager.get_chat(ChatConfig.from_env({"LLM_TYPE": "OLLAMA"}))
    task = asyncio.create_task(manager.aclose())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.closed
    assert manager.readiness()["chat_instances"] == 1
    release.set()
    await manager.aclose()
    assert calls == ["start", "done"]
    assert manager.readiness()["chat_instances"] == 0


@pytest.mark.asyncio
async def test_cancelled_async_action_can_be_retried_without_reopening_manager():
    calls = []

    class Model:
        async def aclose(self):
            calls.append(True)
            if len(calls) == 1:
                raise asyncio.CancelledError

    class Factory:
        def create(self, config):
            return Model()

    manager = ModelManager(chat_factory=Factory())
    manager.get_chat(ChatConfig.from_env({"LLM_TYPE": "OLLAMA"}))
    with pytest.raises(asyncio.CancelledError):
        await manager.aclose()
    await manager.aclose()
    assert calls == [True, True]
    assert manager.closed and manager.readiness()["chat_instances"] == 0


@pytest.mark.asyncio
async def test_cancelled_close_worker_does_not_repeat_running_sync_action():
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Model:
        def close(self):
            calls.append("start")
            entered.set()
            assert release.wait(3)
            calls.append("done")

    class Factory:
        def create(self, config):
            return Model()

    manager = ModelManager(chat_factory=Factory())
    manager.get_chat(ChatConfig.from_env({"LLM_TYPE": "OLLAMA"}))
    task = asyncio.create_task(manager.aclose())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        pending = next(iter(manager._pending_closes.values()))
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await manager.aclose()
        assert calls == ["start", "done"]
    finally:
        release.set()
