"""原文档客户端生命周期约束随模型所有权迁入 infrastructure。"""

import pytest

from app.infrastructure.llm.config import EmbeddingConfig
from app.infrastructure.llm.factory import EmbedModelFactory
from app.infrastructure.llm.manager import ModelManager


def test_sync_close_releases_both_real_ollama_http_clients_without_network():
    resources = ModelManager()
    proxy = resources.get_embedding(
        EmbeddingConfig.from_env({"EMBED_MODEL_TYPE": "OLLAMA", "OLLAMA_BASE_URL": "http://127.0.0.1:1"})
    )
    model = proxy._get_model()
    assert not model._client._client.is_closed
    assert not model._async_client._client.is_closed
    try:
        resources.close()
        assert model._client._client.is_closed
        assert model._async_client._client.is_closed
    finally:
        # Also keep the RED run from leaking the actual clients.
        import asyncio

        model._client.close()
        asyncio.run(model._async_client.close())


@pytest.mark.asyncio
async def test_sync_close_in_running_loop_requires_awaited_async_close_without_losing_clients():
    resources = ModelManager()
    proxy = resources.get_embedding(
        EmbeddingConfig.from_env({"EMBED_MODEL_TYPE": "OLLAMA", "OLLAMA_BASE_URL": "http://127.0.0.1:1"})
    )
    model = proxy._get_model()
    try:
        with pytest.raises(RuntimeError, match="aclose"):
            resources.close()
        assert not model._client._client.is_closed
        assert not model._async_client._client.is_closed
        await resources.aclose()
        await resources.aclose()
        assert model._client._client.is_closed
        assert model._async_client._client.is_closed
    finally:
        model._client.close()
        await model._async_client.close()


@pytest.mark.parametrize("endpoint_setting", ["DASHSCOPE_BASE_URL", "ALIYUN_BASE_URL"])
def test_aliyun_endpoint_snapshot_ignores_later_environment(endpoint_setting, monkeypatch):
    snapshot = {
        "EMBED_MODEL_TYPE": "ALIYUN",
        "DASHSCOPE_API_KEY": "fake-snapshot-key",
        endpoint_setting: "https://configured.example/v1",
    }
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://ambient.example/v1")
    model = EmbedModelFactory().create(EmbeddingConfig.from_env(snapshot))
    try:
        assert str(model.client.base_url) == "https://configured.example/v1/"
    finally:
        model.close()


def test_aliyun_missing_snapshot_key_cannot_borrow_ambient_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-ambient-key")
    with pytest.raises(ValueError, match="API Key"):
        EmbedModelFactory().create(EmbeddingConfig.from_env({"EMBED_MODEL_TYPE": "ALIYUN"}))
