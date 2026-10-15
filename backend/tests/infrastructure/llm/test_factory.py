"""Unit tests for app/infrastructure/llm/factory.py.

Tests cover:
- DashScopeEmbeddingsWrapper.embed_documents / embed_query
- Explicit Chat/Embedding/Reranker construction
- Provider kwargs and environment snapshot isolation
- Model caches and lifecycle are covered by manager tests
"""

from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.llm.config import ChatConfig, EmbeddingConfig, RerankerConfig
from app.infrastructure.llm.factory import (
    ChatModelFactory,
    DashScopeEmbeddingsWrapper,
    EmbedModelFactory,
    RerankerFactory,
    _ollama_client_kwargs,
)

# ---------------------------------------------------------------------------
# DashScopeEmbeddingsWrapper
# ---------------------------------------------------------------------------


class _FakeEmbedItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbedResponse:
    def __init__(self, embeddings):
        self.data = [_FakeEmbedItem(e) for e in embeddings]


class TestDashScopeEmbeddingsWrapper:
    def test_init_uses_api_key(self, monkeypatch):
        """Provided api_key should be used directly."""
        monkeypatch.setattr("os.getenv", lambda k, default=None: "env-key" if k == "DASHSCOPE_API_KEY" else default)
        with patch("openai.OpenAI") as mock_openai:
            DashScopeEmbeddingsWrapper(api_key="explicit-key")
        mock_openai.assert_called_once_with(
            api_key="explicit-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def test_config_resolves_env_key_before_pure_construction(self, monkeypatch):
        """Credential priority belongs to the explicit snapshot, not the constructor."""
        config = EmbeddingConfig.from_env({"EMBED_MODEL_TYPE": "ALIYUN", "DASHSCOPE_API_KEY": "env-key"})
        monkeypatch.setenv("DASHSCOPE_API_KEY", "changed")
        with patch("openai.OpenAI") as mock_openai:
            EmbedModelFactory().create(config)
        mock_openai.assert_called_once_with(
            api_key="env-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def test_init_no_api_key_raises(self, monkeypatch):
        """When no API key is available, raise ValueError."""
        monkeypatch.setattr("os.getenv", lambda k, default=None: None)
        with patch("openai.OpenAI"):
            with pytest.raises(ValueError, match="未设置 API Key"):
                DashScopeEmbeddingsWrapper()

    def test_embed_documents_single_batch(self):
        """embed_documents with <=10 texts should call API once."""
        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        wrapper.model_name = "text-embedding-v4"
        wrapper.client = MagicMock()
        wrapper.client.embeddings.create.return_value = _FakeEmbedResponse([[0.1, 0.2], [0.3, 0.4]])

        result = wrapper.embed_documents(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert wrapper.client.embeddings.create.call_count == 1

    def test_embed_documents_multiple_batches(self):
        """embed_documents with >10 texts should split into multiple batches of 10."""
        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        wrapper.model_name = "text-embedding-v4"
        wrapper.client = MagicMock()
        wrapper.client.embeddings.create.return_value = _FakeEmbedResponse([[0.0]] * 10)

        texts = [f"text{i}" for i in range(25)]
        result = wrapper.embed_documents(texts)
        # 25 / 10 = 3 batches (10 + 10 + 5)
        assert wrapper.client.embeddings.create.call_count == 3
        assert len(result) == 30  # 3 batches of 10 each

    def test_embed_documents_api_error(self):
        """When the API raises, wrap in LLMServiceException."""
        from app.errors.exceptions import LLMServiceException

        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        wrapper.model_name = "text-embedding-v4"
        wrapper.client = MagicMock()
        wrapper.client.embeddings.create.side_effect = Exception("API down")

        with pytest.raises(LLMServiceException):
            wrapper.embed_documents(["hello"])

    def test_embed_query(self):
        """embed_query should return a single embedding vector."""
        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        wrapper.model_name = "text-embedding-v4"
        wrapper.client = MagicMock()
        wrapper.client.embeddings.create.return_value = _FakeEmbedResponse([[0.5, 0.6]])

        result = wrapper.embed_query("test")
        assert result == [0.5, 0.6]

    def test_embed_query_api_error(self):
        """When the API raises, wrap in LLMServiceException."""
        from app.errors.exceptions import LLMServiceException

        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        wrapper.model_name = "text-embedding-v4"
        wrapper.client = MagicMock()
        wrapper.client.embeddings.create.side_effect = Exception("API down")

        with pytest.raises(LLMServiceException):
            wrapper.embed_query("test")

    def test_close_calls_client_close(self):
        """close() should call client.close()."""
        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        wrapper.client = MagicMock()
        client = wrapper.client
        client.close = MagicMock()
        wrapper.close()
        wrapper.close()
        client.close.assert_called_once()
        assert wrapper.client is None

    def test_close_handles_missing_client(self):
        """close() should not raise when client attribute is missing."""
        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        # Don't set wrapper.client at all
        wrapper.close()  # should not raise


@pytest.mark.parametrize("provider", ["OLLAMA", "ALIYUN"])
def test_chat_factory_preserves_explicit_constructor_parameters(provider, monkeypatch):
    config = ChatConfig.from_env(
        {
            "LLM_TYPE": provider,
            "OLLAMA_MODEL_NAME": "local",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "ALIYUN_MODEL_NAME": "cloud",
            "ALIYUN_ACCESS_KEY_SECRET": "snapshot-key",
            "ALIYUN_BASE_URL": "https://snapshot.invalid",
        },
        temperature=0,
        streaming=True,
        top_p=0,
    )
    monkeypatch.setenv("ALIYUN_ACCESS_KEY_SECRET", "ambient-key")
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "ambient-model")
    captured = []

    def constructor(**kwargs):
        captured.append(kwargs)
        return object()

    path = (
        "langchain_ollama.ChatOllama" if provider == "OLLAMA" else "langchain_community.chat_models.tongyi.ChatTongyi"
    )
    with patch(path, constructor):
        factory = ChatModelFactory()
        first, second = factory.create(config), factory.create(config)
    assert first is not second
    if provider == "OLLAMA":
        assert (
            captured
            == [
                {
                    "model": "local",
                    "base_url": "http://localhost:11434",
                    "client_kwargs": {"trust_env": False},
                    "temperature": 0,
                    "streaming": True,
                    "top_p": 0,
                }
            ]
            * 2
        )
    else:
        assert (
            captured
            == [
                {
                    "model": "cloud",
                    "api_key": "snapshot-key",
                    "base_url": "https://snapshot.invalid",
                    "temperature": 0,
                    "streaming": True,
                    "top_p": 0,
                }
            ]
            * 2
        )


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("http://localhost:11434", {"trust_env": False}),
        ("http://127.0.0.1:11434", {"trust_env": False}),
        ("http://[::1]:11434", {"trust_env": False}),
        ("https://ollama.example.test", {}),
    ],
)
def test_embedding_factory_uses_explicit_endpoint_and_loopback_proxy_boundary(base_url, expected, monkeypatch):
    config = EmbeddingConfig.from_env({"OLLAMA_BASE_URL": base_url, "TEXT_EMBEDDING_MODEL_NAME": "snapshot"})
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ambient")
    captured = []
    with patch("langchain_ollama.OllamaEmbeddings", lambda **kwargs: captured.append(kwargs) or object()):
        first = EmbedModelFactory().create(config)
        second = EmbedModelFactory().create(config)
    assert first is not second
    assert captured == [{"model": "snapshot", "base_url": base_url, "client_kwargs": expected}] * 2


def test_remote_ollama_keeps_environment_proxy_support():
    assert _ollama_client_kwargs("https://ollama.example.com") == {}


def test_invalid_provider_cannot_silently_construct_another_backend():
    with pytest.raises(ValueError):
        ChatModelFactory().create(ChatConfig(provider="INVALID"))
    with pytest.raises(ValueError):
        EmbedModelFactory().create(EmbeddingConfig(provider="INVALID"))
    with pytest.raises(ValueError):
        RerankerFactory().create(RerankerConfig(), kind="INVALID")


def test_reranker_factory_creates_independent_lightweight_scorers():
    config = RerankerConfig(local_path="missing")
    first = RerankerFactory().create(config)
    second = RerankerFactory().create(config)
    cross = RerankerFactory().create(config, kind="cross_encoder")
    try:
        assert first is not second and first is not cross
        assert (
            first.readiness()["status"] == second.readiness()["status"] == cross.readiness()["status"] == "not_loaded"
        )
    finally:
        first.close()
        second.close()
        cross.close()
