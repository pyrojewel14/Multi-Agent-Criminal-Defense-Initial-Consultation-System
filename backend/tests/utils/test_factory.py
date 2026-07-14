"""Unit tests for app/utils/factory.py.

Tests cover:
- DashScopeEmbeddingsWrapper.embed_documents / embed_query
- ChatModelFactory.create_model / create_streaming_model / create_precise_model
- EmbedModelFactory.create_embedding_model
- Cache behavior of the factories
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from app.utils.factory import (
    ChatModelFactory,
    DashScopeEmbeddingsWrapper,
    EmbedModelFactory,
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
        mock_openai.assert_called_once_with(api_key="explicit-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def test_init_falls_back_to_env_key(self, monkeypatch):
        """When no api_key is passed, env var is used."""
        env = {"DASHSCOPE_API_KEY": "env-key"}
        monkeypatch.setattr("os.getenv", lambda k, default=None: env.get(k, default))
        with patch("openai.OpenAI") as mock_openai:
            DashScopeEmbeddingsWrapper()
        mock_openai.assert_called_once_with(api_key="env-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

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
        wrapper.client.embeddings.create.return_value = _FakeEmbedResponse(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        result = wrapper.embed_documents(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert wrapper.client.embeddings.create.call_count == 1

    def test_embed_documents_multiple_batches(self):
        """embed_documents with >10 texts should split into multiple batches of 10."""
        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        wrapper.model_name = "text-embedding-v4"
        wrapper.client = MagicMock()
        wrapper.client.embeddings.create.return_value = _FakeEmbedResponse(
            [[0.0]] * 10
        )

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
        wrapper.client.close = MagicMock()
        wrapper.close()
        wrapper.client.close.assert_called_once()

    def test_close_handles_missing_client(self):
        """close() should not raise when client attribute is missing."""
        wrapper = DashScopeEmbeddingsWrapper.__new__(DashScopeEmbeddingsWrapper)
        # Don't set wrapper.client at all
        wrapper.close()  # should not raise


# ---------------------------------------------------------------------------
# ChatModelFactory
# ---------------------------------------------------------------------------


class TestChatModelFactory:
    def _make_factory(self):
        """Create a ChatModelFactory with a mocked _get_llm_config."""
        factory = ChatModelFactory()
        return factory

    def test_get_llm_config_delegates(self):
        """_get_llm_config should return config from config_loader."""
        factory = self._make_factory()
        with patch("app.utils.factory.config_loader") as mock_loader:
            mock_loader.get_llm_config.return_value = {"type": "ALIYUN", "aliyun": {}, "ollama": {}}
            result = factory._get_llm_config()
        assert result["type"] == "ALIYUN"

    def test_create_model_aliyun(self):
        """create_model with default ALIYUN backend should create ChatTongyi."""
        factory = self._make_factory()
        factory.clear_cache()

        with patch.object(factory, "_get_llm_config", return_value={
            "type": "ALIYUN",
            "aliyun": {"model": "qwen3-max", "api_key": "test-key", "base_url": "https://example.com"},
            "ollama": {},
        }):
            with patch("langchain_community.chat_models.tongyi.ChatTongyi") as mock_tongyi:
                mock_tongyi.return_value = "mocked-chat-model"
                result = factory.create_model(temperature=0.1)

        assert result == "mocked-chat-model"
        mock_tongyi.assert_called_once()
        factory.close()

    def test_create_model_ollama(self):
        """create_model with OLLAMA backend should create ChatOllama."""
        factory = self._make_factory()
        factory.clear_cache()

        with patch.object(factory, "_get_llm_config", return_value={
            "type": "OLLAMA",
            "aliyun": {},
            "ollama": {"model": "qwen3:7b", "base_url": "http://localhost:11434"},
        }):
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                mock_ollama.return_value = "mocked-ollama"
                result = factory.create_model(temperature=0.5)

        assert result == "mocked-ollama"
        assert mock_ollama.call_args.kwargs["client_kwargs"] == {"trust_env": False}
        factory.close()

    def test_remote_ollama_keeps_environment_proxy_support(self):
        assert _ollama_client_kwargs("https://ollama.example.com") == {}

    def test_create_streaming_model(self):
        """create_streaming_model should set streaming=True and top_p."""
        factory = self._make_factory()
        factory.clear_cache()

        with patch.object(factory, "_get_llm_config", return_value={
            "type": "ALIYUN",
            "aliyun": {"model": "qwen3-max", "api_key": "k"},
            "ollama": {},
        }):
            with patch("langchain_community.chat_models.tongyi.ChatTongyi") as mock_tongyi:
                mock_tongyi.return_value = "m"
                result = factory.create_streaming_model(top_p=0.9)

        # streaming should be True and top_p should be set
        kwargs = mock_tongyi.call_args.kwargs
        assert kwargs.get("streaming") is True
        assert kwargs.get("top_p") == 0.9
        factory.close()

    def test_create_precise_model(self):
        """create_precise_model should set streaming=False and temperature."""
        factory = self._make_factory()
        factory.clear_cache()

        with patch.object(factory, "_get_llm_config", return_value={
            "type": "ALIYUN",
            "aliyun": {"model": "qwen3-max", "api_key": "k"},
            "ollama": {},
        }):
            with patch("langchain_community.chat_models.tongyi.ChatTongyi") as mock_tongyi:
                mock_tongyi.return_value = "m"
                result = factory.create_precise_model(temperature=0.2)

        kwargs = mock_tongyi.call_args.kwargs
        assert kwargs.get("temperature") == 0.2
        # streaming=False is the default and not explicitly added to params
        assert kwargs.get("streaming") is None or kwargs.get("streaming") is False
        factory.close()

    def test_caches_model(self):
        """Same config should yield the same cached instance."""
        factory = self._make_factory()
        factory.clear_cache()

        with patch.object(factory, "_get_llm_config", return_value={
            "type": "ALIYUN",
            "aliyun": {"model": "qwen3-max", "api_key": "k"},
            "ollama": {},
        }):
            with patch("langchain_community.chat_models.tongyi.ChatTongyi") as mock_tongyi:
                mock_tongyi.return_value = "m"
                a = factory.create_model()
                b = factory.create_model()
        # The same model is returned both times
        assert a is b
        assert mock_tongyi.call_count == 1
        factory.close()

    def test_generator_deprecated(self):
        """generator() should issue a DeprecationWarning and call create_streaming_model."""
        factory = self._make_factory()
        factory.clear_cache()

        with patch.object(factory, "_get_llm_config", return_value={
            "type": "ALIYUN",
            "aliyun": {"model": "qwen3-max", "api_key": "k"},
            "ollama": {},
        }):
            with patch("langchain_community.chat_models.tongyi.ChatTongyi") as mock_tongyi:
                mock_tongyi.return_value = "m"
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    result = factory.generator()

        assert any(issubclass(w.category, DeprecationWarning) for w in captured)
        factory.close()

    def test_clear_cache(self):
        """clear_cache should empty the model cache."""
        factory = self._make_factory()
        factory._model_cache["fake-key"] = "fake-value"
        factory.clear_cache()
        assert factory._model_cache == {}

    def test_close_releases_models(self):
        """close() should call client.close on each cached model that supports it."""
        factory = self._make_factory()
        mock_model = MagicMock()
        mock_model.client.close = MagicMock()
        factory._model_cache["k"] = mock_model
        factory.close()
        mock_model.client.close.assert_called_once()
        assert factory._model_cache == {}


# ---------------------------------------------------------------------------
# EmbedModelFactory
# ---------------------------------------------------------------------------


class TestEmbedModelFactory:
    def _make_factory(self):
        return EmbedModelFactory()

    def test_create_embedding_model_ollama(self, monkeypatch):
        """create_embedding_model with EMBED_MODEL_TYPE=OLLAMA creates OllamaEmbeddings."""
        monkeypatch.setenv("EMBED_MODEL_TYPE", "OLLAMA")
        factory = self._make_factory()
        factory._model_cache.clear()

        with patch("langchain_ollama.OllamaEmbeddings") as mock_emb:
            mock_emb.return_value = "m"
            result = factory.create_embedding_model()

        assert result == "m"
        assert mock_emb.call_args.kwargs["client_kwargs"] == {"trust_env": False}
        factory.close()

    def test_create_embedding_model_aliyun(self, monkeypatch):
        """create_embedding_model with EMBED_MODEL_TYPE=ALIYUN creates DashScopeEmbeddingsWrapper."""
        monkeypatch.setenv("EMBED_MODEL_TYPE", "ALIYUN")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        factory = self._make_factory()
        factory._model_cache.clear()

        with patch("app.utils.factory.DashScopeEmbeddingsWrapper") as mock_wrap:
            mock_wrap.return_value = "wrapper-mock"
            result = factory.create_embedding_model()

        assert result == "wrapper-mock"
        factory.close()

    def test_create_embedding_model_invalid_type(self, monkeypatch):
        """When EMBED_MODEL_TYPE is invalid, raise ValueError."""
        monkeypatch.setenv("EMBED_MODEL_TYPE", "INVALID")
        factory = self._make_factory()
        factory._model_cache.clear()

        with pytest.raises(ValueError, match="不支持的 EMBED_MODEL_TYPE"):
            factory.create_embedding_model()
        factory.close()

    def test_create_embedding_model_caches(self, monkeypatch):
        """Same embed type should return the same instance from cache."""
        monkeypatch.setenv("EMBED_MODEL_TYPE", "OLLAMA")
        factory = self._make_factory()
        factory._model_cache.clear()

        with patch("langchain_ollama.OllamaEmbeddings") as mock_emb:
            mock_emb.return_value = "m"
            a = factory.create_embedding_model()
            b = factory.create_embedding_model()

        assert a is b
        # Model is built only once
        assert mock_emb.call_count == 1
        factory.close()

    def test_generator_deprecated(self, monkeypatch):
        """generator() should issue a DeprecationWarning."""
        monkeypatch.setenv("EMBED_MODEL_TYPE", "OLLAMA")
        factory = self._make_factory()
        factory._model_cache.clear()

        with patch("langchain_ollama.OllamaEmbeddings") as mock_emb:
            mock_emb.return_value = "m"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                factory.generator()

        assert any(issubclass(w.category, DeprecationWarning) for w in captured)
        factory.close()

    def test_close_closes_dashscope_wrappers(self):
        """close() should call close() on each DashScopeEmbeddingsWrapper in cache."""
        factory = self._make_factory()
        wrapper = MagicMock(spec=DashScopeEmbeddingsWrapper)
        factory._model_cache["k"] = wrapper
        factory.close()
        wrapper.close.assert_called_once()
        assert factory._model_cache == {}
