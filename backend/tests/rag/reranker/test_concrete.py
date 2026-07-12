"""Unit tests for the concrete reranker implementations.

Both ``CausalLMReranker`` and ``CrossEncoderReranker`` load Hugging Face models,
so we mock the heavy dependencies (``transformers``, ``torch``,
``sentence_transformers``) to exercise the path-handling, tokenization and
score-computation logic.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.rag.reranker.base import RerankerConfig
from app.rag.reranker.causal_lm import CausalLMReranker
from app.rag.reranker.cross_encoder import CrossEncoderReranker


# ---------------------------------------------------------------------------
# CausalLMReranker._resolve_model_path
# ---------------------------------------------------------------------------


class TestCausalLMResolveModelPath:
    def test_returns_path_with_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CausalLMReranker(cfg)
        assert rr._resolve_model_path() == str(tmp_path)

    def test_finds_config_json_in_subdir(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CausalLMReranker(cfg)
        assert rr._resolve_model_path() == str(sub)

    def test_returns_local_path_if_no_config(self, tmp_path):
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CausalLMReranker(cfg)
        assert rr._resolve_model_path() == str(tmp_path)


# ---------------------------------------------------------------------------
# CausalLMReranker._load_model
# ---------------------------------------------------------------------------


class TestCausalLMLoadModel:
    @pytest.mark.asyncio
    async def test_loads_model_and_tokenizer(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path), device="cpu", positive_token="yes", negative_token="no")
        rr = CausalLMReranker(cfg)

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids = MagicMock(side_effect=[10, 20])

        with patch("app.rag.reranker.causal_lm.AutoTokenizer") as at, \
             patch("app.rag.reranker.causal_lm.AutoModelForCausalLM") as am, \
             patch("app.rag.reranker.causal_lm.torch") as torch_mod:
            at.from_pretrained.return_value = mock_tokenizer
            am.from_pretrained.return_value = mock_model
            torch_mod.float16 = "fp16"
            torch_mod.float32 = "fp32"
            model, tokenizer = await rr._load_model()

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        assert rr._positive_id == 10
        assert rr._negative_id == 20
        # Second call should reuse the cached model
        rr._model = mock_model
        rr._tokenizer = mock_tokenizer
        model2, tokenizer2 = await rr._load_model()
        assert model2 is mock_model
        assert tokenizer2 is mock_tokenizer

    @pytest.mark.asyncio
    async def test_raises_when_tokens_missing(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path), positive_token="missing1", negative_token="missing2")
        rr = CausalLMReranker(cfg)
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids = MagicMock(return_value=None)
        mock_model = MagicMock()
        with patch("app.rag.reranker.causal_lm.AutoTokenizer") as at, \
             patch("app.rag.reranker.causal_lm.AutoModelForCausalLM") as am, \
             patch("app.rag.reranker.causal_lm.torch") as torch_mod:
            at.from_pretrained.return_value = mock_tokenizer
            am.from_pretrained.return_value = mock_model
            torch_mod.float16 = "fp16"
            torch_mod.float32 = "fp32"
            with pytest.raises(ValueError, match="does not have"):
                await rr._load_model()


# ---------------------------------------------------------------------------
# CausalLMReranker._format_pairs
# ---------------------------------------------------------------------------


class TestCausalLMFormatPairs:
    @pytest.mark.asyncio
    async def test_returns_decoded_pairs(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path), max_length=128)
        rr = CausalLMReranker(cfg)
        rr._model = MagicMock()
        rr._tokenizer = MagicMock()
        rr._tokenizer.encode = MagicMock(side_effect=[[1, 2], [3, 4]])
        rr._tokenizer.return_value = {"input_ids": [5, 6]}
        rr._tokenizer.decode = MagicMock(side_effect=["decoded1", "decoded2"])

        result = await rr._format_pairs("query", ["doc1", "doc2"])
        assert result == ["decoded1", "decoded2"]
        assert rr._tokenizer.decode.call_count == 2


# ---------------------------------------------------------------------------
# CausalLMReranker._compute_scores
# ---------------------------------------------------------------------------


class TestCausalLMComputeScores:
    @pytest.mark.asyncio
    async def test_returns_probability_of_positive_token(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CausalLMReranker(cfg)
        rr._model = MagicMock()
        rr._model.device = "cpu"
        rr._tokenizer = MagicMock()
        # First call (in _load_model): positive/negative ids
        rr._positive_id = 7
        rr._negative_id = 8

        # Mock tokenizer(... ) to return inputs dict
        rr._tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value="t_input_ids")),
        }
        rr._tokenizer.__call__ = MagicMock(return_value={
            "input_ids": MagicMock(to=MagicMock(return_value="t_input_ids")),
        })
        # Mock torch tensor-like object that supports [-1, :]
        class _MockTensor:
            def __getitem__(self, item):
                # Return a tensor that supports [:, [ids]]
                if isinstance(item, tuple):
                    # last dim slice
                    return _MockTensor()

            def to(self, device):
                return self

            def __setitem__(self, key, value):
                pass

        # Build a more complete mock for the inputs dict and model output.
        # Easier: build a fully-orchestrated mock for the torch tensor chain.
        mock_t = MagicMock()
        mock_t.to.return_value = mock_t
        mock_t.__getitem__.return_value = mock_t

        rr._tokenizer = MagicMock()
        rr._tokenizer.return_value = {"input_ids": mock_t}

        # Model output: outputs.logits[:, -1, :] is mock_tensor, indexing by id
        # returns a tensor-like, torch.stack + softmax + exp.
        mock_logits = MagicMock()
        mock_logits.__getitem__.return_value = mock_logits
        rr._model.return_value = MagicMock(logits=mock_logits)

        # Mock torch.no_grad, torch.stack, F.log_softmax, exp, tolist
        with patch("app.rag.reranker.causal_lm.torch") as torch_mod:
            # Provide a fake Tensor class so that isinstance() inside the
            # reranker code can succeed when inputs are mocked.
            class _FakeTensor:
                pass

            torch_mod.Tensor = _FakeTensor
            mock_t.__class__ = _FakeTensor

            torch_mod.no_grad.return_value.__enter__ = MagicMock()
            torch_mod.no_grad.return_value.__exit__ = MagicMock()
            torch_mod.stack = MagicMock(return_value="stacked")
            # log_softmax -> [:, 1] -> .exp() -> .tolist()
            slice_mock = MagicMock()
            slice_mock.exp.return_value.tolist.return_value = [0.1, 0.2]
            torch_mod.nn.functional.log_softmax = MagicMock(return_value=MagicMock(__getitem__=MagicMock(return_value=slice_mock)))
            result = await rr._compute_scores(["p1", "p2"])

        assert result == [0.1, 0.2]


# ---------------------------------------------------------------------------
# CrossEncoderReranker._resolve_model_path
# ---------------------------------------------------------------------------


class TestCrossEncoderResolveModelPath:
    def test_returns_path_with_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CrossEncoderReranker(cfg)
        assert rr._resolve_model_path() == str(tmp_path)

    def test_finds_config_json_in_subdir(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CrossEncoderReranker(cfg)
        assert rr._resolve_model_path() == str(sub)

    def test_returns_local_path_if_no_config(self, tmp_path):
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CrossEncoderReranker(cfg)
        assert rr._resolve_model_path() == str(tmp_path)


# ---------------------------------------------------------------------------
# CrossEncoderReranker._load_model
# ---------------------------------------------------------------------------


class TestCrossEncoderLoadModel:
    @pytest.mark.asyncio
    async def test_loads_cross_encoder(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path), max_length=128, device="cpu")
        rr = CrossEncoderReranker(cfg)
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        with patch("app.rag.reranker.cross_encoder.CrossEncoder") as ce:
            ce.return_value = mock_model
            model, tokenizer = await rr._load_model()

        assert model is mock_model
        assert tokenizer is None
        ce.assert_called_once_with(str(tmp_path), max_length=128, device="cpu")

    @pytest.mark.asyncio
    async def test_auto_device_maps_to_cpu(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path), max_length=128, device="auto")
        rr = CrossEncoderReranker(cfg)
        with patch("app.rag.reranker.cross_encoder.CrossEncoder") as ce:
            ce.return_value = MagicMock()
            await rr._load_model()
        # When device="auto", the loader should pass "cpu" to CrossEncoder
        _, kwargs = ce.call_args
        assert kwargs["device"] == "cpu"

    @pytest.mark.asyncio
    async def test_reuses_loaded_model(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path), device="cpu")
        rr = CrossEncoderReranker(cfg)
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        with patch("app.rag.reranker.cross_encoder.CrossEncoder") as ce:
            ce.return_value = mock_model
            await rr._load_model()
            # Second call - CrossEncoder should not be called again
            ce.reset_mock()
            await rr._load_model()
            ce.assert_not_called()


# ---------------------------------------------------------------------------
# CrossEncoderReranker._format_pairs / _compute_scores
# ---------------------------------------------------------------------------


class TestCrossEncoderFormatPairs:
    @pytest.mark.asyncio
    async def test_format_pairs(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CrossEncoderReranker(cfg)
        result = await rr._format_pairs("query", ["d1", "d2"])
        assert result == [("query", "d1"), ("query", "d2")]


class TestCrossEncoderComputeScores:
    @pytest.mark.asyncio
    async def test_compute_scores(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path))
        rr = CrossEncoderReranker(cfg)
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=[0.1, 0.2])))
        rr._model = mock_model
        rr._tokenizer = None
        with patch("app.rag.reranker.cross_encoder.torch") as torch_mod:
            torch_mod.no_grad.return_value.__enter__ = MagicMock()
            torch_mod.no_grad.return_value.__exit__ = MagicMock()
            result = await rr._compute_scores([("q", "d1"), ("q", "d2")])
        assert result == [0.1, 0.2]
        mock_model.predict.assert_called_once()
