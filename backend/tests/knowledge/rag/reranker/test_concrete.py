"""Reranker contracts with real tokenization/tensor math and model I/O doubles."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from app.knowledge.rag.reranker.base import RerankerConfig
from app.knowledge.rag.reranker.causal_lm import CausalLMReranker
from app.knowledge.rag.reranker.cross_encoder import CrossEncoderReranker

QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.fixture
def protocol_tokenizer():
    vocabulary = [
        "[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
        "yes", "no", "true", "false", "Beijing", "Gravity", "query", "document",
        "system", "user", "assistant",
    ]
    tokenizer = Tokenizer(WordLevel({word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
        padding_side="left",
    )


@pytest.fixture
def protocol_reranker(protocol_tokenizer):
    reranker = CausalLMReranker(RerankerConfig(max_length=128, device="cpu"))
    reranker._model = object()
    reranker._tokenizer = protocol_tokenizer
    yield reranker
    reranker.close()

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

        with (
            patch("app.knowledge.rag.reranker.causal_lm.AutoTokenizer") as at,
            patch("app.knowledge.rag.reranker.causal_lm.AutoModelForCausalLM") as am,
            patch("app.knowledge.rag.reranker.causal_lm.torch") as torch_mod,
        ):
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
        with (
            patch("app.knowledge.rag.reranker.causal_lm.AutoTokenizer") as at,
            patch("app.knowledge.rag.reranker.causal_lm.AutoModelForCausalLM") as am,
            patch("app.knowledge.rag.reranker.causal_lm.torch") as torch_mod,
        ):
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
    async def test_preserves_official_template_and_token_ids(self, protocol_reranker, protocol_tokenizer):
        pairs = await protocol_reranker._format_pairs("query", ["Beijing", "document " * 300])
        prefix = protocol_tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
        suffix = protocol_tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
        assert len(pairs) == 2
        for pair in pairs:
            assert isinstance(pair, list)
            assert pair[:len(prefix)] == prefix
            assert pair[-len(suffix):] == suffix
        assert len(pairs[0]) < 128
        assert len(pairs[1]) == 128

    @pytest.mark.asyncio
    async def test_keeps_custom_templates(self, protocol_reranker, protocol_tokenizer):
        protocol_reranker.config.prefix_template = "<|im_start|>user\n"
        protocol_reranker.config.suffix_template = "<|im_end|>\n<|im_start|>assistant\n"
        pairs = await protocol_reranker._format_pairs("query", ["Beijing"])
        assert pairs == [protocol_tokenizer.encode(
            "<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
            "<Query>: query\n<Document>: Beijing<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )]

    @pytest.mark.asyncio
    async def test_rejects_budget_too_short_for_template(self, protocol_reranker):
        protocol_reranker.config.max_length = 1
        with pytest.raises(ValueError, match="max_length"):
            await protocol_reranker._format_pairs("query", ["document"])

    @pytest.mark.asyncio
    async def test_actual_local_qwen_tokenizer_preserves_template(self):
        path = Path(os.getenv("RERANKER_MODEL_PATH", "data/models/Qwen/Qwen3-Reranker-0.6B"))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        if not (path / "tokenizer.json").is_file():
            pytest.skip("configured local tokenizer not present; no download allowed")
        tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side="left", local_files_only=True)
        reranker = CausalLMReranker(RerankerConfig(max_length=512))
        reranker._model, reranker._tokenizer = object(), tokenizer
        try:
            query = "指控我偷了钱，但我否认，没有拿钱"
            pair = (await reranker._format_pairs(query, ["完整正文" * 300]))[0]
            assert isinstance(pair, list)
            assert len(pair) == 512
            text = tokenizer.decode(pair, skip_special_tokens=False)
            assert text.startswith(QWEN_PREFIX)
            assert text.endswith(QWEN_SUFFIX)
            assert query in text
        finally:
            reranker.close()


# ---------------------------------------------------------------------------
# CausalLMReranker._compute_scores
# ---------------------------------------------------------------------------


class TestCausalLMComputeScores:
    @pytest.mark.asyncio
    async def test_rerank_uses_yes_no_last_logits_and_preserves_padded_inputs(self, protocol_tokenizer):
        captured = {}

        def model_forward(**inputs):
            captured.update(inputs)
            ids = inputs["input_ids"]
            logits = torch.zeros((*ids.shape, len(protocol_tokenizer)))
            relevant = (ids == protocol_tokenizer.convert_tokens_to_ids("Beijing")).any(dim=1)
            # Earlier positions and true/false deliberately favour the opposite answer.
            logits[:, :, 6] = 8.0
            logits[:, -1, 6] = torch.where(relevant, 3.0, -3.0)
            logits[:, -1, 7] = torch.where(relevant, 1.0, -1.0)
            logits[:, -1, 8] = torch.where(relevant, -9.0, 9.0)
            return SimpleNamespace(logits=logits)

        model = MagicMock(side_effect=model_forward)
        model.device = "cpu"
        model.eval.return_value = model
        reranker = CausalLMReranker(RerankerConfig(device="cpu"))
        try:
            with (
                patch("app.knowledge.rag.reranker.causal_lm.AutoTokenizer.from_pretrained", return_value=protocol_tokenizer),
                patch("app.knowledge.rag.reranker.causal_lm.AutoModelForCausalLM.from_pretrained", return_value=model),
            ):
                result = await reranker.rerank("query", ["Gravity", "Beijing document document"])
            assert result["success"] is True
            assert [doc["document"] for doc in result["documents"]] == ["Beijing document document", "Gravity"]
            assert [doc["similarity"] for doc in result["documents"]] == pytest.approx([0.880797, 0.119203])
            assert captured["attention_mask"][0, :2].tolist() == [0, 0]
            assert captured["attention_mask"][0, 2:].all()
            prefix = protocol_tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
            suffix = protocol_tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
            for ids, mask in zip(captured["input_ids"], captured["attention_mask"]):
                unpadded = ids[mask.bool()].tolist()
                assert unpadded[:len(prefix)] == prefix
                assert unpadded[-len(suffix):] == suffix
        finally:
            reranker.close()


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

        with patch("app.knowledge.rag.reranker.cross_encoder.CrossEncoder") as ce:
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
        with patch("app.knowledge.rag.reranker.cross_encoder.CrossEncoder") as ce:
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
        with patch("app.knowledge.rag.reranker.cross_encoder.CrossEncoder") as ce:
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
        with patch("app.knowledge.rag.reranker.cross_encoder.torch") as torch_mod:
            torch_mod.no_grad.return_value.__enter__ = MagicMock()
            torch_mod.no_grad.return_value.__exit__ = MagicMock()
            result = await rr._compute_scores([("q", "d1"), ("q", "d2")])
        assert result == [0.1, 0.2]
        mock_model.predict.assert_called_once()
