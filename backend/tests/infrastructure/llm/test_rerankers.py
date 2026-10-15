"""Unified scoring contracts with model I/O replaced at the dependency boundary."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.infrastructure.llm.config import RerankerConfig
from app.infrastructure.llm.rerankers import CrossEncoderReranker, QwenReranker
from app.infrastructure.llm.rerankers.qwen import INSTRUCTION, PREFIX, SUFFIX


class Tokenizer:
    unk_token_id = -1

    def encode(self, text, **kwargs):
        if text in {"yes", "no"}:
            return [{"yes": 11, "no": 12}[text]]
        if text == PREFIX:
            return [101, 102]
        if text == SUFFIX:
            return [103, 104]
        return [ord(char) for char in text]

    def convert_tokens_to_ids(self, token):
        return {"yes": 11, "no": 12}[token]

    def pad(self, inputs, **kwargs):
        return {"input_ids": torch.tensor([[1]] * len(inputs["input_ids"]))}


def test_qwen_preserves_special_ids_query_tail_and_only_truncates_document():
    scorer = QwenReranker(RerankerConfig(max_length=170, instruction="instruction"))
    try:
        query = "这是原始查询，最后是否认：没有偷钱"
        tokens = scorer.encode_pairs(Tokenizer(), query, ["正文" * 1000])[0]
        assert tokens[:2] == [101, 102] and tokens[-2:] == [103, 104]
        assert query + "\n<Document>: " in "".join(chr(t) for t in tokens[2:-2])
        assert len(tokens) == 170
        with pytest.raises(ValueError, match="query"):
            scorer.encode_pairs(Tokenizer(), "否认" * 200, ["正文"])
        assert scorer.decision_tokens(Tokenizer()) == (11, 12)
        assert "criminal law" in INSTRUCTION
    finally:
        scorer.close()


@pytest.mark.parametrize("positive,negative,unknown", [(None, 12, -1), (11, 11, -1), (11, 12, 11), (-1, 12, -2)])
def test_qwen_rejects_invalid_decision_tokens(positive, negative, unknown):
    tokenizer = SimpleNamespace(
        unk_token_id=unknown,
        convert_tokens_to_ids=lambda token: positive if token == "yes" else negative,
    )
    with pytest.raises(ValueError, match="token"):
        QwenReranker.decision_tokens(tokenizer)


def test_qwen_rejects_decision_word_that_encodes_to_multiple_tokens():
    class SplitDecisionTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return [11, 13] if text == "yes" else super().encode(text, **kwargs)

    with pytest.raises(ValueError, match="token"):
        QwenReranker.decision_tokens(SplitDecisionTokenizer())


@pytest.mark.parametrize(
    "setting,value",
    [
        ("positive_token", "true"),
        ("negative_token", "false"),
        ("prefix_template", "prefix"),
        ("suffix_template", "suffix"),
        ("input_template", "{query}{document}"),
    ],
)
def test_qwen_rejects_configuration_that_changes_official_protocol(setting, value):
    with pytest.raises(ValueError, match="protocol"):
        QwenReranker(RerankerConfig(**{setting: value}))


@pytest.mark.parametrize("scorer_type", [QwenReranker, CrossEncoderReranker])
@pytest.mark.parametrize("values", [[0.1], [0.1, float("nan")], [float("inf"), 0.3]])
@pytest.mark.asyncio
async def test_score_rejects_wrong_count_or_nonfinite_values(scorer_type, values):
    scorer = scorer_type(RerankerConfig())
    scorer._score_sync = lambda query, documents: values
    try:
        with pytest.raises(ValueError, match="score"):
            await scorer.score("query", ["one", "two"], timeout_s=1)
    finally:
        scorer.close()


@pytest.mark.parametrize("values", [[-0.1, 0.2], [0.1, 1.01]])
@pytest.mark.asyncio
async def test_qwen_rejects_scores_outside_probability_range(values):
    scorer = QwenReranker(RerankerConfig())
    scorer._score_sync = lambda query, documents: values
    try:
        with pytest.raises(ValueError, match="score"):
            await scorer.score("query", ["one", "two"], timeout_s=1)
    finally:
        scorer.close()


@pytest.mark.parametrize(
    "requested,cuda,mps,expected",
    [
        ("auto", True, True, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
        ("cuda", True, False, "cuda"),
        ("mps", True, True, "mps"),
    ],
)
def test_device_selection_respects_explicit_choice(monkeypatch, requested, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert QwenReranker.resolve_device(requested) == expected


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_unavailable_explicit_device_fails(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        QwenReranker.resolve_device(requested)


@pytest.mark.asyncio
async def test_qwen_uses_local_only_loaders_and_yes_no_logits_in_input_order(tmp_path, monkeypatch):
    import transformers

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
    loader_options = []

    class Model:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, logits_to_keep):
            assert logits_to_keep == 1
            logits = torch.zeros((len(input_ids), 1, 13))
            probabilities = [0.2, 0.8] if len(input_ids) == 2 else [0.4]
            for i, probability in enumerate(probabilities):
                logits[i, 0, 11] = math.log(probability)
                logits[i, 0, 12] = math.log(1 - probability)
            return SimpleNamespace(logits=logits)

    def load_tokenizer(path, **kwargs):
        loader_options.append(kwargs)
        return Tokenizer()

    def load_model(path, **kwargs):
        loader_options.append(kwargs)
        return Model()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    scorer = QwenReranker(RerankerConfig(local_path=str(tmp_path), max_length=512, device="cpu"))
    try:
        assert await scorer.score("query", ("one", "two", "three"), timeout_s=1) == pytest.approx([0.2, 0.8, 0.4])
        assert all(options["local_files_only"] for options in loader_options)
        assert scorer.readiness()["status"] == "ready"
    finally:
        scorer.close()


@pytest.mark.asyncio
async def test_cross_encoder_preserves_unbounded_finite_scores_and_local_only_loading(tmp_path, monkeypatch):
    import sentence_transformers

    (tmp_path / "config.json").write_text("{}")
    options = []

    class Model:
        def __init__(self, path, **kwargs):
            options.append(kwargs)

        def eval(self):
            return self

        def predict(self, pairs, batch_size):
            assert pairs == [("original query", "one"), ("original query", "two")]
            assert batch_size == 1
            return [-3.5, 4.25]

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Model)
    scorer = CrossEncoderReranker(RerankerConfig(local_path=str(tmp_path), device="cpu"))
    try:
        assert await scorer.score("original query", ["one", "two"], timeout_s=1) == [-3.5, 4.25]
        assert options[0]["local_files_only"] is True
    finally:
        scorer.close()


@pytest.mark.parametrize("scorer_type", [QwenReranker, CrossEncoderReranker])
@pytest.mark.asyncio
async def test_missing_or_ambiguous_local_model_fails_without_remote_fallback(tmp_path, scorer_type):
    scorer = scorer_type(RerankerConfig(local_path=str(tmp_path), device="cpu"))
    try:
        with pytest.raises(FileNotFoundError, match="missing|ambiguous"):
            await scorer.score("query", ["one"], timeout_s=1)
        for name in ("first", "second"):
            directory = tmp_path / name
            directory.mkdir()
            (directory / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="ambiguous"):
            await scorer.score("query", ["one"], timeout_s=1)
        assert scorer.readiness()["status"] == "error"
    finally:
        scorer.close()


def test_preparation_import_and_check_are_offline_and_download_is_explicit(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[3] / "scripts" / "prepare_reranker.py"
    spec = importlib.util.spec_from_file_location("prepare_reranker_test", script)
    module = importlib.util.module_from_spec(spec)
    requests = []
    monkeypatch.setitem(
        sys.modules,
        "modelscope",
        SimpleNamespace(snapshot_download=lambda **kwargs: requests.append(kwargs) or str(tmp_path)),
    )
    spec.loader.exec_module(module)
    config = RerankerConfig(local_path=str(tmp_path / "missing"), cache_dir=str(tmp_path / "cache"))
    with pytest.raises(FileNotFoundError):
        module.prepare_model(config)
    assert requests == []
    (tmp_path / "config.json").write_text("{}")
    assert module.prepare_model(config, download=True) == str(tmp_path)
    assert requests == [{"model_id": config.model_name, "cache_dir": config.cache_dir}]
    assert module.prepare_model(RerankerConfig(local_path=str(tmp_path))) == str(tmp_path)
    assert len(requests) == 1


def test_import_constructor_and_readiness_do_not_import_model_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from app.infrastructure.llm.config import RerankerConfig; "
            "from app.infrastructure.llm.rerankers import QwenReranker, CrossEncoderReranker; "
            "scorers = [QwenReranker(RerankerConfig()), CrossEncoderReranker(RerankerConfig())]; "
            "assert all(s.readiness()['status'] == 'not_loaded' for s in scorers); "
            "assert not {'torch', 'transformers', 'sentence_transformers', 'modelscope'} & set(sys.modules); "
            "[s.close() for s in scorers]",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_actual_local_tokenizer_keeps_original_query_and_official_decisions():
    from transformers import AutoTokenizer

    from app.infrastructure.llm.rerankers._execution import resolve_local_path

    backend = Path(__file__).resolve().parents[3]
    config = RerankerConfig.from_env()
    candidates = [
        Path(config.local_path),
        Path(config.cache_dir) / "Qwen/Qwen3-Reranker-0.6B",
        Path(config.cache_dir) / "Qwen/Qwen3-Reranker-0___6B",
    ]
    # 测试复用已有缓存；生产评分器仍只接受显式配置的模型目录。
    for candidate in dict.fromkeys(candidates):
        candidate = candidate if candidate.is_absolute() else backend / candidate
        try:
            path = resolve_local_path(str(candidate))
        except FileNotFoundError:
            continue
        if (path / "tokenizer.json").is_file():
            break
    else:
        pytest.skip("local Qwen tokenizer not present in configured path or cache; download forbidden")
    tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side="left", local_files_only=True)
    scorer = QwenReranker(RerankerConfig(local_path=str(path)))
    try:
        query = "指控我偷了钱，但我否认，没有拿钱"
        tokens = scorer.encode_pairs(tokenizer, query, ["完整正文" * 500])[0]
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        assert query in decoded and "<think>\n\n</think>" in decoded
        assert len(tokens) == 512
        prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        assert tokens[: len(prefix)] == prefix
        assert tokens[-len(suffix) :] == suffix
        assert scorer.decision_tokens(tokenizer) == (9693, 2152)
        assert tokenizer.encode("yes", add_special_tokens=False) == [9693]
        assert tokenizer.encode("no", add_special_tokens=False) == [2152]
        with pytest.raises(ValueError, match="query"):
            scorer.encode_pairs(tokenizer, "前文" * 600 + "最后否认：没有偷钱", ["正文"])
    finally:
        scorer.close()
