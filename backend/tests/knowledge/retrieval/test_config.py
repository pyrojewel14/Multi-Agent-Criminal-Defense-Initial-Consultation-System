"""Configuration rejects unsafe values before components are constructed."""

from dataclasses import FrozenInstanceError

import pytest

from app.infrastructure.llm.config import RerankerConfig
from app.knowledge.retrieval import config as config_module


def test_full_defaults_preserve_recall_and_optional_stage_budgets():
    config = config_module.RetrievalConfig.from_env({})
    assert (config.mode, config.recall_k, config.hyde_k) == ("hybrid", 20, 10)
    assert (config.rerank, config.rerank_timeout) == (True, 30.0)
    assert (config.hyde, config.hyde_timeout, config.hyde_budget) == (False, 10.0, 1)
    with pytest.raises(FrozenInstanceError):
        config.hyde = True


@pytest.mark.parametrize(
    "setting,value",
    [
        ("LAW_FULL_RETRIEVAL_MODE", "ensemble"),
        ("LAW_FULL_RECALL_K", "4"),
        ("LAW_FULL_RECALL_K", "51"),
        ("LAW_FULL_RECALL_K", "20.5"),
        ("LAW_FULL_HYDE_K", "0"),
        ("LAW_FULL_HYDE_K", "21"),
        ("LAW_FULL_HYDE_CALL_BUDGET", "-1"),
        ("LAW_FULL_HYDE_CALL_BUDGET", "2"),
        ("LAW_FULL_HYDE_CALL_BUDGET", "0.5"),
        ("LAW_FULL_HYDE_TIMEOUT_SECONDS", "0"),
        ("LAW_FULL_HYDE_TIMEOUT_SECONDS", "30.1"),
        ("LAW_FULL_RERANK_TIMEOUT_SECONDS", "0.009"),
        ("LAW_FULL_RERANK_TIMEOUT_SECONDS", "121"),
        ("LAW_FULL_RERANK_ENABLED", "yes"),
        ("LAW_FULL_HYDE_ENABLED", " false "),
    ],
)
def test_invalid_environment_settings_are_rejected(setting, value):
    with pytest.raises(ValueError):
        config_module.RetrievalConfig.from_env({setting: value})


@pytest.mark.parametrize(
    "setting",
    [
        "LAW_FULL_RECALL_K",
        "LAW_FULL_HYDE_K",
        "LAW_FULL_HYDE_CALL_BUDGET",
        "LAW_FULL_HYDE_TIMEOUT_SECONDS",
        "LAW_FULL_RERANK_TIMEOUT_SECONDS",
    ],
)
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_configuration_is_rejected(setting, value):
    with pytest.raises(ValueError):
        config_module.RetrievalConfig.from_env({setting: value})


@pytest.mark.parametrize(
    "value,want",
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("on", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("off", False),
    ],
)
def test_flags_preserve_strict_existing_spellings(value, want):
    config = config_module.RetrievalConfig.from_env(
        {
            "LAW_FULL_HYDE_ENABLED": value,
            "LAW_FULL_RERANK_ENABLED": value,
        }
    )
    assert config.hyde is want
    assert config.rerank is want


def test_configuration_is_a_snapshot_of_supplied_environment():
    environment = {"LAW_FULL_RETRIEVAL_MODE": "vector", "LAW_FULL_RECALL_K": "5"}
    config = config_module.RetrievalConfig.from_env(environment)
    environment["LAW_FULL_RECALL_K"] = "50"
    assert config.mode == "vector"
    assert config.recall_k == 5
    assert config_module.RetrievalConfig.from_env(environment).recall_k == 50


@pytest.mark.parametrize(
    "settings",
    [
        {"recall_k": 4},
        {"hyde_k": 21},
        {"hyde_budget": 0.5},
        {"rerank_timeout": float("nan")},
        {"hyde_timeout": float("inf")},
        {"rerank": "false"},
        {"hyde": 1},
        {"mode": "ensemble"},
    ],
)
def test_direct_configuration_cannot_bypass_validation(settings):
    with pytest.raises(ValueError):
        config_module.RetrievalConfig(**settings)


def test_configuration_accepts_inclusive_budget_boundaries():
    low = config_module.RetrievalConfig(
        recall_k=5,
        hyde_k=1,
        hyde_budget=0,
        hyde_timeout=0.01,
        rerank_timeout=0.01,
    )
    high = config_module.RetrievalConfig(
        recall_k=50,
        hyde_k=20,
        hyde_budget=1,
        hyde_timeout=30,
        rerank_timeout=120,
    )
    assert (low.recall_k, low.hyde_k, low.hyde_budget) == (5, 1, 0)
    assert (high.recall_k, high.hyde_k, high.hyde_budget) == (50, 20, 1)


def test_reranker_environment_snapshot_keeps_model_protocol_fields():
    config = RerankerConfig.from_env(
        {
            "RERANKER_MODEL_NAME": "custom/cross-encoder",
            "RERANKER_MODEL_PATH": "./data/models/custom",
            "MODEL_CACHE_DIR": "./data/cache",
            "RERANKER_MAX_LENGTH": "1024",
            "RERANKER_INSTRUCTION": "match query",
            "RERANKER_MAX_CONCURRENCY": "2",
            "LAW_FULL_RERANK_DEVICE": "cpu",
        }
    )
    assert (config.model_name, config.local_path, config.cache_dir) == (
        "custom/cross-encoder",
        "./data/models/custom",
        "./data/cache",
    )
    assert (config.max_length, config.max_concurrency, config.device) == (1024, 2, "cpu")
    assert config.instruction == "match query"
    assert (config.positive_token, config.negative_token) == ("yes", "no")
    assert "{query}" in config.input_template
    assert "{document}" in config.input_template
    assert config.prefix_template and config.suffix_template
    with pytest.raises(FrozenInstanceError):
        config.device = "cuda"


@pytest.mark.parametrize(
    "settings",
    [
        {"max_length": 0},
        {"max_length": True},
        {"max_length": 512.5},
        {"max_concurrency": -1},
        {"max_concurrency": False},
        {"max_concurrency": 1.5},
        {"device": "cuda:0"},
        {"device": "other"},
        {"model_name": ""},
        {"local_path": " "},
        {"cache_dir": ""},
        {"positive_token": ""},
        {"negative_token": ""},
        {"prefix_template": ""},
        {"suffix_template": ""},
        {"input_template": ""},
        {"instruction": 3},
    ],
)
def test_reranker_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        RerankerConfig(**settings)


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "mps"])
def test_reranker_accepts_supported_devices(device):
    assert RerankerConfig(device=device).device == device


@pytest.mark.parametrize("instruction", [None, ""])
def test_reranker_instruction_is_optional(instruction):
    assert RerankerConfig(instruction=instruction).instruction == instruction


@pytest.mark.parametrize(
    "setting,value",
    [
        ("RERANKER_MAX_LENGTH", "NaN"),
        ("RERANKER_MAX_LENGTH", "Infinity"),
        ("RERANKER_MAX_LENGTH", "512.5"),
        ("RERANKER_MAX_LENGTH", "0"),
        ("RERANKER_MAX_CONCURRENCY", "0"),
        ("RERANKER_MAX_CONCURRENCY", "1.5"),
        ("LAW_FULL_RERANK_DEVICE", "cuda:0"),
    ],
)
def test_reranker_environment_cannot_bypass_validation(setting, value):
    with pytest.raises(ValueError):
        RerankerConfig.from_env({setting: value})
