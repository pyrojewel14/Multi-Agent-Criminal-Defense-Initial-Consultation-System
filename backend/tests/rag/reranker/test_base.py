"""Unit tests for ``app.rag.reranker.base.BaseReranker`` and ``RerankerConfig``.

We use a concrete test subclass to exercise the abstract base class without
loading any real model.
"""

import os
from unittest.mock import AsyncMock

import pytest

from app.rag.reranker.base import BaseReranker, RerankerConfig


# ---------------------------------------------------------------------------
# RerankerConfig
# ---------------------------------------------------------------------------


class TestRerankerConfig:
    def test_defaults(self):
        cfg = RerankerConfig()
        assert cfg.model_name == "Qwen/Qwen3-Reranker-0.6B"
        assert cfg.local_path == "./data/models/Qwen/Qwen3-Reranker-0.6B"
        assert cfg.cache_dir == "./data/models"
        assert cfg.max_length == 512
        assert cfg.device == "auto"
        assert cfg.positive_token == "true"
        assert cfg.negative_token == "false"
        assert cfg.instruction is None
        assert "{query}" in cfg.input_template

    def test_from_env_uses_defaults(self, monkeypatch):
        # Clear any env vars that might be set
        for key in ["RERANKER_MODEL_NAME", "RERANKER_MODEL_PATH", "MODEL_CACHE_DIR",
                    "RERANKER_MAX_LENGTH", "RERANKER_INSTRUCTION"]:
            monkeypatch.delenv(key, raising=False)
        cfg = RerankerConfig.from_env()
        assert cfg.model_name == "Qwen/Qwen3-Reranker-0.6B"
        assert cfg.local_path == "./data/models/Qwen/Qwen3-Reranker-0.6B"
        assert cfg.cache_dir == "./data/models"
        assert cfg.max_length == 512
        assert cfg.instruction is None

    def test_from_env_reads_overrides(self, monkeypatch):
        monkeypatch.setenv("RERANKER_MODEL_NAME", "custom-model")
        monkeypatch.setenv("RERANKER_MODEL_PATH", "/custom/path")
        monkeypatch.setenv("MODEL_CACHE_DIR", "/custom/cache")
        monkeypatch.setenv("RERANKER_MAX_LENGTH", "256")
        monkeypatch.setenv("RERANKER_INSTRUCTION", "my instruction")
        cfg = RerankerConfig.from_env()
        assert cfg.model_name == "custom-model"
        assert cfg.local_path == "/custom/path"
        assert cfg.cache_dir == "/custom/cache"
        assert cfg.max_length == 256
        assert cfg.instruction == "my instruction"


# ---------------------------------------------------------------------------
# Concrete subclass used for testing
# ---------------------------------------------------------------------------


class _DummyReranker(BaseReranker):
    def __init__(self, config: RerankerConfig, format_pairs_result=None, scores_result=None):
        super().__init__(config)
        self._format_pairs_result = format_pairs_result or []
        self._scores_result = scores_result or []
        self._load_calls = 0

    async def _load_model(self):
        self._load_calls += 1
        return None, None

    async def _format_pairs(self, query, documents):
        return self._format_pairs_result

    async def _compute_scores(self, pairs):
        return self._scores_result


# ---------------------------------------------------------------------------
# BaseReranker.rerank
# ---------------------------------------------------------------------------


class TestRerank:
    @pytest.mark.asyncio
    async def test_empty_documents_returns_empty_result(self):
        cfg = RerankerConfig()
        rr = _DummyReranker(cfg)
        result = await rr.rerank("q", [])
        assert result == {"success": True, "documents": [], "error": ""}

    @pytest.mark.asyncio
    async def test_sorts_documents_by_score_descending(self):
        cfg = RerankerConfig()
        rr = _DummyReranker(
            cfg,
            format_pairs_result=["p1", "p2", "p3"],
            scores_result=[0.2, 0.9, 0.5],
        )
        result = await rr.rerank("q", ["d1", "d2", "d3"])
        assert result["success"] is True
        docs = result["documents"]
        assert [d["document"] for d in docs] == ["d2", "d3", "d1"]
        assert docs[0]["similarity"] == 0.9

    @pytest.mark.asyncio
    async def test_thinking_callback_invoked(self):
        cfg = RerankerConfig()
        rr = _DummyReranker(
            cfg,
            format_pairs_result=["p1", "p2"],
            scores_result=[0.5, 0.7],
        )
        calls = []

        async def cb(payload):
            calls.append(payload)

        await rr.rerank("q", ["d1", "d2"], thinking_callback=cb)
        # First call: progress message; second call: details with scores
        assert len(calls) >= 2
        assert calls[0]["stage"] == "reorder"
        assert calls[-1]["stage"] == "reorder"
        assert "scores" in calls[-1]["details"]

    @pytest.mark.asyncio
    async def test_long_doc_preview_truncated(self):
        cfg = RerankerConfig()
        rr = _DummyReranker(
            cfg,
            format_pairs_result=["p"],
            scores_result=[0.1],
        )
        long_doc = "x" * 200
        result = await rr.rerank("q", [long_doc])
        assert result["success"] is True
        assert result["documents"][0]["document"] == long_doc

    @pytest.mark.asyncio
    async def test_negative_scores_preserved(self):
        cfg = RerankerConfig()
        rr = _DummyReranker(
            cfg,
            format_pairs_result=["p1", "p2"],
            scores_result=[-0.5, -0.1],
        )
        result = await rr.rerank("q", ["d1", "d2"])
        docs = result["documents"]
        # -0.1 > -0.5 -> d2 first
        assert docs[0]["document"] == "d2"
