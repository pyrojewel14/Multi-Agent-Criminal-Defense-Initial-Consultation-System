"""Unit tests for ``app.rag.reorder_service``.

Model download and reranker inference are mocked. The unit-level public surface
is ``check_and_download_model``, ``ReorderService.reorder_documents`` and
``ReorderService.format_reorder_result``.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.reorder_service import ReorderService, check_and_download_model
from app.rag.reranker.base import RerankerConfig


# ---------------------------------------------------------------------------
# check_and_download_model
# ---------------------------------------------------------------------------


class TestCheckAndDownloadModel:
    def test_returns_local_path_if_config_exists(self, tmp_path):
        # Create a fake config.json inside the local_path
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        cfg = RerankerConfig(local_path=str(tmp_path), model_name="m")
        result = check_and_download_model(cfg)
        assert result == str(tmp_path)

    def test_downloads_when_missing(self, tmp_path):
        cfg = RerankerConfig(local_path=str(tmp_path / "model"), model_name="m", cache_dir=str(tmp_path))
        with patch("modelscope.snapshot_download") as sd:
            sd.return_value = "/downloaded/path"
            result = check_and_download_model(cfg)
        assert result == "/downloaded/path"
        sd.assert_called_once()

    def test_raises_when_download_fails(self, tmp_path):
        cfg = RerankerConfig(local_path=str(tmp_path / "model"), model_name="m", cache_dir=str(tmp_path))
        with patch("modelscope.snapshot_download", side_effect=RuntimeError("network")):
            with pytest.raises(RuntimeError, match="模型下载失败"):
                check_and_download_model(cfg)


# ---------------------------------------------------------------------------
# ReorderService.reorder_documents
# ---------------------------------------------------------------------------


class TestReorderDocuments:
    @pytest.mark.asyncio
    async def test_returns_result_from_reranker(self):
        cfg = RerankerConfig(model_name="m")
        svc = ReorderService(reranker_type="causal_lm", config=cfg)
        fake = {
            "success": True,
            "documents": [
                {"document": "B", "similarity": 0.9},
                {"document": "A", "similarity": 0.5},
            ],
            "error": "",
        }
        svc._reranker.rerank = AsyncMock(return_value=fake)
        result = await svc.reorder_documents("q", ["A", "B"])
        assert result == fake
        svc._reranker.rerank.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_failure_dict_on_exception(self):
        cfg = RerankerConfig(model_name="m")
        svc = ReorderService(reranker_type="causal_lm", config=cfg)
        svc._reranker.rerank = AsyncMock(side_effect=RuntimeError("boom"))
        result = await svc.reorder_documents("q", ["A", "B"])
        assert result["success"] is False
        assert result["documents"] == []
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_documents_in_rerank(self):
        # Reranker returns success but empty documents list
        cfg = RerankerConfig(model_name="m")
        svc = ReorderService(reranker_type="causal_lm", config=cfg)
        svc._reranker.rerank = AsyncMock(return_value={"success": True, "documents": [], "error": ""})
        result = await svc.reorder_documents("q", ["A", "B"])
        assert result["success"] is True
        assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_thinking_callback_forwarded(self):
        cfg = RerankerConfig(model_name="m")
        svc = ReorderService(reranker_type="causal_lm", config=cfg)
        captured = {}

        async def fake_rerank(query, docs, thinking_callback=None):
            captured["cb"] = thinking_callback
            return {"success": True, "documents": [], "error": ""}

        svc._reranker.rerank = fake_rerank

        async def cb(payload):
            pass

        await svc.reorder_documents("q", ["A"], thinking_callback=cb)
        assert captured["cb"] is cb


# ---------------------------------------------------------------------------
# ReorderService.format_reorder_result
# ---------------------------------------------------------------------------


class TestFormatReorderResult:
    @pytest.mark.asyncio
    async def test_format_basic(self):
        docs = [
            {"similarity": 0.95, "document": "doc 1"},
            {"similarity": 0.5, "document": "doc 2"},
        ]
        result = await ReorderService.format_reorder_result(docs)
        assert "重排序后的文档列表" in result
        assert "0.9500" in result
        assert "0.5000" in result
        assert "doc 1" in result
        assert "doc 2" in result
        assert "1. 相似度" in result
        assert "2. 相似度" in result

    @pytest.mark.asyncio
    async def test_format_empty(self):
        result = await ReorderService.format_reorder_result([])
        assert "重排序后的文档列表" in result

    @pytest.mark.asyncio
    async def test_format_handles_missing_similarity(self):
        result = await ReorderService.format_reorder_result([{"document": "x"}])
        assert "0.0000" in result
        assert "x" in result


# ---------------------------------------------------------------------------
# ReorderService.__init__ / create_default_reranker
# ---------------------------------------------------------------------------


class TestReorderServiceInit:
    def test_uses_from_env_when_no_config(self):
        with patch("app.rag.reranker.factory.RerankerFactory.create") as fc:
            fc.return_value = MagicMock()
            svc = ReorderService(reranker_type="causal_lm")
        assert fc.called

    def test_uses_provided_config(self):
        cfg = RerankerConfig(model_name="custom")
        with patch("app.rag.reranker.factory.RerankerFactory.create") as fc:
            fc.return_value = MagicMock()
            svc = ReorderService(reranker_type="cross_encoder", config=cfg)
        assert svc.config is cfg
        assert svc.reranker_type == "cross_encoder"
