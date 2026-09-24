import os

import torch
from sentence_transformers import CrossEncoder

from app.rag.reranker.base import BaseReranker, RerankerConfig
from app.utils.logger import get_logger

_logger = get_logger("RAG.CrossEncoderReranker")


class CrossEncoderReranker(BaseReranker):
    """基于交叉编码器的重排序器。

    使用预训练的交叉编码器模型直接计算查询-文档对的相关性分数。
    """

    async def _load_model(self):
        """加载交叉编码器模型（懒加载）。

        Returns:
            模型和分词器元组。
        """
        return await self._run_blocking(self._load_model_sync)

    def _load_model_sync(self):
        """在线程池工作线程中加载交叉编码器模型。"""
        with self._model_load_lock:
            if self._model is not None:
                self._set_load_status("ready")
                return self._model, self._tokenizer

            self._set_load_status("loading")
            try:
                actual_path = self._resolve_model_path()
                _logger.info("【_load_model】加载模型: %s", actual_path)

                model = CrossEncoder(
                    actual_path,
                    max_length=self.config.max_length,
                    device=self.config.device if self.config.device != "auto" else "cpu",
                )
                model.eval()
                self._tokenizer = None
                self._model = model
                self._set_load_status("ready")
            except Exception as exc:
                self._set_load_status("error", f"{type(exc).__name__}: {exc}")
                _logger.exception("【_load_model】模型加载失败")
                raise

            _logger.info("【_load_model】模型加载成功")
            return self._model, self._tokenizer

    def _resolve_model_path(self) -> str:
        """解析模型路径，查找包含 config.json 的目录。

        Returns:
            模型目录路径。
        """
        path = self.config.local_path
        if os.path.exists(os.path.join(path, "config.json")):
            return path

        for root, dirs, files in os.walk(path):
            if "config.json" in files:
                _logger.info("【_resolve_model_path】找到模型路径: %s", root)
                return root

        _logger.debug("【_resolve_model_path】使用默认路径: %s", path)
        return path

    async def _format_pairs(self, query: str, documents: list[str]) -> list[tuple[str, str]]:
        """格式化查询-文档对。

        Args:
            query: 查询语句。
            documents: 文档列表。

        Returns:
            查询-文档元组列表。
        """
        pairs = [(query, doc) for doc in documents]
        _logger.debug("【_format_pairs】格式化完成，生成 %d 个配对", len(pairs))
        return pairs

    async def _compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """使用交叉编码器计算相关性分数。

        Args:
            pairs: 查询-文档元组列表。

        Returns:
            相关性分数列表。
        """
        return await self._run_blocking(self._compute_scores_sync, pairs)

    def _compute_scores_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """在线程池工作线程中执行交叉编码器推理。"""
        model, _ = self._load_model_sync()
        with torch.no_grad():
            scores = model.predict(pairs, batch_size=1)
        _logger.debug("【_compute_scores】计算完成，返回 %d 个分数", len(scores))
        return scores.tolist()
