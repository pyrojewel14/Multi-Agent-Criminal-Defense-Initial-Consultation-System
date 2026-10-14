"""可选本地 Cross-Encoder，遵循统一评分与生命周期契约。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._execution import LocalScorer, resolve_local_path


class CrossEncoderReranker(LocalScorer):
    """只返回原输入顺序的有限分数，不将模型原始分数强制解释为概率。"""

    def _load_model_sync(self) -> Any:
        """在唯一工作线程内加载本地模型，禁止远程文件回退。"""
        if self._model is not None:
            return self._model
        self._set_load_status("loading")
        try:
            device = self.resolve_device(self.config.device)
            path = resolve_local_path(self.config.local_path)
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(str(path), max_length=self.config.max_length, device=device, local_files_only=True)
            model.eval()
            self._model = model
            self._set_load_status("ready")
            return model
        except Exception as exc:
            self._set_load_status("error", type(exc).__name__)
            raise

    def _score_sync(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """原查询与各正文组成配对，返回模型分数且保留配对顺序。"""
        import torch

        model = self._load_model_sync()
        with torch.inference_mode():
            return model.predict([(query, document) for document in documents], batch_size=1)
