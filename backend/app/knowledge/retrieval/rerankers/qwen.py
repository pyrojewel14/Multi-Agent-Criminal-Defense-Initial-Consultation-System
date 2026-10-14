"""统一的本地 Qwen3 yes/no 评分协议，不负责候选排序。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..config import RerankerConfig
from ._execution import LocalScorer, resolve_local_path

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a criminal law search query, retrieve relevant statutory provisions for further review"
_INPUT_TEMPLATE = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


class QwenReranker(LocalScorer):
    """固定官方协议并只加载本地权重；max_concurrency 保留兼容但实际固定单任务。"""

    probability_scores = True

    def __init__(self, config: RerankerConfig):
        if (
            config.positive_token != "yes" or config.negative_token != "no"
            or config.prefix_template != PREFIX or config.suffix_template != SUFFIX
            or config.input_template != _INPUT_TEMPLATE
        ):
            raise ValueError("Qwen requires the fixed official yes/no protocol")
        super().__init__(config)

    @staticmethod
    def decision_tokens(tokenizer: Any) -> tuple[int, int]:
        """要求 yes/no 分别编码为完整单 token，拒绝未知、拆分或非法 ID。"""
        positive = tokenizer.convert_tokens_to_ids("yes")
        negative = tokenizer.convert_tokens_to_ids("no")
        if (
            type(positive) is not int or type(negative) is not int
            or positive < 0 or negative < 0 or positive == negative
            or getattr(tokenizer, "unk_token_id", None) in {positive, negative}
        ):
            raise ValueError("Qwen yes/no token invalid")
        if (
            tokenizer.encode("yes", add_special_tokens=False) != [positive]
            or tokenizer.encode("no", add_special_tokens=False) != [negative]
        ):
            raise ValueError("Qwen yes/no must each encode to one complete token")
        return positive, negative

    def encode_pairs(self, tokenizer: Any, query: str, documents: Sequence[str]) -> list[list[int]]:
        """直接拼接特殊 token IDs，保留完整原查询，仅截断正文。"""
        prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        remaining = self.config.max_length - len(prefix) - len(suffix)
        if remaining < 2:
            raise ValueError("reranker max length too short")
        header = tokenizer.encode(
            f"<Instruct>: {self.config.instruction or INSTRUCTION}\n<Query>: {query}\n<Document>: ",
            add_special_tokens=False,
        )
        if len(header) >= remaining:
            raise ValueError("original query exceeds reranker token budget")
        return [
            prefix + header + tokenizer.encode(document, add_special_tokens=False)[: remaining - len(header)] + suffix
            for document in documents
        ]

    def _load_model_sync(self) -> tuple[Any, Any]:
        """在唯一工作线程中懒加载本地 Qwen3；加载失败仅保留错误类别。"""
        if self._model is not None:
            return self._model, self._tokenizer
        self._set_load_status("loading")
        try:
            device = self.resolve_device(self.config.device)
            path = resolve_local_path(self.config.local_path)
            model_config = json.loads((path / "config.json").read_text())
            if model_config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in model_config.get("architectures", []):
                raise ValueError("configured reranker is not Qwen3ForCausalLM")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side="left", local_files_only=True)
            self.decision_tokens(tokenizer)
            model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, torch_dtype=torch.float32)
            model = model.to(device).eval()
            self._model, self._tokenizer = model, tokenizer
            self._set_load_status("ready")
            return model, tokenizer
        except Exception as exc:
            self._set_load_status("error", type(exc).__name__)
            raise

    def _score_sync(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """按正文输入顺序分批计算最后位置的 yes/no 概率，不排序或截断候选。"""
        import torch

        model, tokenizer = self._load_model_sync()
        positive, negative = self.decision_tokens(tokenizer)
        pairs = self.encode_pairs(tokenizer, query, documents)
        scores = []
        for start in range(0, len(pairs), 2):
            inputs = tokenizer.pad({"input_ids": pairs[start:start + 2]}, padding=True, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]
                scores.extend(torch.stack([logits[:, negative], logits[:, positive]], dim=1).softmax(dim=1)[:, 1].tolist())
        return scores
