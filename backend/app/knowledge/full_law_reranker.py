"""全量公共检索专用的本地 Qwen3 yes/no 重排协议。"""

from __future__ import annotations

import asyncio
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a criminal law search query, retrieve relevant statutory provisions for further review"


class RerankerBusyError(RuntimeError):
    """上次超时推理尚未结束，拒绝积累后台任务。"""


class QwenFullReranker:
    """只加载已有本地权重；超时不强杀线程，也不允许新增排队推理。"""

    def __init__(self, local_path: str, max_length: int, instruction: str, device: str = "auto"):
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("invalid full reranker device")
        self.local_path = local_path
        self.max_length = max_length
        self.instruction = instruction or INSTRUCTION
        self.device = device
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="full-rerank")
        self._lock = threading.Lock()
        self._future = None
        self._model = self._tokenizer = None

    @staticmethod
    def resolve_device(requested: str) -> str:
        """auto 优先 CUDA、其次 MPS；显式设备不可用时拒绝静默替换。"""
        import torch

        if requested not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("invalid full reranker device")
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("configured CUDA device unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("configured MPS device unavailable")
        return requested

    def close(self) -> None:
        """关闭执行器；正在运行的本地推理仍会自然结束。"""
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def decision_tokens(tokenizer: Any) -> tuple[int, int]:
        """要求 yes/no 为不同的有效单 token，拒绝未知 token。"""
        positive = tokenizer.convert_tokens_to_ids("yes")
        negative = tokenizer.convert_tokens_to_ids("no")
        if (
            positive is None
            or negative is None
            or positive == negative
            or tokenizer.unk_token_id in {positive, negative}
        ):
            raise ValueError("Qwen yes/no token invalid")
        return positive, negative

    def encode_pairs(self, tokenizer: Any, query: str, documents: list[str]) -> list[list[int]]:
        """只截断正文部分，直接拼接特殊 token IDs，不经 decode 往返。"""
        prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        remaining = self.max_length - len(prefix) - len(suffix)
        if remaining < 2:
            raise ValueError("reranker max length too short")
        header = tokenizer.encode(
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: ", add_special_tokens=False
        )
        if len(header) >= remaining:
            raise ValueError("original query exceeds reranker token budget")
        return [
            prefix + header + tokenizer.encode(document, add_special_tokens=False)[: remaining - len(header)] + suffix
            for document in documents
        ]

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        """按指定设备执行 FP32 小批次推理，禁止自动下载或切换模型。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is None:
            device = self.resolve_device(self.device)
            path = Path(self.local_path)
            if not (path / "config.json").is_file():
                candidates = sorted(path.glob("**/config.json")) if path.is_dir() else []
                if len(candidates) != 1:
                    raise FileNotFoundError("configured local reranker missing or ambiguous")
                path = candidates[0].parent
            self._tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side="left", local_files_only=True)
            config = __import__("json").loads((path / "config.json").read_text())
            if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in config.get("architectures", []):
                raise ValueError("configured reranker is not Qwen3ForCausalLM")
            model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, torch_dtype=torch.float32)
            model = model.to(device)
            self._model = model.eval()
        tokenizer, model = self._tokenizer, self._model
        positive, negative = self.decision_tokens(tokenizer)
        pairs = self.encode_pairs(tokenizer, query, documents)
        scores = []
        for start in range(0, len(pairs), 2):
            inputs = tokenizer.pad({"input_ids": pairs[start : start + 2]}, padding=True, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]
                batch = torch.stack([logits[:, negative], logits[:, positive]], dim=1).softmax(dim=1)[:, 1].tolist()
            scores.extend(batch)
        if len(scores) != len(documents) or any(not math.isfinite(s) for s in scores):
            raise ValueError("reranker score count/values invalid")
        return scores

    async def score(self, query: str, documents: list[str], timeout: float) -> list[float]:
        """等待推理到 deadline；超时后保留忙碌状态直到实际工作结束。"""
        with self._lock:
            if self._future is not None and not self._future.done():
                raise RerankerBusyError("previous rerank still running")
            self._future = self._executor.submit(self._score_sync, query, documents)
            future = self._future
        wrapped = asyncio.wrap_future(future)
        # 后台失败必须被消费，避免超时后产生无人读取的 Future 警告。
        wrapped.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        return await asyncio.wait_for(asyncio.shield(wrapped), timeout=timeout)
