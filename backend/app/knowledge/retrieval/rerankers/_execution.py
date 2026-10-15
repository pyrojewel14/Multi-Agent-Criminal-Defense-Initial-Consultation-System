"""本地评分器共用的单工作线程、设备与文件边界。"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..config import RerankerConfig
from ..errors import safe_error_type


class RerankerBusyError(RuntimeError):
    """实际推理仍未结束，拒绝追加排队任务。"""


class RerankerClosedError(RuntimeError):
    """评分器已经关闭，不能再次提交任务。"""


def resolve_local_path(local_path: str) -> Path:
    """定位唯一的本地模型配置目录，缺失或歧义时拒绝远程回退。"""
    path = Path(local_path)
    if (path / "config.json").is_file():
        return path
    candidates = sorted(path.glob("**/config.json")) if path.is_dir() else []
    if len(candidates) != 1:
        raise FileNotFoundError("configured local reranker missing or ambiguous")
    return candidates[0].parent


def resolve_device(requested: str) -> str:
    """auto 按 CUDA、MPS、CPU 选择；显式设备不可用时直接失败。"""
    import torch

    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("invalid reranker device")
    cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend and mps_backend.is_available())
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if (requested == "cuda" and not cuda) or (requested == "mps" and not mps):
        raise RuntimeError(f"configured {requested.upper()} device unavailable")
    return requested


class LocalScorer:
    """共享单实际任务执行边界，超时与取消不会提前释放忙碌状态。"""

    probability_scores = False

    def __init__(self, config: RerankerConfig):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")
        self._lock = threading.Lock()
        self._future: Future | None = None
        self._model = self._tokenizer = None
        self._load_status = "not_loaded"
        self._load_error: str | None = None
        self._closed = False

    resolve_device = staticmethod(resolve_device)

    def _set_load_status(self, status: str, error: BaseException | str | None = None) -> None:
        """原子更新加载状态，仅记录固定诊断类别，避免消息或未知类名泄露。"""
        with self._lock:
            self._load_status, self._load_error = status, safe_error_type(error)

    def readiness(self) -> dict[str, Any]:
        """返回状态快照，不检查模型文件、不加载权重或分配推理设备。"""
        with self._lock:
            return {
                "status": "closed" if self._closed else self._load_status,
                "available": not self._closed and self._load_status == "ready",
                "model": self.config.model_name,
                "error": self._load_error,
                "busy": self._future is not None and not self._future.done(),
                "closed": self._closed,
            }

    def _finished(self, future: Future) -> None:
        """消费后台异常；关闭后待实际工作完成再释放模型引用。"""
        if not future.cancelled():
            future.exception()
        with self._lock:
            if self._closed:
                self._model = self._tokenizer = None

    def close(self) -> None:
        """幂等关闭执行器，运行中的任务自然结束后释放权重引用。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._future is None or self._future.done():
                self._model = self._tokenizer = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _score_sync(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """由具体模型执行同步评分，调用方统一校验数量和有限值。"""
        raise NotImplementedError

    def _run(self, query: str, documents: tuple[str, ...]) -> list[float]:
        """在实际工作线程内校验完整分数，避免 zip 静默丢弃候选。"""
        try:
            scores = [float(score) for score in self._score_sync(query, documents)]
        except (TypeError, OverflowError) as exc:
            raise ValueError("reranker score values invalid") from exc
        if len(scores) != len(documents) or any(not math.isfinite(score) for score in scores):
            raise ValueError("reranker score count/values invalid")
        if self.probability_scores and any(not 0 <= score <= 1 for score in scores):
            raise ValueError("Qwen reranker score outside probability range")
        return scores

    async def score(self, query: str, documents: Sequence[str], *, timeout_s: float) -> Sequence[float]:
        """保留输入顺序评分；等待超时或外层取消后仍保护实际工作线程。"""
        if type(timeout_s) not in (float, int) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("invalid reranker timeout")
        if not isinstance(query, str) or isinstance(documents, (str, bytes)):
            raise ValueError("reranker expects query and document strings")
        copied_documents = tuple(documents)
        if any(not isinstance(document, str) for document in copied_documents):
            raise ValueError("reranker expects document strings")
        with self._lock:
            if self._closed:
                raise RerankerClosedError("reranker closed")
            if self._future is not None and not self._future.done():
                raise RerankerBusyError("previous rerank still running")
            if not copied_documents:
                return []
            future = self._executor.submit(self._run, query, copied_documents)
            self._future = future
        future.add_done_callback(self._finished)
        wrapped = asyncio.wrap_future(future)
        # shield 保留实际任务；超时后也消费异步包装上的异常，避免无人读取警告。
        wrapped.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
        return await asyncio.wait_for(asyncio.shield(wrapped), timeout=timeout_s)
