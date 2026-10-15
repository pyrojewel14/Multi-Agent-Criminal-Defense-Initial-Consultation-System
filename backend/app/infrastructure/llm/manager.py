"""统一持有模型资源，按显式配置复用并协调同步与异步关闭。"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.infrastructure.llm.clients import close_actions
from app.infrastructure.llm.config import ChatConfig, EmbeddingConfig, RerankerConfig
from app.infrastructure.llm.factory import ChatModelFactory, EmbedModelFactory, RerankerFactory
from app.infrastructure.llm.ollama import OllamaClient
from app.infrastructure.llm.rerankers._execution import LocalScorer


class LazyEmbeddings:
    """上传和查询借用同一延迟向量模型；代理不另持有资源所有权。"""

    def __init__(self, owner: ModelManager, config: EmbeddingConfig, factory: Callable[[], Embeddings] | None) -> None:
        self._owner = owner
        self._config = config
        self._factory = factory
        self._model = None

    def _get_model(self) -> Embeddings:
        """创建与关闭使用同一锁，关闭开始后代理不能继续分配资源。"""
        with self._owner._lock:
            self._owner._require_open()
            if self._model is None:
                self._model = (
                    self._factory()
                    if self._factory is not None
                    else self._owner._embedding_factory.create(self._config)
                )
            return self._model

    def embed_query(self, text: str) -> list[float]:
        """供同步语义切片调用同一向量模型。"""
        return self._get_model().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """供同步文档调用同一向量模型。"""
        return self._get_model().embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """将首次同步构造移出事件循环，再调用供应商异步接口。"""
        model = await asyncio.to_thread(self._get_model)
        return await model.aembed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """上传与查询复用相同配置和模型向量空间。"""
        model = await asyncio.to_thread(self._get_model)
        return await model.aembed_documents(texts)


class ModelManager:
    """唯一持有按配置创建的 Chat、Embedding、Reranker 和协议连接。"""

    def __init__(
        self,
        *,
        chat_factory: ChatModelFactory | None = None,
        embedding_factory: EmbedModelFactory | None = None,
        reranker_factory: RerankerFactory | None = None,
        ollama_factory: Callable[[str], OllamaClient] | None = None,
    ) -> None:
        self._chat_factory = chat_factory if chat_factory is not None else ChatModelFactory()
        self._embedding_factory = embedding_factory if embedding_factory is not None else EmbedModelFactory()
        self._reranker_factory = reranker_factory if reranker_factory is not None else RerankerFactory()
        self._ollama_factory = ollama_factory if ollama_factory is not None else OllamaClient
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._chats = {}
        self._embeddings = {}
        self._rerankers = {}
        self._ollama = {}
        self._released = set()
        self._pending_closes = {}
        self._sync_closes = {}
        self._close_executor = None

    @property
    def closed(self) -> bool:
        """返回终态标记，关闭失败也不重新开放创建。"""
        with self._lock:
            return self._closed

    def _require_open(self) -> None:
        """所有获取与延迟创建都在统一锁内检查终态。"""
        if self._closed:
            raise RuntimeError("ModelManager closed")

    def get_chat(self, config: ChatConfig) -> BaseChatModel:
        """按完整不可变配置复用模型，并发首次获取只构造一次。"""
        with self._lock:
            self._require_open()
            if config not in self._chats:
                self._chats[config] = self._chat_factory.create(config)
            return self._chats[config]

    def get_embedding(
        self, config: EmbeddingConfig, *, factory: Callable[[], Embeddings] | None = None
    ) -> LazyEmbeddings:
        """返回轻量代理；显式工厂按对象身份复用并负责自己的配置。"""
        key = ("factory", id(factory)) if factory is not None else ("configured", config)
        with self._lock:
            self._require_open()
            if key not in self._embeddings:
                self._embeddings[key] = LazyEmbeddings(self, config, factory)
            return self._embeddings[key]

    def get_reranker(self, config: RerankerConfig, *, kind: str = "causal_lm") -> LocalScorer:
        """类型与完整配置共同决定轻量评分器身份，不加载权重。"""
        if kind not in {"causal_lm", "cross_encoder"}:
            raise ValueError("unknown reranker type")
        key = (kind, config)
        with self._lock:
            self._require_open()
            if key not in self._rerankers:
                self._rerankers[key] = self._reranker_factory.create(config, kind=kind)
            return self._rerankers[key]

    def get_ollama(self, base_url: str) -> OllamaClient:
        """按明确端点复用通用协议客户端，不发起请求。"""
        key = base_url.rstrip("/")
        with self._lock:
            self._require_open()
            if key not in self._ollama:
                self._ollama[key] = self._ollama_factory(key)
            return self._ollama[key]

    def reranker_readiness(self) -> dict[str, Any]:
        """沿用 Qwen 优先的现有状态选择，只查看已经存在的评分器。"""
        with self._lock:
            entries = tuple(sorted(self._rerankers.items(), key=lambda item: item[0][0]))
        if not entries:
            return {
                "status": "not_loaded",
                "available": False,
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "error": None,
                "type": "causal_lm",
            }
        states = [{**scorer.readiness(), "type": kind} for (kind, _), scorer in entries]
        return next((state for state in states if not state["available"]), states[-1])

    def readiness(self) -> dict[str, Any]:
        """汇总本地实例状态，不展示凭据或把实例存在解释为服务可用。"""
        with self._lock:
            return {
                "closed": self._closed,
                "chat_instances": len(self._chats),
                "embedding_instances": sum(proxy._model is not None for proxy in self._embeddings.values()),
                "embedding_proxies": len(self._embeddings),
                "ollama_clients": len(self._ollama),
                "reranker": self.reranker_readiness(),
            }

    def _entries(self):
        """冻结关闭集合；未使用的代理只清理引用，不触发模型创建。"""
        with self._lock:
            self._closed = True
            entries = []
            caches = (self._chats, self._embeddings, self._rerankers, self._ollama)
            for cache in caches:
                items = (
                    sorted(cache.items(), key=lambda item: item[0][0]) if cache is self._rerankers else cache.items()
                )
                for key, value in items:
                    resource = value._model if isinstance(value, LazyEmbeddings) else value
                    entries.append((cache, key, value, close_actions(resource)))
            return entries

    def _discard_released(self, entries):
        """全部释放成功才移除资源引用，失败对象保留给后续清理。"""
        with self._lock:
            for cache, key, value, actions in entries:
                if all(id(action.target) in self._released for action in actions) and cache.get(key) is value:
                    cache.pop(key)
                    if isinstance(value, LazyEmbeddings):
                        value._model = None
            caches_empty = not any((self._chats, self._embeddings, self._rerankers, self._ollama))
            if caches_empty and self._close_executor is not None:
                self._close_executor.shutdown(wait=False)
                self._close_executor = None

    async def _run_action(self, action):
        """兼容同步 close、异步 close/aclose 及返回 awaitable 的包装方法。"""
        if action.asynchronous:
            result = action.call()
        else:
            identity = id(action.target)
            future = self._sync_closes.get(identity)
            if future is None:
                if self._close_executor is None:
                    self._close_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-close")
                future = self._close_executor.submit(action.call)
                self._sync_closes[identity] = future
            try:
                # concurrent Future 保留实际线程身份，包装 Task 被取消也不能重复关闭。
                result = await asyncio.shield(asyncio.wrap_future(future))
            except asyncio.CancelledError:
                raise
            except Exception:
                self._sync_closes.pop(identity, None)
                raise
            else:
                self._sync_closes.pop(identity, None)
        if inspect.isawaitable(result):
            await result

    async def aclose(self) -> None:
        """逐一尝试全部关闭，成功项去重；失败与取消后仍保留清理所有权。"""
        while not self._close_lock.acquire(blocking=False):
            await asyncio.sleep(0.01)
        entries = self._entries()
        errors = []
        attempted = set()
        try:
            for _, _, _, actions in entries:
                for action in actions:
                    identity = id(action.target)
                    if identity in self._released or identity in attempted:
                        continue
                    attempted.add(identity)
                    task = self._pending_closes.get(identity)
                    if task is not None and task.cancelled():
                        self._pending_closes.pop(identity, None)
                        task = None
                    if task is None:
                        task = asyncio.create_task(self._run_action(action))
                        self._pending_closes[identity] = task
                        # 调用方取消后保留实际关闭任务，也消费无人等待期间的异常。
                        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self._pending_closes.pop(identity, None)
                        errors.append(exc)
                    else:
                        self._pending_closes.pop(identity, None)
                        self._released.add(identity)
        finally:
            self._discard_released(entries)
            self._close_lock.release()
        if errors:
            raise errors[0]

    def close(self) -> None:
        """同步关闭；事件循环中存在异步资源时拒绝假关闭并保留引用。"""
        entries = self._entries()
        actions = [action for _, _, _, group in entries for action in group if id(action.target) not in self._released]
        requires_async = bool(self._pending_closes or self._sync_closes) or any(
            action.asynchronous for action in actions
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        if requires_async:
            raise RuntimeError("异步模型资源需要 await aclose()")
        if not self._close_lock.acquire(blocking=False):
            raise RuntimeError("模型资源正在关闭，需要 await aclose()")
        errors, attempted = [], set()
        try:
            for action in actions:
                identity = id(action.target)
                if identity in attempted:
                    continue
                attempted.add(identity)
                try:
                    result = action.call()
                    if inspect.isawaitable(result):
                        # 同步包装可能返回 awaitable，保留它供同一循环异步清理。
                        task = asyncio.ensure_future(result)
                        self._pending_closes[identity] = task
                        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
                        errors.append(RuntimeError("异步模型资源需要 await aclose()"))
                    else:
                        self._released.add(identity)
                except Exception as exc:
                    errors.append(exc)
        finally:
            self._discard_released(entries)
            self._close_lock.release()
        if errors:
            raise errors[0]


_default_manager = ModelManager()
_provider_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """返回默认所有者；已关闭实例只由应用生命周期显式替换。"""
    with _provider_lock:
        return _default_manager


def set_model_manager(manager: ModelManager) -> None:
    """由应用生命周期或显式注入更换默认所有者，不隐式关闭旧实例。"""
    global _default_manager
    with _provider_lock:
        _default_manager = manager
