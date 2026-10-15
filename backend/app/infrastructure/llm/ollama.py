"""通用 Ollama HTTP 协议；领域提示词、预算与完整性检查由调用方负责。"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

import httpx


class OllamaClient:
    """复用明确端点的异步连接，首次请求才创建连接池。"""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._closed = False
        self._lock = threading.Lock()

    def _get_client(self) -> httpx.AsyncClient:
        """保护请求与关闭竞争，已关闭客户端不得重新创建连接。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("Ollama client closed")
            if self._client is None:
                self._client = httpx.AsyncClient(trust_env=False)
            return self._client

    async def tags(self, *, timeout_s: float = 10) -> Mapping[str, Any]:
        """读取服务模型列表，不解释模型名或 digest。"""
        response = await self._get_client().get(self.base_url + "/api/tags", timeout=timeout_s)
        response.raise_for_status()
        return response.json()

    async def embed(
        self, model: str, texts: Sequence[str], *, truncate: bool, timeout_s: float = 60
    ) -> Mapping[str, Any]:
        """传递批量嵌入请求，保留调用方指定的截断策略。"""
        response = await self._get_client().post(
            self.base_url + "/api/embed",
            json={"model": model, "input": list(texts), "truncate": truncate},
            timeout=timeout_s,
        )
        response.raise_for_status()
        return response.json()

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: Mapping[str, Any],
        think: bool = False,
        timeout_s: float = 30,
    ) -> Mapping[str, Any]:
        """执行非流式生成，不追加提示词、重试或预算扣减。"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "options": dict(options),
        }
        response = await self._get_client().post(self.base_url + "/api/generate", json=payload, timeout=timeout_s)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        """关闭已建立的连接池，失败时保留引用供重试。"""
        with self._lock:
            self._closed = True
            client = self._client
        if client is not None:
            await client.aclose()
            with self._lock:
                if self._client is client:
                    self._client = None
