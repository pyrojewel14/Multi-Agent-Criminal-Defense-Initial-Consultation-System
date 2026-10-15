"""显式注入生成依赖，假设正文仅供额外召回使用。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from app.infrastructure.observability.tracing import (
    current_trace_context,
    session_budget,
    trace_span,
    trace_store,
)


async def generate_hyde(query: str, *, client, model_name: str) -> Mapping[str, Any]:
    """构造检索领域提示词，通用 Ollama 客户端只执行供应商协议。"""
    return await client.generate(
        model_name,
        "为检索生成简短刑法候选正文，不得补造事实、忽略否认或作有罪结论。只输出可能相关的条文用语，不输出解释。检索查询："
        + query,
        options={"temperature": 0, "num_predict": 192, "seed": 0},
        think=False,
        timeout_s=30,
    )


class HyDEExpander:
    """对一次生成保留会话预算、调用树和供应商 token 使用量。"""

    def __init__(
        self, generate: Callable[[str], Awaitable[Mapping[str, Any]]], *, model_name: str, trace_name: str = "full_hyde"
    ):
        self.generate = generate
        self.model_name = model_name
        self.trace_name = trace_name

    async def expand(self, query: str, *, timeout_s: float) -> str:
        """生成非空候选用语；预算耗尽、超时和取消交由上层区分处理。"""
        context = current_trace_context()
        session_id = str(context["session_id"]) if context and context.get("session_id") else None
        with trace_span(trace_store, event_type="llm", name=self.trace_name, model=self.model_name, attempt=1) as event:
            if session_id:
                session_budget.reserve_call(session_id)
            data = await asyncio.wait_for(self.generate(query), timeout=timeout_s)
            event.input_tokens = data.get("prompt_eval_count", "unknown")
            event.output_tokens = data.get("eval_count", "unknown")
            if session_id:
                session_budget.record_tokens(
                    session_id, input_tokens=event.input_tokens, output_tokens=event.output_tokens
                )
            text = data.get("response", "")
            if not isinstance(text, str) or not text.strip():
                event.outcome = "empty"
                raise ValueError("empty HyDE output")
            return text.strip()[:1200]
