import asyncio
import hashlib
import os
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Dict, List, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolCall
from langchain_core.tools import BaseTool

from app.errors.exceptions import LLMServiceException, LLMTimeoutException
from app.observability.tracing import (
    SessionBudgetExceeded,
    build_safe_metadata,
    cost_estimator,
    current_trace_context,
    session_budget,
    trace_span,
    trace_store,
)
from app.utils.factory import chat_model_factory
from app.utils.logger import get_logger

try:
    import httpx
except ImportError:  # pragma: no cover - httpx 是项目依赖，仅保留最小降级兼容
    httpx = None  # type: ignore[assignment]


ResultT = TypeVar("ResultT")


def _model_name(model: object) -> str:
    """从不同供应商模型对象提取稳定名称。"""
    for attribute in ("model", "model_name"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _usage_tokens(response: object) -> tuple[int | str, int | str]:
    """读取常见 LangChain usage 结构；缺失时明确返回 unknown。"""
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            candidate = response_metadata.get("token_usage") or response_metadata.get("usage")
            usage = candidate if isinstance(candidate, dict) else None
    if not isinstance(usage, dict):
        return "unknown", "unknown"

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    normalized_input = input_tokens if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) else "unknown"
    normalized_output = output_tokens if isinstance(output_tokens, int) and not isinstance(output_tokens, bool) else "unknown"
    return normalized_input, normalized_output


def _prompt_version(system_prompt: str) -> str:
    """以 prompt hash 标记版本，不保存 prompt 原文。"""
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _default_jitter(delay: float) -> float:
    """为重试退避增加小幅有界抖动，避免并发请求同步重试。"""
    return random.uniform(delay * 0.8, delay * 1.2)


@dataclass(frozen=True)
class LLMCallPolicy:
    """供应商无关的 LLM deadline、重试上限和退避策略。"""

    total_timeout_seconds: float = 30.0
    attempt_timeout_seconds: float = 15.0
    max_attempts: int = 2
    backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.total_timeout_seconds <= 0 or self.attempt_timeout_seconds <= 0:
            raise ValueError("LLM timeout 必须大于 0")
        if not 1 <= self.max_attempts <= 2:
            raise ValueError("LLM attempt 上限必须为 1 或 2")
        if self.backoff_seconds < 0:
            raise ValueError("LLM retry backoff 不能小于 0")

    @classmethod
    def from_env(cls) -> "LLMCallPolicy":
        """从环境变量加载策略，缺省值保持小而显式。"""
        return cls(
            total_timeout_seconds=float(os.getenv("LLM_TOTAL_TIMEOUT_SECONDS", "30")),
            attempt_timeout_seconds=float(os.getenv("LLM_ATTEMPT_TIMEOUT_SECONDS", "15")),
            max_attempts=int(os.getenv("LLM_MAX_ATTEMPTS", "2")),
            backoff_seconds=float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "0.25")),
        )


def _content_to_text(content: object) -> str:
    """将 LangChain 文本或结构化 content block 归一化为纯文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return "".join(text_parts)


class LLMGateway:
    """统一 LLM 调用入口。

    支持 ALIYUN（DashScope / Qwen3）和 OLLAMA（本地）两种后端，
    通过 LLM_TYPE 环境变量切换。法律查询强制使用 temperature=0。
    """

    def __init__(
        self,
        policy: LLMCallPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float], float] = _default_jitter,
    ):
        """初始化 LLM 网关。"""
        self._logger = get_logger("LLMGateway")
        self._policy = policy or LLMCallPolicy.from_env()
        self._sleep = sleep
        self._jitter = jitter
        self._logger.info("【__init__】LLM 网关初始化完成")

    @staticmethod
    def _status_code(exc: BaseException) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    @staticmethod
    def _is_network_error(exc: BaseException) -> bool:
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        return bool(httpx is not None and isinstance(exc, httpx.TransportError))

    @classmethod
    def _is_retryable(cls, exc: BaseException) -> bool:
        status_code = cls._status_code(exc)
        return (
            status_code == 429
            or status_code in {500, 502, 503, 504}
            or cls._is_network_error(exc)
        )

    async def _invoke_with_policy(
        self,
        operation: str,
        invoke: Callable[[int], Awaitable[ResultT]],
    ) -> ResultT:
        """在统一总 deadline 内执行有限、可审计的瞬态重试。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._policy.total_timeout_seconds
        last_error: BaseException | None = None

        for attempt in range(1, self._policy.max_attempts + 1):
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            attempt_timeout = min(self._policy.attempt_timeout_seconds, remaining)
            self._logger.info(
                "【llm_attempt】operation=%s attempt=%d outcome=start",
                operation,
                attempt,
            )
            try:
                async with asyncio.timeout(attempt_timeout):
                    result = await invoke(attempt)
            except SessionBudgetExceeded:
                raise
            except Exception as exc:
                last_error = exc
                status_code = self._status_code(exc)
                retryable = self._is_retryable(exc)
                outcome = "timeout" if isinstance(exc, TimeoutError) else "error"
                self._logger.warning(
                    "【llm_attempt】operation=%s attempt=%d outcome=%s retryable=%s status=%s error_type=%s",
                    operation,
                    attempt,
                    outcome,
                    retryable,
                    status_code if status_code is not None else "none",
                    type(exc).__name__,
                )
                if not retryable or attempt >= self._policy.max_attempts:
                    break

                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                retry_delay = max(
                    0.0,
                    min(self._jitter(self._policy.backoff_seconds), remaining),
                )
                try:
                    async with asyncio.timeout(remaining):
                        await self._sleep(retry_delay)
                except TimeoutError as exc:
                    last_error = exc
                    break
            else:
                self._logger.info(
                    "【llm_attempt】operation=%s attempt=%d outcome=success",
                    operation,
                    attempt,
                )
                return result

        status_code = self._status_code(last_error) if last_error else None
        detail = (
            f"operation={operation}, attempts<={self._policy.max_attempts}, "
            f"error_type={type(last_error).__name__ if last_error else 'deadline'}, "
            f"status={status_code if status_code is not None else 'none'}"
        )
        if isinstance(last_error, TimeoutError) or status_code == 504 or loop.time() >= deadline:
            raise LLMTimeoutException(detail=detail) from last_error
        raise LLMServiceException(detail=detail) from last_error

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        is_legal: bool = False,
    ) -> str:
        """调用 LLM 并返回文本响应。

        Args:
            system_prompt: 系统的指令内容。
            user_message: 用户的输入文本。
            temperature: 采样温度 (0.0-1.0)，法律场景强制为 0。
            is_legal: 是否为法律场景，为 True 时强制使用 temperature=0。

        Returns:
            模型的文本输出。

        Raises:
            LLMTimeoutException: 上游 API 超时。
            LLMServiceException: 上游 API 返回错误。
        """
        actual_temp = 0.0 if is_legal else temperature
        self._logger.debug(
            "【generate】LLM 调用: temp=%.2f, is_legal=%s, msg_len=%d",
            actual_temp,
            is_legal,
            len(user_message),
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        async def invoke(attempt: int):
            model = chat_model_factory.create_precise_model(actual_temp)
            context = current_trace_context()
            session_id = str(context["session_id"]) if context and context["session_id"] else "unknown"
            if session_id != "unknown":
                session_budget.reserve_call(session_id)
            with trace_span(
                trace_store,
                event_type="llm",
                name="generate",
                model=_model_name(model),
                prompt_version=_prompt_version(system_prompt),
                attempt=attempt,
                cache_hit=False,
            ) as event:
                try:
                    response = await model.ainvoke(messages)
                except TimeoutError:
                    event.outcome = "timeout"
                    raise
                input_tokens, output_tokens = _usage_tokens(response)
                event.input_tokens = input_tokens
                event.output_tokens = output_tokens
                event.cost_usd = cost_estimator.estimate(
                    event.model or "unknown", input_tokens, output_tokens
                )
                if session_id != "unknown":
                    session_budget.record_tokens(
                        session_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                return response

        response = await self._invoke_with_policy("generate", invoke)

        content = _content_to_text(response.content)
        self._logger.debug("【generate】LLM 响应: len=%d", len(content))
        return content

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[BaseTool],
        temperature: float = 0.1,
        is_legal: bool = False,
    ) -> Dict[str, Any]:
        """调用 LLM 并支持 Function Calling。

        Args:
            system_prompt: 系统的指令内容。
            user_message: 用户的输入文本。
            tools: 可用的工具列表（使用 @tool 装饰的函数）。
            temperature: 采样温度 (0.0-1.0)，法律场景强制为 0。
            is_legal: 是否为法律场景，为 True 时强制使用 temperature=0。

        Returns:
            {
                "content": str,           # 文本内容（如果模型选择不调用工具）
                "tool_calls": List[Dict], # 工具调用列表
                "has_tool_call": bool,    # 是否调用了工具
            }

        Raises:
            LLMTimeoutException: 上游 API 超时。
            LLMServiceException: 上游 API 返回错误。
        """
        actual_temp = 0.0 if is_legal else temperature
        tool_names = [t.name for t in tools]
        self._logger.info(
            "【generate_with_tools】请求开始: temp=%.2f, is_legal=%s, tools=%s, user_msg_len=%d",
            actual_temp,
            is_legal,
            tool_names,
            len(user_message),
        )

        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        async def invoke(attempt: int):
            model = chat_model_factory.create_precise_model(actual_temp)
            model_with_tools = model.bind_tools(tools)
            self._logger.debug("【generate_with_tools】绑定工具: %s", tool_names)
            context = current_trace_context()
            session_id = str(context["session_id"]) if context and context["session_id"] else "unknown"
            if session_id != "unknown":
                session_budget.reserve_call(session_id)
            with trace_span(
                trace_store,
                event_type="llm",
                name="generate_with_tools",
                model=_model_name(model),
                prompt_version=_prompt_version(system_prompt),
                attempt=attempt,
                cache_hit=False,
                metadata=build_safe_metadata({"tool_names": sorted(tool_names)}),
            ) as event:
                try:
                    response = await model_with_tools.ainvoke(messages)
                except TimeoutError:
                    event.outcome = "timeout"
                    raise
                input_tokens, output_tokens = _usage_tokens(response)
                event.input_tokens = input_tokens
                event.output_tokens = output_tokens
                event.cost_usd = cost_estimator.estimate(
                    event.model or "unknown", input_tokens, output_tokens
                )
                if session_id != "unknown":
                    session_budget.record_tokens(
                        session_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                return response

        response = await self._invoke_with_policy("generate_with_tools", invoke)
        self._logger.debug("【generate_with_tools】LLM 响应完成")

        # 提取工具调用
        tool_calls: List[ToolCall] = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls.extend(response.tool_calls)
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                # 记录调用的工具名和参数键（不记录参数值，避免敏感信息）
                arg_keys = list(tool_args.keys()) if isinstance(tool_args, dict) else "N/A"
                self._logger.info(
                    "【generate_with_tools】工具调用: name=%s, args_keys=%s",
                    tool_name,
                    arg_keys,
                )

        # 记录返回内容
        content = _content_to_text(response.content) if hasattr(response, "content") else ""
        self._logger.info(
            "【generate_with_tools】响应完成: has_tool_call=%s, content_len=%d",
            len(tool_calls) > 0,
            len(content),
        )

        result = {
            "content": content,
            "tool_calls": tool_calls,
            "has_tool_call": len(tool_calls) > 0,
        }
        return result


llm_gateway = LLMGateway()
