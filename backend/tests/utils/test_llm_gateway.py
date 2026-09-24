"""Unit tests for ``app.utils.llm_gateway.LLMGateway``.

The ``chat_model_factory`` and underlying LLM are mocked so tests run without
any real model calls.  Both happy and error paths are exercised for
``generate`` and ``generate_with_tools``.
"""

import asyncio
import logging
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool, tool

from app.errors.exceptions import LLMServiceException, LLMTimeoutException
from app.observability.tracing import (
    BoundedTraceStore,
    SessionBudget,
    SessionBudgetExceeded,
    TokenCostEstimator,
    trace_span,
)
from app.utils import llm_gateway as gateway_module
from app.utils.llm_gateway import LLMCallPolicy, LLMGateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(content: object = "hello", tool_calls=None):
    """Build a fake ``AIMessage``-like response with ``content`` and ``tool_calls``."""
    response = MagicMock()
    response.content = content
    response.tool_calls = tool_calls or []
    return response


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_records_model_usage_and_duration_for_each_attempt(self, monkeypatch):
        """漏记 usage 或把 retry 合并成一次会让成本数据失真。"""
        store = BoundedTraceStore(max_events=20)
        budget = SessionBudget(max_calls=5, max_tokens=100)
        monkeypatch.setattr(gateway_module, "trace_store", store, raising=False)
        monkeypatch.setattr(gateway_module, "session_budget", budget, raising=False)
        monkeypatch.setattr(
            gateway_module,
            "cost_estimator",
            TokenCostEstimator({"qwen-test": {"input": 1.0, "output": 2.0}}),
            raising=False,
        )
        gateway = LLMGateway(policy=LLMCallPolicy(max_attempts=2, backoff_seconds=0))

        class UpstreamError(Exception):
            status_code = 503

        success_response = _make_response("ok")
        success_response.usage_metadata = {"input_tokens": 11, "output_tokens": 7}
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.model = "qwen-test"
            model.ainvoke = AsyncMock(side_effect=[UpstreamError("temporary"), success_response])
            mock_factory.create_precise_model.return_value = model

            with trace_span(
                store,
                event_type="node",
                name="risk_assessor",
                request_id="req-usage",
                session_id="session-usage",
                node="risk_assessor",
            ):
                assert await gateway.generate("system-v1", "private facts") == "ok"

        attempts = [event for event in store.events() if event.event_type == "llm"]
        assert [(event.attempt, event.outcome) for event in attempts] == [(1, "error"), (2, "success")]
        assert attempts[1].model == "qwen-test"
        assert attempts[1].input_tokens == 11
        assert attempts[1].output_tokens == 7
        assert attempts[1].duration_ms is not None
        assert attempts[1].prompt_version.startswith("sha256:")
        assert attempts[1].cost_usd == 0.000025
        assert budget.snapshot("session-usage") == {"calls": 2, "tokens": 18}
        assert "private facts" not in str(attempts[1].to_dict())

    @pytest.mark.asyncio
    async def test_missing_provider_usage_is_recorded_as_unknown(self, monkeypatch):
        store = BoundedTraceStore(max_events=10)
        monkeypatch.setattr(gateway_module, "trace_store", store, raising=False)
        monkeypatch.setattr(
            gateway_module,
            "session_budget",
            SessionBudget(max_calls=5, max_tokens=100),
            raising=False,
        )
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.model_name = "provider-model"
            response = _make_response("ok")
            response.usage_metadata = None
            response.response_metadata = {}
            model.ainvoke = AsyncMock(return_value=response)
            mock_factory.create_precise_model.return_value = model
            with trace_span(
                store,
                event_type="node",
                name="fact_digger",
                request_id="req-unknown",
                session_id="session-unknown",
            ):
                await gateway.generate("system", "facts")

        attempt = next(event for event in store.events() if event.event_type == "llm")
        assert attempt.input_tokens == "unknown"
        assert attempt.output_tokens == "unknown"

    @pytest.mark.asyncio
    async def test_call_budget_exhaustion_remains_typed(self, monkeypatch):
        store = BoundedTraceStore(max_events=10)
        budget = SessionBudget(max_calls=1, max_tokens=100)
        monkeypatch.setattr(gateway_module, "trace_store", store, raising=False)
        monkeypatch.setattr(gateway_module, "session_budget", budget, raising=False)
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(return_value=_make_response("ok"))
            mock_factory.create_precise_model.return_value = model
            with trace_span(
                store,
                event_type="node",
                name="fact_digger",
                request_id="req-budget",
                session_id="session-budget",
            ):
                await gateway.generate("system", "first")
                with pytest.raises(SessionBudgetExceeded):
                    await gateway.generate("system", "second")

    @pytest.mark.asyncio
    async def test_returns_content_on_success(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(return_value=_make_response("ok"))
            mock_factory.create_precise_model.return_value = model

            result = await gateway.generate("system", "user")

        assert result == "ok"
        mock_factory.create_precise_model.assert_called_once_with(0.1)

    @pytest.mark.asyncio
    async def test_is_legal_forces_zero_temperature(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(return_value=_make_response("legal answer"))
            mock_factory.create_precise_model.return_value = model

            await gateway.generate("system", "user", temperature=0.7, is_legal=True)

        mock_factory.create_precise_model.assert_called_once_with(0.0)

    @pytest.mark.asyncio
    async def test_uses_supplied_temperature_when_not_legal(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(return_value=_make_response("ok"))
            mock_factory.create_precise_model.return_value = model

            await gateway.generate("system", "user", temperature=0.5, is_legal=False)

        mock_factory.create_precise_model.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_raises_timeout_exception_on_asyncio_timeout(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(side_effect=TimeoutError("timed out"))
            mock_factory.create_precise_model.return_value = model

            with pytest.raises(LLMTimeoutException):
                await gateway.generate("system", "user")

    @pytest.mark.asyncio
    async def test_raises_service_exception_on_generic_error(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(side_effect=RuntimeError("api error"))
            mock_factory.create_precise_model.return_value = model

            with pytest.raises(LLMServiceException):
                await gateway.generate("system", "user")

    @pytest.mark.asyncio
    async def test_handles_empty_content(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(return_value=_make_response(""))
            mock_factory.create_precise_model.return_value = model

            result = await gateway.generate("system", "user")

        assert result == ""

    @pytest.mark.asyncio
    async def test_normalizes_structured_content_blocks_to_text(self):
        """LangChain 的结构化 content block 应转换为稳定的文本返回值。"""
        content = ["前缀", {"type": "text", "text": "正文"}, {"type": "image_url"}]
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(return_value=_make_response(content))
            mock_factory.create_precise_model.return_value = model

            result = await gateway.generate("system", "user")

        assert result == "前缀正文"

    @pytest.mark.asyncio
    async def test_never_returning_model_hits_typed_total_deadline(self):
        """永不返回的模型必须在总 deadline 内终止为类型化 504。"""
        gateway = LLMGateway(
            policy=LLMCallPolicy(
                total_timeout_seconds=0.02,
                attempt_timeout_seconds=0.1,
                max_attempts=2,
                backoff_seconds=0,
            )
        )
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()

            async def never_returns(*_args, **_kwargs):
                await asyncio.Event().wait()

            model.ainvoke = AsyncMock(side_effect=never_returns)
            mock_factory.create_precise_model.return_value = model

            with pytest.raises(LLMTimeoutException) as exc_info:
                await gateway.generate("system", "user")

        assert exc_info.value.status_code == 504
        assert model.ainvoke.await_count == 1

    @pytest.mark.asyncio
    async def test_per_attempt_timeout_can_retry_within_total_deadline(self):
        """单次超时可重试，但第二次仍共享同一总时间预算。"""
        gateway = LLMGateway(
            policy=LLMCallPolicy(
                total_timeout_seconds=0.05,
                attempt_timeout_seconds=0.01,
                max_attempts=2,
                backoff_seconds=0,
            )
        )
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()

            async def first_times_out_then_succeeds(*_args, **_kwargs):
                if model.ainvoke.await_count == 1:
                    await asyncio.Event().wait()
                return _make_response("ok")

            model.ainvoke = AsyncMock(side_effect=first_times_out_then_succeeds)
            mock_factory.create_precise_model.return_value = model

            assert await gateway.generate("system", "user") == "ok"

        assert model.ainvoke.await_count == 2

    @pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_retries_only_explicit_transient_http_statuses(self, status_code):
        """明确的限流和短暂 5xx 允许在上限内重试。"""

        class UpstreamError(Exception):
            def __init__(self, status: int):
                self.status_code = status

        sleeps = []

        async def record_sleep(delay: float):
            sleeps.append(delay)

        gateway = LLMGateway(
            policy=LLMCallPolicy(max_attempts=2, backoff_seconds=0.25),
            sleep=record_sleep,
            jitter=lambda delay: delay,
        )
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(
                side_effect=[UpstreamError(status_code), _make_response("ok")]
            )
            mock_factory.create_precise_model.return_value = model

            assert await gateway.generate("system", "user") == "ok"

        assert model.ainvoke.await_count == 2
        assert sleeps == [0.25]

    @pytest.mark.asyncio
    async def test_retry_backoff_uses_injected_jitter(self):
        """生产退避可带抖动，测试可注入确定性 jitter。"""
        sleeps = []

        async def record_sleep(delay: float):
            sleeps.append(delay)

        gateway = LLMGateway(
            policy=LLMCallPolicy(max_attempts=2, backoff_seconds=0.2),
            sleep=record_sleep,
            jitter=lambda delay: delay + 0.05,
        )
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(
                side_effect=[ConnectionError("reset"), _make_response("ok")]
            )
            mock_factory.create_precise_model.return_value = model

            assert await gateway.generate("system", "user") == "ok"

        assert sleeps == [0.25]

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    @pytest.mark.asyncio
    async def test_non_retryable_4xx_fails_after_one_attempt(self, status_code):
        """普通 4xx 不得消耗第二次模型调用。"""

        class UpstreamError(Exception):
            def __init__(self, status: int):
                self.status_code = status

        gateway = LLMGateway(policy=LLMCallPolicy(max_attempts=2))
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(side_effect=UpstreamError(status_code))
            mock_factory.create_precise_model.return_value = model

            with pytest.raises(LLMServiceException):
                await gateway.generate("system", "user")

        assert model.ainvoke.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_network_connection_error_once(self):
        """网络连接瞬态错误可重试，但仍受两次 attempt 上限约束。"""
        gateway = LLMGateway(
            policy=LLMCallPolicy(max_attempts=2, backoff_seconds=0),
        )
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            model = MagicMock()
            model.ainvoke = AsyncMock(
                side_effect=[ConnectionError("reset"), _make_response("ok")]
            )
            mock_factory.create_precise_model.return_value = model

            assert await gateway.generate("system", "user") == "ok"

        assert model.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_attempt_logs_exclude_prompt_fact_and_raw_exception_text(self):
        """审计日志只记录 attempt/outcome，不泄露 prompt、事实或异常原文。"""

        class UpstreamError(Exception):
            status_code = 400

        secret = "敏感案件事实-不可记录"
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        gateway = LLMGateway(policy=LLMCallPolicy(max_attempts=2))
        gateway._logger.addHandler(handler)
        try:
            with patch.object(gateway_module, "chat_model_factory") as mock_factory:
                model = MagicMock()
                model.ainvoke = AsyncMock(side_effect=UpstreamError(secret))
                mock_factory.create_precise_model.return_value = model

                with pytest.raises(LLMServiceException):
                    await gateway.generate(secret, secret)
        finally:
            gateway._logger.removeHandler(handler)

        logged = stream.getvalue()
        assert "attempt=1" in logged
        assert "outcome=error" in logged
        assert secret not in logged


# ---------------------------------------------------------------------------
# generate_with_tools
# ---------------------------------------------------------------------------


@tool
def _echo_tool(message: str) -> str:
    """Echo the input back."""
    return message


class TestGenerateWithTools:
    @pytest.mark.asyncio
    async def test_returns_no_tool_call_when_response_has_none(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            bound = MagicMock()
            bound.ainvoke = AsyncMock(return_value=_make_response("just text"))
            model = MagicMock()
            model.bind_tools = MagicMock(return_value=bound)
            mock_factory.create_precise_model.return_value = model

            result = await gateway.generate_with_tools(
                "system", "user", [_echo_tool]
            )

        assert result["content"] == "just text"
        assert result["tool_calls"] == []
        assert result["has_tool_call"] is False
        model.bind_tools.assert_called_once_with([_echo_tool])

    @pytest.mark.asyncio
    async def test_extracts_tool_calls(self):
        gateway = LLMGateway()
        tool_call = {"name": "_echo_tool", "args": {"message": "hi"}, "id": "call-1"}
        response_with_tc = MagicMock()
        response_with_tc.content = ""
        response_with_tc.tool_calls = [tool_call]

        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            bound = MagicMock()
            bound.ainvoke = AsyncMock(return_value=response_with_tc)
            model = MagicMock()
            model.bind_tools = MagicMock(return_value=bound)
            mock_factory.create_precise_model.return_value = model

            result = await gateway.generate_with_tools(
                "system", "user", [_echo_tool]
            )

        assert result["has_tool_call"] is True
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "_echo_tool"
        assert result["tool_calls"][0]["args"] == {"message": "hi"}

    @pytest.mark.asyncio
    async def test_is_legal_forces_zero_temperature(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            bound = MagicMock()
            bound.ainvoke = AsyncMock(return_value=_make_response("ok"))
            model = MagicMock()
            model.bind_tools = MagicMock(return_value=bound)
            mock_factory.create_precise_model.return_value = model

            await gateway.generate_with_tools(
                "system", "user", [_echo_tool], temperature=0.7, is_legal=True
            )

        mock_factory.create_precise_model.assert_called_once_with(0.0)

    @pytest.mark.asyncio
    async def test_raises_timeout_exception(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            bound = MagicMock()
            bound.ainvoke = AsyncMock(side_effect=TimeoutError("slow"))
            model = MagicMock()
            model.bind_tools = MagicMock(return_value=bound)
            mock_factory.create_precise_model.return_value = model

            with pytest.raises(LLMTimeoutException):
                await gateway.generate_with_tools(
                    "system", "user", [_echo_tool]
                )

    @pytest.mark.asyncio
    async def test_raises_service_exception_on_error(self):
        gateway = LLMGateway()
        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            bound = MagicMock()
            bound.ainvoke = AsyncMock(side_effect=RuntimeError("model error"))
            model = MagicMock()
            model.bind_tools = MagicMock(return_value=bound)
            mock_factory.create_precise_model.return_value = model

            with pytest.raises(LLMServiceException):
                await gateway.generate_with_tools(
                    "system", "user", [_echo_tool]
                )

    @pytest.mark.asyncio
    async def test_handles_response_without_tool_calls_attribute(self):
        """If the response has no ``tool_calls`` attribute, treat as no tool call."""
        gateway = LLMGateway()
        # Build a response object without ``tool_calls`` (use spec= to limit attrs)
        response = MagicMock(spec=["content"])
        response.content = "ok"

        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            bound = MagicMock()
            bound.ainvoke = AsyncMock(return_value=response)
            model = MagicMock()
            model.bind_tools = MagicMock(return_value=bound)
            mock_factory.create_precise_model.return_value = model

            result = await gateway.generate_with_tools(
                "system", "user", [_echo_tool]
            )

        assert result["has_tool_call"] is False
        assert result["tool_calls"] == []
        assert result["content"] == "ok"

    @pytest.mark.asyncio
    async def test_empty_tool_calls_list_treated_as_no_call(self):
        """A response whose ``tool_calls`` is an empty list should not flag a call."""
        gateway = LLMGateway()
        response = _make_response("ok", tool_calls=[])

        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            bound = MagicMock()
            bound.ainvoke = AsyncMock(return_value=response)
            model = MagicMock()
            model.bind_tools = MagicMock(return_value=bound)
            mock_factory.create_precise_model.return_value = model

            result = await gateway.generate_with_tools(
                "system", "user", [_echo_tool]
            )

        assert result["has_tool_call"] is False

    @pytest.mark.asyncio
    async def test_normalizes_structured_content_blocks(self):
        """Function Calling 响应也必须向上层暴露纯文本 content。"""
        gateway = LLMGateway()
        response = _make_response([{"type": "text", "text": "工具前说明"}])

        with patch.object(gateway_module, "chat_model_factory") as mock_factory:
            bound = MagicMock()
            bound.ainvoke = AsyncMock(return_value=response)
            model = MagicMock()
            model.bind_tools = MagicMock(return_value=bound)
            mock_factory.create_precise_model.return_value = model

            result = await gateway.generate_with_tools("system", "user", [_echo_tool])

        assert result["content"] == "工具前说明"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_llm_gateway_singleton_exists(self):
        """The module exposes a single ``llm_gateway`` instance."""
        assert isinstance(gateway_module.llm_gateway, LLMGateway)
