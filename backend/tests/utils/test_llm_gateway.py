"""Unit tests for ``app.utils.llm_gateway.LLMGateway``.

The ``chat_model_factory`` and underlying LLM are mocked so tests run without
any real model calls.  Both happy and error paths are exercised for
``generate`` and ``generate_with_tools``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool, tool

from app.errors.exceptions import LLMServiceException, LLMTimeoutException
from app.utils import llm_gateway as gateway_module
from app.utils.llm_gateway import LLMGateway


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
