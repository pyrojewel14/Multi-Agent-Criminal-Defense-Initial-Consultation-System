from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolCall
from langchain_core.tools import BaseTool

from app.errors.exceptions import LLMServiceException, LLMTimeoutException
from app.utils.factory import chat_model_factory
from app.utils.logger import get_logger


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

    def __init__(self):
        """初始化 LLM 网关。"""
        self._logger = get_logger("LLMGateway")
        self._logger.info("【__init__】LLM 网关初始化完成")

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

        try:
            model = chat_model_factory.create_precise_model(actual_temp)
            response = await model.ainvoke(messages)
        except TimeoutError as e:
            self._logger.error("【generate】LLM 超时: %s", e)
            raise LLMTimeoutException(detail=str(e)) from e
        except Exception as e:
            self._logger.error("【generate】LLM 服务错误: %s", e)
            raise LLMServiceException(detail=str(e)) from e

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
        user_msg_preview = user_message[:200] + "..." if len(user_message) > 200 else user_message
        sys_prompt_preview = system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt

        self._logger.info(
            "【generate_with_tools】请求开始: temp=%.2f, is_legal=%s, tools=%s, user_msg_len=%d",
            actual_temp,
            is_legal,
            tool_names,
            len(user_message),
        )
        self._logger.debug(
            "【generate_with_tools】System prompt: %s",
            sys_prompt_preview,
        )
        self._logger.debug(
            "【generate_with_tools】User message: %s",
            user_msg_preview,
        )

        messages: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        try:
            model = chat_model_factory.create_precise_model(actual_temp)
            model_with_tools = model.bind_tools(tools)
            self._logger.debug("【generate_with_tools】绑定工具: %s", tool_names)

            response = await model_with_tools.ainvoke(messages)
            self._logger.debug("【generate_with_tools】LLM 响应完成")
        except TimeoutError as e:
            self._logger.error("【generate_with_tools】LLM 超时: %s", e)
            raise LLMTimeoutException(detail=str(e)) from e
        except Exception as e:
            self._logger.error("【generate_with_tools】LLM 服务错误: %s", e)
            raise LLMServiceException(detail=str(e)) from e

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
                self._logger.debug(
                    "【generate_with_tools】工具 %s 完整参数: %s",
                    tool_name,
                    tool_args,
                )

        # 记录返回内容
        content = _content_to_text(response.content) if hasattr(response, "content") else ""
        content_preview = content[:200] + "..." if len(content) > 200 else content
        self._logger.info(
            "【generate_with_tools】响应完成: has_tool_call=%s, content_len=%d",
            len(tool_calls) > 0,
            len(content),
        )
        if content:
            self._logger.debug(
                "【generate_with_tools】响应内容: %s",
                content_preview,
            )

        result = {
            "content": content,
            "tool_calls": tool_calls,
            "has_tool_call": len(tool_calls) > 0,
        }
        return result


llm_gateway = LLMGateway()
