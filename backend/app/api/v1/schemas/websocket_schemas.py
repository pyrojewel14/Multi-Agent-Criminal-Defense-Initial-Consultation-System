from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class WebSocketMessage(BaseModel):
    """WebSocket消息基类模型"""

    type: str = Field(..., description="消息类型")
    session_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")
    message_id: Optional[str] = Field(None, description="消息ID")
    correlation_id: Optional[str] = Field(None, description="关联ID用于追踪")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class WSAgentMessage(BaseModel):
    """WebSocket Agent消息模型"""

    type: Literal["agent_message"] = Field(default="agent_message", description="消息类型")
    session_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")
    message_id: Optional[str] = Field(None, description="消息ID")
    correlation_id: Optional[str] = Field(None, description="关联ID")
    agent_name: str = Field(..., description="Agent名称")
    content: str = Field(..., description="消息内容")
    format: Literal["text", "markdown", "structured"] = Field(default="text", description="内容格式")
    actions_available: Optional[List[str]] = Field(None, description="可用操作列表")


class WSActionRequired(BaseModel):
    """WebSocket需要操作消息模型"""

    type: Literal["action_required"] = Field(default="action_required", description="消息类型")
    session_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")
    message_id: Optional[str] = Field(None, description="消息ID")
    correlation_id: Optional[str] = Field(None, description="关联ID")
    action_type: str = Field(..., description="操作类型")
    title: str = Field(..., description="操作标题")
    description: str = Field(..., description="操作描述")
    options: List[dict[str, str]] = Field(default_factory=list, description="操作选项列表")
    required: bool = Field(default=True, description="是否必需")
    timeout: Optional[int] = Field(None, description="超时时间（秒）")


class WSNotice(BaseModel):
    """WebSocket通知消息模型"""

    type: Literal["notice"] = Field(default="notice", description="消息类型")
    session_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")
    message_id: Optional[str] = Field(None, description="消息ID")
    correlation_id: Optional[str] = Field(None, description="关联ID")
    notice_type: Literal["info", "warning", "success", "progress"] = Field(..., description="通知类型")
    title: str = Field(..., description="通知标题")
    content: str = Field(..., description="通知内容")
    auto_dismiss: bool = Field(default=False, description="是否自动消失")
    dismiss_after: Optional[int] = Field(None, description="自动消失时间（秒）")


class WSError(BaseModel):
    """WebSocket错误消息模型"""

    type: Literal["error"] = Field(default="error", description="消息类型")
    session_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")
    message_id: Optional[str] = Field(None, description="消息ID")
    correlation_id: Optional[str] = Field(None, description="关联ID")
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误消息")
    details: Optional[dict[str, Any]] = Field(None, description="错误详情")
    recoverable: bool = Field(default=True, description="是否可恢复")
    retry_available: bool = Field(default=False, description="是否可重试")
