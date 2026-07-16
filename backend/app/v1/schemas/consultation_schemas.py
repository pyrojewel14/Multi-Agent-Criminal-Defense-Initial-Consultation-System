from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class ConsultationCreateRequest(BaseModel):
    user_type: str = Field(..., description="用户类型: suspect, victim, family")


class ConsultationUpdateRequest(BaseModel):
    user_type: Optional[str] = None
    consent_given: Optional[bool] = None
    facts_structured: Optional[str] = None
    applied_laws: Optional[str] = None
    final_output: Optional[str] = None


class MessageCreateRequest(BaseModel):
    content: str = Field(..., description="消息内容")
    sender_type: str = Field(..., description="发送者类型: user, agent, lawyer")
    sender_id: Optional[str] = None
    agent_name: Optional[str] = None
    message_type: str = "text"


class ConsultationResponse(BaseModel):
    id: str
    client_id: str
    client_username: Optional[str] = None
    client_real_name: Optional[str] = None
    assigned_lawyer_id: Optional[str] = None
    assigned_lawyer_name: Optional[str] = None
    user_type: str
    consent_given: bool
    status: str
    facts_structured: Optional[str] = None
    applied_laws: Optional[str] = None
    final_output: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: str
    consultation_id: str
    sender_type: str
    sender_id: Optional[str] = None
    content: str
    agent_name: Optional[str] = None
    message_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConsultationListResponse(BaseModel):
    consultations: List[ConsultationResponse]
    total: int
    page: int
    page_size: int


class AssignLawyerRequest(BaseModel):
    consultation_id: str = Field(..., description="咨询记录ID")
    lawyer_id: str = Field(..., description="律师ID")


class UpdateStatusRequest(BaseModel):
    status: str = Field(..., description="状态: pending, in_progress, completed, cancelled")


class CreateSessionRequest(BaseModel):
    """会话创建请求模型"""

    client_id: Optional[str] = Field(
        None,
        description="兼容字段；实际会话所有者始终取自 access token",
    )
    user_type: Literal["suspect", "victim", "family"] = Field(..., description="用户类型")
    initial_message: Optional[str] = Field(None, description="初始消息内容")
    source: Optional[str] = Field(None, description="来源渠道")


class CreateSessionResponse(BaseModel):
    """会话创建响应模型"""

    session_id: str = Field(..., description="会话ID")
    consultation_id: str = Field(..., description="数据库咨询记录ID")
    welcome_message: str = Field(..., description="欢迎语")
    current_agent: str = Field(..., description="当前Agent")
    created_at: datetime = Field(..., description="创建时间")


class SendMessageRequest(BaseModel):
    """发送消息请求模型"""

    session_id: str = Field(..., description="会话ID")
    content: str = Field(..., min_length=1, description="消息内容")
    message_type: Literal["text", "action", "system"] = Field(default="text", description="消息类型")
    metadata: Optional[dict[str, Any]] = Field(None, description="附加元数据")


class SendMessageResponse(BaseModel):
    """发送消息响应模型"""

    session_id: str = Field(..., description="会话ID")
    message_id: str = Field(..., description="消息ID")
    agent_name: str = Field(..., description="Agent名称")
    response_content: str = Field(default="", description="响应内容")
    is_complete: bool = Field(default=False, description="是否完成")
    pending_questions: Optional[list] = Field(None, description="待提问列表")
    alert_triggered: bool = Field(default=False, description="是否触发告警")
    created_at: datetime = Field(..., description="创建时间")


class ConfirmConsentRequest(BaseModel):
    """隐私同意确认请求模型"""

    session_id: str = Field(..., description="会话ID")
    consent_given: bool = Field(..., description="是否同意隐私协议")
    consent_timestamp: datetime = Field(..., description="同意时间戳")
    consent_version: str = Field(..., description="隐私协议版本")
    ip_address: Optional[str] = Field(None, description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理信息")
    identity_info: Optional[dict] = Field(None, description="身份信息")


class ConfirmConsentResponse(BaseModel):
    """隐私同意确认响应模型"""

    session_id: str = Field(..., description="会话ID")
    success: bool = Field(..., description="操作是否成功")
    current_agent: str = Field(..., description="当前Agent")
    next_prompt: str = Field(..., description="下一步提示")
    conversation_started: bool = Field(..., description="是否已开始对话")


class SessionStateResponse(BaseModel):
    """会话状态响应模型"""

    session_id: str = Field(..., description="会话ID")
    consultation_id: Optional[str] = Field(None, description="咨询记录ID")
    user_id: str = Field(default="", description="用户ID")
    user_type: str = Field(default="suspect", description="用户类型")
    consent_given: bool = Field(default=False, description="是否已同意隐私协议")
    current_agent: Optional[str] = Field(None, description="当前活跃的Agent名称")
    conversation_history: list = Field(default_factory=list, description="对话历史")
    facts_raw: list = Field(default_factory=list, description="原始叙述段落")
    facts_structured: Optional[dict] = Field(None, description="结构化事实")
    applied_laws: list = Field(default_factory=list, description="适用法律")
    pending_questions: list = Field(default_factory=list, description="待提问列表")
    alert_triggered: bool = Field(default=False, description="是否触发告警")
    risk_assessment: Optional[dict] = Field(None, description="风险评估")
    final_output: Optional[str] = Field(None, description="最终输出")
    lawyer_id: Optional[str] = Field(None, description="律师ID")
    status: str = Field(default="active", description="会话状态")


class ReportDraftResponse(BaseModel):
    """工作流报告草案响应模型。"""

    session_id: str = Field(..., description="会话ID")
    consultation_id: Optional[str] = Field(None, description="数据库咨询记录ID")
    report_draft: str = Field(..., description="待律师审核的报告草案")
    service_plan: Optional[dict[str, Any]] = Field(None, description="服务方案")
    awaiting_lawyer_review: bool = Field(False, description="是否等待律师审核")


class LawyerReviewRequest(BaseModel):
    """律师审核反馈请求模型"""

    decision: Literal["approved", "revise_facts", "revise_risk"] = Field(
        ..., description="律师审核决定: approved=批准报告, revise_facts=要求修改事实, revise_risk=要求修改风险评估"
    )
    feedback: Optional[str] = Field(None, description="律师反馈意见")
    final_output: Optional[str] = Field(None, description="律师确认的最终报告内容(仅当decision=approved时)")


class LawyerReviewResponse(BaseModel):
    """律师审核反馈响应模型"""

    session_id: str = Field(..., description="会话ID")
    decision: str = Field(..., description="审核决定")
    feedback: Optional[str] = Field(None, description="反馈意见")
    next_agent: Optional[str] = Field(None, description="下一步将执行的Agent")
    processed_at: datetime = Field(..., description="处理时间")


class SessionCloseRequest(BaseModel):
    """关闭会话请求模型"""

    reason: Optional[str] = Field(None, description="关闭原因")


class SessionCloseResponse(BaseModel):
    """关闭会话响应模型"""

    session_id: str = Field(..., description="会话ID")
    success: bool = Field(True, description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    closed_at: datetime = Field(..., description="关闭时间")


class SessionListItem(BaseModel):
    """会话列表项模型"""

    session_id: str = Field(..., description="会话ID")
    consultation_id: str = Field(..., description="咨询记录ID")
    user_id: str = Field(..., description="用户ID")
    user_type: str = Field(..., description="用户类型")
    current_agent: str = Field(..., description="当前Agent")
    status: str = Field(..., description="会话状态")
    consent_given: bool = Field(False, description="是否已同意隐私条款")
    alert_triggered: bool = Field(False, description="是否触发高风险告警")
    awaiting_lawyer_review: bool = Field(False, description="是否等待律师审核")
    risk_level: Optional[str] = Field(None, description="风险等级")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class SessionListResponse(BaseModel):
    """会话列表响应模型"""

    sessions: List[SessionListItem] = Field(default_factory=list, description="会话列表")
    total: int = Field(0, description="总数")
