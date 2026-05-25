from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class LawyerSessionItem(BaseModel):
    """律师端会话列表项"""

    id: str = Field(..., description="会话ID")
    client_id: str = Field(..., description="客户ID")
    client_username: Optional[str] = Field(None, description="客户用户名")
    client_real_name: Optional[str] = Field(None, description="客户真实姓名")
    user_type: str = Field(..., description="用户类型: suspect, victim, family")
    status: str = Field(..., description="会话状态")
    risk_level: Optional[str] = Field(None, description="风险等级: low, medium, high, critical")
    alert_triggered: bool = Field(False, description="是否触发高风险告警")
    lawyer_review_needed: bool = Field(False, description="是否需要律师审核")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    class Config:
        from_attributes = True


class MessageHistoryItem(BaseModel):
    """消息历史项"""

    id: str
    sender_type: str = Field(..., description="发送者类型: user, agent, lawyer")
    sender_id: Optional[str] = None
    content: str
    agent_name: Optional[str] = None
    message_type: str = "text"
    created_at: datetime

    class Config:
        from_attributes = True


class LawyerSessionDetail(BaseModel):
    """律师端会话详情"""

    id: str
    client_id: str
    client_username: Optional[str] = None
    client_real_name: Optional[str] = None
    user_type: str
    consent_given: bool
    status: str
    risk_level: Optional[str] = None
    alert_triggered: bool
    lawyer_review_needed: bool
    facts_raw: Optional[List[str]] = Field(None, description="原始叙述段落")
    facts_structured: Optional[dict] = Field(None, description="结构化案件事实")
    applied_laws: Optional[List[dict]] = Field(None, description="适用的法律法规")
    risk_assessment: Optional[dict] = Field(None, description="风险评估结果")
    report_draft: Optional[str] = Field(None, description="报告草稿")
    service_plan: Optional[dict] = Field(None, description="服务计划")
    final_output: Optional[str] = Field(None, description="最终输出")
    conversation_history: Optional[List[MessageHistoryItem]] = Field(None, description="对话历史")
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ApproveReportRequest(BaseModel):
    """审核报告请求"""

    final_output: str = Field(..., description="律师确认的最终报告内容")
    feedback: Optional[str] = Field(None, description="律师反馈意见")

    class Config:
        from_attributes = True


class ApproveReportResponse(BaseModel):
    """审核报告响应"""

    success: bool = Field(True, description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    session_id: str = Field(..., description="会话ID")
    approved_at: datetime = Field(..., description="批准时间")

    class Config:
        from_attributes = True


class RejectSessionRequest(BaseModel):
    """退回会话请求"""

    target_node: str = Field(..., description="退回目标节点: fact_digger, risk_assessor")
    reason: Optional[str] = Field(None, description="退回原因")
    feedback: Optional[str] = Field(None, description="具体修改要求")

    class Config:
        from_attributes = True


class RejectSessionResponse(BaseModel):
    """退回会话响应"""

    success: bool = Field(True, description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    session_id: str = Field(..., description="会话ID")
    target_node: str = Field(..., description="退回目标节点")
    rejected_at: datetime = Field(..., description="退回时间")

    class Config:
        from_attributes = True


class InterventionResponse(BaseModel):
    """人工接管会话响应"""

    success: bool = Field(True, description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    session_id: str = Field(..., description="会话ID")
    intervened_at: datetime = Field(..., description="接管时间")
    current_agent: str = Field("Lawyer", description="当前活跃的agent")

    class Config:
        from_attributes = True


class RiskAlertItem(BaseModel):
    """高风险告警项"""

    id: str
    session_id: str = Field(..., description="关联的会话ID")
    client_id: str = Field(..., description="客户ID")
    client_real_name: Optional[str] = Field(None, description="客户真实姓名")
    risk_type: str = Field(..., description="风险类型")
    risk_level: str = Field(..., description="风险等级: low, medium, high, critical")
    details: Optional[str] = Field(None, description="风险详情")
    is_read: bool = Field(False, description="是否已读")
    created_at: datetime = Field(..., description="告警时间")

    class Config:
        from_attributes = True


class LawyerSessionListResponse(BaseModel):
    """律师端会话列表响应"""

    sessions: List[LawyerSessionItem]
    total: int
    page: int
    page_size: int
