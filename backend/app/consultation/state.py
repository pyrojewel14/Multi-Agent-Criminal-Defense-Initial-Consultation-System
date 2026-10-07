from datetime import datetime
from typing import List, Optional, TypedDict

from pydantic import TypeAdapter, ValidationError

from app.consultation.memory.schemas import MemoryState


class ConsultationState(TypedDict, total=False):
    """LangGraph 工作流中供所有 Agent 共享的全局状态。"""

    # 会话标识与用户身份
    consultation_id: str  # 咨询会话的唯一标识符，用于追踪和关联整个咨询过程
    user_id: str  # 用户 ID，标识进行咨询的用户身份
    session_id: str  # 会话 ID，用于标识当前咨询会话
    user_type: Optional[str]  # 用户类型；接待阶段尚未识别时为 None
    user_role: Optional[str]  # 用户在系统中的角色（client/lawyer/admin），用于 RBAC 权限判断
    identity_info: Optional[dict]  # 用户身份详细信息（姓名、联系方式等；当前不掩码），Receptionist 阶段收集
    consent_given: bool  # 是否已获得用户的知情同意

    # 当前输入与多轮对话
    current_input: Optional[str]  # 用户最新一条输入消息，Agent 节点从中读取当前轮内容
    current_message_id: Optional[str]  # 本轮外部消息来源 ID；内部调用可为空
    memory: MemoryState  # checkpoint 唯一保存派生摘要、近期上下文与案件字段记忆
    message_audit: Optional[dict]  # 有界待修复回执，修复不得再次运行图节点
    lifecycle_audit: Optional[dict]  # 当前生命周期原始载荷，审计修复时复用
    conversation_history: List[dict]  # 对话历史记录，用于维护多轮对话的上下文
    pending_questions: List[str]  # 待提问的后续问题列表

    # 案件事实与构成要件覆盖度
    facts_raw: List[str]  # 近期原始陈述；完整外部原文以业务消息表为准
    facts_structured: dict  # 通过 LLM 函数调用提取的结构化案件事实
    facts_coverage_rate: Optional[float]  # 构成要件覆盖度（0.0-1.0），FactDigger 用于判断是否继续追问
    element_to_law_mapping: Optional[dict]  # 构成要件到法条的映射，LawRef 生成后供 FactDigger 计算覆盖度

    # 法条检索与适用依据
    applied_laws: List[dict]  # 法条候选，包含枚举来源、权威 required_elements 及模型判定
    law_search_status: Optional[str]  # 法条检索状态（success/text_only/missing_facts/no_law_match/dependency_failure）
    law_text_candidates: List[dict]  # 可读取的正文参考，不参与罪名适用或覆盖分母
    law_research: dict  # LawRef 局部工具循环的非敏感轨迹、计数与终止原因
    rag_only: bool  # 是否仅命中未经 JSON 知识库验证的 RAG 结果

    # 事实与法条循环：次数、终止原因及失败记录
    fact_law_loop_count: int  # FactDigger/LawRef 循环次数，用于防止工作流无限追问
    fact_law_attempts: List[dict]  # 每次事实与法条循环的结构化审计记录
    fact_law_termination_reason: Optional[str]  # 循环终止原因，供降级处理和审计使用
    fact_law_failure_streak: int  # 当前重试窗口内任意非事实失败的连续总数
    fact_law_last_failure: Optional[str]  # 当前重试窗口内最近一次非事实失败类型

    # 风险识别与评估
    alert_triggered: bool  # 是否触发高风险陈述警报
    risk_assessment: Optional[dict]  # 可选的风险评估结果

    # 律师分配与人工审核
    lawyer_id: Optional[str]  # 可选的律师 ID，分配给此案件的律师标识
    lawyer_review_needed: bool  # 是否需要律师审核
    awaiting_lawyer_review: Optional[bool]  # 是否正在等待律师审核，HumanReview 节点设置
    lawyer_decision: Optional[str]  # 律师审核决定（approved/revise_facts/revise_risk），外部 API 写入
    lawyer_feedback: Optional[str]  # 律师审核反馈意见，退回时附带的修改建议

    # 报告生成与后续服务
    report_draft: Optional[str]  # 可选的报告草稿，在咨询过程中生成的中期报告
    final_output: str  # 最终输出内容，包括完整的咨询报告和建议
    service_plan: Optional[dict]  # 可选的服务计划，包含后续法律服务建议

    # 工作流运行与跨存储一致性
    current_agent: str  # 当前活跃的 Agent 名称
    workflow_status: Optional[str]  # 工作流运行状态，依赖失败终止时为 degraded
    repair_required: bool  # 生命周期命令跨存储失败后，是否必须先执行修复
    consistency_error: Optional[dict]  # 非敏感的一致性失败阶段和操作标记
    command_processed_at: datetime  # 应用命令首次成功时间，仅用于稳定重放响应

    # 结构化产物、来源与校验结果
    artifact_results: dict  # 按 fact/law/risk/service 保存结构化产物状态
    degraded_reason: Optional[str]  # 最近一次产物降级原因
    source: Optional[str]  # 最近一次产物来源
    validation_errors: List[dict]  # 最近一次 schema 错误（不含原始输入）


_STATE_ADAPTER: TypeAdapter[ConsultationState] = TypeAdapter(ConsultationState)


def validate_consultation_state(value: object) -> ConsultationState:
    """校验动态来源的咨询状态，并保留 LangGraph 元数据键。

    Args:
        value: Checkpointer、Redis 或 LangGraph 返回的动态值。

    Returns:
        类型校验通过的咨询状态副本。

    Raises:
        ValueError: 状态不是字典，或已知字段的值类型不合法。
    """
    if not isinstance(value, dict):
        raise ValueError("咨询状态无效：状态必须是字典")

    known_state = {key: value[key] for key in ConsultationState.__annotations__ if key in value}
    try:
        _STATE_ADAPTER.validate_python(known_state)
    except ValidationError as exc:
        raise ValueError("咨询状态无效") from exc

    return ConsultationState(**value)
