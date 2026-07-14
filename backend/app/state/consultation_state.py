from typing import List, Optional, TypedDict

from pydantic import TypeAdapter, ValidationError


class ConsultationState(TypedDict, total=False):
    """LangGraph 工作流中供所有 Agent 共享的全局状态。"""

    consultation_id: str  # 咨询会话的唯一标识符，用于追踪和关联整个咨询过程
    user_id: str  # 用户ID，标识进行咨询的用户身份
    session_id: str  # 会话ID，用于标识当前咨询会话
    user_type: Optional[str]  # 用户类型；接待阶段尚未识别时为 None
    consent_given: bool  # 是否已获得用户的知情同意
    facts_raw: List[str]  # 原始用户叙述段落，包含未经处理的案件描述
    facts_structured: dict  # 通过LLM函数调用提取的结构化案件事实
    applied_laws: List[dict]  # 法律法规检索结果，包含罪名、条款、相关案例
    current_agent: str  # 当前活跃的agent名称
    pending_questions: List[str]  # 待提问的后续问题列表
    alert_triggered: bool  # 是否触发高风险陈述警报
    risk_assessment: Optional[dict]  # 可选的风险评估结果
    lawyer_review_needed: bool  # 是否需要律师审核
    final_output: str  # 最终输出内容，包括完整的咨询报告和建议
    conversation_history: List[dict]  # 对话历史记录，用于维护多轮对话的上下文
    report_draft: Optional[str]  # 可选的报告草稿，在咨询过程中生成的中期报告
    service_plan: Optional[dict]  # 可选的服务计划，包含后续法律服务建议
    lawyer_id: Optional[str]  # 可选的律师ID，分配给此案件的律师标识
    current_input: Optional[str]  # 用户最新一条输入消息，Agent 节点从中读取当前轮内容
    facts_coverage_rate: Optional[float]  # 构成要件覆盖度（0.0-1.0），FactDigger 用于判断是否继续追问
    fact_law_loop_count: int  # FactDigger/LawRef 循环次数，用于防止工作流无限追问
    element_to_law_mapping: Optional[dict]  # 构成要件到法条的映射，LawRef 生成后供 FactDigger 计算覆盖度
    identity_info: Optional[dict]  # 用户身份详细信息（姓名脱敏、联系方式等），Receptionist 阶段收集
    user_role: Optional[str]  # 用户在系统中的角色（client/lawyer/admin），用于 RBAC 权限判断
    awaiting_lawyer_review: Optional[bool]  # 是否正在等待律师审核，HumanReview 节点设置
    lawyer_decision: Optional[str]  # 律师审核决定（approved/revise_facts/revise_risk），外部 API 写入
    lawyer_feedback: Optional[str]  # 律师审核反馈意见，退回时附带的修改建议
    rag_only: bool  # 是否仅命中未经 JSON 知识库验证的 RAG 结果


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
