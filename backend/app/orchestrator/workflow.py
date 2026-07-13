from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from pydantic import TypeAdapter, ValidationError

from app.agents.fact_digger import fact_digger_node
from app.agents.human_alert import human_alert_node
from app.agents.law_ref import law_ref_node
from app.agents.receptionist import receptionist_node
from app.agents.risk_assessor import risk_assessor_node
from app.agents.service_planner import service_planner_node
from app.errors.exceptions import LLMServiceException, LLMTimeoutException
from app.security.disclaimer import disclaimer
from app.state.consultation_state import ConsultationState
from app.utils.logger import get_logger

_logger = get_logger("Orchestrator")

COVERAGE_THRESHOLD = 0.8

# 需要等待外部输入的节点，执行后自动中断
INTERRUPT_AFTER_NODES = ["receptionist", "wait_for_user", "human_review", "human_alert"]

WorkflowGraph = StateGraph[ConsultationState, None, ConsultationState, ConsultationState]
CompiledWorkflowGraph = CompiledStateGraph[ConsultationState, None, ConsultationState, ConsultationState]
_STATE_ADAPTER: TypeAdapter[ConsultationState] = TypeAdapter(ConsultationState)


def _validate_workflow_state(value: object) -> ConsultationState:
    """校验 LangGraph 状态值，同时保留框架使用的元数据键。"""
    if not isinstance(value, dict):
        raise ValueError("工作流返回了无效状态: 结果不是字典")

    known_state = {key: value[key] for key in ConsultationState.__annotations__ if key in value}
    try:
        _STATE_ADAPTER.validate_python(known_state)
    except ValidationError as exc:
        raise ValueError("工作流返回了无效状态") from exc

    return ConsultationState(**value)


def check_consent(state: ConsultationState) -> Literal["continue", "end"]:
    """条件边：根据用户是否同意决定流程走向。

    Args:
        state: 当前咨询状态

    Returns:
        "continue" - 同意，继续到 FactDigger
        "end" - 不同意，结束流程
    """
    if state.get("consent_given"):
        _logger.debug("用户已同意，继续流程")
        return "continue"
    _logger.debug("用户未同意，结束流程")
    return "end"


def check_facts_sufficient(state: ConsultationState) -> Literal["complete", "loop", "alert", "max_loop"]:
    """条件边：根据事实收集覆盖度决定流程走向。

    Args:
        state: 当前咨询状态

    Returns:
        "complete" - 覆盖度 >= 80%，继续到 RiskAssessor
        "loop" - 覆盖度 < 80%，等待用户输入后继续追问
        "alert" - 触发人工介入
        "max_loop" - 达到最大循环次数，强制进入 RiskAssessor
    """
    if state.get("alert_triggered"):
        _logger.info("检测到高风险内容，触发人工介入")
        return "alert"

    # 循环计数保护
    loop_count = state.get("fact_law_loop_count", 0)
    max_loops = 10
    if loop_count >= max_loops:
        _logger.warning(
            "【check_facts_sufficient】达到最大循环次数 %d，强制进入风险评估",
            max_loops,
        )
        return "max_loop"

    coverage_rate = state.get("facts_coverage_rate") or 0.0

    if coverage_rate >= COVERAGE_THRESHOLD:
        _logger.info("事实覆盖度 %.2f >= %.2f，流程完成", coverage_rate, COVERAGE_THRESHOLD)
        return "complete"

    _logger.info("事实覆盖度 %.2f < %.2f，需要继续追问", coverage_rate, COVERAGE_THRESHOLD)
    return "loop"


def lawyer_decision(state: ConsultationState) -> Literal["approved", "revise_facts", "revise_risk", "wait"]:
    """条件边：根据律师审核决策决定流程走向。

    Args:
        state: 当前咨询状态

    Returns:
        "approved" - 报告已批准，结束流程
        "revise_facts" - 需要修改事实，返回 FactDigger
        "revise_risk" - 需要修改风险评估，返回 RiskAssessor
        "wait" - 尚无有效律师决定，保持在 HumanReview 断点
    """
    decision = state.get("lawyer_decision")

    if decision == "revise_facts":
        _logger.info("律师决策：需要修改事实，返回 FactDigger")
        return "revise_facts"
    elif decision == "revise_risk":
        _logger.info("律师决策：需要修改风险评估，返回 RiskAssessor")
        return "revise_risk"

    if decision == "approved":
        _logger.info("律师决策：报告已批准，流程结束")
        return "approved"

    _logger.info("尚未收到有效律师决策，继续等待人工审核")
    return "wait"


async def human_review_node(state: ConsultationState) -> ConsultationState:
    """HumanReview Agent 节点函数 - 律师审核节点

    等待律师对服务方案和报告草案进行审核，
    根据律师决策更新状态。

    Args:
        state: 当前 ConsultationState

    Returns:
        更新后的 ConsultationState
    """
    _logger.info("【human_review_node】律师审核节点开始执行")

    session_id = state.get("session_id", "unknown")
    decision = state.get("lawyer_decision")

    if decision in {"approved", "revise_facts", "revise_risk"}:
        state["awaiting_lawyer_review"] = False
        state["current_agent"] = "HumanReview"
        if "conversation_history" not in state:
            state["conversation_history"] = []
        state["conversation_history"].append(
            {
                "agent": "HumanReview",
                "action": "review_decision",
                "session_id": session_id,
                "decision": decision,
                "feedback": state.get("lawyer_feedback"),
            }
        )
        _logger.info("【human_review_node】收到律师决策: %s", decision)
        return state

    report_draft = state.get("report_draft", "")
    service_plan = state.get("service_plan", {})

    if report_draft:
        state["final_output"] = disclaimer.inject(f"""【律师审核请求】

您好，以下是系统生成的初期咨询报告草案，请您审核：

{report_draft}

请选择：
1. 批准此报告
2. 要求修改事实收集
3. 要求修改风险评估
""")
    else:
        state["final_output"] = disclaimer.inject("报告草案尚未生成，请稍后重试。")

    state["awaiting_lawyer_review"] = True
    state["current_agent"] = "HumanReview"

    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append(
        {
            "agent": "HumanReview",
            "action": "awaiting_review",
            "session_id": session_id,
            "has_report": bool(report_draft),
            "has_service_plan": bool(service_plan),
        }
    )

    _logger.info("【human_review_node】等待律师审核，session_id: %s", session_id)

    return state


async def wait_for_user_node(state: ConsultationState) -> ConsultationState:
    """等待用户输入的中转节点。

    当 FactDigger 判定覆盖度不足时，流程经此节点后中断，
    等待用户发送下一条消息。恢复后自动进入 LawRef 检索法条，
    再回到 FactDigger 继续收集事实。
    """
    _logger.info("【wait_for_user_node】等待用户输入，session_id: %s", state.get("session_id", "unknown"))
    return state


def _calculate_coverage_rate(state: ConsultationState) -> float:
    """
    已废弃
    计算当前事实覆盖度。

    注意：跳过未验证的 RAG 检索结果，只使用 JSON 知识库的结果计算覆盖度，
    与 fact_digger._analyze_coverage 保持一致。

    Args:
        state: 当前咨询状态

    Returns:
        覆盖度百分比 (0.0 - 1.0)
    """
    facts_structured = state.get("facts_structured", {})
    applied_laws = state.get("applied_laws", [])

    if not applied_laws:
        return 0.0

    # 延迟导入避免循环依赖
    from app.agents.law_ref import _is_unverified_rag_result

    # 过滤掉未验证的 RAG 结果，只用 JSON 知识库验证过的结果计算覆盖度
    json_laws = [law for law in applied_laws if not _is_unverified_rag_result(law)]

    if not json_laws:
        return 0.0

    total_elements = 0
    covered_elements = 0

    for law in json_laws:
        elements = law.get("elements", [])
        total_elements += len(elements)

        for element in elements:
            if isinstance(element, dict):
                element_key = element.get("key", element.get("name", ""))
            else:
                element_key = str(element)
            fact_value = _get_fact_value(facts_structured, element_key)

            # 与 _analyze_coverage 一致：空列表和 False 算作已覆盖（弱要素）
            if fact_value is not None and fact_value != "":
                covered_elements += 1

    if total_elements == 0:
        return 0.0

    return covered_elements / total_elements


def _get_fact_value(facts_structured: Dict[str, Any], key: str) -> Any:
    """从结构化事实中获取指定键的值。

    Args:
        facts_structured: 结构化事实数据
        key: 要获取的键名

    Returns:
        键对应的值，如果不存在返回 None
    """
    key_mapping = {
        "time": "incident_time",
        "location": "incident_location",
        "parties": "parties",
        "behavior": "behavior_sequence",
        "consequence": "consequence",
        "evidence": "evidence_mentioned",
        "arrest": "arrest_status",
        "surrender": "surrender",
        "forgiveness": "victim_forgiveness",
        "record": "prior_record",
    }

    mapped_key = key_mapping.get(key, key)
    return facts_structured.get(mapped_key)


class ConsultationOrchestrator:
    """封装完整多 Agent 咨询流程的 LangGraph StateGraph。

    工作流拓扑（使用 checkpointer + interrupt_after 实现自动流转与人工断点）：

        START → Receptionist ──[interrupt]──→ [consent_given?]
                                               ├── continue → FactDigger → [coverage?]
                                               │     ├── complete → RiskAssessor → ServicePlanner → HumanReview ──[interrupt]
                                               │     │                                                              ↓
                                               │     │                                                        [lawyer_decision]
                                               │     │                                                              ↓
                                               │     │                                                              ├── wait → HumanReview(保持中断)
                                               │     ├── loop → WaitForUser ──[interrupt]──→ LawRef → FactDigger(循环)
                                               │     ├── alert → HumanAlert ──[interrupt]──→ END
                                               │     └── max_loop → RiskAssessor
                                               └── end → END

    核心方法：
        start_workflow()  - 启动新会话，从 START 执行到第一个中断点
        resume_workflow() - 恢复会话，从中断点继续执行到下一个中断点
    """

    def __init__(self):
        self._logger = get_logger("Orchestrator")
        self._checkpointer = MemorySaver()
        self._compiled: Optional[CompiledWorkflowGraph] = None
        # 保留 _active_sessions 用于 get_active_sessions 等兼容接口
        self._active_sessions: Dict[str, ConsultationState] = {}

    def _build_workflow(self) -> WorkflowGraph:
        """构建工作流 DAG，包含所有 Agent 节点和条件边。

        Returns:
            配置完成的 StateGraph 实例
        """
        workflow: WorkflowGraph = StateGraph(ConsultationState)

        workflow.add_node("receptionist", receptionist_node)
        workflow.add_node("fact_digger", fact_digger_node)
        workflow.add_node("law_ref", law_ref_node)
        workflow.add_node("risk_assessor", risk_assessor_node)
        workflow.add_node("service_planner", service_planner_node)
        workflow.add_node("human_review", human_review_node)
        workflow.add_node("human_alert", human_alert_node)
        workflow.add_node("wait_for_user", wait_for_user_node)

        workflow.set_entry_point("receptionist")

        # Receptionist → 根据同意状态分流
        workflow.add_conditional_edges("receptionist", check_consent, {"continue": "fact_digger", "end": END})

        # FactDigger → 根据覆盖度分流（移除了旧的无条件边 fact_digger → law_ref）
        workflow.add_conditional_edges(
            "fact_digger",
            check_facts_sufficient,
            {
                "complete": "risk_assessor",
                "loop": "wait_for_user",  # 覆盖度不足 → 等待用户输入
                "alert": "human_alert",
                "max_loop": "risk_assessor",
            },
        )

        # WaitForUser → LawRef → FactDigger（用户输入后自动检索法条再回到事实收集）
        workflow.add_edge("wait_for_user", "law_ref")
        workflow.add_edge("law_ref", "fact_digger")

        # 后半段线性链 + 律师审核条件边
        workflow.add_edge("risk_assessor", "service_planner")
        workflow.add_edge("service_planner", "human_review")
        workflow.add_conditional_edges(
            "human_review",
            lawyer_decision,
            {
                "approved": END,
                "revise_facts": "fact_digger",
                "revise_risk": "risk_assessor",
                "wait": "human_review",
            },
        )

        workflow.add_edge("human_alert", END)

        return workflow

    def _ensure_compiled(self):
        """确保工作流已编译（带 checkpointer 和 interrupt_after）。"""
        if self._compiled is None:
            workflow = self._build_workflow()
            self._compiled = workflow.compile(
                checkpointer=self._checkpointer,
                interrupt_after=INTERRUPT_AFTER_NODES,
            )
            self._logger.debug("工作流编译完成（含 checkpointer + interrupt_after）")

    def _compiled_graph(self) -> CompiledWorkflowGraph:
        """获取已编译工作流，供类型检查器识别非 None。"""
        self._ensure_compiled()
        if self._compiled is None:
            raise RuntimeError("工作流编译失败")
        return self._compiled

    def _config(self, session_id: str) -> RunnableConfig:
        """生成 LangGraph checkpointer 配置。"""
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        return config

    # ------------------------------------------------------------------
    # 核心方法：start / resume
    # ------------------------------------------------------------------

    async def start_workflow(self, initial_state: ConsultationState) -> ConsultationState:
        """启动新会话的工作流，从 START 执行到第一个中断点。

        Args:
            initial_state: 初始咨询状态

        Returns:
            执行到中断点时的状态

        Raises:
            LLMServiceException: LLM 服务异常
            LLMTimeoutException: LLM 调用超时
        """
        compiled = self._compiled_graph()

        session_id = initial_state.get("session_id", "unknown")
        config = self._config(session_id)

        self._logger.info("启动工作流: session_id=%s", session_id)
        self._active_sessions[session_id] = initial_state.copy()

        try:
            result = _validate_workflow_state(await compiled.ainvoke(initial_state, config))
            self._active_sessions[session_id] = result
            self._logger.info(
                "工作流中断: session_id=%s, current_agent=%s",
                session_id,
                result.get("current_agent", "unknown"),
            )
            return result
        except (LLMServiceException, LLMTimeoutException) as e:
            self._logger.error("工作流异常终止: session_id=%s, error=%s", session_id, e.code.value)
            self._cleanup_session(session_id)
            raise
        except Exception as e:
            self._logger.error("工作流执行失败: session_id=%s, error=%s", session_id, str(e))
            self._cleanup_session(session_id)
            raise

    async def resume_workflow(
        self,
        session_id: str,
        state_updates: Optional[Dict[str, Any]] = None,
    ) -> ConsultationState:
        """从断点恢复工作流，执行到下一个中断点。

        先通过 aupdate_state 更新状态（如用户消息、律师决策等），
        然后调用 ainvoke(None, config) 从断点继续执行。

        Args:
            session_id: 会话 ID
            state_updates: 需要合并到当前状态中的更新字段

        Returns:
            执行到下一个中断点时的状态

        Raises:
            ValueError: 会话不存在
            LLMServiceException: LLM 服务异常
            LLMTimeoutException: LLM 调用超时
        """
        compiled = self._compiled_graph()

        config = self._config(session_id)

        # 检查会话是否存在
        snapshot = await compiled.aget_state(config)
        if snapshot.values is None or not snapshot.values:
            raise ValueError(f"会话不存在: {session_id}")

        # 如果有状态更新，先写入 checkpointer
        if state_updates:
            await compiled.aupdate_state(config, state_updates, as_node=None)

        snapshot = await compiled.aget_state(config)
        self._logger.info("恢复工作流: session_id=%s, next=%s", session_id, snapshot.next)

        try:
            result = _validate_workflow_state(await compiled.ainvoke(None, config))
            self._active_sessions[session_id] = result
            self._logger.info(
                "工作流中断: session_id=%s, current_agent=%s",
                session_id,
                result.get("current_agent", "unknown"),
            )
            return result
        except (LLMServiceException, LLMTimeoutException) as e:
            self._logger.error("工作流异常终止: session_id=%s, error=%s", session_id, e.code.value)
            self._cleanup_session(session_id)
            raise
        except Exception as e:
            self._logger.error("工作流执行失败: session_id=%s, error=%s", session_id, str(e))
            self._cleanup_session(session_id)
            raise

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    async def get_snapshot(self, session_id: str) -> Optional[StateSnapshot]:
        """获取会话的 LangGraph 状态快照。

        Args:
            session_id: 会话 ID

        Returns:
            StateSnapshot，如果会话不存在返回 None
        """
        compiled = self._compiled_graph()
        config = self._config(session_id)
        snapshot = await compiled.aget_state(config)
        if snapshot.values is None or not snapshot.values:
            return None
        return snapshot

    async def get_next_node(self, session_id: str) -> Optional[str]:
        """获取会话的下一个待执行节点名称。

        Args:
            session_id: 会话 ID

        Returns:
            下一个节点名称，如果流程已结束返回 None
        """
        snapshot = await self.get_snapshot(session_id)
        if snapshot is None:
            return None
        next_nodes = snapshot.next
        if not next_nodes:
            return None
        return next_nodes[0]

    async def is_workflow_finished(self, session_id: str) -> bool:
        """判断工作流是否已执行完毕（到达 END）。

        Args:
            session_id: 会话 ID

        Returns:
            True 表示已结束
        """
        snapshot = await self.get_snapshot(session_id)
        if snapshot is None:
            return True
        return len(snapshot.next) == 0

    # ------------------------------------------------------------------
    # 兼容接口（供现有代码逐步迁移）
    # ------------------------------------------------------------------

    def get_session_context(self, session_id: str) -> Optional[ConsultationState]:
        """获取会话上下文（同步，从内存缓存读取）。

        注意：此方法从 _active_sessions 读取，可能不是最新状态。
        如需最新状态，请使用 get_snapshot()。

        Args:
            session_id: 会话 ID

        Returns:
            会话状态，如果不存在返回 None
        """
        return self._active_sessions.get(session_id)

    def update_session_context(self, session_id: str, updates: ConsultationState) -> bool:
        """更新会话上下文（同步，写入内存缓存）。

        注意：此方法仅更新 _active_sessions，不写入 checkpointer。
        如需持久化到工作流，请使用 resume_workflow()。

        Args:
            session_id: 会话 ID
            updates: 要更新的字段

        Returns:
            更新是否成功
        """
        if session_id not in self._active_sessions:
            self._logger.warning("会话不存在: %s", session_id)
            return False

        self._active_sessions[session_id].update(updates)
        self._logger.debug("会话上下文已更新: %s", session_id)
        return True

    def _update_session_context(self, session_id: str, state: ConsultationState) -> None:
        """更新会话上下文（内部使用）。"""
        self.update_session_context(session_id, state)

    def _cleanup_session(self, session_id: str) -> None:
        """清理会话上下文。"""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            self._logger.debug("会话上下文已清理: %s", session_id)

    def get_active_sessions(self) -> Dict[str, ConsultationState]:
        """获取所有活跃会话。"""
        return self._active_sessions.copy()

    async def process_lawyer_feedback(
        self, session_id: str, decision: str, feedback: Optional[str] = None
    ) -> ConsultationState:
        """处理律师反馈，更新状态并通过工作流自动路由到下一节点。

        Args:
            session_id: 会话 ID
            decision: 律师决策 (approved/revise_facts/revise_risk)
            feedback: 律师反馈内容

        Returns:
            更新后的状态
        """
        self._logger.info("处理律师反馈: session_id=%s, decision=%s", session_id, decision)

        state_updates = {
            "lawyer_decision": decision,
            "lawyer_feedback": feedback,
            "awaiting_lawyer_review": False,
        }

        result = await self.resume_workflow(session_id, state_updates)
        return result


orchestrator = ConsultationOrchestrator()


if __name__ == "__main__":
    print("Orchestrator 模块加载成功")
    import asyncio

    async def test_workflow():
        test_state: ConsultationState = {
            "session_id": "test_session",
            "consultation_id": "test_consultation",
            "user_id": "test_user",
            "consent_given": True,
            "user_type": "suspect",
            "facts_raw": [],
            "facts_structured": {},
            "applied_laws": [],
            "pending_questions": [],
            "alert_triggered": False,
            "conversation_history": [],
        }

        result = await orchestrator.start_workflow(test_state)
        print(f"工作流启动完成，当前节点: {result.get('current_agent', 'unknown')}")

    asyncio.run(test_workflow())
