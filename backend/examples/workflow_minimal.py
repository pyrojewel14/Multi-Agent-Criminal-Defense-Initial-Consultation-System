"""用于验证 LangGraph 咨询拓扑的确定性冒烟示例。

该示例用小型确定性函数替代依赖 LLM/RAG 的 Agent 实现，无需模型凭证即可验证
图编译、基于检查点的恢复、律师审核暂停以及显式批准流程。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

from app.orchestrator.workflow import ConsultationOrchestrator
from app.state.consultation_state import ConsultationState


async def _receptionist(state: ConsultationState) -> ConsultationState:
    state["current_agent"] = "Receptionist"
    return state


async def _fact_digger(state: ConsultationState) -> ConsultationState:
    state["current_agent"] = "FactDigger"
    state["facts_structured"] = {"incident_time": "2026-07-12", "incident_location": "示例地点"}
    state["facts_coverage_rate"] = 1.0
    state["fact_law_loop_count"] = 1
    return state


async def _risk_assessor(state: ConsultationState) -> ConsultationState:
    state["current_agent"] = "RiskAssessor"
    state["risk_assessment"] = {"risk_level": "demo"}
    return state


async def _service_planner(state: ConsultationState) -> ConsultationState:
    state["current_agent"] = "ServicePlanner"
    state["service_plan"] = {"next_step": "lawyer_review"}
    state["report_draft"] = "示例报告草案"
    return state


async def _checkpoint(orchestrator: ConsultationOrchestrator, session_id: str, stage: str) -> dict[str, Any]:
    snapshot = await orchestrator.get_snapshot(session_id)
    if snapshot is None:
        raise RuntimeError(f"missing workflow snapshot: {session_id}")
    next_node = snapshot.next[0] if snapshot.next else None
    return {
        "stage": stage,
        "current_agent": snapshot.values.get("current_agent"),
        "next": next_node,
        "finished": await orchestrator.is_workflow_finished(session_id),
    }


async def run_demo() -> list[dict[str, Any]]:
    """运行无需网络、最终由律师显式批准的工作流路径。"""
    session_id = "workflow-minimal-demo"
    initial_state: ConsultationState = {
        "consultation_id": "consultation-minimal-demo",
        "user_id": "user-minimal-demo",
        "session_id": session_id,
        "user_type": "suspect",
        "consent_given": True,
        "facts_raw": ["这是不调用真实模型的工作流拓扑示例。"],
        "facts_structured": {},
        "applied_laws": [],
        "pending_questions": [],
        "alert_triggered": False,
        "conversation_history": [],
        "lawyer_decision": None,
    }

    with (
        patch("app.orchestrator.workflow.receptionist_node", _receptionist),
        patch("app.orchestrator.workflow.fact_digger_node", _fact_digger),
        patch("app.orchestrator.workflow.risk_assessor_node", _risk_assessor),
        patch("app.orchestrator.workflow.service_planner_node", _service_planner),
    ):
        orchestrator = ConsultationOrchestrator()
        await orchestrator.start_workflow(initial_state)
        reception = await _checkpoint(orchestrator, session_id, "reception")

        await orchestrator.resume_workflow(session_id)
        review = await _checkpoint(orchestrator, session_id, "review")

        await orchestrator.resume_workflow(
            session_id,
            {
                "lawyer_decision": "approved",
                "awaiting_lawyer_review": False,
                "final_output": "律师确认后的示例输出",
            },
        )
        approved = await _checkpoint(orchestrator, session_id, "approved")

    return [reception, review, approved]


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), ensure_ascii=False, indent=2))
