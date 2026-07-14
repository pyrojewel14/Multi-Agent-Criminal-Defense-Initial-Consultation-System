"""使用标准 case 运行可复现的 LangGraph Demo。

该示例保留真实工作流拓扑、条件边、中断和恢复机制，并用 case 中明确标记的
确定性夹具替代 LLM 与 RAG 节点输出。它用于验证端到端控制流和状态契约，
不用于证明模型抽取质量、检索命中率或法律结论准确性。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from app.agents.human_alert import human_alert_node
from app.orchestrator.workflow import ConsultationOrchestrator
from app.state.consultation_state import ConsultationState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = PROJECT_ROOT / "demos" / "consultation" / "cases"
REQUIRED_CASE_KEYS = {
    "id",
    "title",
    "category",
    "user_input",
    "expected_fact_fields",
    "expected_law_keywords",
    "should_follow_up",
    "should_trigger_human",
    "expected_workflow_finished",
    "expected_requires_human_intervention",
    "expected_final_output_type",
    "demo_fixture",
}


def load_case(case_id: str) -> dict[str, Any]:
    """加载并校验标准 Demo case。"""
    case_path = CASE_DIR / f"{case_id}.json"
    if not case_path.is_file():
        available = ", ".join(sorted(path.stem for path in CASE_DIR.glob("*.json")))
        raise ValueError(f"Demo case 不存在: {case_id}；可用 case: {available}")

    case = json.loads(case_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CASE_KEYS - case.keys())
    if missing:
        raise ValueError(f"Demo case 缺少字段: {', '.join(missing)}")
    return case


def _law_keywords_present(case: dict[str, Any], laws: list[dict[str, Any]]) -> bool:
    searchable = json.dumps(laws, ensure_ascii=False)
    return all(keyword in searchable for keyword in case["expected_law_keywords"])


async def run_case(case_id: str = "ordinary_assault") -> dict[str, Any]:
    """运行一个标准 Demo case，并返回机器可读的状态轨迹。"""
    case = load_case(case_id)
    fixture = case["demo_fixture"]
    trace: list[dict[str, Any]] = []

    async def receptionist(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "Receptionist"
        trace.append({"stage": "consent_gate", "agent": "Receptionist", "consent_given": False})
        return state

    async def fact_digger(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "FactDigger"
        state["facts_structured"] = case["expected_fact_fields"]
        state["fact_law_loop_count"] = 1

        if case["should_trigger_human"]:
            state["alert_triggered"] = True
            state["risk_assessment"] = fixture["risk_assessment"]
            trace.append(
                {
                    "stage": "risk_gate",
                    "agent": "FactDigger",
                    "alert_triggered": True,
                    "risk_assessment": fixture["risk_assessment"],
                }
            )
            return state

        laws = fixture.get("candidate_laws", [])
        state["applied_laws"] = laws
        if case["should_follow_up"]:
            questions = fixture["pending_questions"]
            state["facts_coverage_rate"] = 0.4
            state["pending_questions"] = questions
            state["final_output"] = "为了更准确地分析案件，请补充：\n" + "\n".join(
                f"{index}. {question}" for index, question in enumerate(questions, 1)
            )
        else:
            state["facts_coverage_rate"] = 1.0
            state["pending_questions"] = []

        trace.append(
            {
                "stage": "fact_and_law_contract",
                "agent": "FactDigger",
                "facts_structured": state["facts_structured"],
                "candidate_laws": laws,
                "facts_coverage_rate": state["facts_coverage_rate"],
                "pending_questions": state["pending_questions"],
            }
        )
        return state

    async def law_ref(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "LawRef"
        state["applied_laws"] = fixture.get("candidate_laws", [])
        return state

    async def risk_assessor(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "RiskAssessor"
        state["risk_assessment"] = fixture["risk_assessment"]
        trace.append(
            {
                "stage": "risk_assessment",
                "agent": "RiskAssessor",
                "risk_assessment": state["risk_assessment"],
            }
        )
        return state

    async def service_planner(state: ConsultationState) -> ConsultationState:
        state["current_agent"] = "ServicePlanner"
        state["service_plan"] = fixture["service_plan"]
        state["report_draft"] = fixture["report_draft"]
        state["lawyer_review_needed"] = True
        trace.append(
            {
                "stage": "service_plan_and_draft",
                "agent": "ServicePlanner",
                "service_plan": state["service_plan"],
                "report_draft": state["report_draft"],
            }
        )
        return state

    session_id = f"demo-{case_id}"
    initial_state: ConsultationState = {
        "consultation_id": f"consultation-{case_id}",
        "user_id": "demo-client",
        "session_id": session_id,
        "user_type": "family" if case_id == "missing_facts" else "suspect",
        "consent_given": False,
        "current_input": case["user_input"],
        "facts_raw": [case["user_input"]],
        "facts_structured": {},
        "applied_laws": [],
        "pending_questions": [],
        "alert_triggered": False,
        "conversation_history": [],
        "lawyer_decision": None,
    }

    with (
        patch("app.orchestrator.workflow.receptionist_node", receptionist),
        patch("app.orchestrator.workflow.fact_digger_node", fact_digger),
        patch("app.orchestrator.workflow.law_ref_node", law_ref),
        patch("app.orchestrator.workflow.risk_assessor_node", risk_assessor),
        patch("app.orchestrator.workflow.service_planner_node", service_planner),
        patch("app.orchestrator.workflow.human_alert_node", human_alert_node),
    ):
        orchestrator = ConsultationOrchestrator()
        await orchestrator.start_workflow(initial_state)
        state = await orchestrator.resume_workflow(session_id, {"consent_given": True})

        if case["should_follow_up"]:
            snapshot = await orchestrator.get_snapshot(session_id)
            result = {
                "case_id": case_id,
                "mode": "deterministic_workflow_contract",
                "finished": False,
                "requires_human_intervention": False,
                "current_agent": state.get("current_agent"),
                "next_node": snapshot.next[0] if snapshot and snapshot.next else None,
                "output_type": "follow_up_questions",
                "trace": trace,
            }
        elif case["should_trigger_human"]:
            result = {
                "case_id": case_id,
                "mode": "deterministic_workflow_contract",
                "finished": await orchestrator.is_workflow_finished(session_id),
                "requires_human_intervention": True,
                "current_agent": state.get("current_agent"),
                "output_type": "human_intervention_notice",
                "alert_triggered": state.get("alert_triggered"),
                "final_output": state.get("final_output"),
                "trace": trace,
            }
        else:
            trace.append(
                {
                    "stage": "lawyer_review",
                    "agent": "HumanReview",
                    "awaiting_lawyer_review": state.get("awaiting_lawyer_review"),
                    "review_draft": state.get("report_draft"),
                }
            )
            state = await orchestrator.resume_workflow(
                session_id,
                {
                    "lawyer_decision": "approved",
                    "lawyer_feedback": "确定性 Demo 审核通过",
                    "awaiting_lawyer_review": False,
                    "final_output": fixture["lawyer_final_output"],
                },
            )
            trace.append(
                {
                    "stage": "approved",
                    "agent": "HumanReview",
                    "final_output": state.get("final_output"),
                }
            )
            result = {
                "case_id": case_id,
                "mode": "deterministic_workflow_contract",
                "finished": await orchestrator.is_workflow_finished(session_id),
                "requires_human_intervention": False,
                "current_agent": state.get("current_agent"),
                "output_type": "lawyer_reviewed_report",
                "law_keywords_present": _law_keywords_present(case, state.get("applied_laws", [])),
                "final_output": state.get("final_output"),
                "trace": trace,
            }

    return result


def main() -> None:
    """解析命令行参数并输出 Demo JSON。"""
    parser = argparse.ArgumentParser(description="运行确定性的 LangGraph 完整 Demo")
    parser.add_argument("--case", default="ordinary_assault", help="咨询 Demo 的 case id")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_case(args.case)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
