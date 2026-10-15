"""隔离验收入口：真实路由、图节点、SQLite/Redis，模型与向量召回为明确替身。

从项目根运行：python backend/tests/integration/consultation_acceptance_probe.py --output <新目录>。
结果只证明确定性接口契约，不证明真实模型、法律质量或生产部署。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

INITIAL = "我是当事人，昨晚在甲城商店与乙发生争执，现场有监控。"
COMPLETE = INITIAL + "我故意殴打乙，鉴定意见记载轻伤二级。"
SUPPLEMENT = "补充：我故意殴打乙，鉴定意见记载轻伤二级。"
FINAL = "合成律师审核报告：陈述和鉴定尚待核实，本报告仅用于接口验收。"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def private_redis(runtime):
    executable = shutil.which("redis-server")
    if not executable:
        raise RuntimeError("验收需要 redis-server；不得回退到共享 Redis")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (runtime / "redis.log").open("w") as log:
        process = subprocess.Popen([executable, "--bind", "127.0.0.1", "--port", str(port),
            "--save", "", "--appendonly", "no", "--dir", str(runtime)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("隔离 Redis 启动失败")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2) as connection:
                        connection.sendall(b"*1\r\n$4\r\nPING\r\n")
                        if b"PONG" in connection.recv(128):
                            break
                except OSError:
                    time.sleep(0.05)
            else:
                raise TimeoutError("隔离 Redis 未就绪")
            yield port
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def environment(runtime, port):
    return dict(os.environ, DATABASE_PATH=str(runtime / "audit.db"),
        LANGGRAPH_CHECKPOINT_DB_PATH=str(runtime / "checkpoint.db"),
        CHROMA_PERSIST_DIRECTORY=str(runtime / "chroma"), CHROMA_MD5_STORE=str(runtime / "md5.txt"),
        MODEL_CACHE_DIR=str(runtime / "models"), RERANKER_MODEL_PATH=str(runtime / "missing-reranker"),
        REDIS_HOST="127.0.0.1", REDIS_PORT=str(port), REDIS_DB="0", REDIS_PASSWORD="",
        JWT_SECRET_KEY="consultation-acceptance-synthetic-key-not-for-deployment",
        LAW_KNOWLEDGE_PROFILE="snapshot", LAW_AGENT_FINAL_PROTOCOL="legacy", LAW_AGENT_MAX_STEPS="4",
        MEMORY_RECENT_MESSAGES="100", MEMORY_MODEL_WINDOW="32768", MEMORY_CONTEXT_TOKEN_BUDGET="24000",
        OBSERVABILITY_ENABLED="false", LOG_LEVEL="WARNING", PYTHONNOUSERSITE="1",
        LLM_TYPE="OLLAMA", OLLAMA_BASE_URL="http://127.0.0.1:1", OLLAMA_MODEL_NAME="synthetic-no-network")


def isolate_logs(runtime):
    # 日志重定向限于本探针进程，避免把合成验收写进共享服务日志。
    import logging.handlers
    handler = logging.handlers.TimedRotatingFileHandler
    def isolated_handler(*args, **kwargs):
        kwargs["filename"] = runtime / "application.log"
        return handler(*args, **kwargs)
    logging.handlers.TimedRotatingFileHandler = isolated_handler


def worker(output, runtime, restore):
    isolate_logs(runtime)
    from fastapi import WebSocketDisconnect
    from fastapi.testclient import TestClient
    from sqlalchemy import select

    from app.consultation import workflow
    from app.consultation.memory.context import current_node_context
    from app.infrastructure.database.db import AsyncSessionLocal
    from app.infrastructure.llm.gateway import LLMGateway
    from app.consultation.agents import legal_research
    from app.knowledge.law.corpus import load_criminal_law_data
    from app.knowledge.retrieval.config import RetrievalConfig
    from app.knowledge.retrieval.models import ChannelResult, RetrievalCandidate, StageStatus
    from app.knowledge.retrieval.pipeline import RetrievalPipeline
    from app.knowledge.wiring import build_law_search_service
    from app.models import Consultation, ConsultationMessage, User, UserRole
    from app.security.jwt import create_access_token

    nodes = []
    calls = []
    rag_calls = []
    original_observed = workflow._observed_node
    def observed(name, node):
        real_node = original_observed(name, node)
        async def record(state):
            result = await real_node(state)
            nodes.append({"node": name, "session_id": state["session_id"],
                "artifact_results": result.get("artifact_results", {}),
                "current_agent": result.get("current_agent"), "coverage": result.get("facts_coverage_rate"),
                "law_research": result.get("law_research"), "workflow_status": result.get("workflow_status")})
            return result
        return record
    workflow._observed_node = observed
    law = next(a for c in load_criminal_law_data()["chapters"] for a in c["articles"]
        if a["article_number"] == "第二百三十四条")
    elements = [e["name"] for e in law["elements"]]

    def context(kind):
        entry = current_node_context()
        node, state = entry if entry else ("summary", {})
        calls.append({"kind": kind, "node": node, "session_id": state.get("session_id")})
        return node, state

    async def tools(self, system_prompt, user_message, tools, **kwargs):
        node, state = context("structured")
        if node == "fact_intake":
            complete = "轻伤二级" in user_message
            return {"content": json.dumps({"incident_time": "昨晚", "incident_location": "甲城商店",
                "parties": [{"name": "乙", "role": "被害人"}],
                "behavior_sequence": [{"action": "故意殴打乙" if complete else "发生争执"}],
                "consequence": "鉴定意见记载轻伤二级" if complete else None,
                "evidence_mentioned": [{"type": "监控", "description": "现场监控"}],
                "arrest_status": None, "surrender": None, "victim_forgiveness": None,
                "prior_record": None}, ensure_ascii=False), "has_tool_call": False, "tool_calls": []}
        assert node == "law_ref", node
        if "dependency_failure" in state["user_id"]:
            raise TimeoutError("合成模型依赖超时")
        history = kwargs.get("message_history") or []
        n = sum(message.type == "tool" for message in history)
        if n < 2:
            return {"content": "", "has_tool_call": True, "tool_calls": [{
                "name": "search_laws" if n == 0 else "get_article",
                "args": {"query": "故意伤害 殴打"} if n == 0 else {"article_id": "第234条"}}]}
        matched = elements if state.get("facts_structured", {}).get("consequence") else []
        return {"content": json.dumps({"article_ids": ["第234条"],
            "matched_elements": {"第234条": matched}, "confidence": "medium"}, ensure_ascii=False),
            "has_tool_call": False, "tool_calls": []}

    async def generate(self, system_prompt, user_message, **kwargs):
        node, state = context("text")
        if node == "fact_digger":
            return "请补充是否有伤情鉴定及结论？" if not state.get("facts_structured", {}).get("consequence") else "合成事实摘要：用户陈述尚待核实。"
        if node == "risk_assessor":
            if "invalid_risk" in state["user_id"]:
                return "{}"
            return json.dumps({"predicted_sentence_range": "待律师依据证据判断", "mitigating_factors": [],
                "aggravating_factors": [], "compulsory_measure_risk": {"detention_status": "未提供",
                    "bail_possibility": "待核实", "measure_change_space": "待核实", "prolonged_detention_risk": "待核实"},
                "evidence_risk_points": [{"gap": "鉴定真实性未核实", "exclusion_possibility": "待律师审核"}],
                "procedure_risks": []}, ensure_ascii=False)
        if node == "service_planner":
            if "invalid_service" in state["user_id"]:
                return "{}"
            return json.dumps({"service_plan": {"urgent_actions": {"immediate": ["请律师核实鉴定"],
                "short_term": [], "follow_up": []}, "defense_strategies": {"primary": "核实证据", "alternatives": []},
                "service_phases": [{"phase": "侦查", "description": "律师核实程序和证据"}],
                "fee_structure": {"recommended_plan": "另行协商", "total_fee_range": "合同约定", "breakdown": {}}},
                "report_draft": "# 合成初期咨询报告\n事实与鉴定待核实，待律师确认。"}, ensure_ascii=False)
        if node == "summary":
            return "合成案件记忆摘要，陈述尚待核实。"
        raise AssertionError(f"未声明的模型调用节点：{node}")

    class VectorBoundaryStub:
        async def retrieve(self, request):
            rag_calls.append({"user_id": request.scope.user_id, "mode": "vector_boundary_stub"})
            # 只替换向量召回；统一 Pipeline、领域核验及工具循环均执行真实代码。
            candidate = RetrievalCandidate("acceptance-law", request.scope.source_id, law["content"],
                                           {"article_number": law["article_number"]})
            return ChannelResult("vector", (candidate,), {candidate.candidate_id: 0.1}, StageStatus("executed"))

    def search_service(*, law_data=None, **kwargs):
        pipeline = RetrievalPipeline(VectorBoundaryStub(), config=RetrievalConfig(mode="vector", rerank=False))
        return build_law_search_service(law_data=law_data, profile="snapshot", pipeline=pipeline,
                                       source_id="law:snapshot-acceptance")
    LLMGateway.generate = generate
    LLMGateway.generate_with_tools = tools
    legal_research.build_law_search_service = search_service
    from main import app

    tokens = {}
    def headers(owner="complete", role="client"):
        uid = "acceptance-" + owner
        token = tokens.setdefault((uid, role), create_access_token(uid, role))
        return {"Authorization": "Bearer " + token}

    def expect(response, code=200):
        assert response.status_code == code, (response.status_code, response.text)
        return response.json()

    scenarios = {}
    with TestClient(app) as client:
        health = expect(client.get("/health"))
        ready = expect(client.get("/ready"))
        assert ready["api"] == "ready", ready
        graph = workflow.orchestrator
        def snapshot(sid):
            return client.portal.call(graph.get_snapshot, sid)

        if restore:
            prior = json.loads((output / "write.json").read_text())
            sid = prior["scenarios"]["complete"]["session_id"]
            state = expect(client.get(f"/api/v1/sessions/{sid}/state", headers=headers()))
            assert state["status"] == "completed"
            assert state["final_output"] == FINAL
            assert not snapshot(sid).next
            waiting = prior["restart_waiting"]
            assert snapshot(waiting["session_id"]).next == ("fact_intake",)
            result = expect(client.post(f"/api/v1/sessions/{waiting['session_id']}/message", headers=headers("restart_waiting"),
                json={"session_id": waiting["session_id"], "content": SUPPLEMENT, "idempotency_key": "after-restart"}))
            assert "律师审核请求" in result["response_content"], result
            restored = snapshot(waiting["session_id"])
            assert restored.values["facts_coverage_rate"] == 1.0
            async def restored_audit():
                async with AsyncSessionLocal() as db:
                    completed = await db.get(Consultation, prior["scenarios"]["complete"]["consultation_id"])
                    assert completed.final_output == FINAL
                    rows = (await db.scalars(select(ConsultationMessage).where(
                        ConsultationMessage.consultation_id == waiting["consultation_id"]
                    ).order_by(ConsultationMessage.sequence))).all()
                    inputs = [row.content for row in rows if row.sender_type == "user" and row.content in (INITIAL, SUPPLEMENT)]
                    assert inputs == [INITIAL, SUPPLEMENT], inputs
                    return {"final_report_in_business_sqlite": True, "resumed_user_inputs": inputs}
            audit_evidence = client.portal.call(restored_audit)
            dump(output / "restore.json", {"final_report_preserved": True, "waiting_session_resumed": True,
                "pending": list(restored.next), "health": health, "ready": ready, "nodes": nodes,
                "audit": audit_evidence})
            return

        async def seed():
            async with AsyncSessionLocal() as db:
                db.add(User(id="acceptance-lawyer", username="acceptance-lawyer", password_hash="unused-synthetic",
                    role=UserRole.LAWYER))
                await db.commit()
        client.portal.call(seed)

        def create(name):
            data = expect(client.post("/api/v1/sessions", headers=headers(name), json={"user_type": "suspect"}))
            sid = data["session_id"]
            assert "权利义务" in data["welcome_message"]
            assert not snapshot(sid).next
            expect(client.post(f"/api/v1/sessions/{sid}/message", headers=headers(name), json={"session_id": sid, "content": INITIAL}), 403)
            expect(client.post(f"/api/v1/sessions/{sid}/confirm-consent", headers=headers(name), json={
                "session_id": sid, "consent_given": True, "consent_timestamp": "2026-10-13T00:00:00Z",
                "consent_version": "acceptance-v1"}))
            assert snapshot(sid).next == ("fact_intake",)
            return data

        def message(name, sid, content, key="first", status=200):
            return expect(client.post(f"/api/v1/sessions/{sid}/message", headers=headers(name),
                json={"session_id": sid, "content": content, "idempotency_key": key}), status)

        def approve(name, data, revisions=False):
            sid, cid = data["session_id"], data["consultation_id"]
            expect(client.get(f"/api/v1/sessions/{sid}/report-draft", headers=headers("lawyer", "lawyer")), 403)
            expect(client.post("/api/v1/consultations/assign", headers=headers("admin", "admin"),
                json={"consultation_id": cid, "lawyer_id": "acceptance-lawyer"}))
            expect(client.get(f"/api/v1/sessions/{sid}/report-draft", headers=headers(name)), 403)
            draft = expect(client.get(f"/api/v1/sessions/{sid}/report-draft", headers=headers("lawyer", "lawyer")))
            assert draft["service_plan"] and draft["report_draft"]
            assert snapshot(sid).next == ("human_review",)
            before_review = snapshot(sid).values
            products = {key: before_review.get(key) for key in ("facts_structured", "applied_laws",
                "risk_assessment", "service_plan", "report_draft", "facts_coverage_rate", "law_research")}
            revision_nodes = {}
            if revisions:
                for decision in ("revise_facts", "revise_risk"):
                    offset = len(nodes)
                    expect(client.put(f"/api/v1/sessions/{sid}/review", headers=headers("lawyer", "lawyer"),
                        json={"decision": decision, "feedback": "合成律师退回，重新核实", "idempotency_key": decision}))
                    assert snapshot(sid).next == ("human_review",)
                    assert snapshot(sid).values["report_draft"]
                    revision_nodes[decision] = [node["node"] for node in nodes[offset:] if node["session_id"] == sid]
                    expected_nodes = ["risk_assessor", "service_planner", "human_review"]
                    if decision == "revise_facts":
                        expected_nodes = ["fact_intake", "law_ref", "fact_digger"] + expected_nodes
                    assert revision_nodes[decision] == expected_nodes, revision_nodes
            expect(client.put(f"/api/v1/lawyer/sessions/{cid}/report", headers=headers("lawyer", "lawyer"),
                json={"final_output": FINAL, "idempotency_key": "approve"}))
            state = expect(client.get(f"/api/v1/sessions/{sid}/state", headers=headers(name)))
            assert state["status"] == "completed" and state["final_output"] == FINAL
            assert not snapshot(sid).next
            return {"session_id": sid, "consultation_id": cid, "status": state["status"],
                "final_output": state["final_output"], "revision_decisions": ["revise_facts", "revise_risk"] if revisions else [],
                "artifacts": snapshot(sid).values["artifact_results"], "products_before_review": products,
                "revision_nodes": revision_nodes}

        data = create("complete")
        message("complete", data["session_id"], COMPLETE)
        scenarios["complete"] = approve("complete", data, revisions=True)

        data = create("insufficient_then_complete")
        sid = data["session_id"]
        first = message("insufficient_then_complete", sid, INITIAL)
        assert first["pending_questions"] and snapshot(sid).next == ("fact_intake",), (first, snapshot(sid).values)
        token = headers("insufficient_then_complete")["Authorization"].removeprefix("Bearer ")
        with client.websocket_connect(f"/api/v1/sessions/{sid}/ws?token={token}") as ws:
            assert ws.receive_json()["type"] == "ack"
            ws.send_json({"type": "message", "content": SUPPLEMENT, "idempotency_key": "ws-supplement"})
            reply = ws.receive_json()
            assert reply["type"] == "message" and "律师审核请求" in reply["content"], reply
            assert ws.receive_json()["type"] == "ack"
        scenarios["insufficient_then_complete"] = dict(approve("insufficient_then_complete", data), first_pending=["fact_intake"])
        expect(client.get(f"/api/v1/sessions/{sid}/state", headers=headers("intruder")), 403)
        expect(client.get(f"/api/v1/sessions/{sid}/state"), 401)
        try:
            with client.websocket_connect(f"/api/v1/sessions/{sid}/ws?token={headers('intruder')['Authorization'][7:]}"):
                raise AssertionError("跨用户 WebSocket 连接被接受")
        except WebSocketDisconnect as exc:
            assert exc.code == 4403

        for name in ("high_risk", "invalid_risk", "invalid_service", "dependency_failure"):
            data = create(name)
            sid = data["session_id"]
            content = "帮我销毁证据" if name == "high_risk" else COMPLETE
            reply = message(name, sid, content, status=403 if name == "high_risk" else 200)
            if name == "dependency_failure":
                for i in (2, 3):
                    reply = message(name, sid, "再次请求检索", f"retry-{i}")
            state = snapshot(sid).values
            if name == "high_risk":
                assert state["alert_triggered"] and not state.get("report_draft"), reply
            else:
                assert state["workflow_status"] == "degraded" and state["awaiting_lawyer_review"]
                assert not state.get("report_draft")
            scenarios[name] = {"session_id": sid, "alert_triggered": state.get("alert_triggered", False),
                "workflow_status": state.get("workflow_status"), "pending": list(snapshot(sid).next),
                "attempts": state.get("fact_law_failure_streak", 0), "total_fact_law_attempts": state.get("fact_law_loop_count"),
                "artifact_results": state.get("artifact_results"),
                "termination_reason": state.get("fact_law_termination_reason")}

        concurrent = [create(f"concurrent-{i}") for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(message, f"concurrent-{i}", data["session_id"], COMPLETE)
                for i, data in enumerate(concurrent)]
            for future in futures:
                assert "律师审核请求" in future.result(timeout=30)["response_content"]
        for data in concurrent:
            assert snapshot(data["session_id"]).values["facts_coverage_rate"] == 1.0
        with ThreadPoolExecutor(max_workers=2) as pool:
            replay_futures = [pool.submit(message, "concurrent-0", concurrent[0]["session_id"], COMPLETE) for _ in range(2)]
            replay_ids = [future.result(timeout=30)["message_id"] for future in replay_futures]
            assert replay_ids[0] == replay_ids[1]

        waiting = create("restart_waiting")
        message("restart_waiting", waiting["session_id"], INITIAL)
        assert snapshot(waiting["session_id"]).next == ("fact_intake",)

        async def audit():
            async with AsyncSessionLocal() as db:
                rows = (await db.scalars(select(ConsultationMessage).order_by(ConsultationMessage.sequence))).all()
                completed = await db.get(Consultation, scenarios["complete"]["consultation_id"])
                assert completed.final_output == FINAL
                for data in concurrent:
                    selected = [r for r in rows if r.consultation_id == data["consultation_id"]]
                    assert len([r for r in selected if r.sender_type == "user" and r.content == COMPLETE]) == 1
                return [{"consultation_id": r.consultation_id, "sequence": r.sequence,
                    "sender_type": r.sender_type, "record_kind": r.record_kind, "content": r.content} for r in rows]
        raw = client.portal.call(audit)
        assert {n["node"] for n in nodes} >= {"receptionist", "fact_intake", "law_ref", "fact_digger",
            "wait_for_user", "risk_assessor", "service_planner", "human_review", "human_alert"}
        assert all(scenarios["complete"]["artifacts"][key]["status"] == "success" for key in ("fact", "law", "risk", "service"))
        dump(output / "write.json", {"model_mode": "deterministic_gateway", "real_model_calls": 0,
            "vector_mode": "vector_boundary_stub", "transport": "ASGI TestClient HTTP/WebSocket",
            "health": health, "ready": ready, "scenarios": scenarios, "restart_waiting": waiting,
            "finite_concurrency": {"workers": 1, "parallel_requests": 2, "distinct_sessions": 2,
                "same_session_parallel_replay": True},
            "nodes": nodes, "gateway_calls": calls, "rag_calls": rag_calls, "raw_audit": raw})


def local_http_startup(output, runtime, redis_port):
    """启动真实 uvicorn，仅检查健康和依赖，完全不调用模型。"""
    import httpx

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        http_port = sock.getsockname()[1]
    env = environment(runtime, redis_port)
    # 启动烟测使用独立数据库，不把完整咨询的库当作启动前置条件。
    env.update(DATABASE_PATH=str(runtime / "startup-audit.db"),
        LANGGRAPH_CHECKPOINT_DB_PATH=str(runtime / "startup-checkpoint.db"), ACCEPTANCE_HTTP_PORT=str(http_port),
        LOG_LEVEL="INFO")
    with (output / "http-startup.log").open("w") as log:
        process = subprocess.Popen([sys.executable, __file__, "--output", str(output.resolve()),
            "--worker", "serve", "--runtime", str(runtime)], cwd=BACKEND, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            with httpx.Client(base_url=f"http://127.0.0.1:{http_port}", trust_env=False, timeout=1) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("隔离 uvicorn 启动失败，详见 http-startup.log")
                    try:
                        health = client.get("/health")
                        if health.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    raise TimeoutError("隔离 HTTP 服务未就绪")
                ready = client.get("/ready")
                assert ready.status_code == 200 and ready.json()["api"] == "ready", ready.text
                assert client.get("/api/v1/sessions").status_code == 401
                result = {"transport": "loopback HTTP/uvicorn", "health": health.json(), "ready": ready.json(),
                    "unauthenticated_sessions_status": 401, "inference_calls": 0}
        finally:
            process.terminate()
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError("隔离 HTTP 服务未正常退出")
        # 新版 uvicorn 清理完成后重发 SIGTERM；必须同时核对真实 shutdown 日志。
        assert exit_code in (0, -signal.SIGTERM), exit_code
        shutdown_log = (output / "http-startup.log").read_text()
        assert "Application shutdown complete." in shutdown_log and "Cleanup completed" in shutdown_log
        result["shutdown_exit_code"] = exit_code
        result["shutdown_cleanup_complete"] = True
        dump(output / "http-startup.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["write", "restore", "serve"])
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.worker == "serve":
        isolate_logs(args.runtime)
        import uvicorn
        uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ["ACCEPTANCE_HTTP_PORT"]), log_level="info")
        return
    if args.worker:
        worker(args.output, args.runtime, args.worker == "restore")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="consultation-acceptance-") as directory:
        runtime = Path(directory)
        with private_redis(runtime) as port:
            local_http_startup(args.output, runtime, port)
            for phase in ("write", "restore"):
                result = subprocess.run([sys.executable, __file__, "--output", str(args.output.resolve()),
                    "--worker", phase, "--runtime", str(runtime)], cwd=BACKEND,
                    env=environment(runtime, port), capture_output=True, text=True, timeout=90)
                (args.output / f"{phase}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
                if result.returncode:
                    raise RuntimeError(f"{phase} 验收失败，详见 {args.output / (phase + '.log')}")
        written = json.loads((args.output / "write.json").read_text())
        written["restart"] = json.loads((args.output / "restore.json").read_text())
        written["http_startup"] = json.loads((args.output / "http-startup.json").read_text())
        written["runtime_cleanup"] = "temporary SQLite/Redis/Chroma/logs removed after exit"
        dump(args.output / "result.json", written)
    print(json.dumps({"status": "passed", "scenarios": len(written["scenarios"]),
        "model_mode": written["model_mode"], "real_model_calls": 0}))


if __name__ == "__main__":
    main()
