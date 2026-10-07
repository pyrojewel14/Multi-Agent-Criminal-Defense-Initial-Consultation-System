"""通过挂载 HTTP、真实图和 RAG 服务检查调用树；只替换外部供应商。"""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.consultation import service, workflow
from app.consultation.agents import fact_digger, legal_research
from app.consultation.memory import coordinator
from app.infrastructure.database.db import get_db
from app.infrastructure.llm import gateway
from app.infrastructure.observability.tracing import BoundedTraceStore, SessionBudget, current_trace_context
from app.knowledge.rag import rag_service
from app.models import ConsultationMessage
from app.security.jwt import create_access_token
from main import app
from tests.consultation.test_memory_persistence import memory_db as memory_db
from tests.consultation.test_memory_persistence import start


@pytest.fixture
async def traced_http(memory_db, monkeypatch):
    store = BoundedTraceStore(max_events=200)
    budget = SessionBudget(max_calls=20, max_tokens=10000)
    for module in (service, workflow, coordinator, gateway, rag_service, legal_research):
        monkeypatch.setattr(module, "trace_store", store)
    for module in (gateway, rag_service, legal_research):
        monkeypatch.setattr(module, "session_budget", budget)
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "snapshot")
    monkeypatch.setenv("LAW_AGENT_FINAL_PROTOCOL", "legacy")

    # 真实 LawRef 工具循环按请求分别消费供应商响应，保留并发时的异步切换。
    class SyntheticLawProvider:
        model = "synthetic-law-provider"

        def __init__(self):
            self.calls = {}

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            context = current_trace_context()
            request_id = context["request_id"]
            index = self.calls.get(request_id, 0)
            self.calls[request_id] = index + 1
            await asyncio.sleep(0)
            responses = [
                AIMessage(content="", tool_calls=[{"name": "search_laws", "args": {"query": "盗窃"}, "id": "search"}]),
                AIMessage(content="", tool_calls=[{"name": "get_article", "args": {"article_id": "第264条"}, "id": "read"}]),
                AIMessage(content=json.dumps({"article_ids": ["第264条"], "matched_elements": {"第264条": []}, "confidence": "low"})),
            ]
            return responses[index].model_copy(update={"usage_metadata": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}})

    law_provider = SyntheticLawProvider()
    monkeypatch.setattr(gateway.chat_model_factory, "create_precise_model", lambda *_: law_provider)

    async def hyde_provider(_messages):
        await asyncio.sleep(0)
        return AIMessage(content="合成法律检索")
    monkeypatch.setattr(rag_service, "get_chat_model", lambda: RunnableLambda(hyde_provider))

    class SyntheticVectorStore:
        def get_dynamic_weights(self, query):
            return (0.5, 0.5)

        async def get_retriever(self, query, user_id, include_public):
            async def retrieve(_):
                await asyncio.sleep(0)
                return []
            return RunnableLambda(retrieve)

    monkeypatch.setattr(rag_service, "get_vector_store", SyntheticVectorStore)

    async def receptionist(state):
        return dict(state, consent_given=True, current_agent="FactDigger", final_output="欢迎")

    async def extract(_facts):
        from app.consultation.schemas.artifacts import ArtifactSource
        return {"incident_time": None, "incident_location": None, "parties": [],
                "behavior_sequence": [{"action": "我没有拿走任何物品"}], "consequence": None,
                "evidence_mentioned": [], "arrest_status": None, "surrender": None,
                "victim_forgiveness": None, "prior_record": None}, ArtifactSource.CONTENT_JSON

    async def coverage(state):
        return dict(state, facts_coverage_rate=0.0, final_output="请继续补充")

    monkeypatch.setattr(workflow, "receptionist_node", receptionist)
    monkeypatch.setattr(fact_digger, "_extract_structured_facts", extract)
    monkeypatch.setattr(workflow, "_fact_digger_workflow_node", coverage)
    graph = workflow.ConsultationOrchestrator()
    monkeypatch.setattr(service, "orchestrator", graph)
    factory = async_sessionmaker(memory_db.bind, expire_on_commit=False)

    async def isolated_db():
        async with factory() as db:
            yield db

    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = isolated_db
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client, store, graph
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)


def assert_tree(store, request_id, session_id):
    tree = store.export_tree(request_id)
    roots = [event for event in tree if event["parent_span_id"] is None]
    assert len(roots) == 1
    assert roots[0]["event_type"] == "http"
    assert {event["session_id"] for event in tree} == {session_id}
    assert {event["request_id"] for event in tree} == {request_id}
    assert {event["trace_id"] for event in tree} == {roots[0]["trace_id"]}
    spans = {event["span_id"]: event for event in tree}
    for event in tree:
        if event["parent_span_id"]:
            assert event["parent_span_id"] in spans
    resumed = next(event for event in tree if event["name"] == "resume_workflow")
    assert resumed["parent_span_id"] == roots[0]["span_id"]
    node = next(event for event in tree if event["event_type"] == "node" and event["name"] == "law_ref")
    assert node["parent_span_id"] == resumed["span_id"]
    llm = next(event for event in tree if event["name"] == "generate_with_tools")
    rag = next(event for event in tree if event["name"] == "retrieve_document")
    decision = spans[llm["parent_span_id"]]
    tool = spans[rag["parent_span_id"]]
    assert decision["event_type"] == "law_agent_step"
    assert tool["event_type"] == "law_tool"
    assert decision["parent_span_id"] == node["span_id"]
    assert tool["parent_span_id"] == node["span_id"]
    assert llm["input_tokens"] == 5
    assert llm["output_tokens"] == 2
    assert rag["metadata"]["query"]["length"] == 2
    assert any(event["route_reason"] == "facts_accepted" for event in tree)
    serialized = str(tree)
    for raw in ("张三", "13800138000", "青禾市", "合成案情检索", "我没有拿走任何物品"):
        assert raw not in serialized
    return tree


@pytest.mark.asyncio
async def test_fixed_http_request_exports_workflow_node_llm_rag_tree(traced_http, memory_db):
    client, store, graph = traced_http
    sid = "session-fixed"
    await start(graph, memory_db, sid)
    request_id = "33333333-3333-4333-8333-333333333333"
    content = "我叫张三，电话13800138000，住在青禾市明月区长宁路18号。我没有拿走任何物品"
    response = await client.post(f"/api/v1/sessions/{sid}/message", json={"session_id": sid, "content": content},
        headers={"Authorization": f"Bearer {create_access_token('synthetic-owner', 'client')}", "X-Request-ID": request_id})
    assert response.status_code == 200, response.text
    assert response.headers["X-Request-ID"] == request_id
    assert (await graph.get_snapshot(sid)).values["law_research"]["termination_reason"] == "final_answer"
    assert_tree(store, request_id, sid)
    assert current_trace_context() is None
    rows = (await memory_db.scalars(select(ConsultationMessage).where(ConsultationMessage.sender_type == "user"))).all()
    assert [row.content for row in rows] == [content]


@pytest.mark.asyncio
async def test_concurrent_mounted_http_requests_keep_separate_trees(traced_http, memory_db):
    client, store, graph = traced_http
    sessions = ("trace-alpha", "trace-beta")
    ids = ("44444444-4444-4444-8444-444444444444", "55555555-5555-4555-8555-555555555555")
    for sid in sessions:
        await start(graph, memory_db, sid)

    async def send(sid, request_id):
        return await client.post(f"/api/v1/sessions/{sid}/message", json={"session_id": sid, "content": "我叫张三，电话13800138000。我没有拿走任何物品"},
            headers={"Authorization": f"Bearer {create_access_token('synthetic-owner', 'client')}", "X-Request-ID": request_id})

    responses = await asyncio.gather(*(send(sid, rid) for sid, rid in zip(sessions, ids)))
    assert [response.status_code for response in responses] == [200, 200]
    trees = [assert_tree(store, rid, sid) for sid, rid in zip(sessions, ids)]
    assert {event["span_id"] for event in trees[0]}.isdisjoint({event["span_id"] for event in trees[1]})
    assert trees[0][0]["trace_id"] != trees[1][0]["trace_id"]
    assert current_trace_context() is None
