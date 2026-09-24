from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.observability.tracing import BoundedTraceStore, SessionBudget, trace_span
from app.orchestrator import workflow as workflow_module
from app.utils import llm_gateway as gateway_module
from app.utils.llm_gateway import LLMGateway
from app.v1.service import consultation_service
from tests.factories import make_consultation_state


@pytest.mark.asyncio
async def test_fixed_http_request_exports_workflow_node_llm_rag_tree(monkeypatch):
    """任一层丢失 contextvars 关联都会让固定请求的调用树断链。"""
    store = BoundedTraceStore(max_events=50)
    budget = SessionBudget(max_calls=10, max_tokens=1000)
    monkeypatch.setattr(consultation_service, "trace_store", store)
    monkeypatch.setattr(workflow_module, "trace_store", store)
    monkeypatch.setattr(gateway_module, "trace_store", store)
    monkeypatch.setattr(gateway_module, "session_budget", budget)

    async def law_ref_node(state):
        gateway = LLMGateway()
        await gateway.generate("law-ref-v1", "张三 13800138000")
        with trace_span(
            store,
            event_type="rag",
            name="retrieve_document",
            cache_hit=False,
        ):
            pass
        workflow_module._record_route("check_facts_sufficient", state, "coverage_sufficient")
        return {
            **state,
            "final_output": "ok",
            "current_agent": "HumanReview",
            "alert_triggered": False,
            "conversation_history": [],
        }

    observed_law_ref = workflow_module._observed_node("law_ref", law_ref_node)

    class FakeOrchestrator:
        async def get_snapshot(self, _session_id):
            return SimpleNamespace(values=state, next=("law_ref",))

        async def resume_workflow(self, _session_id, _updates):
            with trace_span(
                store,
                event_type="workflow",
                name="resume_workflow",
                session_id="session-fixed",
            ):
                return await observed_law_ref(state)

        async def is_workflow_finished(self, _session_id):
            return False

    state = make_consultation_state(
        session_id="session-fixed",
        user_id="user-fixed",
        consent_given=True,
        current_agent="LawRef",
        conversation_history=[],
    )
    monkeypatch.setattr(consultation_service, "orchestrator", FakeOrchestrator())
    monkeypatch.setattr(consultation_service, "persist_state", AsyncMock())

    response = MagicMock()
    response.content = "ok"
    response.tool_calls = []
    response.usage_metadata = {"input_tokens": 5, "output_tokens": 2}
    with patch.object(gateway_module, "chat_model_factory") as model_factory:
        model = MagicMock()
        model.model = "qwen-fixed"
        model.ainvoke = AsyncMock(return_value=response)
        model_factory.create_precise_model.return_value = model
        result = await consultation_service.process_message(
            "session-fixed",
            "张三 13800138000",
            state,
            "LawRef",
            request_id="33333333-3333-4333-8333-333333333333",
            transport="http",
        )

    tree = store.export_tree("33333333-3333-4333-8333-333333333333")
    assert [event["event_type"] for event in tree] == [
        "http",
        "workflow",
        "node",
        "llm",
        "rag",
        "route",
    ]
    by_type = {event["event_type"]: event for event in tree}
    assert by_type["workflow"]["parent_span_id"] == by_type["http"]["span_id"]
    assert by_type["node"]["parent_span_id"] == by_type["workflow"]["span_id"]
    assert by_type["llm"]["parent_span_id"] == by_type["node"]["span_id"]
    assert by_type["rag"]["parent_span_id"] == by_type["node"]["span_id"]
    assert by_type["route"]["route_reason"] == "coverage_sufficient"
    assert by_type["llm"]["input_tokens"] == 5
    assert result.request_id == "33333333-3333-4333-8333-333333333333"
    serialized = str(tree)
    assert "张三" not in serialized
    assert "13800138000" not in serialized
