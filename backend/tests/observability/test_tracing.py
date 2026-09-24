import json

import pytest

from app.observability.tracing import (
    BoundedTraceStore,
    SessionBudget,
    SessionBudgetExceeded,
    TokenCostEstimator,
    TraceEvent,
    build_safe_metadata,
    trace_span,
)


def test_nested_spans_export_explicit_parent_child_call_tree():
    """删除任一 parent_span_id 传播都会破坏固定调用树。"""
    store = BoundedTraceStore(max_events=20)

    with trace_span(
        store,
        event_type="http",
        name="POST /sessions/{session_id}/message",
        request_id="req-fixed",
        session_id="session-fixed",
    ):
        with trace_span(store, event_type="workflow", name="resume_workflow"):
            with trace_span(store, event_type="node", name="law_ref", node="law_ref"):
                with trace_span(store, event_type="llm", name="extract_laws", attempt=1):
                    pass
                with trace_span(store, event_type="rag", name="retrieve_document"):
                    pass

    tree = store.export_tree("req-fixed")
    by_name = {item["name"]: item for item in tree}
    assert [item["name"] for item in tree] == [
        "POST /sessions/{session_id}/message",
        "resume_workflow",
        "law_ref",
        "extract_laws",
        "retrieve_document",
    ]
    assert by_name["resume_workflow"]["parent_span_id"] == by_name[
        "POST /sessions/{session_id}/message"
    ]["span_id"]
    assert by_name["law_ref"]["parent_span_id"] == by_name["resume_workflow"]["span_id"]
    assert by_name["extract_laws"]["parent_span_id"] == by_name["law_ref"]["span_id"]
    assert by_name["retrieve_document"]["parent_span_id"] == by_name["law_ref"]["span_id"]


def test_trace_event_has_unified_optional_contract_and_never_serializes_pii_values():
    """事件若保存 prompt、事实或工具参数值，PII 扫描必须失败。"""
    raw_metadata = {
        "prompt": "张三身份证110101199001011234，手机号13800138000",
        "tool_args": {"name": "张三", "phone": "13800138000"},
    }
    metadata = build_safe_metadata(raw_metadata)
    event = TraceEvent(
        event_type="llm",
        name="generate",
        trace_id="trace-1",
        span_id="span-1",
        request_id="request-1",
        session_id="session-1",
        node="fact_digger",
        route_reason=None,
        model="qwen-test",
        prompt_version="sha256:abc",
        input_tokens="unknown",
        output_tokens="unknown",
        duration_ms=1.5,
        attempt=1,
        outcome="success",
        cache_hit=False,
        metadata=raw_metadata,
    )

    serialized = json.dumps(event.to_dict(), ensure_ascii=False)
    assert "张三" not in serialized
    assert "110101199001011234" not in serialized
    assert "13800138000" not in serialized
    assert metadata["field_names"] == ["prompt", "tool_args"]
    assert metadata["prompt"]["length"] == 38
    assert metadata["prompt"]["sha256"].startswith("sha256:")
    assert event.metadata == metadata
    assert event.to_dict()["input_tokens"] == "unknown"


def test_trace_store_evicts_oldest_events_at_configured_bound():
    """无界 append 会让单进程观测存储随流量持续增长。"""
    store = BoundedTraceStore(max_events=2)
    for index in range(3):
        store.append(
            TraceEvent(
                event_type="node",
                name=f"node-{index}",
                trace_id="trace",
                span_id=f"span-{index}",
                request_id="request",
                session_id="session",
            )
        )

    assert [event.name for event in store.events()] == ["node-1", "node-2"]


def test_session_budget_rejects_call_and_token_overruns_with_typed_error():
    """预算检查缺失或把 unknown 当 0 都会允许无限继续调用。"""
    budget = SessionBudget(max_calls=2, max_tokens=10)
    budget.reserve_call("session")
    budget.record_tokens("session", input_tokens=4, output_tokens=3)
    budget.reserve_call("session")

    with pytest.raises(SessionBudgetExceeded) as call_error:
        budget.reserve_call("session")
    assert call_error.value.dimension == "calls"

    with pytest.raises(SessionBudgetExceeded) as token_error:
        budget.record_tokens("session", input_tokens=4, output_tokens=0)
    assert token_error.value.dimension == "tokens"

    before = budget.snapshot("session")
    budget.record_tokens("session", input_tokens="unknown", output_tokens="unknown")
    assert budget.snapshot("session") == before
    assert before == {"calls": 2, "tokens": 11}


def test_cost_is_only_estimated_from_explicit_model_price_and_known_usage():
    """缺价格或缺 usage 时若返回 0，会伪造成本证据。"""
    estimator = TokenCostEstimator({"qwen-priced": {"input": 1.0, "output": 2.0}})

    assert estimator.estimate("qwen-priced", 1_000_000, 500_000) == 2.0
    assert estimator.estimate("unpriced", 10, 20) == "unknown"
    assert estimator.estimate("qwen-priced", "unknown", 20) == "unknown"


def test_budget_registry_is_bounded_without_resetting_existing_sessions():
    """通过淘汰旧 session 限内存会重置其预算并重新允许调用。"""
    budget = SessionBudget(max_calls=2, max_tokens=10, max_sessions=2)
    budget.reserve_call("session-1")
    budget.reserve_call("session-2")

    with pytest.raises(SessionBudgetExceeded) as error:
        budget.reserve_call("session-3")

    assert error.value.dimension == "sessions"
    assert budget.snapshot("session-1") == {"calls": 1, "tokens": 0}
