"""新法律服务保留有界排名指纹与主查询会话预算。"""

import pytest

from app.infrastructure.observability.tracing import BoundedTraceStore, SessionBudget, SessionBudgetExceeded, trace_span
from tests.knowledge.law.test_search_service import Vector, service
from app.knowledge.retrieval.models import RetrievalCandidate


@pytest.mark.asyncio
async def test_ranked_results_are_bounded_ordered_and_fingerprinted(monkeypatch):
    from app.knowledge.law import service as module

    store = BoundedTraceStore(max_events=50)
    monkeypatch.setattr(module, "trace_store", store)
    vector = Vector(
        [
            RetrievalCandidate(str(i), "test", "第999条 私密正文" + str(i), {"article_number": f"第{900 + i}条"})
            for i in range(9)
        ]
    )
    selected = service(vector=vector)
    with trace_span(store, event_type="law_tool", name="search_laws", request_id="ranked", session_id="sid"):
        result = await selected.search("私密查询", user_id="alice")
    events = store.export_tree("ranked")
    assert len([event for event in events if event["event_type"] == "rag_ranked_result_set"]) == 1
    ranked = sorted(
        (event for event in events if event["event_type"] == "rag_ranked_result"), key=lambda event: event["attempt"]
    )
    assert [event["attempt"] for event in ranked] == [1, 2, 3, 4, 5]
    assert len(result.search.candidates) == 5
    assert all(event["metadata"]["content"]["sha256"] for event in ranked)
    assert all(event["metadata"]["origin"]["sha256"] for event in ranked)
    assert "私密正文" not in str(events) and "私密查询" not in str(events)


@pytest.mark.asyncio
async def test_empty_retrieval_emits_result_set_and_budget_failure_is_not_swallowed(monkeypatch):
    from app.knowledge.law import service as module

    store = BoundedTraceStore(max_events=50)
    budget = SessionBudget(max_calls=1, max_tokens=100)
    monkeypatch.setattr(module, "trace_store", store)
    monkeypatch.setattr(module, "session_budget", budget)
    selected = service(vector=Vector([]))
    with trace_span(store, event_type="law_tool", name="search_laws", request_id="empty", session_id="bounded"):
        result = await selected.search("无匹配", user_id="alice")
        assert not result.search.candidates
        with pytest.raises(SessionBudgetExceeded):
            await selected.search("无匹配", user_id="alice")
    events = store.export_tree("empty")
    assert any(event["event_type"] == "rag_ranked_result_set" for event in events)
    assert not any(event["event_type"] == "rag_ranked_result" for event in events)
    assert budget.snapshot("bounded")["calls"] == 1
