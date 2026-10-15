"""HyDE keeps explicit configuration, session budgets, and provider usage."""

import asyncio
from functools import partial
from unittest.mock import AsyncMock
import httpx
import pytest
from app.infrastructure.observability.tracing import BoundedTraceStore, SessionBudget, SessionBudgetExceeded, trace_span


async def test_expander_preserves_usage_trace_and_limits_generated_text(monkeypatch):
    from app.knowledge.retrieval import hyde

    store, budget = BoundedTraceStore(), SessionBudget(max_calls=2, max_tokens=100)
    monkeypatch.setattr(hyde, "trace_store", store)
    monkeypatch.setattr(hyde, "session_budget", budget)
    generator = AsyncMock(return_value={"response": "候选" * 900, "prompt_eval_count": 11, "eval_count": 7})
    expander = hyde.HyDEExpander(generator, model_name="explicit-model")
    with trace_span(store, event_type="tool", name="search", session_id="session") as parent:
        text = await expander.expand("否认拿钱", timeout_s=1)
    assert len(text) == 1200
    assert budget.snapshot("session") == {"calls": 1, "tokens": 18}
    event = next(e for e in store.events() if e.name == "full_hyde")
    assert (event.event_type, event.model, event.attempt, event.parent_span_id) == (
        "llm",
        "explicit-model",
        1,
        parent.span_id,
    )
    assert (event.input_tokens, event.output_tokens) == (11, 7)
    generator.assert_awaited_once_with("否认拿钱")


async def test_exhausted_session_reserves_before_provider_call(monkeypatch):
    from app.knowledge.retrieval import hyde

    budget = SessionBudget(max_calls=1, max_tokens=100)
    budget.reserve_call("bounded")
    monkeypatch.setattr(hyde, "session_budget", budget)
    generator = AsyncMock()
    with trace_span(BoundedTraceStore(), event_type="tool", name="search", session_id="bounded"):
        with pytest.raises(SessionBudgetExceeded):
            await hyde.HyDEExpander(generator, model_name="model").expand("query", timeout_s=1)
    generator.assert_not_awaited()


@pytest.mark.parametrize("text", [None, "", "  \n", 4])
async def test_empty_or_nontext_generation_is_rejected(text):
    from app.knowledge.retrieval.hyde import HyDEExpander

    with pytest.raises(ValueError):
        await HyDEExpander(AsyncMock(return_value={"response": text}), model_name="model").expand("query", timeout_s=1)


async def test_expander_deadline_and_external_cancel_propagate():
    from app.knowledge.retrieval.hyde import HyDEExpander

    entered = asyncio.Event()

    async def blocked(_query):
        entered.set()
        await asyncio.Event().wait()

    expander = HyDEExpander(blocked, model_name="model")
    with pytest.raises(TimeoutError):
        await expander.expand("query", timeout_s=0.01)
    entered.clear()
    task = asyncio.create_task(expander.expand("query", timeout_s=10))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_ollama_http_payload_is_explicit_and_ignores_environment(monkeypatch):
    from app.knowledge.retrieval.hyde import HyDEExpander, ollama_generate

    seen = []

    def handler(request):
        import json

        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"response": "条文用语", "prompt_eval_count": 2, "eval_count": 3})

    real_client = httpx.AsyncClient
    options = []

    def client(**kwargs):
        options.append(kwargs)
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://wrong.invalid")
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "wrong-model")
    expander = HyDEExpander(
        partial(ollama_generate, base_url="http://configured:11434/", model_name="configured-model"),
        model_name="configured-model",
    )
    assert await expander.expand("否认拿钱", timeout_s=1) == "条文用语"
    url, payload = seen[0]
    assert url == "http://configured:11434/api/generate" and payload["model"] == "configured-model"
    assert payload["stream"] is False and payload["think"] is False
    assert payload["options"] == {"temperature": 0, "num_predict": 192, "seed": 0}
    assert "不得补造事实、忽略否认或作有罪结论" in payload["prompt"] and payload["prompt"].endswith("否认拿钱")
    assert options[0]["trust_env"] is False


async def test_truncation_does_not_return_whitespace_only_output():
    from app.knowledge.retrieval.hyde import HyDEExpander

    generator = AsyncMock(return_value={"response": " " * 1500 + "候选条文"})
    assert await HyDEExpander(generator, model_name="model").expand("query", timeout_s=1) == "候选条文"


async def test_token_budget_overrun_records_usage_and_fails_trace(monkeypatch):
    from app.knowledge.retrieval import hyde

    store, budget = BoundedTraceStore(), SessionBudget(max_calls=2, max_tokens=10)
    monkeypatch.setattr(hyde, "trace_store", store)
    monkeypatch.setattr(hyde, "session_budget", budget)
    generator = AsyncMock(return_value={"response": "候选", "prompt_eval_count": 8, "eval_count": 5})
    with trace_span(store, event_type="tool", name="search", session_id="tokens"):
        with pytest.raises(SessionBudgetExceeded):
            await hyde.HyDEExpander(generator, model_name="model").expand("query", timeout_s=1)
    assert budget.snapshot("tokens") == {"calls": 1, "tokens": 13}
    event = next(e for e in store.events() if e.name == "full_hyde")
    assert event.outcome == "error" and (event.input_tokens, event.output_tokens) == (8, 5)
