"""通用 Ollama 客户端仅传输既有协议，不引入领域逻辑。"""

import json

import httpx
import pytest

from app.infrastructure.llm.ollama import OllamaClient


@pytest.mark.asyncio
async def test_protocol_payloads_timeouts_reuse_and_lazy_close(monkeypatch):
    requests, pools = [], []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"response": "public", "embeddings": [[1]], "models": []})

    real_client = httpx.AsyncClient

    def create(**kwargs):
        assert kwargs == {"trust_env": False}
        client = real_client(transport=httpx.MockTransport(handler), **kwargs)
        pools.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", create)
    client = OllamaClient("http://localhost:11434/")
    assert pools == []
    assert (await client.tags(timeout_s=7))["models"] == []
    assert (await client.embed("embedding", ["query"], truncate=False, timeout_s=17))["embeddings"] == [[1]]
    assert (
        await client.generate(
            "chat", "prompt", options={"temperature": 0, "seed": 0, "num_predict": 192}, think=False, timeout_s=27
        )
    )["response"] == "public"
    assert len(pools) == 1
    assert [request.url.path for request in requests] == ["/api/tags", "/api/embed", "/api/generate"]
    assert json.loads(requests[1].content) == {"model": "embedding", "input": ["query"], "truncate": False}
    assert json.loads(requests[2].content) == {
        "model": "chat",
        "prompt": "prompt",
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 0, "num_predict": 192},
    }
    assert [request.extensions["timeout"]["read"] for request in requests] == [7, 17, 27]
    await client.aclose()
    await client.aclose()
    assert pools[0].is_closed
    with pytest.raises(RuntimeError):
        await client.tags()


@pytest.mark.asyncio
async def test_http_failure_is_raised_without_retry(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    client = OllamaClient("http://public.invalid")
    with pytest.raises(httpx.HTTPStatusError):
        await client.tags()
    assert len(requests) == 1
    await client.aclose()
