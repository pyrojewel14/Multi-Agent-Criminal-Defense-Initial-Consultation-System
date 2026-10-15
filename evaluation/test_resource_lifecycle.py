"""独立评测 owner 在正常、失败和真实取消路径均归还资源。"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.infrastructure.llm.config import ChatConfig
from app.infrastructure.llm.gateway import llm_gateway
from app.infrastructure.llm.manager import ModelManager, get_model_manager


class Client:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class Factory:
    def __init__(self):
        self.clients = []

    def create(self, config):
        client = Client()
        self.clients.append(client)
        return client


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["normal", "error", "cancel"])
@pytest.mark.parametrize("injected", [False, True])
async def test_evaluation_runtime_closes_only_owned_manager_and_restores_gateway(monkeypatch, outcome, injected):
    import app.infrastructure.llm.manager as manager_module

    from evaluation.model_runtime import evaluation_runtime

    factory = Factory()
    owner = ModelManager(chat_factory=factory)
    monkeypatch.setattr(manager_module, "ModelManager", lambda: owner)
    original = llm_gateway._model_manager
    application = get_model_manager()
    application_closed = application.closed
    ready = asyncio.Event()
    observed = []

    async def run():
        async with evaluation_runtime(model_manager=owner if injected else None) as (
            manager,
            resources,
        ):
            observed.append((manager, resources))
            assert llm_gateway._model_manager is owner
            first = manager.get_chat(ChatConfig(provider="OLLAMA"))
            assert manager.get_chat(ChatConfig(provider="OLLAMA")) is first
            if outcome == "error":
                raise ValueError("evaluation failed")
            if outcome == "cancel":
                ready.set()
                await asyncio.Future()

    if outcome == "cancel":
        task = asyncio.create_task(run())
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(ValueError, match="evaluation failed"):
            await run()
    else:
        await run()
    assert observed[0][0] is owner
    assert llm_gateway._model_manager is original
    assert get_model_manager() is application and application.closed is application_closed
    assert owner.closed is not injected
    assert factory.clients[0].closed is not injected
    if injected:
        await owner.aclose()
        assert factory.clients[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["live-rag", "live-lawref"])
@pytest.mark.parametrize("outcome", ["normal", "error", "cancel"])
async def test_full_runner_owns_manager_and_restores_legal_factory(tmp_path, monkeypatch, mode, outcome):
    import json

    import app.infrastructure.llm.manager as manager_module
    from app.consultation.agents import legal_research
    from app.knowledge import wiring
    from app.knowledge.retrieval.config import RetrievalConfig
    from app.knowledge.retrieval.pipeline import RetrievalPipeline

    from evaluation import run_full_eval

    owner = ModelManager(chat_factory=Factory())
    monkeypatch.setattr(manager_module, "ModelManager", lambda: owner)
    original_builder = wiring.build_law_search_service
    original_factory = legal_research.build_law_search_service
    application = get_model_manager()
    application_closed = application.closed
    used = []
    ready = asyncio.Event()

    def build(**kwargs):
        assert kwargs["model_manager"] is owner
        used.append(kwargs["resources"])
        owner.get_chat(ChatConfig(provider="OLLAMA"))
        kwargs["pipeline"] = RetrievalPipeline(None, config=RetrievalConfig(rerank=False))
        return original_builder(**kwargs)

    async def suspend():
        if outcome == "cancel":
            ready.set()
            await asyncio.Future()

    async def law_node(state):
        legal_research.build_law_search_service()
        await suspend()
        return {
            "applied_laws": [],
            "law_research": {"termination_reason": "final_answer", "trajectory": []},
        }

    monkeypatch.setattr(wiring, "build_law_search_service", build)
    monkeypatch.setattr("app.consultation.agents.law_ref.law_ref_node", law_node)
    if mode == "live-rag" and outcome == "cancel":

        async def search(*args, **kwargs):
            await suspend()

        monkeypatch.setattr("app.knowledge.law.service.LawSearchService.search", search)
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "id": "theft",
                "query": "第264条",
                "article_id": "第264条",
                "contains": "盗窃",
                "coverage_eligible": True,
            }
        )
        + "\n"
    )
    output = tmp_path if outcome == "error" else tmp_path / "result.json"
    call = run_full_eval.run_evaluation(mode, output, cases_path=cases)
    if outcome == "cancel":
        task = asyncio.create_task(call)
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(IsADirectoryError):
            await call
    else:
        assert (await call)["total"] == 1
    assert used and owner.closed
    assert legal_research.build_law_search_service is original_factory
    assert get_model_manager() is application and application.closed is application_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["normal", "error", "cancel"])
async def test_snapshot_runtime_owns_manager_and_restores_factory(tmp_path, monkeypatch, outcome):
    import app.infrastructure.llm.manager as manager_module
    from app.consultation.agents import legal_research

    from evaluation import run_live_eval
    from evaluation.build_live_index import build_index

    owner = ModelManager(chat_factory=Factory())
    monkeypatch.setattr(manager_module, "ModelManager", lambda: owner)
    index = tmp_path / "snapshot"
    build_index(
        run_live_eval.LAW,
        index,
        "snapshot",
        "test",
        "digest",
        lambda texts: [[1.0] for _ in texts],
    )
    monkeypatch.setenv("TEXT_EMBEDDING_MODEL_NAME", "test")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.invalid")
    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "false")
    monkeypatch.setenv("LAW_FULL_HYDE_ENABLED", "false")
    original = legal_research.build_law_search_service
    ready = asyncio.Event()

    async def run():
        async with run_live_eval.snapshot_search_runtime(
            index_dir=index,
            preflight_result={
                "index_collection": "snapshot",
                "embedding_digest": "digest",
            },
        ):
            service = legal_research.build_law_search_service()
            assert service.repository.source_id.startswith("law:snapshot-evaluation:")
            assert llm_gateway._model_manager is owner
            owner.get_chat(ChatConfig(provider="OLLAMA"))
            if outcome == "error":
                raise ValueError("snapshot failed")
            if outcome == "cancel":
                ready.set()
                await asyncio.Future()

    if outcome == "cancel":
        task = asyncio.create_task(run())
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(ValueError, match="snapshot failed"):
            await run()
    else:
        await run()
    assert owner.closed
    assert legal_research.build_law_search_service is original


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["normal", "error", "cancel"])
async def test_memory_compare_reuses_owned_chat_and_closes_on_all_exits(monkeypatch, outcome):
    from types import SimpleNamespace

    import app.infrastructure.llm.manager as manager_module
    from langchain_core.messages import HumanMessage

    from evaluation import run_memory_eval

    ready = asyncio.Event()

    class Chat(Client):
        def bind(self, **kwargs):
            assert kwargs == {"max_tokens": 333}
            return self

        async def ainvoke(self, messages):
            if outcome == "error":
                raise ValueError("provider failed")
            if outcome == "cancel":
                ready.set()
                await asyncio.Future()
            return SimpleNamespace(
                usage_metadata={"input_tokens": 1, "output_tokens": 1},
                content="公开回答",
            )

    chat = Chat()
    factory = Factory()
    factory.create = lambda config: chat
    owner = ModelManager(chat_factory=factory)
    monkeypatch.setattr(manager_module, "ModelManager", lambda: owner)
    built = SimpleNamespace(messages=[HumanMessage(content="公开事实")], estimated_input_tokens=1)
    monkeypatch.setattr(run_memory_eval, "context_pair", lambda *args: (built, built))
    monkeypatch.setenv("LLM_TYPE", "OLLAMA")
    original = llm_gateway._model_manager
    call = run_memory_eval.compare_context("live", {}, [], "回忆问题")
    if outcome == "cancel":
        task = asyncio.create_task(call)
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(ValueError, match="provider failed"):
            await call
    else:
        rows = await call
        assert [row["answer"] for row in rows] == ["公开回答", "公开回答"]
    assert owner.closed and chat.closed
    assert llm_gateway._model_manager is original


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cloud", "local"])
async def test_capture_task_cancel_preserves_partial_stream_and_closes_both_clients(tmp_path, monkeypatch, kind):
    import json

    import httpx
    from app.consultation.agents.legal_research import _SYSTEM_PROMPT
    from app.infrastructure.llm.factory import ChatModelFactory
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    from evaluation.run_cloud_lawref import CloudCapture, masked_case_facts
    from evaluation.run_local_lawref import LocalCapture

    query = "我没有拿走任何物品。"
    cases = [{"query": query}]
    ready = asyncio.Event()
    requested = "gpt-oss:120b-cloud" if kind == "cloud" else "qwen3.5:9b"
    returned = "gpt-oss:120b" if kind == "cloud" else requested

    class Partial(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                json.dumps(
                    {
                        "model": returned,
                        "message": {"role": "assistant", "content": "部分回答"},
                        "done": False,
                    }
                )
                + "\n"
            ).encode()
            ready.set()
            await asyncio.Future()

    def respond(request):
        return httpx.Response(200, stream=Partial())

    transport = httpx.MockTransport(respond)
    if kind == "cloud":
        model = ChatOllama(model=requested, client_kwargs={"transport": transport, "trust_env": False})
        monkeypatch.setattr(ChatModelFactory, "create", lambda self, config: model)
        capture = CloudCapture(
            requested,
            "low",
            tmp_path / "raw.json",
            masked_case_facts(cases, "article"),
            {},
        )
    else:
        capture = LocalCapture(requested, "http://ollama.invalid", tmp_path / "raw.json", cases, {})
        capture.client_kwargs["transport"] = transport
    original_manager = llm_gateway._model_manager
    original_stream = ChatOllama._acreate_chat_stream
    original_policy = llm_gateway._policy
    observed = []

    async def run():
        async with capture:
            model = capture.model_manager.get_chat(ChatConfig(provider="OLLAMA", model_name=requested))
            observed.append(model)
            await model.ainvoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "facts": json.dumps(
                                    {"behavior_sequence": [query], "consequence": ""},
                                    ensure_ascii=False,
                                )
                            }
                        )
                    ),
                ]
            )

    task = asyncio.create_task(run())
    await asyncio.wait_for(ready.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert capture.records[0]["content"] == "部分回答"
    assert capture.records[0]["exception_type"] == "CancelledError"
    assert capture.model_manager.closed
    assert observed[0]._client._client.is_closed
    assert observed[0]._async_client._client.is_closed
    assert llm_gateway._model_manager is original_manager
    assert llm_gateway._policy is original_policy
    assert ChatOllama._acreate_chat_stream is original_stream


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong", ["config", "returned_model"])
async def test_cloud_invalid_identity_rejects_before_create_or_closes_owned_model(tmp_path, monkeypatch, wrong):
    from app.infrastructure.llm.factory import ChatModelFactory
    from langchain_ollama import ChatOllama

    from evaluation.run_cloud_lawref import CloudCapture

    created = []

    def create(self, config):
        model = ChatOllama(
            model="wrong-model",
            base_url="http://ollama.invalid",
            client_kwargs={"trust_env": False},
        )
        created.append(model)
        return model

    monkeypatch.setattr(ChatModelFactory, "create", create)
    capture = CloudCapture("gpt-oss:120b-cloud", "low", tmp_path / "raw.json", set(), {})
    async with capture:
        with pytest.raises(ValueError, match="禁止回落"):
            capture.model_manager.get_chat(
                ChatConfig(
                    provider="OLLAMA",
                    model_name="wrong-config" if wrong == "config" else "gpt-oss:120b-cloud",
                )
            )
    assert capture.model_manager.closed
    if wrong == "config":
        assert created == []
    else:
        assert len(created) == 1
        assert created[0]._client._client.is_closed
        assert created[0]._async_client._client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["normal", "error", "cancel"])
async def test_full_ablation_closes_owned_manager_on_preflight_exit(tmp_path, monkeypatch, outcome):
    import app.infrastructure.llm.manager as manager_module
    import httpx

    from evaluation import run_full_retrieval_ablation as runner

    owner = ModelManager(chat_factory=Factory())
    monkeypatch.setattr(manager_module, "ModelManager", lambda: owner)
    original = llm_gateway._model_manager
    ready = asyncio.Event()

    def identity():
        owner.get_chat(ChatConfig(provider="OLLAMA"))
        return {"hyde_model": "test", "index_manifest": {"embedding_model": "test"}}

    async def tags(self, url):
        if outcome == "error":
            raise ValueError("tags unavailable")
        if outcome == "cancel":
            ready.set()
            await asyncio.Future()
        return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr(runner, "identity", identity)
    monkeypatch.setattr(httpx.AsyncClient, "get", tags)
    output = tmp_path / "ablation.json"
    call = runner.run([], [], output, 1)
    if outcome == "cancel":
        task = asyncio.create_task(call)
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await call
        assert result["complete"] is (outcome == "normal")
        if outcome == "error":
            assert result["preflight_error"] == "ValueError"
    assert owner.closed and output.exists()
    assert llm_gateway._model_manager is original
