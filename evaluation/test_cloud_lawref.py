"""云对照适配器的真实消息、权限错误与模型隔离回归。"""

import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

QUERY = "我没有拿走任何物品。"
SAFE_FACTS = '{"behavior_sequence": ["我没有拿走任[NAME-MASKED]。"], "consequence": ""}'


def test_public_six_cases_remain_jsonl_and_match_published_replay_hash():
    root = Path(__file__).parent
    cases = root / "lawref_cloud_cases.jsonl"
    rows = [json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
    assert len(rows) == 6
    evidence = json.loads((root / "evidence" / "lawref_cloud_gptoss120_facts_v1_capture_v2_six_run1_2026-10-01.json").read_text())
    assert "sha256:" + hashlib.sha256(cases.read_bytes()).hexdigest() == evidence["cases_sha256"]


def model_and_wire(status=200, returned_model="gpt-oss:120b"):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={"error": "access denied"})
        body = {"model": returned_model, "message": {"role": "assistant", "content": "{}"},
                "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5}
        return httpx.Response(200, text=json.dumps(body) + "\n")

    model = ChatOllama(model="gpt-oss:120b-cloud", base_url="http://127.0.0.1:11434",
                       client_kwargs={"transport": httpx.MockTransport(respond), "trust_env": False})
    return model, requests


def messages():
    return [SystemMessage(content="公开合成测试"), HumanMessage(content=json.dumps({"facts": SAFE_FACTS}))]


@pytest.mark.asyncio
async def test_cloud_copy_sends_low_and_restores_factory_after_use(tmp_path, monkeypatch):
    from app.infrastructure.llm.factory import chat_model_factory
    from evaluation.run_cloud_lawref import CloudCapture, masked_case_facts

    model, wire = model_and_wire()
    def original(temperature):
        return model

    monkeypatch.setattr(chat_model_factory, "create_precise_model", original)
    with CloudCapture("gpt-oss:120b-cloud", "low", tmp_path / "raw.json",
                      masked_case_facts([{"query": QUERY}], "article"), {}) as capture:
        copied = chat_model_factory.create_precise_model(0)
        response = await copied.ainvoke(messages())
        assert response.content == "{}"
    assert wire[0]["think"] == "low"
    assert "format" not in wire[0]
    assert model.reasoning is None
    assert chat_model_factory.create_precise_model is original
    assert capture.records[0]["http_status"] == 200
    assert capture.records[0]["last_response"]["model"] == "gpt-oss:120b"


def test_payload_accepts_existing_mask_transform_but_rejects_other_facts(tmp_path):
    from evaluation.run_cloud_lawref import CloudCapture, masked_case_facts

    capture = CloudCapture("gpt-oss:120b-cloud", "low", tmp_path / "raw.json",
                           masked_case_facts([{"query": QUERY}], "article"), {})
    request = {"model": "gpt-oss:120b-cloud", "think": "low", "messages": [
        {"role": "system", "content": "公开测试"},
        {"role": "user", "content": json.dumps({"facts": SAFE_FACTS})}]}
    capture.verify_payload(request)
    request["messages"][1]["content"] = json.dumps({"facts": '{"private_case":"unapproved"}'})
    with pytest.raises(ValueError, match="案例"):
        capture.verify_payload(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [402, 410])
async def test_service_refusal_records_actual_status_without_model_fallback(tmp_path, monkeypatch, status):
    from app.infrastructure.llm.factory import chat_model_factory
    from evaluation.run_cloud_lawref import CloudCapture

    model, wire = model_and_wire(status=status)
    monkeypatch.setattr(chat_model_factory, "create_precise_model", lambda temperature: model)
    with (
        CloudCapture("gpt-oss:120b-cloud", "low", tmp_path / "raw.json", {SAFE_FACTS}, {}) as capture,
        pytest.raises(Exception, match="access denied"),
    ):
        await chat_model_factory.create_precise_model(0).ainvoke(messages())
    assert len(wire) == 1
    assert wire[0]["model"] == "gpt-oss:120b-cloud"
    assert capture.records[0]["http_status"] == status
    assert capture.records[0]["last_response"] is None


@pytest.mark.asyncio
async def test_returned_unrequested_model_is_rejected_and_preserved(tmp_path, monkeypatch):
    from app.infrastructure.llm.factory import chat_model_factory
    from evaluation.run_cloud_lawref import CloudCapture

    model, _ = model_and_wire(returned_model="qwen3.5:0.8b")
    monkeypatch.setattr(chat_model_factory, "create_precise_model", lambda temperature: model)
    with (
        CloudCapture("gpt-oss:120b-cloud", "low", tmp_path / "raw.json", {SAFE_FACTS}, {}) as capture,
        pytest.raises(ValueError, match="返回模型"),
    ):
        await chat_model_factory.create_precise_model(0).ainvoke(messages())
    assert capture.records[0]["last_response"]["model"] == "qwen3.5:0.8b"


@pytest.fixture
def evaluation_setup(tmp_path, monkeypatch):
    from argparse import Namespace

    from app.infrastructure.llm.factory import chat_model_factory

    model, wire = model_and_wire(status=402)
    monkeypatch.setattr(chat_model_factory, "create_precise_model", lambda temperature: model)
    calls = []
    original_client = httpx.AsyncClient

    def preflight(request):
        calls.append(request.url.path)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"thinking": {"values": ["low"]}, "capabilities": ["tools"]})
        return httpx.Response(200, json={"models": []})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(preflight)))
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "control", "query": QUERY, "article_id": "第264条",
                                 "coverage_eligible": True}) + "\n")
    args = Namespace(model="gpt-oss:120b-cloud", thinking="low", output=tmp_path / "result.json",
                     raw_output=tmp_path / "raw.json", cases=cases, case=None, query_mode="article",
                     base_url="http://127.0.0.1:11434")
    return args, calls, wire


@pytest.mark.asyncio
async def test_summary_and_raw_same_path_rejected_before_external_request(evaluation_setup):
    from evaluation.run_cloud_lawref import evaluate

    args, calls, wire = evaluation_setup
    args.raw_output = args.output
    with pytest.raises(ValueError, match="相同"):
        await evaluate(args)
    assert calls == wire == []
    assert not args.output.exists()


@pytest.mark.asyncio
async def test_empty_cases_rejected_before_external_request(evaluation_setup):
    from evaluation.run_cloud_lawref import evaluate

    args, calls, wire = evaluation_setup
    args.cases.write_text("")
    with pytest.raises(ValueError, match="为空"):
        await evaluate(args)
    assert calls == wire == []
    assert not args.output.exists()


@pytest.mark.asyncio
async def test_no_selected_case_rejected_before_external_request(evaluation_setup):
    from evaluation.run_cloud_lawref import evaluate

    args, calls, wire = evaluation_setup
    args.case = ["missing-case"]
    with pytest.raises(ValueError, match="案例"):
        await evaluate(args)
    assert calls == wire == []
    assert not args.output.exists()


@pytest.mark.asyncio
async def test_model_refusal_does_not_claim_rag_executed(evaluation_setup):
    from evaluation.run_cloud_lawref import evaluate

    args, calls, wire = evaluation_setup
    result = await evaluate(args)
    assert len(wire) == 1
    assert calls == ["/api/show", "/api/tags"]
    assert result["passed"] == 0 and result["total"] == 1
    assert result["rag_executed"] is False
    assert result["llm_executed"] is False
    assert result["actual_execution"]["http_statuses"] == [402]
