"""本地六例对照的预算、网络诊断和真实消息边界回归。"""

import json
import os
from argparse import Namespace

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from evaluation import run_local_lawref as runner


def test_six_cases_and_original_scoring_input_are_locked(tmp_path):
    cases = runner.load_cases(runner.CASES)
    assert len(cases) == 6
    path = tmp_path / "changed.jsonl"
    path.write_text(runner.CASES.read_text().replace("5000", "5001"))
    with pytest.raises(ValueError, match="固定六例"):
        runner.load_cases(path)


@pytest.mark.asyncio
async def test_unreachable_service_is_not_a_zero_of_six_model_run():
    calls = []

    def fail(request):
        calls.append(request.url.path)
        raise httpx.ConnectError("offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        report = await runner.probe(client, "http://ollama.invalid:11434", ["qwen3.5:9b"])
    assert report["available_models"] == []
    assert report["llm_executed"] is False
    assert "passed" not in report and "total" not in report
    assert "/api/chat" not in calls


@pytest.mark.asyncio
async def test_preflight_detects_missing_and_cloud_models_without_pulling():
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "qwen3.5:9b", "digest": "digest9"},
                {"name": "qwen3.5:4b", "digest": "proxy", "remote_host": "https://ollama.com"},
            ]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["tools", "thinking"]})
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        report = await runner.probe(client, "http://ollama.invalid:11434", ["qwen3.5:4b", "qwen3.5:9b"])
    assert report["available_models"] == ["qwen3.5:9b"]
    assert report["models"]["qwen3.5:4b"]["status"] == "cloud_manifest_rejected"
    assert "/api/chat" not in calls and "/api/pull" not in calls


def test_invalid_budget_and_existing_outputs_fail_before_network(tmp_path):
    args = Namespace(output_dir=tmp_path, models=["qwen3.5:9b"], rounds=3)
    with pytest.raises(ValueError, match="轮"):
        runner.validate_args(args)
    args.rounds = 1
    (tmp_path / "preflight.json").write_text("preserve")
    with pytest.raises(ValueError, match="已存在"):
        runner.validate_args(args)


def capture(tmp_path, max_requests=48):
    cases = [{"query": "我没有拿走任何物品。"}]
    articles = {"第264条": {"content": "公开正文", "elements": [{"name": "公开要件"}]}}
    return runner.LocalCapture("qwen3.5:9b", "http://ollama.invalid:11434", tmp_path / "raw.json",
                               cases, articles, max_requests=max_requests)


def test_current_read_prompt_accepts_facts_object_but_rejects_snapshot_changes(tmp_path):
    cap = capture(tmp_path)
    from app.consultation.agents.legal_research import _SYSTEM_PROMPT

    facts = {"behavior_sequence": ["我没有拿走任何物品。"], "consequence": ""}
    payload = {"model": "qwen3.5:9b", "think": False, "options": runner.OPTIONS,
               "messages": [{"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps({"facts": facts, "read_articles": [
                                {"article_id": "第264条", "content": "公开正文", "required_elements": ["公开要件"]}
                            ]})}]}
    cap.verify_payload(payload)
    altered = json.loads(payload["messages"][1]["content"])
    altered["read_articles"][0]["content"] = "不属于快照的正文"
    payload["messages"][1]["content"] = json.dumps(altered)
    with pytest.raises(ValueError, match="快照"):
        cap.verify_payload(payload)
    payload["messages"][1]["content"] = json.dumps({"facts": {"private": "case"}})
    with pytest.raises(ValueError, match="案例"):
        cap.verify_payload(payload)


@pytest.mark.asyncio
async def test_capture_forces_direct_client_and_restores_factory_on_budget_exhaustion(tmp_path, monkeypatch):
    from app.consultation.agents.legal_research import _SYSTEM_PROMPT
    from app.infrastructure.llm.factory import chat_model_factory

    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, text=json.dumps({"model": "qwen3.5:9b", "message": {
            "role": "assistant", "content": "{}"}, "done": True, "eval_count": 2}) + "\n")

    cap = capture(tmp_path, max_requests=1)
    cap.client_kwargs["transport"] = httpx.MockTransport(respond)
    original = chat_model_factory.create_precise_model
    from app.infrastructure.llm.gateway import LLMCallPolicy, llm_gateway
    monkeypatch.setattr(llm_gateway, "_policy", LLMCallPolicy(total_timeout_seconds=7, attempt_timeout_seconds=3))
    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=json.dumps({
        "facts": json.dumps({"behavior_sequence": ["我没有拿走任何物品。"], "consequence": ""}, ensure_ascii=False)}))]
    with cap:
        model = chat_model_factory.create_precise_model(0)
        assert llm_gateway._policy == LLMCallPolicy()
        assert isinstance(model, ChatOllama)
        assert model.client_kwargs["trust_env"] is False
        assert (await model.ainvoke(messages)).content == "{}"
        with pytest.raises(ValueError, match="请求预算"):
            await model.ainvoke(messages)
    assert chat_model_factory.create_precise_model == original
    assert llm_gateway._policy.total_timeout_seconds == 7
    assert len(sent) == 1
    assert sent[0]["think"] is False
    assert cap.records[0]["last_response"]["eval_count"] == 2
    assert json.loads((tmp_path / "raw.json").read_text())[0]["http_status"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refusal", "wrong_model", "thinking_only"])
async def test_raw_failure_and_thinking_channels_remain_distinct(tmp_path, mode):
    from app.consultation.agents.legal_research import _SYSTEM_PROMPT
    from app.infrastructure.llm.factory import chat_model_factory
    from ollama import ResponseError

    def respond(request):
        if mode == "refusal":
            return httpx.Response(502, text="proxy upstream unavailable")
        return httpx.Response(200, text=json.dumps({
            "model": "other" if mode == "wrong_model" else "qwen3.5:9b",
            "message": {"role": "assistant", "content": "", "thinking": "原始思考片段"},
            "done": True, "done_reason": "length", "eval_count": 1024}) + "\n")
    cap = capture(tmp_path)
    cap.client_kwargs["transport"] = httpx.MockTransport(respond)
    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=json.dumps({
        "facts": {"behavior_sequence": ["我没有拿走任何物品。"], "consequence": ""}}))]
    with cap:
        if mode in {"refusal", "wrong_model"}:
            with pytest.raises(ResponseError if mode == "refusal" else ValueError):
                await chat_model_factory.create_precise_model(0).ainvoke(messages)
        else:
            result = await chat_model_factory.create_precise_model(0).ainvoke(messages)
            assert result.content == ""
    row = cap.records[0]
    if mode == "refusal":
        assert row["http_status"] == 502 and row["last_response"] is None
        assert row["error_response_body"] == "proxy upstream unavailable"
    elif mode == "wrong_model":
        assert row["last_response"]["model"] == "other"
    else:
        assert row["thinking"] == "原始思考片段" and row["content"] == ""
        assert row["last_response"]["done_reason"] == "length"


@pytest.mark.asyncio
async def test_residency_failure_does_not_discard_completed_inference(monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(503))))
    assert await runner.residency("http://ollama.invalid:11434") == {"error_type": "HTTPStatusError"}


@pytest.mark.asyncio
async def test_unavailable_run_writes_diagnosis_without_importing_real_evaluation(tmp_path, monkeypatch):
    async def unavailable(*_):
        return {"available_models": [], "models": {}, "llm_executed": False}

    monkeypatch.setattr(runner, "probe", unavailable)
    args = Namespace(output_dir=tmp_path / "new", models=["qwen3.5:9b"], rounds=1,
                     base_url="http://ollama.invalid:11434", index_directory=tmp_path / "missing")
    result = await runner.run(args)
    assert result["status"] == "blocked_service_or_models"
    assert result["runs"] == []
    assert (args.output_dir / "preflight.json").exists()
    assert not (args.output_dir / "state").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["response", "no_inference", "interrupted", "partial_then_interrupted"])
async def test_available_run_uses_isolated_dotted_index_and_fixed_gateway_budget(tmp_path, monkeypatch, outcome):
    from app.consultation.agents.legal_research import _SYSTEM_PROMPT
    from app.infrastructure.llm.gateway import LLMCallPolicy, llm_gateway

    from evaluation import run_full_eval

    index = tmp_path / "full-law-2026-09-30.2.integration1-qwen3"
    index.mkdir()
    (index / "public-only.txt").write_text("public snapshot")
    manifest = {"embedding_model": "embedding", "embedding_model_digest": "emb-digest", "collection": "public"}
    index.parent.joinpath(index.name + ".manifest.json").write_text(json.dumps(manifest))
    async def available(*_):
        return {"available_models": ["qwen3.5:9b"], "models": {
            "qwen3.5:9b": {"catalog_entry": {"digest": "nine"}}}, "requests": [
                {"endpoint": "/api/tags", "body": {"models": [{"name": "embedding", "digest": "emb-digest"}]}}
            ]}
    monkeypatch.setattr(runner, "probe", available)
    sent = []
    def respond(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b", "size_vram": 999}]})
        sent.append(json.loads(request.content))
        return httpx.Response(200, text=json.dumps({"model": "qwen3.5:9b", "message": {
            "role": "assistant", "content": '{"article_ids":[],"matched_elements":{},"confidence":"low"}'},
            "done": True, "prompt_eval_count": 20, "eval_count": 4}) + "\n")
    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        **{**kwargs, "transport": httpx.MockTransport(respond)}))
    original_init = runner.LocalCapture.__init__
    def init(cap, *args, **kwargs):
        original_init(cap, *args, **kwargs)
        cap.client_kwargs["transport"] = httpx.MockTransport(respond)
    monkeypatch.setattr(runner.LocalCapture, "__init__", init)
    inspected = []
    async def evaluate(mode, output, **kwargs):
        assert kwargs["cases_path"] == runner.CASES and kwargs["query_mode"] == "article"
        assert os.environ["DATABASE_PATH"].startswith(str(tmp_path / "output" / "state"))
        assert os.environ["LAW_AGENT_FINAL_PROTOCOL"] == "legacy"
        assert LLMCallPolicy.from_env() == LLMCallPolicy()
        copied = runner.Path(os.environ["LAW_FULL_INDEX_DIRECTORY"])
        assert copied != index and (copied / "public-only.txt").read_text() == "public snapshot"
        inspected.append(True)
        if outcome == "interrupted":
            raise TimeoutError("bounded round")
        facts = {"behavior_sequence": [runner.load_cases(runner.CASES)[0]["query"]], "consequence": ""}
        if outcome in {"response", "partial_then_interrupted"}:
            decision = await llm_gateway.generate_with_tools(_SYSTEM_PROMPT, json.dumps({
                "facts": json.dumps(facts, ensure_ascii=False)}, ensure_ascii=False), [], is_legal=True)
            assert decision["content"]
            if outcome == "partial_then_interrupted":
                raise TimeoutError("after one real captured response")
        return {"passed": 0, "total": 6, "cases": [{"trajectory": []}] * 6}
    monkeypatch.setattr(run_full_eval, "run_evaluation", evaluate)
    args = Namespace(output_dir=tmp_path / "output", models=["qwen3.5:9b"], rounds=1,
                     base_url="http://ollama.invalid:11434", index_directory=index)
    old_model = os.environ.get("OLLAMA_MODEL_NAME")
    result = await runner.run(args)
    assert inspected == [True]
    assert os.environ.get("OLLAMA_MODEL_NAME") == old_model
    if outcome in {"interrupted", "partial_then_interrupted"}:
        assert result["status"] == "runner_interrupted" and result["error_type"] == "TimeoutError"
        assert result["llm_executed"] is (outcome == "partial_then_interrupted")
        assert len(result["runs"]) == 1
        row = result["runs"][0]
        assert row["model"] == "qwen3.5:9b" and row["round"] == 1
        assert row["status"] == "runner_interrupted"
        assert row["passed"] is None and row["total"] is None
        assert (args.output_dir / row["raw"]).exists()
        summary = json.loads((args.output_dir / row["summary"]).read_text())
        assert summary["llm_executed"] is (outcome == "partial_then_interrupted")
        assert summary["passed"] is None and summary["total"] is None
        assert summary["rag_executed"] is None
        assert summary["actual_execution"]["raw_sha256"] == runner.hashlib.sha256(
            (args.output_dir / row["raw"]).read_bytes()).hexdigest()
        assert json.loads((args.output_dir / "comparison.json").read_text())["status"] == "runner_interrupted"
        return
    summary = json.loads((args.output_dir / "qwen3.5-9b-run1.json").read_text())
    if outcome == "no_inference":
        assert result["status"] == "blocked_no_inference"
        assert result["runs"][0]["passed"] is None and result["runs"][0]["total"] is None
        assert summary["llm_executed"] is False and summary["attempted_case_count"] == 6
        return
    assert result["status"] == "evaluated"
    assert result["runs"][0]["total"] == 6 and result["runs"][0]["passed"] == 0
    assert sent[0]["options"]["num_ctx"] == 8192
    assert sent[0]["options"]["num_predict"] == 1024
    assert summary["llm_executed"] is True and summary["rag_executed"] is False
    assert (index / "public-only.txt").read_text() == "public snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize("returned_model", ["qwen3.5:9b", "unrequested-model", "mixed-models"])
async def test_fixed_six_replay_through_real_lawref_and_original_scorer(tmp_path, monkeypatch, returned_model):
    """仅供应商与RAG网络用替身，实际工具消息、读取和评分走正式入口。"""
    from unittest.mock import AsyncMock

    from app.knowledge import law_retrieval

    index = tmp_path / "public.index"
    index.mkdir()
    (index.parent / (index.name + ".manifest.json")).write_text(json.dumps({
        "embedding_model": "embedding", "embedding_model_digest": "emb", "collection": "public"}))
    async def available(*_):
        return {"available_models": ["qwen3.5:9b"], "models": {
            "qwen3.5:9b": {"catalog_entry": {"digest": "nine"}}}, "requests": [
                {"endpoint": "/api/tags", "body": {"models": [{"name": "embedding", "digest": "emb"}]}}
            ]}
    monkeypatch.setattr(runner, "probe", available)
    monkeypatch.setattr(law_retrieval, "search_laws_by_rag", AsyncMock(return_value=[]))
    sent = []
    def respond(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        payload = json.loads(request.content)
        sent.append(payload)
        prompt = json.loads(payload["messages"][1]["content"])
        facts = prompt["facts"]
        facts = json.loads(facts) if isinstance(facts, str) else facts
        query = facts["behavior_sequence"][0]
        article = "第133条之一" if "车" in query or "133" in query else "第264条"
        tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
        if prompt.get("read_articles"):
            message = {"role": "assistant", "content": json.dumps({
                "article_ids": [article], "matched_elements": {article: []}, "confidence": "low"})}
        else:
            function = {"name": "get_article", "arguments": {"article_id": article}} if tool_messages else {
                "name": "search_laws", "arguments": {"query": article}}
            message = {"role": "assistant", "content": "", "tool_calls": [{"function": function}]}
        actual_model = ("qwen3.5:9b" if len(sent) == 1 else "unrequested-model") if returned_model == "mixed-models" else returned_model
        return httpx.Response(200, text=json.dumps({"model": actual_model, "message": message,
            "done": True, "eval_count": 10, "prompt_eval_count": 20}) + "\n")
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **{**kwargs, "transport": httpx.MockTransport(respond)}))
    args = Namespace(output_dir=tmp_path / "output", models=["qwen3.5:9b"], rounds=1,
                     base_url="http://ollama.invalid:11434", index_directory=index)
    result = await runner.run(args)
    if returned_model != "qwen3.5:9b":
        assert result["status"] == "blocked_identity_mismatch"
        assert result["llm_executed"] is True
        assert result["candidate_llm_executed"] is (returned_model == "mixed-models")
        row = result["runs"][0]
        assert row["status"] == "identity_mismatch"
        assert row["passed"] is None and row["total"] is None
        summary = json.loads((args.output_dir / row["summary"]).read_text())
        assert summary["actual_execution"]["identity_verified"] is False
        assert summary["actual_execution"]["returned_models"] == (
            ["qwen3.5:9b", "unrequested-model"] if returned_model == "mixed-models" else ["unrequested-model"])
        assert summary["passed"] is None and summary["total"] is None
        return
    assert result["status"] == "evaluated"
    assert result["runs"][0]["passed"] == result["runs"][0]["total"] == 6
    assert len(sent) == 18
    summary = json.loads((args.output_dir / "qwen3.5-9b-run1.json").read_text())
    assert all(case["termination_reason"] == "final_answer" for case in summary["cases"])
    assert all(case["guardrail_passed"] for case in summary["cases"][-2:])
    assert "expected_matched_elements" not in json.dumps(sent)


@pytest.mark.parametrize("status", ["runner_interrupted", "blocked_identity_mismatch"])
def test_cli_keeps_exit_two_for_incomplete_or_invalid_rounds(tmp_path, monkeypatch, status):
    async def evaluate(args):
        return {"status": status, "runs": [{"model": "qwen3.5:9b", "round": 1,
                                            "passed": None, "total": None}]}
    monkeypatch.setattr(runner, "run", evaluate)
    monkeypatch.setattr(runner.sys, "argv", ["run_local_lawref.py", "--base-url", "http://ollama.invalid:11434",
        "--models", "qwen3.5:9b", "--index-directory", str(tmp_path / "index"),
        "--output-dir", str(tmp_path / "output")])
    assert runner.main() == 2
