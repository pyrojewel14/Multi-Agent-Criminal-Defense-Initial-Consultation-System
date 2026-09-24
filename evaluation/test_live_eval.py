"""Live runner 的前置条件和证据保留契约。"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from evaluation.build_live_index import build_index

from evaluation.run_live_eval import preflight, summarize_events, run_cases, _run_ablation, _execute_chain, configure_runtime, _metadata


def test_metadata_records_actual_llm_deadlines(monkeypatch):
    monkeypatch.setenv("LLM_TOTAL_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("LLM_ATTEMPT_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "0.5")
    metadata = _metadata({}, [{"id": "ordinary_theft", "input": "test"}])
    assert metadata["llm_policy"] == {
        "total_timeout_seconds": 180.0,
        "attempt_timeout_seconds": 90.0,
        "max_attempts": 2,
        "backoff_seconds": 0.5,
    }


def test_preflight_separates_mutable_chroma_snapshot_from_build_manifest(tmp_path):
    law = Path(__file__).resolve().parents[1] / "backend/data/law_knowledge/criminal_law_chapters.json"
    index = tmp_path / "index"
    build_manifest = build_index(
        law, index, "rag_collection", "qwen3-embedding:0.6b", "embed-digest",
        lambda texts: [[float(i), 1.0] for i in range(len(texts))],
    )
    reranker = tmp_path / "reranker"
    reranker.mkdir()
    (reranker / "config.json").write_text("{}")
    (reranker / "model.safetensors").write_bytes(b"test")
    kwargs = dict(
        base_url="http://127.0.0.1:11434", model="qwen3.5:0.8b",
        embedding_model="qwen3-embedding:0.6b", index_dir=index,
        reranker_path=reranker, law_path=law,
        fetch_tags=lambda _: {"models": [
            {"name": "qwen3.5:0.8b", "digest": "chat-digest"},
            {"name": "qwen3-embedding:0.6b", "digest": "embed-digest"},
        ]},
    )
    first = preflight(**kwargs)
    second = preflight(**kwargs)
    manifest_path = index.parent / "index.manifest.json"
    stable_hash = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert first["ready"] and second["ready"]
    assert first["index_hash_scope"] == "mutable_chroma_directory_snapshot"
    assert first["index_build_manifest_sha256"] == second["index_build_manifest_sha256"] == stable_hash
    assert first["index_build_documents_sha256"] == build_manifest["documents_sha256"]
    assert first["index_build_embeddings_sha256"] == build_manifest["embeddings_sha256"]


def test_preflight_blocks_missing_index_and_records_model_digest(tmp_path, monkeypatch):
    law = tmp_path / "law.json"
    law.write_text('{"metadata":{"dataset_version":"test-v1"}}', encoding="utf-8")
    reranker = tmp_path / "reranker"
    reranker.mkdir()

    def tags(_url):
        return {"models": [{"name": "qwen3.5:0.8b", "digest": "sha256:chat"},
                           {"name": "qwen3-embedding:0.6b", "digest": "sha256:embed"}]}

    result = preflight(
        base_url="http://127.0.0.1:11434",
        model="qwen3.5:0.8b",
        embedding_model="qwen3-embedding:0.6b",
        index_dir=tmp_path / "absent",
        reranker_path=reranker,
        law_path=law,
        fetch_tags=tags,
    )
    assert result["ready"] is False
    assert "index_missing" in result["issues"]
    assert result["model_digest"] == "sha256:chat"
    assert result["embedding_digest"] == "sha256:embed"
    assert result["law_dataset_version"] == "test-v1"


def test_preflight_rejects_non_loopback_and_missing_model(tmp_path):
    result = preflight(
        base_url="http://example.com:11434",
        model="missing",
        embedding_model="embed",
        index_dir=tmp_path,
        reranker_path=tmp_path,
        law_path=tmp_path / "missing.json",
        fetch_tags=lambda _: {"models": []},
    )
    assert "non_loopback_ollama" in result["issues"]
    assert "model_missing" in result["issues"]
    assert "law_snapshot_missing" in result["issues"]


def test_preflight_rejects_empty_chroma_collection(tmp_path):
    import chromadb

    index = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(index))
    client.get_or_create_collection("rag_collection")
    law = tmp_path / "law.json"
    law.write_text('{"metadata":{"dataset_version":"v1"}}', encoding="utf-8")
    result = preflight(
        base_url="http://127.0.0.1:11434", model="chat", embedding_model="embed",
        index_dir=index, reranker_path=tmp_path / "reranker", law_path=law,
        fetch_tags=lambda _: {"models": [{"name": "chat", "digest": "chat-digest"},
                                      {"name": "embed", "digest": "embed-digest"}]},
    )
    assert "index_empty" in result["issues"]
    assert result["index_document_count"] == 0


def test_preflight_rejects_reranker_config_without_weights(tmp_path):
    reranker = tmp_path / "reranker"
    reranker.mkdir()
    (reranker / "config.json").write_text("{}", encoding="utf-8")
    law = tmp_path / "law.json"
    law.write_text('{"metadata":{"dataset_version":"v1"}}', encoding="utf-8")
    result = preflight(
        base_url="http://127.0.0.1:11434", model="chat", embedding_model="embed",
        index_dir=tmp_path / "missing", reranker_path=reranker, law_path=law,
        fetch_tags=lambda _: {"models": [{"name": "chat"}, {"name": "embed"}]},
    )
    assert "reranker_weights_missing" in result["issues"]


def test_runtime_paths_match_production_environment(monkeypatch):
    for name in ("RERANKER_MODEL_PATH", "CHROMA_PERSIST_DIRECTORY"):
        monkeypatch.delenv(name, raising=False)
    paths = configure_runtime()
    assert paths["reranker_path"].is_absolute()
    assert paths["index_dir"].is_absolute()
    assert os.environ["RERANKER_MODEL_PATH"] == str(paths["reranker_path"])
    assert os.environ["CHROMA_PERSIST_DIRECTORY"] == str(paths["index_dir"])


def test_summary_counts_attempts_and_routes_without_claiming_accuracy():
    events = [
        {"event_type": "llm", "attempt": 1, "outcome": "error", "duration_ms": 10},
        {"event_type": "llm", "attempt": 2, "outcome": "success", "duration_ms": 20},
        {"event_type": "rag", "attempt": None, "outcome": "success", "duration_ms": 5},
        {"event_type": "route", "route_reason": "dependency_degraded", "duration_ms": 1},
    ]
    summary = summarize_events(events)
    assert summary == {
        "llm_attempts": 2, "rag_calls": 1,
        "route_reasons": ["dependency_degraded"],
        "retrieval_top_k_status": "unavailable", "retrieval_top_k": [],
    }


def test_run_cases_keeps_failed_case_and_continues(tmp_path):
    async def execute(case):
        if case["id"] == "broken":
            raise RuntimeError("service failed")
        return {"status": "wait_for_user", "llm_attempts": 1}

    output = tmp_path / "result.json"
    result = asyncio.run(run_cases([{"id": "broken", "input": "A"},
                                    {"id": "ok", "input": "B"}], execute, output))
    assert all(case["id"].startswith("sha256:") for case in result["cases"])
    assert result["cases"][0]["status"] == "error"
    assert result["cases"][0]["error_type"] == "RuntimeError"
    assert result["cases"][1]["status"] == "wait_for_user"
    assert json.loads(output.read_text(encoding="utf-8"))["cases"] == result["cases"]


def test_reception_ablation_records_both_disclaimers(tmp_path, monkeypatch):
    from app.agents.receptionist import llm_gateway
    from app.observability.tracing import trace_span, trace_store

    async def response(**_kwargs):
        with trace_span(trace_store, event_type="llm", name="test-provider-attempt", attempt=1):
            pass
        return "请选择身份类型"

    monkeypatch.setattr(llm_gateway, "generate", response)
    output = tmp_path / "ablation.json"
    result = asyncio.run(_run_ablation([{"id": "one", "input": "我是当事人"}], output, {}))
    assert result["cases"][0]["status"] == "success"
    assert result["cases"][0]["template_disclaimer"] is True
    assert result["cases"][0]["llm_disclaimer"] is True
    assert result["cases"][0]["llm_attempts"] == 1
    assert result["mode"] == "component-ablation"


def test_reception_ablation_keeps_failed_attempt_count(tmp_path, monkeypatch):
    from app.agents.receptionist import llm_gateway
    from app.observability.tracing import trace_span, trace_store

    secret = "张三13800138000110105199001011234"

    async def response(**_kwargs):
        with trace_span(trace_store, event_type="llm", name="test-provider-attempt", attempt=1):
            pass
        raise TimeoutError(f"provider timeout {secret}")

    monkeypatch.setattr(llm_gateway, "generate", response)
    output = tmp_path / "out.json"
    result = asyncio.run(_run_ablation([{"id": "failure", "input": "我是当事人"}], output, {}))
    assert result["cases"][0]["status"] == "error"
    assert result["cases"][0]["error_type"] == "TimeoutError"
    assert result["cases"][0]["llm_attempts"] == 1
    assert result["aggregate"]["llm_attempts"] == 1
    assert secret not in output.read_text(encoding="utf-8")


def test_chain_result_reports_wait_node_instead_of_successor(monkeypatch):
    import app.orchestrator.workflow as workflow

    class Orchestrator:
        async def start_workflow(self, _state):
            return {}

        async def resume_workflow(self, _session_id, _updates=None):
            return {"current_agent": "FactDigger", "facts_structured": {}}

        async def get_next_node(self, _session_id):
            return "fact_intake"

    monkeypatch.setattr(workflow, "ConsultationOrchestrator", Orchestrator)
    result = asyncio.run(_execute_chain({"id": "one", "input": "案情不完整"}))
    assert result["status"] == "wait_for_user"


def test_chain_result_counts_attempt_before_failure(monkeypatch):
    import app.orchestrator.workflow as workflow
    from app.observability.tracing import trace_span, trace_store

    class Orchestrator:
        async def start_workflow(self, state):
            with trace_span(trace_store, event_type="llm", name="test-attempt", session_id=state["session_id"]):
                pass
            raise RuntimeError("startup failed")

    monkeypatch.setattr(workflow, "ConsultationOrchestrator", Orchestrator)
    result = asyncio.run(_execute_chain({"id": "one", "input": "案情"}))
    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"
    assert result["llm_attempts"] == 1


def test_real_compiled_graph_consumes_case_once(monkeypatch):
    import app.agents.fact_digger as fact_digger
    import app.orchestrator.workflow as workflow
    from app.schemas.llm_artifacts import ArtifactSource

    seen = []

    async def extract(facts_raw):
        seen.append(list(facts_raw))
        return {
            "incident_time": None, "incident_location": None, "parties": [],
            "behavior_sequence": [], "consequence": None, "evidence_mentioned": [],
            "arrest_status": None, "surrender": None, "victim_forgiveness": None,
            "prior_record": None,
        }, ArtifactSource.CONTENT_JSON

    monkeypatch.setattr(fact_digger, "_extract_structured_facts", extract)
    monkeypatch.setattr(workflow, "INTERRUPT_AFTER_NODES", ["receptionist", "fact_intake"])
    result = asyncio.run(_execute_chain({"id": "once", "input": "上周在超市发生盗窃"}))
    assert result["status"] != "error"
    assert len(seen) == 1
    assert len(seen[0]) == 1


def test_failed_case_omits_sensitive_exception_text(tmp_path):
    secret = "张三13800138000110105199001011234"

    async def execute(_case):
        raise RuntimeError(f"prompt leaked: {secret}")

    output = tmp_path / "failure.json"
    result = asyncio.run(run_cases([{"id": "failure", "input": "synthetic"}], execute, output))
    serialized = output.read_text(encoding="utf-8")
    assert secret not in serialized
    assert "prompt leaked" not in serialized
    assert result["cases"][0]["error_type"] == "RuntimeError"


def test_chain_public_result_excludes_untrusted_artifact_and_law_text(monkeypatch, tmp_path):
    import app.orchestrator.workflow as workflow
    secret = "张三13800138000110105199001011234"

    class Orchestrator:
        async def start_workflow(self, _state):
            return {}

        async def resume_workflow(self, _session_id, _updates=None):
            return {
                "facts_structured": {secret: secret},
                "applied_laws": [
                    {"article_number": secret, "data_source": secret, "content": secret},
                    {"article_number": "第13800138000条", "data_source": "rag_unverified"},
                ],
                "artifact_results": {"fact": {"status": "degraded", "source": "content_json",
                                               "validation_errors": [{"msg": secret, "loc": [secret]}]}},
                "law_search_status": secret,
            }

        async def get_next_node(self, _session_id):
            return None

    monkeypatch.setattr(workflow, "ConsultationOrchestrator", Orchestrator)
    output = tmp_path / "public.json"
    result = asyncio.run(run_cases([{"id": "redacted", "input": "synthetic"}], _execute_chain, output))
    assert secret not in output.read_text(encoding="utf-8")
    assert "13800138000" not in output.read_text(encoding="utf-8")
    row = result["cases"][0]
    assert row["artifact_results"]["fact"]["validation_error_count"] == 1


def test_ranked_trace_is_not_confused_with_final_law_candidates():
    digest_one = "sha256:" + "a" * 64
    digest_two = "sha256:" + "b" * 64
    origin_one = "sha256:" + "c" * 64
    origin_two = "sha256:" + "d" * 64
    events = [
        {"event_type": "rag_ranked_result_set"},
        {"event_type": "rag_ranked_result", "attempt": 2,
         "metadata": {"content": {"sha256": digest_two}, "origin": {"sha256": origin_two}}},
        {"event_type": "rag_ranked_result", "attempt": 1,
         "metadata": {"content": {"sha256": digest_one}, "origin": {"sha256": origin_one}}},
    ]
    summary = summarize_events(events)
    assert summary["retrieval_top_k_status"] == "observed"
    assert summary["retrieval_top_k"] == [
        {"rank": 1, "document_id": digest_one, "source": "rag_returned_content", "source_id": origin_one},
        {"rank": 2, "document_id": digest_two, "source": "rag_returned_content", "source_id": origin_two},
    ]


def test_chain_result_reads_actual_ranked_events_separately_from_laws(monkeypatch):
    import app.orchestrator.workflow as workflow
    from app.observability.tracing import trace_span, trace_store

    class Orchestrator:
        async def start_workflow(self, _state):
            return {}

        async def resume_workflow(self, session_id, _updates=None):
            with trace_span(trace_store, event_type="rag_ranked_result_set", name="retrieve_documents", session_id=session_id):
                pass
            with trace_span(trace_store, event_type="rag_ranked_result", name="reranked_document",
                            session_id=session_id, attempt=1, metadata={"content": "检索文档"}):
                pass
            return {"applied_laws": [{"article_number": "第264条", "data_source": "json_keyword"}]}

        async def get_next_node(self, _session_id):
            return None

    monkeypatch.setattr(workflow, "ConsultationOrchestrator", Orchestrator)
    result = asyncio.run(_execute_chain({"id": "ranked", "input": "synthetic"}))
    assert result["retrieval_top_k_status"] == "observed"
    assert len(result["retrieval_top_k"]) == 1
    assert result["retrieval_top_k"][0]["source"] == "rag_returned_content"
    assert result["laws"] == [{"article_number": "第264条", "data_source": "json_keyword"}]


def test_empty_ranked_result_set_is_observed_empty():
    summary = summarize_events([{"event_type": "rag_ranked_result_set"}])
    assert summary["retrieval_top_k_status"] == "observed"
    assert summary["retrieval_top_k"] == []


def test_chain_failure_omits_sensitive_exception_text(monkeypatch):
    import app.orchestrator.workflow as workflow
    secret = "张三13800138000110105199001011234"

    class Orchestrator:
        async def start_workflow(self, _state):
            raise RuntimeError(secret)

    monkeypatch.setattr(workflow, "ConsultationOrchestrator", Orchestrator)
    result = asyncio.run(_execute_chain({"id": "failed", "input": "synthetic"}))
    assert secret not in json.dumps(result, ensure_ascii=False)
    assert result["error_type"] == "RuntimeError"


def test_untrusted_case_id_is_not_written_verbatim(tmp_path):
    secret = "张三13800138000"

    async def execute(_case):
        return {"status": "end"}

    output = tmp_path / "case.json"
    result = asyncio.run(run_cases([{"id": secret, "input": "synthetic"}], execute, output))
    assert secret not in output.read_text(encoding="utf-8")
    assert result["cases"][0]["id"].startswith("sha256:")


def test_numeric_phone_and_identity_case_ids_are_hashed(tmp_path):
    async def execute(_case):
        return {"status": "end"}

    for secret in ("13800138000", "110105199001011234"):
        output = tmp_path / f"{secret}.json"
        result = asyncio.run(run_cases([{"id": secret, "input": "synthetic"}], execute, output))
        assert secret not in output.read_text(encoding="utf-8")
        assert result["cases"][0]["id"].startswith("sha256:")


def test_untrusted_exception_class_name_is_not_written(tmp_path):
    secret = "13800138000"
    LeakError = type(f"LeakError_{secret}", (Exception,), {})

    async def execute(_case):
        raise LeakError("sensitive")

    output = tmp_path / "error.json"
    result = asyncio.run(run_cases([{"id": "safe", "input": "synthetic"}], execute, output))
    assert secret not in output.read_text(encoding="utf-8")
    assert result["cases"][0]["error_type"] == "UnknownError"


def test_preflight_does_not_echo_url_credentials(tmp_path):
    result = preflight(
        base_url="http://user:topsecret@127.0.0.1:11434",
        model="chat", embedding_model="embed",
        index_dir=tmp_path, reranker_path=tmp_path, law_path=tmp_path / "none",
        fetch_tags=lambda _: {"models": []},
    )
    assert "topsecret" not in json.dumps(result)
    assert result["ready"] is False


def test_preflight_hashes_nondefault_model_names(tmp_path):
    model = "secretChat_13800138000"
    embedding = "secretEmbed_110105199001011234"
    result = preflight(
        base_url="http://127.0.0.1:11434", model=model, embedding_model=embedding,
        index_dir=tmp_path, reranker_path=tmp_path, law_path=tmp_path / "none",
        fetch_tags=lambda _: {"models": [{"name": model, "digest": "chat-digest"},
                                      {"name": embedding, "digest": "embed-digest"}]},
    )
    encoded = json.dumps(result)
    assert model not in encoded
    assert embedding not in encoded
    assert result["model"].startswith("sha256:")
    assert result["embedding_model"].startswith("sha256:")


def test_tracked_evidence_has_no_raw_exception_message_field():
    from evaluation.run_live_eval import ROOT

    for path in (ROOT / "evaluation/evidence").glob("*.json"):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        for case in evidence.get("cases", []):
            assert "error" not in case, path.name
