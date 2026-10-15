"""真实 Chroma 与 HTTP 替身连接完整性、唯一向量回调及服务生命周期。"""

import json

import httpx
import pytest

from app.knowledge.retrieval.config import RerankerConfig, RetrievalConfig


@pytest.fixture
def full_runtime(tmp_path, monkeypatch):
    from app.knowledge.law.corpus import FULL_CORPUS_PATH
    from app.knowledge.law.index import build_full_index

    directory = tmp_path / "public-index"
    build_full_index(FULL_CORPUS_PATH, directory, "public", "fixture", "fixture-digest",
                     lambda docs: [[1.0, 0.0] if "醉酒驾驶机动车" in doc else [0.0, 1.0] for doc in docs])
    for key, value in {"LAW_KNOWLEDGE_PROFILE": "full", "LAW_FULL_INDEX_DIRECTORY": str(directory),
                       "LAW_FULL_INDEX_COLLECTION": "public", "LAW_FULL_EMBEDDING_DIGEST": "fixture-digest",
                       "TEXT_EMBEDDING_MODEL_NAME": "fixture", "EMBED_MODEL_TYPE": "OLLAMA",
                       "LAW_FULL_RERANK_ENABLED": "false", "LAW_FULL_HYDE_ENABLED": "false",
                       "LAW_FULL_DEMO_ANNOTATIONS": "off"}.items():
        monkeypatch.setenv(key, value)
    requests = []
    actual = httpx.AsyncClient

    def response(request):
        requests.append((request.url.path, json.loads(request.content) if request.content else None))
        data = {"models": [{"name": "fixture", "digest": "fixture-digest"}]} if request.url.path == "/api/tags" else {"embeddings": [[1.0, 0.0]]}
        return httpx.Response(200, json=data)

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        return actual(transport=httpx.MockTransport(response), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return directory, requests


@pytest.mark.asyncio
async def test_full_binding_checks_actual_digest_and_manifest_each_request_then_embed(full_runtime):
    from app.knowledge.wiring import build_law_search_service

    directory, requests = full_runtime
    service = build_law_search_service()
    first = await service.search("醉酒驾驶机动车", user_id="alice")
    assert first.search.candidates[0].metadata["article_number"] == "第一百三十三条之一"
    assert first.search.retrieval_status["integrity_verified"] is True
    assert first.search.retrieval_status["bm25"] == "executed"
    assert requests[1] == ("/api/embed", {"model": "fixture", "input": ["醉酒驾驶机动车"], "truncate": False})
    path = directory.parent / "public-index.manifest.json"
    data = json.loads(path.read_text())
    data["embeddings_sha256"] = "tampered"
    path.write_text(json.dumps(data))
    second = await service.search("盗窃", user_id="alice")
    assert second.search.dependency_failed
    assert second.search.candidates and second.as_laws()[0]["data_source"] == "json_keyword"
    assert [path for path, _ in requests] == ["/api/tags", "/api/embed", "/api/tags"]


@pytest.mark.asyncio
async def test_exact_still_checks_index_without_query_embedding(full_runtime):
    from app.knowledge.wiring import build_law_search_service

    _, requests = full_runtime
    result = await build_law_search_service().search("第133条之1 第264条", user_id="alice")
    assert result.as_laws()[0]["article_number"] == "第一百三十三条之一"
    assert [path for path, _ in requests] == ["/api/tags"]
    assert result.search.retrieval_status["rerank"] == "skipped_exact"


@pytest.mark.asyncio
async def test_full_qwen_name_gate_degrades_without_loading_wrong_model(full_runtime, monkeypatch):
    from app.knowledge.wiring import build_law_search_service, KnowledgeResources

    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANKER_MODEL_NAME", "other/Qwen3-lookalike")
    resources = KnowledgeResources()
    monkeypatch.setattr(resources, "scorer", lambda config: pytest.fail("unsupported model must never be created"))
    result = await build_law_search_service(resources=resources).search("盗窃", user_id="alice")
    assert result.search.candidates
    assert any(law["data_source"] == "json_keyword" for law in result.as_laws())
    assert result.search.retrieval_status["rerank"] == "failed"
    assert result.search.retrieval_status["rerank_error"] == "ValueError"
    assert result.search.retrieval_status["degraded"] is True
    assert not result.search.dependency_failed


def test_readiness_and_close_never_load_model(monkeypatch):
    from app.knowledge.wiring import KnowledgeResources

    resources = KnowledgeResources()
    scorer = resources.scorer(RerankerConfig(local_path="/definitely/missing/model"))
    monkeypatch.setattr(scorer, "_load_model_sync", lambda: pytest.fail("readiness loaded weights"))
    state = resources.readiness()
    assert state["status"] == "not_loaded" and state["available"] is False
    assert "/definitely" not in json.dumps(state)
    resources.close()
    assert scorer.readiness()["closed"] is True


def test_moved_assets_load_from_backend_data_directory():
    from app.knowledge.law.corpus import LAW_KNOWLEDGE_PATH, load_criminal_law_data

    assert LAW_KNOWLEDGE_PATH.is_file()
    assert LAW_KNOWLEDGE_PATH.parent.parent.name == "data"
    assert len(load_criminal_law_data(profile="snapshot")["chapters"]) > 0


def test_explicit_environment_snapshot_selects_matching_assets(monkeypatch):
    from app.knowledge.wiring import build_law_search_service
    from app.knowledge.law.verification import is_lawref_eligible

    monkeypatch.setenv("LAW_FULL_DEMO_ANNOTATIONS", "on")
    selected = build_law_search_service(env={"LAW_KNOWLEDGE_PROFILE": "full", "LAW_FULL_DEMO_ANNOTATIONS": "off", "LAW_FULL_RERANK_ENABLED": "false"})
    assert not is_lawref_eligible(selected.repository.article_index["第133条之一"])


@pytest.mark.asyncio
async def test_preflight_rejects_repository_from_another_corpus_even_if_index_valid(full_runtime):
    from app.knowledge.wiring import build_law_search_service

    selected = build_law_search_service()
    selected.repository.law_data["metadata"]["corpus_sha256"] = "wrong-corpus"
    result = await selected.search("盗窃", user_id="alice")
    assert result.search.dependency_failed
    assert result.search.retrieval_status["integrity_error"] == "ValueError"
