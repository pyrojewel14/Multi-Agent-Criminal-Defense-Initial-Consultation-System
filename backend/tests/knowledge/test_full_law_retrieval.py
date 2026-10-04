"""真实公共 Chroma/JSON/工具路径契约；外部模型 I/O 使用明确替身。"""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.consultation.agents.legal_research import LegalToolRegistry
from app.knowledge import full_law_index as fi
from app.knowledge import law_knowledge as lk
from app.knowledge import law_retrieval as lr
from app.knowledge.full_law_corpus import FULL_CORPUS_PATH


@pytest.fixture(scope="module")
def public_index(tmp_path_factory):
    directory = tmp_path_factory.mktemp("full-public") / "index"
    fi.build_full_index(
        FULL_CORPUS_PATH,
        directory,
        "public",
        "fixture",
        "fixture-digest",
        lambda docs: [[1.0, 0.0] if "醉酒驾驶机动车" in d else [0.0, 1.0] for d in docs],
    )
    return directory


@pytest.fixture
def runtime(public_index, monkeypatch):
    for key, value in {
        "LAW_KNOWLEDGE_PROFILE": "full",
        "LAW_FULL_DEMO_ANNOTATIONS": "off",
        "LAW_FULL_INDEX_DIRECTORY": str(public_index),
        "LAW_FULL_INDEX_COLLECTION": "public",
        "LAW_FULL_EMBEDDING_DIGEST": "fixture-digest",
        "TEXT_EMBEDDING_MODEL_NAME": "fixture",
        "EMBED_MODEL_TYPE": "OLLAMA",
        "LAW_FULL_RETRIEVAL_MODE": "hybrid",
        "LAW_FULL_RERANK_ENABLED": "false",
        "LAW_FULL_HYDE_ENABLED": "false",
    }.items():
        monkeypatch.setenv(key, value)
    actual_client, requests = httpx.AsyncClient, []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content) if request.content else None))
        data = (
            {"models": [{"name": "fixture", "digest": "fixture-digest"}]}
            if request.url.path == "/api/tags"
            else {"embeddings": [[0.0, 1.0]]}
        )
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: actual_client(transport=httpx.MockTransport(respond), **kw))
    lk._load_law_path.cache_clear()
    yield requests
    lk._load_law_path.cache_clear()


@pytest.mark.asyncio
async def test_chinese_recall_real_index_rag_registry(runtime):
    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    hits = await fi.search_full_index(facts)
    assert "第133条之一" in [lk._normalize_article_number(h["article_number"]) for h in hits]
    assert hits[0]["retrieval_status"]["bm25"] == "executed"
    raw = await lr.search_laws_by_rag(facts, "alice")
    assert raw[0]["source"] and raw[0]["content_sha256"]
    result = await LegalToolRegistry(facts, "alice", lk.load_criminal_law_data())._search_laws("醉酒驾驶机动车")
    candidate = result["candidates"][0]
    assert candidate["article_id"] == "第133条之一"
    assert candidate["rank"] == 1 and candidate["score"] is not None
    assert candidate["retrieval_status"]["bm25"] == "executed"


@pytest.mark.asyncio
async def test_identity_isolation_exact_paths(runtime, monkeypatch):
    assert await lr.search_laws_by_rag({"behavior_sequence": ["醉驾"]}, None) == []
    assert runtime == []
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/nonexistent-user-upload-index")
    for query, expected in [
        ("第133条之1", "第133条之一"),
        ("第133条", "第133条"),
        ("第199条", None),
        ("第999条", None),
    ]:
        hits = await lr.search_laws_by_rag({"behavior_sequence": [query]}, "bob")
        assert [lk._normalize_article_number(h["article_number"]) for h in hits] == ([expected] if expected else [])
    assert all(path == "/api/tags" for path, _ in runtime)


@pytest.mark.asyncio
async def test_json_keeps_rank_status_source_without_promoting_elements(runtime):
    hits = await fi.search_full_index({"behavior_sequence": ["第133条之一"]})
    hits[0].update(rank=1, fusion_score=0.03, retrieval_status={"rerank": "timeout"})
    enriched = lr._verify_and_enrich_with_json(hits, lk._build_article_index(lk.load_criminal_law_data()))
    assert enriched[0]["rank"] == 1 and enriched[0]["fusion_score"] == 0.03
    assert enriched[0]["retrieval_status"]["rerank"] == "timeout"
    assert enriched[0]["source"] == hits[0]["source"]
    assert enriched[0]["data_source"] == "text_only" and enriched[0]["required_elements"] == []


@pytest.mark.asyncio
async def test_registry_reranks_union_once_source_cannot_override(runtime, monkeypatch):
    from app.knowledge import full_law_retrieval as retrieval

    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "true")
    seen = []

    async def score(query, documents, timeout):
        seen.extend(documents)
        return [1.0 if "醉酒驾驶机动车" in d else 0.01 for d in documents]

    monkeypatch.setattr(retrieval, "score_candidates", score)
    result = await LegalToolRegistry({}, "alice", lk.load_criminal_law_data())._search_laws("醉酒驾驶机动车")
    assert len(seen) > 5
    assert result["candidates"][0]["article_id"] == "第133条之一"
    assert result["candidates"][0]["source"] == "text_only"
    assert result["candidates"][0]["score"] == 1.0
    assert result["retrieval_status"]["rerank"] == "executed"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("missing local model"), asyncio.TimeoutError()])
async def test_rerank_failure_keeps_fusion_order(runtime, monkeypatch, failure):
    from app.knowledge import full_law_retrieval as retrieval

    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    baseline = await fi.search_full_index(facts)
    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "true")
    monkeypatch.setattr(retrieval, "score_candidates", AsyncMock(side_effect=failure))
    hits = await fi.search_full_index(facts)
    assert [h["article_number"] for h in hits] == [h["article_number"] for h in baseline]
    assert hits[0]["retrieval_status"]["rerank"] in {"failed", "timeout"}
    assert hits[0]["retrieval_status"]["degraded"] is True


@pytest.mark.asyncio
async def test_hyde_additional_budget_timeout_fallback(runtime, monkeypatch):
    from app.knowledge import full_law_retrieval as retrieval

    monkeypatch.setenv("LAW_FULL_HYDE_ENABLED", "true")
    monkeypatch.setenv("LAW_FULL_HYDE_CALL_BUDGET", "0")
    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    hits = await fi.search_full_index(facts)
    assert hits[0]["retrieval_status"]["hyde"] == "budget_exhausted"
    monkeypatch.setenv("LAW_FULL_HYDE_CALL_BUDGET", "1")
    monkeypatch.setattr(retrieval, "generate_hyde", AsyncMock(side_effect=asyncio.TimeoutError()))
    hits = await fi.search_full_index(facts)
    assert hits[0]["retrieval_status"]["hyde"] == "timeout" and hits[0]["retrieval_status"]["vector"] == "executed"
    monkeypatch.setattr(retrieval, "generate_hyde", AsyncMock(return_value="刑法危险驾驶醉驾候选正文"))
    hits = await fi.search_full_index(facts)
    assert hits[0]["retrieval_status"]["hyde"] == "executed"
    embeds = [body["input"][0] for path, body in runtime if path == "/api/embed"]
    assert "醉酒驾驶机动车" in embeds and "刑法危险驾驶醉驾候选正文" in embeds
    assert facts == {"behavior_sequence": ["醉酒驾驶机动车"]}


@pytest.mark.asyncio
async def test_digest_manifest_reject_no_upload_fallback(runtime, public_index, monkeypatch):
    monkeypatch.setenv("LAW_FULL_EMBEDDING_DIGEST", "wrong")
    hits = await lr.search_laws_by_rag({"behavior_sequence": ["醉驾"]}, "alice")
    assert not hits and hits.dependency_failed
    monkeypatch.setenv("LAW_FULL_EMBEDDING_DIGEST", "fixture-digest")
    path = public_index.parent / "index.manifest.json"
    original = path.read_bytes()
    try:
        changed = json.loads(original)
        changed["corpus_sha256"] = "wrong"
        path.write_text(json.dumps(changed))
        hits = await lr.search_laws_by_rag({"behavior_sequence": ["醉驾"]}, "alice")
        assert not hits and hits.dependency_failed
    finally:
        path.write_bytes(original)


def test_qwen_protocol_token_ids_preserve_special_tokens_and_yes_no():
    from app.knowledge.full_law_reranker import PREFIX, SUFFIX, QwenFullReranker

    class Tokenizer:
        def encode(self, text, **kw):
            if text == PREFIX:
                return [101, 102]
            if text == SUFFIX:
                return [103, 104]
            if text.startswith("<Instruct>"):
                return [201]
            return [202, 203]

        def __call__(self, text, **kw):
            return {"input_ids": [201, 202, 203][: kw["max_length"]]}

        def convert_tokens_to_ids(self, token):
            return {"yes": 91, "no": 92}[token]

        unk_token_id = -1

    scorer = QwenFullReranker("unused", 6, "test")
    encoded = scorer.encode_pairs(Tokenizer(), "query", ["body"])
    assert encoded == [[101, 102, 201, 202, 103, 104]]
    assert scorer.decision_tokens(Tokenizer()) == (91, 92)
    assert "<|im_start|>system" in PREFIX and '"yes" or "no"' in PREFIX
    assert "<think>\n\n</think>" in SUFFIX
    scorer.close()


def test_reranker_retains_query_tail_and_only_truncates_document():
    from app.knowledge.full_law_reranker import PREFIX, SUFFIX, QwenFullReranker

    class Tokenizer:
        def encode(self, text, **kw):
            if text == PREFIX:
                return [100]
            if text == SUFFIX:
                return [200]
            return [ord(c) for c in text]

    scorer = QwenFullReranker("unused", 170, "instruction")
    tokens = scorer.encode_pairs(Tokenizer(), "这是原始查询，最后是否认：没有偷钱", ["正文" * 1000])[0]
    decoded = "".join(chr(t) for t in tokens[1:-1])
    assert "没有偷钱\n<Document>:" in decoded
    assert len(tokens) == 170
    with pytest.raises(ValueError, match="query"):
        scorer.encode_pairs(Tokenizer(), "否认" * 200, ["正文"])
    scorer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["corpus_version", "embeddings_sha256", "documents_sha256"])
async def test_full_rag_rejects_version_vector_document_hash(runtime, public_index, field):
    path = public_index.parent / "index.manifest.json"
    original = path.read_bytes()
    try:
        changed = json.loads(original)
        changed[field] = "tampered"
        path.write_text(json.dumps(changed))
        results = await lr.search_laws_by_rag({"behavior_sequence": ["拿走手机"]}, "alice")
        assert not results and results.dependency_failed
    finally:
        path.write_bytes(original)


@pytest.mark.asyncio
async def test_vector_switch_keeps_existing_five_and_hyde_off(runtime, monkeypatch):
    monkeypatch.setenv("LAW_FULL_RETRIEVAL_MODE", "vector")
    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    hits = await fi.search_full_index(facts)
    assert len(hits) == 5
    assert all(h["retrieval_method"] == "vector_index" for h in hits)
    assert hits[0]["retrieval_status"]["bm25"] == "disabled"
    assert "第133条之一" not in [lk._normalize_article_number(h["article_number"]) for h in hits]


@pytest.mark.asyncio
async def test_actual_hyde_deadline_and_session_budget_fallback(runtime, monkeypatch):
    from app.infrastructure.observability.tracing import SessionBudgetExceeded
    from app.knowledge import full_law_retrieval as retrieval

    monkeypatch.setenv("LAW_FULL_HYDE_ENABLED", "true")
    monkeypatch.setenv("LAW_FULL_HYDE_TIMEOUT_SECONDS", ".01")

    async def slow(query):
        await asyncio.sleep(0.2)
        return "hypothetical"

    monkeypatch.setattr(retrieval, "generate_hyde", slow)
    hits = await fi.search_full_index({"behavior_sequence": ["醉酒驾驶机动车"]})
    assert hits[0]["retrieval_status"]["hyde"] == "timeout"
    monkeypatch.setattr(
        retrieval, "generate_hyde", AsyncMock(side_effect=SessionBudgetExceeded(session_id="test", dimension="calls"))
    )
    hits = await fi.search_full_index({"behavior_sequence": ["醉酒驾驶机动车"]})
    assert hits[0]["retrieval_status"]["hyde"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_timeout_does_not_queue_another_background_inference():
    import threading

    from app.knowledge.full_law_reranker import QwenFullReranker, RerankerBusyError

    scorer = QwenFullReranker("unused", 512, "instruction")
    started, release = threading.Event(), threading.Event()

    def blocked(query, docs):
        started.set()
        release.wait(1)
        return [0.5]

    scorer._score_sync = blocked
    try:
        with pytest.raises(asyncio.TimeoutError):
            await scorer.score("query", ["doc"], 0.01)
        assert started.is_set()
        with pytest.raises(RerankerBusyError):
            await scorer.score("query", ["doc"], 0.01)
    finally:
        release.set()
        scorer.close()


def test_bm25_cache_changes_with_public_content_and_tokenizer_version():
    from app.knowledge.full_law_retrieval import _bm25, chinese_tokens

    assert "醉酒" in chinese_tokens("醉酒驾驶机动车")
    assert "醉酒驾驶机动车" not in chinese_tokens("醉酒驾驶机动车")
    first = _bm25("hash-a", "v1", ("醉酒驾驶", "偷钱"))
    assert first is _bm25("hash-a", "v1", ("醉酒驾驶", "偷钱"))
    assert first is not _bm25("hash-b", "v1", ("醉酒驾驶", "偷钱"))
    assert first is not _bm25("hash-a", "v2", ("醉酒驾驶", "偷钱"))
    assert first is not _bm25("hash-a", "v1", ("醉酒驾驶", "偷手机"))


@pytest.mark.asyncio
async def test_vector_registry_baseline_keeps_original_source_priority(runtime, monkeypatch):
    corpus = lk.load_criminal_law_data()
    index = lk._build_article_index(corpus)
    laws = []
    for number in ["第133条之一", "第264条"]:
        article = index[number]
        laws.append(
            {
                "article_number": article["article_number"],
                "content": article["article_number"] + " " + article["content"],
                "corpus_sha256": article["corpus_sha256"],
                "corpus_version": article["corpus_version"],
                "fusion_score": 1.0 / (61 + len(laws)),
                "recall_ranks": {"vector": len(laws) + 1},
            }
        )
    monkeypatch.setenv("LAW_FULL_RETRIEVAL_MODE", "vector")
    monkeypatch.setattr(
        lr,
        "search_laws_by_rag",
        AsyncMock(
            return_value=lr.LawSearchResults(laws, retrieval_status={"method": "vector_index", "mode": "vector"})
        ),
    )
    result = await LegalToolRegistry({}, "alice", corpus)._search_laws("查询")
    assert result["candidates"][0]["article_id"] == "第264条"
    assert result["retrieval_status"]["ranking_contract"] == "legacy_source_priority"


@pytest.mark.asyncio
async def test_worker_cancel_busy_recovery_and_background_error_consumed():
    import threading

    from app.knowledge.full_law_reranker import QwenFullReranker, RerankerBusyError

    scorer = QwenFullReranker("unused", 512, "instruction")
    started, release = threading.Event(), threading.Event()

    def blocked(query, docs):
        started.set()
        release.wait(1)
        raise RuntimeError("controlled worker failure")

    scorer._score_sync = blocked
    task = asyncio.create_task(scorer.score("query", ["doc"], 1))
    try:
        for _ in range(50):
            if started.is_set():
                break
            await asyncio.sleep(0.002)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RerankerBusyError):
            await scorer.score("query", ["doc"], 1)
        release.set()
        for _ in range(50):
            if scorer._future.done():
                break
            await asyncio.sleep(0.002)
        assert scorer._future.done()
        await asyncio.sleep(0.01)
        scorer._score_sync = lambda query, docs: [0.75]
        assert await scorer.score("query", ["doc"], 1) == [0.75]
    finally:
        release.set()
        scorer.close()


@pytest.mark.asyncio
async def test_hyde_session_budget_exhaustion_makes_no_http_request(runtime, monkeypatch):
    from app.infrastructure.observability.tracing import SessionBudget, SessionBudgetExceeded
    from app.knowledge import full_law_retrieval as retrieval

    budget = SessionBudget(max_calls=1, max_tokens=100)
    budget.reserve_call("bounded")
    monkeypatch.setattr(retrieval, "session_budget", budget)
    monkeypatch.setattr(retrieval, "current_trace_context", lambda: {"session_id": "bounded"})
    with pytest.raises(SessionBudgetExceeded):
        await retrieval.generate_hyde("不会传给 HTTP 的查询")
    assert runtime == []


def test_actual_local_qwen_tokenizer_retains_special_tokens_and_negation():
    import os
    from pathlib import Path

    from transformers import AutoTokenizer

    from app.knowledge.full_law_reranker import PREFIX, SUFFIX, QwenFullReranker

    path = Path(os.getenv("RERANKER_MODEL_PATH", "data/models/Qwen/Qwen3-Reranker-0.6B"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not (path / "tokenizer.json").is_file():
        pytest.skip("configured local tokenizer not present; no download allowed")
    tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side="left", local_files_only=True)
    scorer = QwenFullReranker(str(path), 512, "instruction")
    try:
        query = "指控我偷了钱，但我否认，没有拿钱"
        tokens = scorer.encode_pairs(tokenizer, query, ["完整正文" * 300])[0]
        assert len(tokens) == 512
        assert tokens[: len(tokenizer.encode(PREFIX, add_special_tokens=False))] == tokenizer.encode(
            PREFIX, add_special_tokens=False
        )
        assert tokens[-len(tokenizer.encode(SUFFIX, add_special_tokens=False)) :] == tokenizer.encode(
            SUFFIX, add_special_tokens=False
        )
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        assert query in decoded and "<think>\n\n</think>" in decoded
        assert scorer.decision_tokens(tokenizer) == (9693, 2152)
        assert tokenizer.encode("yes", add_special_tokens=False) == [9693]
        assert tokenizer.encode("no", add_special_tokens=False) == [2152]
        with pytest.raises(ValueError, match="query"):
            scorer.encode_pairs(tokenizer, "前文" * 600 + "最后否认：没有偷钱", ["正文"])
    finally:
        scorer.close()


@pytest.mark.asyncio
async def test_hyde_records_actual_session_usage_without_saving_generated_facts(monkeypatch):
    from app.infrastructure.observability.tracing import SessionBudget
    from app.knowledge import full_law_retrieval as retrieval

    budget = SessionBudget(max_calls=2, max_tokens=100)
    requests = []
    actual_client = httpx.AsyncClient

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"response": "用于额外召回的假设正文", "prompt_eval_count": 12, "eval_count": 8}
        )

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: actual_client(transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(retrieval, "session_budget", budget)
    monkeypatch.setattr(retrieval, "current_trace_context", lambda: {"session_id": "usage"})
    assert await retrieval.generate_hyde("否认拿钱") == "用于额外召回的假设正文"
    assert budget.snapshot("usage") == {"calls": 1, "tokens": 20}
    assert len(requests) == 1 and requests[0]["options"]["num_predict"] == 192
    assert requests[0]["stream"] is False and requests[0]["think"] is False


@pytest.mark.asyncio
async def test_user_identity_does_not_change_public_pool(runtime):
    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    alice = await lr.search_laws_by_rag(facts, "alice")
    bob = await lr.search_laws_by_rag(facts, "bob")
    assert [hit["article_number"] for hit in alice] == [hit["article_number"] for hit in bob]
    assert all(hit["is_public"] is True and not hit.get("user_id") for hit in [*alice, *bob])
