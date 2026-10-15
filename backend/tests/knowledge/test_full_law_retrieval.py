"""真实公共 Chroma/JSON/工具路径契约；外部模型 I/O 使用明确替身。"""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.consultation.agents.legal_research import LegalToolRegistry
from app.knowledge.law import corpus as lk
from app.knowledge.law import index as fi
from app.knowledge.law.corpus import FULL_CORPUS_PATH
from app.knowledge.law.identifiers import normalize_article_number
from app.knowledge.law.repository import _build_article_index
from tests.knowledge.law.helpers import verified_laws


async def _search_result(facts, user_id="alice", *, keyword_terms=None):
    from app.knowledge.law.keyword import facts_query
    from app.knowledge.wiring import build_law_search_service
    return await build_law_search_service().search(facts_query(facts), user_id=user_id, keyword_terms=keyword_terms)


async def _search(facts):
    from app.knowledge.law.keyword import facts_terms
    return (await _search_result(facts, keyword_terms=facts_terms(facts))).as_laws()


def _set_scorer(monkeypatch, callback):
    from app.knowledge.wiring import knowledge_resources
    class Scorer:
        probability_scores = True
        async def score(self, query, documents, *, timeout_s):
            return await callback(query, documents, timeout_s)
    monkeypatch.setattr(knowledge_resources, "scorer", lambda config: Scorer())


def _set_expander(monkeypatch, callback):
    from app.knowledge.retrieval.hyde import HyDEExpander
    async def expand(self, query, *, timeout_s):
        return await callback(query)
    monkeypatch.setattr(HyDEExpander, "expand", expand)


async def _generate_hyde(query):
    from functools import partial

    from app.knowledge.retrieval.hyde import HyDEExpander, ollama_generate
    return await HyDEExpander(partial(ollama_generate, base_url="http://127.0.0.1:11434", model_name="fixture"), model_name="fixture").expand(query, timeout_s=10)


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
    hits = await _search(facts)
    assert "第133条之一" in [normalize_article_number(h["article_number"]) for h in hits]
    assert hits[0]["retrieval_status"]["bm25"] == "executed"
    raw = (await _search_result(facts, "alice")).as_laws()
    assert raw[0]["source"] and raw[0]["content_sha256"]
    result = await LegalToolRegistry(facts, "alice", lk.load_criminal_law_data())._search_laws("醉酒驾驶机动车")
    candidate = result["candidates"][0]
    assert candidate["article_id"] == "第133条之一"
    assert candidate["rank"] == 1 and candidate["score"] is not None
    assert candidate["retrieval_status"]["bm25"] == "executed"


@pytest.mark.asyncio
async def test_identity_isolation_exact_paths(runtime, monkeypatch):
    skipped = await _search_result({"behavior_sequence": ["醉驾"]}, None)
    assert skipped.search.retrieval_status["vector"] == "skipped_unauthorized"
    assert runtime == []
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/nonexistent-user-upload-index")
    for query, expected in [
        ("第133条之1", "第133条之一"),
        ("第133条", "第133条"),
        ("第199条", None),
        ("第999条", None),
    ]:
        hits = (await _search_result({"behavior_sequence": [query]}, "bob")).as_laws()
        assert [normalize_article_number(h["article_number"]) for h in hits] == ([expected] if expected else [])
    assert all(path == "/api/tags" for path, _ in runtime)


@pytest.mark.asyncio
async def test_json_keeps_rank_status_source_without_promoting_elements(runtime):
    corpus = lk.load_criminal_law_data()
    row = next(row for row in fi._rows(corpus, corpus["metadata"]["corpus_sha256"]) if row["metadata"]["article_id"] == "第133条之一")
    hits = [{**row["metadata"], "content": row["document"]}]
    hits[0].update(rank=1, fusion_score=0.03, retrieval_status={"rerank": "timeout"})
    enriched = verified_laws(hits, _build_article_index(lk.load_criminal_law_data()))
    assert enriched[0]["rank"] == 1 and enriched[0]["fusion_score"] == 0.03
    assert enriched[0]["retrieval_status"]["rerank"] == "timeout"
    assert enriched[0]["source"] == hits[0]["source"]
    assert enriched[0]["data_source"] == "text_only" and enriched[0]["required_elements"] == []


@pytest.mark.asyncio
async def test_registry_reranks_union_once_source_cannot_override(runtime, monkeypatch):

    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "true")
    seen = []

    async def score(query, documents, timeout):
        seen.extend(documents)
        return [1.0 if "醉酒驾驶机动车" in d else 0.01 for d in documents]

    _set_scorer(monkeypatch, score)
    result = await LegalToolRegistry({}, "alice", lk.load_criminal_law_data())._search_laws("醉酒驾驶机动车")
    assert len(seen) > 5
    assert result["candidates"][0]["article_id"] == "第133条之一"
    assert result["candidates"][0]["source"] == "text_only"
    assert result["candidates"][0]["score"] == 1.0
    assert result["retrieval_status"]["rerank"] == "executed"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("missing local model"), asyncio.TimeoutError()])
async def test_rerank_failure_keeps_fusion_order(runtime, monkeypatch, failure):

    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    baseline = await _search(facts)
    monkeypatch.setenv("LAW_FULL_RERANK_ENABLED", "true")
    _set_scorer(monkeypatch, AsyncMock(side_effect=failure))
    hits = await _search(facts)
    assert [h["article_number"] for h in hits] == [h["article_number"] for h in baseline]
    assert hits[0]["retrieval_status"]["rerank"] in {"failed", "timeout"}
    assert hits[0]["retrieval_status"]["degraded"] is True


@pytest.mark.asyncio
async def test_hyde_additional_budget_timeout_fallback(runtime, monkeypatch):

    monkeypatch.setenv("LAW_FULL_HYDE_ENABLED", "true")
    monkeypatch.setenv("LAW_FULL_HYDE_CALL_BUDGET", "0")
    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    hits = await _search(facts)
    assert hits[0]["retrieval_status"]["hyde"] == "budget_exhausted"
    monkeypatch.setenv("LAW_FULL_HYDE_CALL_BUDGET", "1")
    _set_expander(monkeypatch, AsyncMock(side_effect=asyncio.TimeoutError()))
    hits = await _search(facts)
    assert hits[0]["retrieval_status"]["hyde"] == "timeout" and hits[0]["retrieval_status"]["vector"] == "executed"
    _set_expander(monkeypatch, AsyncMock(return_value="刑法危险驾驶醉驾候选正文"))
    hits = await _search(facts)
    assert hits[0]["retrieval_status"]["hyde"] == "executed"
    embeds = [body["input"][0] for path, body in runtime if path == "/api/embed"]
    assert "醉酒驾驶机动车" in embeds and "刑法危险驾驶醉驾候选正文" in embeds
    assert facts == {"behavior_sequence": ["醉酒驾驶机动车"]}


@pytest.mark.asyncio
async def test_digest_manifest_reject_no_upload_fallback(runtime, public_index, monkeypatch):
    monkeypatch.setenv("LAW_FULL_EMBEDDING_DIGEST", "wrong")
    hits = await _search_result({"behavior_sequence": ["醉驾"]}, "alice")
    assert hits.search.dependency_failed
    monkeypatch.setenv("LAW_FULL_EMBEDDING_DIGEST", "fixture-digest")
    path = public_index.parent / "index.manifest.json"
    original = path.read_bytes()
    try:
        changed = json.loads(original)
        changed["corpus_sha256"] = "wrong"
        path.write_text(json.dumps(changed))
        hits = await _search_result({"behavior_sequence": ["醉驾"]}, "alice")
        assert hits.search.dependency_failed
    finally:
        path.write_bytes(original)


def test_qwen_protocol_token_ids_preserve_special_tokens_and_yes_no():
    from app.knowledge.retrieval.config import RerankerConfig
    from app.knowledge.retrieval.rerankers.qwen import PREFIX, SUFFIX, QwenReranker

    class Tokenizer:
        def encode(self, text, **kw):
            if text == "yes":
                return [91]
            if text == "no":
                return [92]
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

    scorer = QwenReranker(RerankerConfig(local_path="unused", max_length=6, instruction="test"))
    encoded = scorer.encode_pairs(Tokenizer(), "query", ["body"])
    assert encoded == [[101, 102, 201, 202, 103, 104]]
    assert scorer.decision_tokens(Tokenizer()) == (91, 92)
    assert "<|im_start|>system" in PREFIX and '"yes" or "no"' in PREFIX
    assert "<think>\n\n</think>" in SUFFIX
    scorer.close()


def test_reranker_retains_query_tail_and_only_truncates_document():
    from app.knowledge.retrieval.config import RerankerConfig
    from app.knowledge.retrieval.rerankers.qwen import PREFIX, SUFFIX, QwenReranker

    class Tokenizer:
        def encode(self, text, **kw):
            if text == PREFIX:
                return [100]
            if text == SUFFIX:
                return [200]
            return [ord(c) for c in text]

    scorer = QwenReranker(RerankerConfig(local_path="unused", max_length=170, instruction="instruction"))
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
        results = await _search_result({"behavior_sequence": ["拿走手机"]}, "alice")
        assert not results.search.candidates and results.search.dependency_failed
    finally:
        path.write_bytes(original)


@pytest.mark.asyncio
async def test_vector_mode_keeps_five_with_keyword_channel_and_hyde_off(runtime, monkeypatch):
    monkeypatch.setenv("LAW_FULL_RETRIEVAL_MODE", "vector")
    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    hits = await _search(facts)
    assert len(hits) == 5
    assert all(h["retrieval_method"] in {"keyword", "vector_index"} for h in hits)
    assert hits[0]["retrieval_status"]["bm25"] == "disabled"
    assert hits[0]["recall_ranks"].get("keyword") == 1
    assert hits[0]["retrieval_status"]["keyword"] == "executed"


@pytest.mark.asyncio
async def test_actual_hyde_deadline_and_session_budget_fallback(runtime, monkeypatch):
    from app.infrastructure.observability.tracing import SessionBudgetExceeded

    monkeypatch.setenv("LAW_FULL_HYDE_ENABLED", "true")
    monkeypatch.setenv("LAW_FULL_HYDE_TIMEOUT_SECONDS", ".01")

    async def slow(query):
        await asyncio.sleep(0.2)
        return "hypothetical"

    _set_expander(monkeypatch, slow)
    hits = await _search({"behavior_sequence": ["醉酒驾驶机动车"]})
    assert hits[0]["retrieval_status"]["hyde"] == "timeout"
    _set_expander(monkeypatch, AsyncMock(side_effect=SessionBudgetExceeded(session_id="test", dimension="calls")))
    hits = await _search({"behavior_sequence": ["醉酒驾驶机动车"]})
    assert hits[0]["retrieval_status"]["hyde"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_timeout_does_not_queue_another_background_inference():
    import threading

    from app.knowledge.retrieval.config import RerankerConfig
    from app.knowledge.retrieval.rerankers import QwenReranker, RerankerBusyError

    scorer = QwenReranker(RerankerConfig(local_path="unused", max_length=512, instruction="instruction"))
    started, release = threading.Event(), threading.Event()

    def blocked(query, docs):
        started.set()
        release.wait(1)
        return [0.5]

    scorer._score_sync = blocked
    try:
        with pytest.raises(asyncio.TimeoutError):
            await scorer.score("query", ["doc"], timeout_s=0.01)
        assert started.is_set()
        with pytest.raises(RerankerBusyError):
            await scorer.score("query", ["doc"], timeout_s=0.01)
    finally:
        release.set()
        scorer.close()


@pytest.mark.asyncio
async def test_bm25_cache_changes_with_public_content_and_tokenizer_version(monkeypatch):
    from app.knowledge.retrieval import tokenization
    from app.knowledge.retrieval.bm25 import BM25Retriever
    from app.knowledge.retrieval.models import RetrievalRequest, RetrievalScope
    from app.knowledge.storage.chroma import ChromaSource, CorpusSnapshot, StoredRow
    class Store:
        source = ChromaSource("public", None, "public")
        fingerprint = "one"
        def read_snapshot(self, *, where):
            return CorpusSnapshot("public", (StoredRow("a", "醉酒驾驶", {}), StoredRow("b", "偷钱", {})), self.fingerprint)
    store = Store()
    retriever = BM25Retriever(store)
    request = RetrievalRequest("醉酒", RetrievalScope("public", "alice", True), 20, 5)
    await retriever.retrieve(request)
    first = next(iter(retriever._cache.values()))
    await retriever.retrieve(request)
    assert next(iter(retriever._cache.values())) is first
    store.fingerprint = "two"
    await retriever.retrieve(request)
    assert len(retriever._cache) == 2
    monkeypatch.setattr(tokenization, "TOKENIZER_VERSION", "changed")
    await retriever.retrieve(request)
    assert len(retriever._cache) == 3


@pytest.mark.asyncio
async def test_vector_registry_uses_shared_fusion_without_source_priority(runtime, monkeypatch):
    from app.knowledge.retrieval.config import RetrievalConfig
    from app.knowledge.retrieval.models import ChannelResult, RetrievalCandidate, StageStatus
    from app.knowledge.retrieval.pipeline import RetrievalPipeline
    from app.knowledge.wiring import build_law_search_service
    corpus = lk.load_criminal_law_data()
    index = _build_article_index(corpus)
    class Vector:
        async def retrieve(self, request):
            candidates = []
            for number in ["第133条之一", "第264条"]:
                article = index[number]
                candidates.append(RetrievalCandidate(number, request.scope.source_id, article["article_number"]+" "+article["content"],
                                  {**article, "content_sha256": article["text_provenance"]["content_sha256"]}))
            return ChannelResult("vector", tuple(candidates), {}, StageStatus("executed"))
    service = build_law_search_service(law_data=corpus, pipeline=RetrievalPipeline(Vector(), config=RetrievalConfig(mode="vector", rerank=False)))
    result = await LegalToolRegistry({}, "alice", corpus, search_service=service)._search_laws("查询")
    assert result["candidates"][0]["article_id"] == "第133条之一"
    assert result["candidates"][0]["source"] == "text_only"
    assert result["candidates"][0]["coverage_eligible"] is False
    assert result["candidates"][1]["source"] == "rag_verified"
    assert result["candidates"][1]["coverage_eligible"] is True
    assert result["retrieval_status"]["bm25"] == "disabled"
    assert "ranking_contract" not in result["retrieval_status"]


@pytest.mark.asyncio
async def test_worker_cancel_busy_recovery_and_background_error_consumed():
    import threading

    from app.knowledge.retrieval.config import RerankerConfig
    from app.knowledge.retrieval.rerankers import QwenReranker, RerankerBusyError

    scorer = QwenReranker(RerankerConfig(local_path="unused", max_length=512, instruction="instruction"))
    started, release = threading.Event(), threading.Event()

    def blocked(query, docs):
        started.set()
        release.wait(1)
        raise RuntimeError("controlled worker failure")

    scorer._score_sync = blocked
    task = asyncio.create_task(scorer.score("query", ["doc"], timeout_s=1))
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
            await scorer.score("query", ["doc"], timeout_s=1)
        release.set()
        for _ in range(50):
            if scorer._future.done():
                break
            await asyncio.sleep(0.002)
        assert scorer._future.done()
        await asyncio.sleep(0.01)
        scorer._score_sync = lambda query, docs: [0.75]
        assert await scorer.score("query", ["doc"], timeout_s=1) == [0.75]
    finally:
        release.set()
        scorer.close()


@pytest.mark.asyncio
async def test_hyde_session_budget_exhaustion_makes_no_http_request(runtime, monkeypatch):
    from app.infrastructure.observability.tracing import SessionBudget, SessionBudgetExceeded
    from app.knowledge.retrieval import hyde as retrieval

    budget = SessionBudget(max_calls=1, max_tokens=100)
    budget.reserve_call("bounded")
    monkeypatch.setattr(retrieval, "session_budget", budget)
    monkeypatch.setattr(retrieval, "current_trace_context", lambda: {"session_id": "bounded"})
    with pytest.raises(SessionBudgetExceeded):
        await _generate_hyde("不会传给 HTTP 的查询")
    assert runtime == []


def test_actual_local_qwen_tokenizer_retains_special_tokens_and_negation():
    import os
    from pathlib import Path

    from transformers import AutoTokenizer

    from app.knowledge.retrieval.config import RerankerConfig
    from app.knowledge.retrieval.rerankers.qwen import PREFIX, SUFFIX, QwenReranker

    path = Path(os.getenv("RERANKER_MODEL_PATH", "data/models/Qwen/Qwen3-Reranker-0.6B"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not (path / "tokenizer.json").is_file():
        pytest.skip("configured local tokenizer not present; no download allowed")
    tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side="left", local_files_only=True)
    scorer = QwenReranker(RerankerConfig(local_path=str(path), max_length=512, instruction="instruction"))
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
    from app.knowledge.retrieval import hyde as retrieval

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
    assert await _generate_hyde("否认拿钱") == "用于额外召回的假设正文"
    assert budget.snapshot("usage") == {"calls": 1, "tokens": 20}
    assert len(requests) == 1 and requests[0]["options"]["num_predict"] == 192
    assert requests[0]["stream"] is False and requests[0]["think"] is False


@pytest.mark.asyncio
async def test_user_identity_does_not_change_public_pool(runtime):
    facts = {"behavior_sequence": ["醉酒驾驶机动车"]}
    alice = (await _search_result(facts, "alice")).as_laws()
    bob = (await _search_result(facts, "bob")).as_laws()
    assert [hit["article_number"] for hit in alice] == [hit["article_number"] for hit in bob]
    assert all(hit["is_public"] is True and not hit.get("user_id") for hit in [*alice, *bob])
