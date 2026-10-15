"""领域缓存不持有模型，公开获取接口在并发和清理后保持一致。"""

import threading
from concurrent.futures import ThreadPoolExecutor

from app.knowledge.wiring import KnowledgeResources


def test_domain_resources_do_not_own_model_caches_or_factories():
    resources = KnowledgeResources()
    assert not hasattr(resources, "_scorers")
    assert not hasattr(resources, "_document_embeddings")
    assert not hasattr(resources, "scorer")
    assert not hasattr(resources, "document_embeddings")


def test_public_document_service_getter_is_atomic_and_clearable():
    resources = KnowledgeResources()
    barrier = threading.Barrier(6)
    created = []

    def factory():
        service = object()
        created.append(service)
        return service

    def acquire(_):
        barrier.wait(timeout=5)
        return resources.document_service(factory)

    with ThreadPoolExecutor(max_workers=6) as pool:
        services = list(pool.map(acquire, range(6)))
    assert len(created) == 1
    assert all(service is created[0] for service in services)
    resources.close()
    assert resources.document_service(factory) is not services[0]
    assert len(created) == 2


def test_bm25_resource_cache_evicts_oldest_identity_at_eight_entries():
    resources = KnowledgeResources()
    store = object()
    first = resources.bm25(store, identity=("first",))
    second = resources.bm25(store, identity=("second",))
    for number in range(7):
        resources.bm25(store, identity=(number,))
    assert resources.bm25(store, identity=("second",)) is second
    assert resources.bm25(store, identity=("first",)) is not first
    resources.close()


def test_unsupported_law_configuration_is_domain_state_without_model_creation():
    from app.knowledge.wiring import build_law_search_service

    class Manager:
        def get_reranker(self, config, **kwargs):
            raise AssertionError("unsupported law model must not be created")

    resources = KnowledgeResources()
    service = build_law_search_service(
        resources=resources,
        model_manager=Manager(),
        env={"RERANKER_MODEL_NAME": "other/model", "LAW_FULL_RERANK_ENABLED": "true"},
    )
    state = resources.law_reranker_readiness()
    assert state["status"] == "error" and state["error"] == "ValueError"
    assert service.pipeline.reranker is not None
    resources.close()
    assert resources.law_reranker_readiness() is None


def test_domain_cleanup_does_not_close_borrowed_model_manager(isolated_model_manager):
    from app.infrastructure.llm.config import RerankerConfig

    resources = KnowledgeResources()
    scorer = isolated_model_manager.get_reranker(RerankerConfig())
    resources.close()
    assert isolated_model_manager.get_reranker(RerankerConfig()) is scorer
    assert scorer.readiness()["closed"] is False
