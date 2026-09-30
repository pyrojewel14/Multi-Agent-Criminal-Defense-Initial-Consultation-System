"""Real Chroma consistency, public isolation and manifest regressions."""

import json
import sys
from pathlib import Path

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def test_full_index_excludes_deleted_and_checks_documents(tmp_path):
    from app.knowledge.full_law_corpus import FULL_CORPUS_PATH
    from app.knowledge.full_law_index import build_full_index, verify_full_index

    target = tmp_path / "full"
    manifest = build_full_index(
        FULL_CORPUS_PATH,
        target,
        "full_law",
        "test",
        "test-digest",
        lambda texts: [[1.0, float(len(t))] for t in texts],
        embedding_engine="deterministic-test",
    )
    assert manifest["document_count"] == 504
    assert manifest["embedding_engine"] == "deterministic-test"
    verified = verify_full_index(
        FULL_CORPUS_PATH,
        target,
        "full_law",
        "test",
        "test-digest",
        allow_test_embeddings=True,
    )
    assert verified["corpus_version"] == "2026-09-30.2.integration1"
    collection = chromadb.PersistentClient(path=str(target)).get_collection("full_law")
    assert collection.get(where={"article_id": "第199条"})["ids"] == []
    assert len(collection.get(where={"article_id": "第133条之一"})["ids"]) == 1
    assert collection.get(where={"user_id": "someone"})["ids"] == []
    hit = collection.get(where={"article_id": "第264条"})["ids"][0]
    collection.update(ids=[hit], documents=["tampered"], embeddings=[[1.0, 2.0]])
    with pytest.raises(ValueError, match="documents"):
        verify_full_index(
            FULL_CORPUS_PATH,
            target,
            "full_law",
            "test",
            "test-digest",
            allow_test_embeddings=True,
        )


def test_full_index_rejects_version_model_and_synthetic_runtime(tmp_path):
    from app.knowledge.full_law_corpus import FULL_CORPUS_PATH
    from app.knowledge.full_law_index import build_full_index, verify_full_index

    target = tmp_path / "full"
    build_full_index(
        FULL_CORPUS_PATH,
        target,
        "full_law",
        "test",
        "test-digest",
        lambda texts: [[1.0, 0.0] for t in texts],
        embedding_engine="deterministic-test",
    )
    with pytest.raises(ValueError, match="embedding"):
        verify_full_index(FULL_CORPUS_PATH, target, "full_law", "test", "test-digest")
    with pytest.raises(ValueError, match="model"):
        verify_full_index(
            FULL_CORPUS_PATH,
            target,
            "full_law",
            "other",
            "test-digest",
            allow_test_embeddings=True,
        )
    altered = json.loads(FULL_CORPUS_PATH.read_text())
    altered["metadata"]["dataset_version"] = "new"
    other = tmp_path / "other.json"
    other.write_text(json.dumps(altered, ensure_ascii=False))
    with pytest.raises(ValueError, match="corpus"):
        verify_full_index(
            other, target, "full_law", "test", "test-digest", allow_test_embeddings=True
        )
    with pytest.raises(FileExistsError):
        build_full_index(
            FULL_CORPUS_PATH,
            target,
            "full_law",
            "test",
            "test-digest",
            lambda texts: [],
        )


def test_invalid_embeddings_do_not_publish_index(tmp_path):
    from app.knowledge.full_law_corpus import FULL_CORPUS_PATH
    from app.knowledge.full_law_index import build_full_index

    for bad in [
        lambda texts: [],
        lambda texts: [[float("nan")] for t in texts],
        lambda texts: [[] for t in texts],
    ]:
        with pytest.raises(ValueError, match="embedding"):
            build_full_index(
                FULL_CORPUS_PATH,
                tmp_path / "bad",
                "full_law",
                "test",
                "test-digest",
                bad,
            )
        assert not (tmp_path / "bad").exists()


@pytest.mark.asyncio
async def test_exact_id_uses_chroma_document_not_embedding_recall(
    tmp_path, monkeypatch
):
    import httpx
    from app.knowledge.full_law_corpus import FULL_CORPUS_PATH
    from app.knowledge.full_law_index import build_full_index, search_full_index

    target = tmp_path / "full"
    build_full_index(
        FULL_CORPUS_PATH,
        target,
        "full_law",
        "test",
        "test-digest",
        lambda texts: [[1.0, 0.0] for t in texts],
    )
    monkeypatch.setenv("LAW_FULL_INDEX_DIRECTORY", str(target))
    monkeypatch.setenv("LAW_FULL_INDEX_COLLECTION", "full_law")
    monkeypatch.setenv("LAW_FULL_EMBEDDING_DIGEST", "test-digest")
    monkeypatch.setenv("TEXT_EMBEDDING_MODEL_NAME", "test")
    monkeypatch.setenv("EMBED_MODEL_TYPE", "OLLAMA")

    async def tags(self, url):
        return httpx.Response(
            200,
            json={"models": [{"name": "test", "digest": "test-digest"}]},
            request=httpx.Request("GET", url),
        )

    async def reject_post(*args, **kwargs):
        raise AssertionError("Exact article IDs must not depend on vector similarity")

    monkeypatch.setattr(httpx.AsyncClient, "get", tags)
    monkeypatch.setattr(httpx.AsyncClient, "post", reject_post)
    result = await search_full_index({"behavior_sequence": ["第133条之1"]})
    assert result[0]["article_number"] == "第一百三十三条之一"
    assert "醉酒驾驶机动车" in result[0]["content"]
    assert result[0]["retrieval_method"] == "article_id_index"
    assert await search_full_index({"behavior_sequence": ["第199条"]}) == []
