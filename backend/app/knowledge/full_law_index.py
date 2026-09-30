"""独立全量公共 Chroma 索引的构建、完整性核验与检索。"""

from __future__ import annotations

import asyncio
import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Callable, cast

from app.knowledge.full_law_corpus import FULL_CORPUS_PATH, sha256, validate_full_corpus
from app.knowledge.law_knowledge import _normalize_article_number, is_article_in_force


def _json_hash(value: Any) -> str:
    """对稳定 JSON 表示计算指纹。"""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def _rows(corpus: dict[str, Any], corpus_hash: str) -> list[dict[str, Any]]:
    """每个有效条文仅建一个完整正文文档，标注不进入 embedding 文本。"""
    meta = corpus["metadata"]
    rows = []
    for chapter in corpus["chapters"]:
        for a in chapter["articles"]:
            if not is_article_in_force(a):
                continue
            number = _normalize_article_number(a["article_number"])
            rows.append(
                {
                    "id": sha256(f"{meta['dataset_version']}:{number}".encode()),
                    "document": f"{a['article_number']} {a['content']}",
                    "metadata": {
                        "article_id": number,
                        "article_number": a["article_number"],
                        "is_public": True,
                        "source": a["text_provenance"]["url"],
                        "official_text_source": a["official_text_source"],
                        "corpus_version": meta["dataset_version"],
                        "corpus_sha256": corpus_hash,
                        "content_sha256": a["text_provenance"]["content_sha256"],
                        "text_review_status": "source_attributed",
                        "legal_review_status": "pending",
                    },
                }
            )
    return sorted(rows, key=lambda row: row["id"])


def _vectors(vectors: list[list[float]], count: int) -> list[list[float]]:
    """检查有限、等维向量并规范为 Chroma 存储的 float32 精度。"""
    if len(vectors) != count or not vectors or not vectors[0]:
        raise ValueError("embedding count/dimension invalid")
    dim = len(vectors[0])
    if any(len(v) != dim or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("embedding vector invalid")
    try:
        result = [[struct.unpack("f", struct.pack("f", x))[0] for x in v] for v in vectors]
    except (OverflowError, struct.error) as exc:
        raise ValueError("embedding float32 overflow") from exc
    if any(not math.isfinite(x) for v in result for x in v):
        raise ValueError("embedding float32 invalid")
    return result


def _load(corpus_path: Path) -> tuple[dict[str, Any], str]:
    """读取公共 JSON 原始字节并核验导入契约。"""
    raw = corpus_path.read_bytes()
    data = json.loads(raw)
    validate_full_corpus(data)
    return data, sha256(raw)


def build_full_index(
    corpus_path: Path,
    index_dir: Path,
    collection: str,
    embedding_model: str,
    embedding_digest: str,
    embed_documents: Callable[[list[str]], list[list[float]]],
    *,
    embedding_engine: str = "ollama",
) -> dict[str, Any]:
    """构建不可覆盖的独立版本目录，只有完整读回核验后才发布 manifest。"""
    import chromadb
    from chromadb.api.types import Embedding, Metadata

    corpus_path, index_dir = Path(corpus_path).resolve(), Path(index_dir).resolve()
    manifest_path = index_dir.parent / f"{index_dir.name}.manifest.json"
    if index_dir.exists() or manifest_path.exists():
        raise FileExistsError(index_dir)
    if not embedding_model or not embedding_digest or embedding_engine not in {"ollama", "deterministic-test"}:
        raise ValueError("embedding identity invalid")
    corpus, corpus_hash = _load(corpus_path)
    rows = _rows(corpus, corpus_hash)
    documents = [r["document"] for r in rows]
    embeddings: list[list[float]] = []
    for start in range(0, len(documents), 32):
        batch = documents[start : start + 32]
        embeddings.extend(_vectors(embed_documents(batch), len(batch)))
    embeddings = _vectors(embeddings, len(rows))
    index_dir.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    target = client.create_collection(collection, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        target.add(
            ids=[r["id"] for r in batch],
            documents=[r["document"] for r in batch],
            metadatas=cast(list[Metadata], [r["metadata"] for r in batch]),
            embeddings=cast(list[Embedding], embeddings[start : start + 32]),
        )
    stored = target.get(ids=[r["id"] for r in rows], include=["documents", "metadatas", "embeddings"])
    if stored["documents"] is None or stored["metadatas"] is None or stored["embeddings"] is None:
        raise ValueError("documents/embedding missing")
    by_id = {identifier: i for i, identifier in enumerate(stored["ids"])}
    if target.count() != len(rows) or len(by_id) != len(rows):
        raise RuntimeError("index count mismatch; incomplete index has no manifest")
    retrieved_vectors = [list(stored["embeddings"][by_id[r["id"]]]) for r in rows]
    for r in rows:
        i = by_id[r["id"]]
        if stored["documents"][i] != r["document"] or stored["metadatas"][i] != r["metadata"]:
            raise RuntimeError("index documents mismatch; incomplete index has no manifest")
    manifest = {
        "schema_version": "full-index-v1",
        "corpus_version": corpus["metadata"]["dataset_version"],
        "corpus_sha256": corpus_hash,
        "official_text_source": corpus["metadata"]["official_text"]["source_id"],
        "document_count": len(rows),
        "excluded_article_ids": ["第199条"],
        "collection": collection,
        "embedding_engine": embedding_engine,
        "embedding_model": embedding_model,
        "embedding_model_digest": embedding_digest,
        "embedding_dimension": len(embeddings[0]),
        "documents_sha256": _json_hash(rows),
        "embeddings_sha256": _json_hash(retrieved_vectors),
        "legal_review_completed": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_full_index(
    corpus_path: Path,
    index_dir: Path,
    collection: str,
    embedding_model: str,
    embedding_digest: str,
    *,
    allow_test_embeddings: bool = False,
) -> dict[str, Any]:
    """读回全部文档、来源、向量与 manifest；任一不一致均拒绝使用。"""
    import chromadb

    corpus, corpus_hash = _load(Path(corpus_path))
    index_dir = Path(index_dir).resolve()
    manifest = json.loads((index_dir.parent / f"{index_dir.name}.manifest.json").read_text())
    if (
        manifest.get("schema_version") != "full-index-v1"
        or manifest.get("corpus_sha256") != corpus_hash
        or manifest.get("corpus_version") != corpus["metadata"]["dataset_version"]
    ):
        raise ValueError("corpus manifest mismatch")
    if manifest.get("embedding_model") != embedding_model or manifest.get("embedding_model_digest") != embedding_digest:
        raise ValueError("embedding model/digest mismatch")
    if manifest.get("embedding_engine") != "ollama" and not (
        allow_test_embeddings and manifest.get("embedding_engine") == "deterministic-test"
    ):
        raise ValueError("test embedding cannot serve runtime")
    if manifest.get("collection") != collection or manifest.get("legal_review_completed") is not False:
        raise ValueError("collection/review manifest mismatch")
    if not index_dir.is_dir():
        raise ValueError("index missing")
    target = chromadb.PersistentClient(path=str(index_dir)).get_collection(collection)
    rows = _rows(corpus, corpus_hash)
    stored = target.get(include=["documents", "metadatas", "embeddings"])
    if stored["documents"] is None or stored["metadatas"] is None or stored["embeddings"] is None:
        raise ValueError("documents/embedding missing")
    by_id = {identifier: i for i, identifier in enumerate(stored["ids"])}
    if (
        set(by_id) != {r["id"] for r in rows}
        or target.count() != len(rows)
        or manifest.get("document_count") != len(rows)
    ):
        raise ValueError("documents count/ids mismatch")
    for r in rows:
        i = by_id[r["id"]]
        if stored["documents"][i] != r["document"] or stored["metadatas"][i] != r["metadata"]:
            raise ValueError("documents/content/source mismatch")
    vectors = [list(stored["embeddings"][by_id[r["id"]]]) for r in rows]
    if _json_hash(rows) != manifest.get("documents_sha256") or _json_hash(vectors) != manifest.get("embeddings_sha256"):
        raise ValueError("documents/embedding hash mismatch")
    if len(_vectors(vectors, len(rows))[0]) != manifest.get("embedding_dimension"):
        raise ValueError("embedding dimension mismatch")
    return manifest


async def search_full_index(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """查询显式配置的公共全量索引，拒绝语料、模型及实际模型 digest 不一致。"""
    import chromadb
    import httpx

    from app.knowledge.law_retrieval import _extract_article_number_from_text, _flatten_fact_terms

    directory = os.getenv("LAW_FULL_INDEX_DIRECTORY")
    digest = os.getenv("LAW_FULL_EMBEDDING_DIGEST")
    if not directory or not digest or os.getenv("EMBED_MODEL_TYPE", "OLLAMA") != "OLLAMA":
        raise ValueError("全量 RAG 需要独立目录、Ollama embedding 与明确 digest")
    model = os.getenv("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
    collection = os.getenv("LAW_FULL_INDEX_COLLECTION", "criminal_law_full")
    corpus_path = Path(os.getenv("LAW_FULL_CORPUS_PATH", str(FULL_CORPUS_PATH)))
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        response = await client.get(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
    if not any(m.get("name") == model and m.get("digest") == digest for m in models):
        raise ValueError("实际 embedding 模型 digest 不一致")
    manifest = await asyncio.to_thread(verify_full_index, corpus_path, Path(directory), collection, model, digest)
    query = " ".join(
        _flatten_fact_terms(facts.get("behavior_sequence", [])) + _flatten_fact_terms(facts.get("consequence", ""))
    )
    target = chromadb.PersistentClient(path=str(Path(directory).resolve())).get_collection(collection)
    exact_id = _normalize_article_number(_extract_article_number_from_text(query))
    if exact_id:
        exact = await asyncio.to_thread(target.get, where={"article_id": exact_id}, include=["documents", "metadatas"])
        if exact["documents"] is None or exact["metadatas"] is None:
            return []
        return [
            {
                "article_number": m["article_number"],
                "content": document,
                "corpus_sha256": m["corpus_sha256"],
                "corpus_version": m["corpus_version"],
                "data_source": "rag_unverified",
                "retrieval_method": "article_id_index",
            }
            for document, m in zip(exact["documents"], exact["metadatas"])
        ]
    async with httpx.AsyncClient(trust_env=False, timeout=90) as client:
        response = await client.post(
            os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/embed",
            json={"model": model, "input": [query], "truncate": False},
        )
        response.raise_for_status()
        vector = response.json()["embeddings"][0]
    vector = _vectors([vector], 1)[0]
    if len(vector) != manifest["embedding_dimension"]:
        raise ValueError("查询 embedding 维度不一致")
    target = chromadb.PersistentClient(path=str(Path(directory).resolve())).get_collection(collection)
    results = await asyncio.to_thread(
        target.query,
        query_embeddings=[vector],
        n_results=5,
        where={"is_public": True},
        include=["documents", "metadatas", "distances"],
    )
    if results["documents"] is None or results["metadatas"] is None:
        raise ValueError("query documents missing")
    return [
        {
            "article_number": m["article_number"],
            "content": document,
            "corpus_sha256": m["corpus_sha256"],
            "corpus_version": m["corpus_version"],
            "data_source": "rag_unverified",
            "retrieval_method": "vector_index",
        }
        for document, m in zip(results["documents"][0], results["metadatas"][0])
    ]
