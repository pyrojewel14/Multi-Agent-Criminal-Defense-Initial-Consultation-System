"""Build an isolated Chroma index from the six-article public law snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Callable

import chromadb


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "backend/data/law_knowledge/criminal_law_chapters.json"
EXPECTED_ARTICLES = {
    "第二百三十二条", "第二百三十四条", "第二百六十三条",
    "第二百六十四条", "第二百六十六条", "第二百九十三条",
}


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def build_index(
    snapshot_path: Path,
    index_dir: Path,
    collection: str,
    embedding_model: str,
    embedding_digest: str,
    embed_documents: Callable[[list[str]], list[list[float]]],
) -> dict:
    """Validate provenance first; never overwrite an existing index directory."""
    snapshot_path = Path(snapshot_path).resolve()
    index_dir = Path(index_dir).resolve()
    if index_dir.exists():
        raise FileExistsError(index_dir)
    raw = snapshot_path.read_bytes()
    data = json.loads(raw)
    metadata = data["metadata"]
    source = metadata["official_text"]["source_id"]
    annotation = metadata["annotation_provenance"]["annotation_id"]
    if source != "criminal-law-consolidated-amendment-12" or annotation != "project-maintained-v1":
        raise ValueError("snapshot source mismatch")
    articles = [article for chapter in data["chapters"] for article in chapter["articles"]]
    if len(articles) != 6 or {article["article_number"] for article in articles} != EXPECTED_ARTICLES:
        raise ValueError("snapshot article coverage mismatch")
    for article in articles:
        if article.get("official_text_source") != source or article.get("annotation_source") != annotation:
            raise ValueError("article source mismatch")
        if not isinstance(article.get("content"), str) or not article["content"].strip():
            raise ValueError("article content missing")
    articles.sort(key=lambda article: article["article_number"])
    documents = [f"{article['article_number']} {article['content']}" for article in articles]
    embeddings = embed_documents(documents)
    if len(embeddings) != len(documents):
        raise ValueError("embedding count mismatch")
    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1 or not dimensions.pop() or any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for vector in embeddings for value in vector
    ):
        raise ValueError("embedding vector invalid")
    ids = [_hash_bytes(article["article_number"].encode("utf-8")) for article in articles]
    chroma_metadata = [
        {
            "article_number": article["article_number"],
            "is_public": True,
            "official_text_source": source,
            "annotation_source": annotation,
            "source": f"{source}:{article['article_number']}",
            "snapshot_version": metadata["dataset_version"],
        }
        for article in articles
    ]
    index_dir.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    target = client.create_collection(collection)
    target.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=chroma_metadata)
    if target.count() != 6 or not target.query(
        query_embeddings=[embeddings[0]], n_results=1, where={"is_public": True}
    )["ids"][0]:
        raise RuntimeError("public index query failed")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(snapshot_path)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0
    manifest = {
        "snapshot_path": str(snapshot_path.relative_to(ROOT)) if snapshot_path.is_relative_to(ROOT) else str(snapshot_path),
        "snapshot_sha256": _hash_bytes(raw),
        "snapshot_git_tracked": tracked,
        "snapshot_version": metadata["dataset_version"],
        "official_text_source": source,
        "annotation_source": annotation,
        "document_count": 6,
        "collection": collection,
        "embedding_model": embedding_model,
        "embedding_model_digest": embedding_digest,
        "embedding_dimension": len(embeddings[0]),
        "documents_sha256": _hash_bytes(json.dumps(documents, ensure_ascii=False, separators=(",", ":")).encode()),
        "embeddings_sha256": _hash_bytes(json.dumps(embeddings, separators=(",", ":")).encode()),
        "build_command": "PYTHONNOUSERSITE=1 conda run -n Agent_dev python evaluation/build_live_index.py --index-dir <isolated-path>",
    }
    manifest_path = index_dir.parent / f"{index_dir.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--collection", default="rag_collection")
    args = parser.parse_args()
    from run_live_eval import _ollama_tags, configure_runtime

    configure_runtime()
    import os

    model = os.environ["TEXT_EMBEDDING_MODEL_NAME"]
    tags = _ollama_tags(os.environ["OLLAMA_BASE_URL"])
    digests = {item.get("name"): item.get("digest") for item in tags.get("models", [])}
    if not digests.get(model):
        raise RuntimeError("embedding model unavailable")
    from app.utils.factory import embed_model

    manifest = build_index(
        SNAPSHOT, args.index_dir, args.collection, model, digests[model],
        embed_model.embed_documents,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
