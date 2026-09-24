"""The isolated live index must expose only the six public snapshot articles."""

import json
import os
from pathlib import Path
import subprocess
import sys

import chromadb
import pytest

from evaluation.build_live_index import build_index


SNAPSHOT = Path(__file__).resolve().parents[1] / "backend/data/law_knowledge/criminal_law_chapters.json"


def test_build_index_is_public_and_records_provenance(tmp_path):
    index = tmp_path / "eval-index"
    manifest = build_index(
        SNAPSHOT, index, "rag_collection", "qwen3-embedding:0.6b",
        "test-model-digest", lambda texts: [[float(i), 1.0] for i in range(len(texts))],
    )
    collection = chromadb.PersistentClient(path=str(index)).get_collection("rag_collection")
    visible = collection.get(where={"is_public": True}, include=["documents", "metadatas"])
    assert collection.count() == 6
    assert len(visible["ids"]) == 6
    assert {item["article_number"] for item in visible["metadatas"]} == {
        "第二百三十二条", "第二百三十四条", "第二百六十三条",
        "第二百六十四条", "第二百六十六条", "第二百九十三条",
    }
    assert all(item["official_text_source"] == "criminal-law-consolidated-amendment-12" for item in visible["metadatas"])
    assert all(item["annotation_source"] == "project-maintained-v1" for item in visible["metadatas"])
    assert all(
        item["source"] == f"criminal-law-consolidated-amendment-12:{item['article_number']}"
        for item in visible["metadatas"]
    )
    assert all("第二百" in text for text in visible["documents"])
    public_query = collection.query(query_embeddings=[[0.0, 1.0]], n_results=3, where={"is_public": True})
    assert len(public_query["ids"][0]) == 3
    private_query = collection.query(query_embeddings=[[0.0, 1.0]], n_results=3, where={"user_id": "other"})
    assert private_query["ids"] == [[]]
    assert manifest["snapshot_version"] == "2026-09-21.1"
    assert manifest["document_count"] == 6
    assert manifest["embedding_model_digest"] == "test-model-digest"
    assert json.loads((index.parent / "eval-index.manifest.json").read_text()) == manifest


def test_build_index_rejects_existing_target_without_changing_it(tmp_path):
    index = tmp_path / "existing"
    index.mkdir()
    marker = index / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        build_index(SNAPSHOT, index, "rag_collection", "m", "d", lambda texts: [])
    assert marker.read_text() == "keep"


def test_build_index_rejects_non_public_or_incomplete_snapshot(tmp_path):
    data = json.loads(SNAPSHOT.read_text())
    data["chapters"][0]["articles"][0]["official_text_source"] = "untrusted"
    snapshot = tmp_path / "bad.json"
    snapshot.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source"):
        build_index(snapshot, tmp_path / "index", "rag_collection", "m", "d", lambda texts: [])
    assert not (tmp_path / "index").exists()


def test_direct_script_entrypoint_resolves_sibling_runner(tmp_path):
    env = {**os.environ, "OLLAMA_BASE_URL": "http://127.0.0.1:1"}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("build_live_index.py")),
         "--index-dir", str(tmp_path / "index")],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stderr
    assert "URLError" in result.stderr
