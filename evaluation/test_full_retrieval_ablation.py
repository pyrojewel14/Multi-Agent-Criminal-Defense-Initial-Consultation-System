"""评分只消费已经返回的候选，不向生产检索传 gold。"""

from evaluation.run_full_retrieval_ablation import score_result, summarize


def test_identity_fingerprints_active_pipeline_and_marks_changed_semantics(tmp_path, monkeypatch):
    import hashlib
    import json
    import transformers
    from evaluation.run_full_retrieval_ablation import ROOT, identity

    directory = tmp_path / "index"
    directory.mkdir()
    (tmp_path / "index.manifest.json").write_text(json.dumps({"embedding_model": "test"}))
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json"):
        (model / name).write_text("{}")
    monkeypatch.setenv("LAW_FULL_INDEX_DIRECTORY", str(directory))
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(model))
    monkeypatch.setenv("LAW_FULL_RERANK_DEVICE", "cpu")

    class Tokenizer:
        def convert_tokens_to_ids(self, word):
            return {"yes": 1, "no": 2}[word]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    result = identity()
    paths = result["retrieval_code_sha256"]
    assert "backend/app/knowledge/retrieval/pipeline.py" in paths
    assert "backend/app/knowledge/law/service.py" in paths
    assert "backend/app/knowledge/wiring.py" in paths
    for path, digest in paths.items():
        assert digest == "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert result["historical_strict_comparison_supported"] is False
    assert result["vector_rag_channels"] == ["vector"]
    assert result["common_domain_channels"] == ["keyword"]


def test_partial_multi_article_recall_and_first_relevant_mrr():
    result = score_result(["第1条", "第234条", "第264条"], ["第234条", "第264条"])
    assert result["recall_at_1"] == 0
    assert result["recall_at_5"] == 1
    assert result["mrr"] == 0.5
    assert score_result(["第234条"], ["第234条", "第264条"])["recall_at_5"] == 0.5


def test_empty_gold_excluded_and_failure_not_disappeared():
    assert score_result(["第1条"], [])["recall_at_5"] is None
    summary = summarize(
        [
            {
                "metrics": score_result(["第264条"], ["第264条"]),
                "latency_ms": 10.0,
                "retrieval_status": {"degraded": False},
            },
            {
                "metrics": score_result([], ["第234条"]),
                "latency_ms": 20.0,
                "retrieval_status": {"degraded": True},
                "error": "TimeoutError",
            },
            {"metrics": score_result([], []), "latency_ms": 30.0, "retrieval_status": {"degraded": False}},
        ]
    )
    assert summary["recall_at_5"] == 0.5
    assert summary["scored_count"] == 2 and summary["failed_count"] == 1
