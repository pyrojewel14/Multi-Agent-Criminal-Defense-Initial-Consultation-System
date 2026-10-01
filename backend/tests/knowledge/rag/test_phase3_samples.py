import json
from pathlib import Path

import pytest

from app.knowledge.law_knowledge import _build_article_index, _normalize_article_number
from app.knowledge.law_retrieval import _extract_article_number_from_text
from examples.rag_samples import load_queries, run_samples

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_phase3_query_file_has_five_structured_cases():
    queries = load_queries()

    assert len(queries) >= 5
    for case in queries:
        assert case["query"]
        assert case["expected_article_number"]
        assert case["expected_law_title"]
        assert case["facts_structured"]["behavior_sequence"]


@pytest.mark.asyncio
async def test_offline_json_samples_return_real_local_results(monkeypatch):
    monkeypatch.setenv("LAW_KNOWLEDGE_PROFILE", "snapshot")
    result = await run_samples(probe_live=False, with_reranker=False)

    assert result["run_mode"] == "offline_json_keyword_contract"
    assert len(result["samples"]) >= 5
    assert all(sample["live_rag_top_k"] == [] for sample in result["samples"])
    samples = {sample["id"]: sample for sample in result["samples"]}
    for case_id, article_number in {
        "intentional_injury": "第二百三十四条",
        "theft": "第二百六十四条",
        "fraud": "第二百六十六条",
    }.items():
        sample = samples[case_id]
        assert [item["article_number"] for item in sample["actual_top_k"]] == [article_number]
        assert sample["expected_hit"] is True
        assert sample["json_verification_passed"] is True

    for case_id in ("unlawful_detention", "dangerous_driving"):
        sample = samples[case_id]
        assert sample["actual_top_k"] == []
        assert sample["expected_hit"] is False
        assert sample["json_verification_passed"] is False


def test_subarticle_number_does_not_collide_with_base_article():
    assert _normalize_article_number("第一百三十三条") == "第133条"
    assert _normalize_article_number("第一百三十三条之一") == "第133条之一"
    assert _extract_article_number_from_text("依据第一百三十三条之一处理") == "第一百三十三条之一"

    article_index = _build_article_index(
        {
            "chapters": [
                {
                    "chapter": "测试章节",
                    "articles": [
                        {"article_number": "第一百三十三条", "title": "交通肇事罪"},
                        {"article_number": "第一百三十三条之一", "title": "危险驾驶罪"},
                    ],
                }
            ],
        }
    )
    assert article_index["第133条"]["title"] == "交通肇事罪"
    assert article_index["第133条之一"]["title"] == "危险驾驶罪"


def test_recorded_results_match_query_contract_when_present():
    result_path = PROJECT_ROOT / "demos" / "rag" / "results" / "2026-07-13.json"
    if not result_path.exists():
        pytest.skip("正式运行结果尚未生成")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["samples"]) >= 5
    assert result["dependencies"]["chroma"]["document_count"] == 0
    assert all("failure_reason" in sample for sample in result["samples"])


def test_recorded_live_results_keep_sources_and_layers_separate():
    result_path = PROJECT_ROOT / "demos" / "rag" / "results" / "2026-07-14-ollama.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["schema_version"] == 2
    assert result["run_mode"] == "live_hyde_chroma_bm25_reranker_json_verification"
    assert result["dependencies"]["chroma"]["document_count"] == 608
    assert result["dependencies"]["reranker"]["live_case_statuses"] == ["success"] * 5
    assert len(result["samples"]) == 5

    for sample in result["samples"]:
        assert sample["live_rag_top_k"]
        assert sample["json_fallback_top_k"]
        assert sample["live_expected_hit"] is True
        assert sample["json_verification_passed"] is True
        assert sample["layers"]["retriever"]["uses_ensemble"] is True
        assert sample["layers"]["reranker"]["status"] == "success"
        assert sample["failure_reason"] is None
        assert all(item["metadata_is_public"] is True for item in sample["live_rag_top_k"])
