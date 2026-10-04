"""评分只消费已经返回的候选，不向生产检索传 gold。"""

from evaluation.run_full_retrieval_ablation import score_result, summarize


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
