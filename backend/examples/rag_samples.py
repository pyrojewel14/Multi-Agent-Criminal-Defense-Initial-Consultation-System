"""运行 Phase 3 的五条本地 RAG 契约样例。

当前样例把完整在线 RAG 与可离线复现的 JSON 关键词回退层分开记录。
只有实际执行过的层才会写入结果，空 Chroma 或模型失败不会生成虚构 top-k。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.knowledge.law.corpus import load_criminal_law_data
from app.knowledge.law.identifiers import normalize_article_number as _normalize_article_number
from app.knowledge.law.keyword import facts_terms
from app.knowledge.wiring import KnowledgeResources, build_law_search_service

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_FILE = PROJECT_ROOT / "demos" / "rag" / "queries.json"
LIVE_VALIDATION_USER_ID = "phase3-rag-validation-user"


def load_queries() -> list[dict[str, Any]]:
    """加载五条结构化 RAG 查询样例。"""
    queries = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    if len(queries) < 5:
        raise ValueError("Phase 3 至少需要五条 RAG 查询样例")
    return queries


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """压缩 live 结果中的重复字段，保留验收所需证据。"""
    compact_samples = []
    for sample in result["samples"]:
        compact_samples.append(
            {
                "id": sample["id"],
                "query": sample["query"],
                "expected_law": sample["expected_law"],
                "retrieval_mode": sample["retrieval_mode"],
                "hypothetical_document": sample["hypothetical_document"],
                "live_rag_top_k": [
                    {
                        key: item.get(key)
                        for key in (
                            "rank",
                            "article_number",
                            "title",
                            "data_source",
                            "rerank_score",
                            "json_verified",
                            "source",
                            "metadata_user_id",
                            "metadata_is_public",
                        )
                    }
                    for item in sample["live_rag_top_k"]
                ],
                "json_fallback_top_k": [
                    {
                        key: item.get(key)
                        for key in (
                            "rank",
                            "article_number",
                            "title",
                            "relevance_score",
                            "json_verified",
                        )
                    }
                    for item in sample["json_fallback_top_k"]
                ],
                "live_expected_hit": sample["live_expected_hit"],
                "fallback_expected_hit": sample["fallback_expected_hit"],
                "json_verification_passed": sample["json_verification_passed"],
                "verification_method": sample["verification_method"],
                "layers": sample["layers"],
                "failure_reason": sample["failure_reason"],
            }
        )

    return {
        "schema_version": result["schema_version"],
        "retrieval_semantics": result["retrieval_semantics"],
        "historical_strict_comparison_supported": result["historical_strict_comparison_supported"],
        "generated_at": result["generated_at"],
        "run_mode": result["run_mode"],
        "top_k": result["top_k"],
        "dependencies": result["dependencies"],
        "samples": compact_samples,
    }


def _error_result(exc: Exception) -> dict[str, Any]:
    """将依赖异常转换为可记录的结构化结果。"""
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
    }


async def probe_dependencies(probe_live: bool) -> dict[str, Any]:
    """离线只加载受控资产；显式在线探针通过同一法律服务报告实际阶段。"""
    law_data = load_criminal_law_data()
    result = {
        "json_knowledge": {"status": "available", "article_count": sum(len(c["articles"]) for c in law_data["chapters"])},
        "chroma": {"status": "not_probed", "document_count": None},
        "bm25": {"status": "not_probed"}, "hyde": {"status": "not_probed"},
        "embedding": {"status": "not_probed"}, "vector_search": {"status": "not_run"},
    }
    if not probe_live:
        return result
    resources = KnowledgeResources()
    try:
        service = build_law_search_service(law_data=law_data, resources=resources)
        searched = await service.search("故意伤害", user_id=LIVE_VALIDATION_USER_ID)
        status = dict(searched.search.retrieval_status)
        for name in ("bm25", "hyde"):
            result[name] = {"status": status.get(name, "unconfigured")}
        result["chroma"]["status"] = status.get("vector", "unconfigured")
        result["embedding"] = {"status": status.get("vector", "unconfigured")}
        result["vector_search"] = {"status": status.get("vector", "unconfigured"),
                                   "dependency_failed": searched.search.dependency_failed,
                                   "retrieval_status": status}
    except Exception as exc:
        result["vector_search"] = _error_result(exc)
    finally:
        resources.close()
    return result


async def _run_live_rag_sample(query: str, law_data: dict[str, Any], top_k: int,
                               resources: KnowledgeResources) -> dict[str, Any]:
    """调用统一法律服务，保留关键词、向量及核验的独立来源状态。"""
    service = build_law_search_service(law_data=law_data, resources=resources)
    searched = await service.search(query, user_id=LIVE_VALIDATION_USER_ID)
    status = dict(searched.search.retrieval_status)
    results = []
    for law in searched.as_laws()[:top_k]:
        origins = law.get("origins", [])
        metadata = next((origin["metadata"] for origin in origins if origin["channel"] == "vector"), {})
        results.append({
            "rank": law.get("rank"), "article_number": law.get("article_number", ""),
            "title": law.get("title", ""), "data_source": law["data_source"],
            "rerank_score": law.get("rerank_score"), "fusion_score": law.get("fusion_score"),
            "json_verified": law["data_source"] in {"rag_verified", "json_keyword", "text_only"},
            "content_preview": law.get("content", "")[:240], "source": metadata.get("source"),
            "metadata_user_id": metadata.get("user_id"), "metadata_is_public": metadata.get("is_public"),
            "recall_ranks": law.get("recall_ranks", {}),
        })
    return {"hypothetical_document": None, "live_rag_top_k": results,
            "dependency_failed": searched.search.dependency_failed,
            "layers": {"retrieval_status": status,
                       "retriever": {"status": status.get("vector"), "type": "RetrievalPipeline", "uses_ensemble": False},
                       "hyde": {"status": status.get("hyde")}, "reranker": {"status": status.get("rerank")},
                       "json_verification": {"status": "executed", "verified_count": sum(r["json_verified"] for r in results)}}}


async def run_samples(
    *,
    probe_live: bool = False,
    with_reranker: bool = False,
    live_rag: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """运行五条样例，返回实际 JSON 关键词召回与依赖探测结果。"""
    queries = load_queries()
    law_data = load_criminal_law_data()
    dependencies = await probe_dependencies(probe_live or live_rag)
    sample_results = []
    reranker_statuses = []
    resources = KnowledgeResources()
    try:
        offline_service = build_law_search_service(law_data=law_data, resources=resources, env={
            **os.environ, "LAW_FULL_RERANK_ENABLED": str(with_reranker and not live_rag).lower(),
            "LAW_FULL_HYDE_ENABLED": "false"})

        for sample in queries:
            searched = await offline_service.search(sample["query"], user_id=None,
                                                     keyword_terms=facts_terms(sample["facts_structured"]))
            ranked_results = searched.as_laws()
            reranker = {"status": searched.search.retrieval_status.get("rerank", "disabled")}
            reranker_statuses.append(reranker["status"])

            actual_top_k = []
            for rank, law in enumerate(ranked_results[:top_k], 1):
                actual_top_k.append(
                    {
                        "rank": rank,
                        "article_number": law.get("article_number", ""),
                        "title": law.get("title", ""),
                        "data_source": law["data_source"],
                        "relevance_score": law.get("relevance_score"),
                        "rerank_score": law.get("rerank_score"),
                        "matched_tags": law.get("matched_tags", []),
                        "json_verified": True,
                    }
                )

            expected_normalized = _normalize_article_number(sample["expected_article_number"])
            fallback_expected_hit = any(
                _normalize_article_number(item["article_number"]) == expected_normalized for item in actual_top_k
            )

            live_result = {
                "hypothetical_document": None,
                "live_rag_top_k": [],
                "layers": {
                    "status": "not_run",
                    "reason": "离线模式不执行 HyDE、Chroma、BM25 或 reranker",
                },
            }
            live_error = None
            if live_rag:
                try:
                    live_result = await _run_live_rag_sample(
                        sample["query"],
                        law_data,
                        top_k,
                        resources,
                    )
                except Exception as exc:
                    live_error = _error_result(exc)
                    live_result = {
                        "hypothetical_document": None,
                        "live_rag_top_k": [],
                        "layers": {"status": "failed", "error": live_error},
                    }

            live_expected_hit = any(
                _normalize_article_number(item["article_number"]) == expected_normalized
                for item in live_result["live_rag_top_k"]
            )
            live_json_verified = any(
                _normalize_article_number(item["article_number"]) == expected_normalized and item["json_verified"]
                for item in live_result["live_rag_top_k"]
            )

            if live_rag and live_result.get("dependency_failed"):
                failure_reason = "法律索引依赖失败；候选只表示独立受控关键词回退"
            elif live_rag and live_error:
                failure_reason = f"真实检索失败: {live_error['error_type']}"
            elif live_rag and not live_expected_hit:
                failure_reason = "真实 RAG top-k 未命中预期法条"
            elif live_rag and not live_json_verified:
                failure_reason = "真实 RAG 命中预期编号，但未通过本地 JSON 验证"
            elif not live_rag and not fallback_expected_hit:
                failure_reason = "JSON 关键词召回未命中预期法条"
            else:
                failure_reason = None

            sample_results.append(
                {
                    "id": sample["id"],
                    "query": sample["query"],
                    "expected_law": {
                        "article_number": sample["expected_article_number"],
                        "title": sample["expected_law_title"],
                    },
                    "retrieval_mode": "live_rag" if live_rag else "json_keyword_fallback",
                    "hypothetical_document": live_result["hypothetical_document"],
                    "live_rag_top_k": live_result["live_rag_top_k"],
                    "json_fallback_top_k": actual_top_k,
                    "actual_top_k": live_result["live_rag_top_k"] if live_rag else actual_top_k,
                    "live_expected_hit": live_expected_hit,
                    "fallback_expected_hit": fallback_expected_hit,
                    "expected_hit": live_expected_hit if live_rag else fallback_expected_hit,
                    "json_verification_passed": live_json_verified if live_rag else fallback_expected_hit,
                    "verification_method": (
                        "unified_law_service_domain_verification"
                        if live_rag
                        else "candidate_originates_from_local_json_keyword_search"
                    ),
                    "layers": live_result["layers"],
                    "fallback_reranker": reranker,
                    "failure_reason": failure_reason,
                    "live_rag_failure_reason": failure_reason if live_rag else "真实 RAG 未执行",
                }
            )

        live_reranker_statuses = [
            sample.get("layers", {}).get("reranker", {}).get("status", "not_run") for sample in sample_results
        ]
        dependencies["reranker"] = {
            "status": (
                "success_in_live_cases"
                if live_rag and all(status == "executed" for status in live_reranker_statuses)
                else "success"
                if reranker_statuses and all(status == "executed" for status in reranker_statuses)
                else "partial_or_not_run"
            ),
            "live_case_statuses": live_reranker_statuses if live_rag else [],
            "fallback_case_statuses": reranker_statuses,
        }
        if live_rag:
            states = [sample["layers"].get("retrieval_status", {}).get("vector", "failed") for sample in sample_results]
            dependencies["vector_search"] = {
                "status": states[0] if states and len(set(states)) == 1 else "mixed",
                "executed_case_count": sum(state == "executed" for state in states),
                "case_states": states,
                "case_result_counts": [len(sample["live_rag_top_k"]) for sample in sample_results],
            }
            dependencies["live_validation_user_id"] = LIVE_VALIDATION_USER_ID

        return {
            "schema_version": 3,
            "retrieval_semantics": "unified-pipeline-rrf-verified-keyword-v1",
            "historical_strict_comparison_supported": False,
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "run_mode": (
                "live_unified_law_search"
                if live_rag
                else "offline_json_keyword_plus_local_reranker"
                if with_reranker
                else "offline_json_keyword_contract"
            ),
            "top_k": top_k,
            "dependencies": dependencies,
            "samples": sample_results,
        }

    finally:
        resources.close()


def main() -> None:
    """解析命令行参数并输出样例结果。"""
    parser = argparse.ArgumentParser(description="运行 Phase 3 RAG 查询样例")
    parser.add_argument("--probe-live", action="store_true", help="探测本机 HyDE 与 embedding")
    parser.add_argument("--with-reranker", action="store_true", help="运行本地 reranker")
    parser.add_argument("--live-rag", action="store_true", help="运行真实 HyDE、Chroma/BM25、reranker 与 JSON 验证子链")
    parser.add_argument("--compact", action="store_true", help="仅输出验收所需的紧凑字段")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    result = asyncio.run(
        run_samples(
            probe_live=args.probe_live,
            with_reranker=args.with_reranker,
            live_rag=args.live_rag,
            top_k=args.top_k,
        )
    )
    if args.compact:
        result = compact_result(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
