"""运行 Phase 3 的五条本地 RAG 契约样例。

当前样例把完整在线 RAG 与可离线复现的 JSON 关键词回退层分开记录。
只有实际执行过的层才会写入结果，空 Chroma 或模型失败不会生成虚构 top-k。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.agents.law_ref import (
    _build_article_index,
    _extract_article_number_from_text,
    _normalize_article_number,
    _verify_and_enrich_with_json,
    load_criminal_law_data,
    search_laws_by_keyword,
)


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
        "message": str(exc) or "依赖返回空错误信息",
        "status_code": getattr(exc, "status_code", None),
    }


async def probe_dependencies(probe_live: bool) -> dict[str, Any]:
    """探测本地知识库、Chroma、BM25、HyDE 和 embedding 状态。"""
    from app.rag.vector_store import get_vector_store

    law_data = load_criminal_law_data()
    article_count = sum(len(chapter.get("articles", [])) for chapter in law_data.get("chapters", []))

    vector_store = get_vector_store()
    collection = await asyncio.to_thread(
        vector_store.vectors_store.get,
        include=["documents", "metadatas"],
    )
    collection_count = len(collection.get("documents", []))
    collection_metadata = collection.get("metadatas", [])
    bm25 = await vector_store.hybrid_retriever.get_bm25_retriever(
        user_id="phase3-rag-samples",
        include_public=True,
    )

    dependencies: dict[str, Any] = {
        "json_knowledge": {"status": "available", "article_count": article_count},
        "chroma": {
            "status": "available" if collection_count else "available_empty",
            "collection_name": "rag_collection",
            "document_count": collection_count,
            "owner_counts": dict(Counter(item.get("user_id") for item in collection_metadata)),
            "public_counts": {
                str(key).lower(): value
                for key, value in Counter(item.get("is_public") for item in collection_metadata).items()
            },
            "source_counts": dict(
                Counter(item.get("original_filename") for item in collection_metadata)
            ),
        },
        "bm25": {
            "status": "available" if bm25 is not None else "unavailable_empty_corpus",
        },
        "hyde": {"status": "not_probed"},
        "embedding": {"status": "not_probed"},
        "vector_search": {
            "status": "not_run",
            "reason": (
                "等待 live 样例执行"
                if collection_count
                else "当前 Chroma collection 为空，不能产生真实向量 top-k"
            ),
        },
    }

    if not probe_live:
        dependencies["hyde"]["reason"] = "离线模式不调用本机 LLM"
        dependencies["embedding"]["reason"] = "离线模式不调用本机 embedding"
        return dependencies

    probe_query = "故意伤害他人身体"

    try:
        from app.rag.rag_service import RagService

        service = RagService(user_id="phase3-rag-samples", include_public=True)
        hypothetical = await service.generate_hypothetical_document(probe_query)
        dependencies["hyde"] = {
            "status": "fallback_to_query" if hypothetical == probe_query else "success",
            "returned_original_query": hypothetical == probe_query,
            "model": getattr(service.chat_model, "model", None),
            "output_length": len(hypothetical),
        }
    except Exception as exc:
        dependencies["hyde"] = _error_result(exc)

    try:
        from app.utils.factory import embed_model

        vector = await asyncio.to_thread(embed_model.embed_query, probe_query)
        dependencies["embedding"] = {
            "status": "success",
            "model": getattr(embed_model, "model", None),
            "dimensions": len(vector),
        }
    except Exception as exc:
        dependencies["embedding"] = _error_result(exc)

    return dependencies


async def _run_live_rag_sample(
    query: str,
    law_data: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """按真实 RAG 子链运行单条查询并记录各层结果。"""
    from app.rag.rag_service import RagService, _deduplicate_documents
    from app.rag.reorder_service import reorder_service

    service = RagService(user_id=LIVE_VALIDATION_USER_ID, include_public=True)
    hypothetical = await service.generate_hypothetical_document(query)
    retriever = await service.vector_store.get_retriever(
        hypothetical,
        LIVE_VALIDATION_USER_ID,
        True,
    )

    documents = await retriever.ainvoke(hypothetical)
    deduplicated = _deduplicate_documents(documents)
    content_to_metadata = {doc.page_content: doc.metadata for doc in deduplicated}
    contents = list(content_to_metadata)
    rerank_result = await reorder_service.reorder_documents(query, contents)

    if rerank_result.get("success"):
        ranked_items = rerank_result.get("documents", [])
        reranker = {
            "status": "success",
            "model": reorder_service.config.model_name,
        }
    else:
        ranked_items = [
            {"document": content, "similarity": None}
            for content in contents
        ]
        reranker = {
            "status": "fallback_to_retrieval_order",
            "error": rerank_result.get("error", "未知 rerank 错误"),
        }

    rag_candidates = []
    ranked_metadata = []
    for item in ranked_items[:top_k]:
        content = item.get("document", "")
        rag_candidates.append(
            {
                "article_number": _extract_article_number_from_text(content),
                "title": "",
                "content": content,
                "elements": [],
                "base_sentence": "",
                "charge_tags": [],
                "common_keywords": [],
                "chapter": "",
                "relevance_score": 1.0,
                "matched_tags": ["RAG 真实检索"],
                "data_source": "rag",
            }
        )
        ranked_metadata.append(
            {
                "rerank_score": (
                    round(float(item["similarity"]), 6)
                    if item.get("similarity") is not None
                    else None
                ),
                "metadata": content_to_metadata.get(content, {}),
            }
        )

    verified = _verify_and_enrich_with_json(
        rag_candidates,
        _build_article_index(law_data),
    )
    top_results = []
    for rank, (law, trace) in enumerate(zip(verified, ranked_metadata), 1):
        metadata = trace["metadata"]
        top_results.append(
            {
                "rank": rank,
                "article_number": law.get("article_number", ""),
                "title": law.get("title", ""),
                "data_source": law.get("data_source", "rag_unverified"),
                "rerank_score": trace["rerank_score"],
                "json_verified": law.get("data_source") == "rag_verified",
                "content_preview": law.get("content", "")[:240],
                "source": metadata.get("original_filename", metadata.get("source", "")),
                "metadata_user_id": metadata.get("user_id"),
                "metadata_is_public": metadata.get("is_public"),
            }
        )

    return {
        "hypothetical_document": hypothetical,
        "live_rag_top_k": top_results,
        "layers": {
            "hyde": {
                "status": "success" if hypothetical != query else "fallback_to_query",
                "model": getattr(service.chat_model, "model", None),
                "output_length": len(hypothetical),
            },
            "retriever": {
                "status": "success",
                "type": type(retriever).__name__,
                "uses_ensemble": type(retriever).__name__ == "EnsembleRetriever",
                "raw_count": len(documents),
            },
            "deduplicate": {
                "status": "success",
                "before": len(documents),
                "after": len(deduplicated),
            },
            "reranker": reranker,
            "json_verification": {
                "status": "success",
                "verified_count": sum(item["json_verified"] for item in top_results),
                "unverified_count": sum(not item["json_verified"] for item in top_results),
            },
        },
    }


def _candidate_text(law: dict[str, Any]) -> str:
    """构造本地 reranker 使用的法条文本。"""
    return " ".join(
        part
        for part in (
            law.get("article_number", ""),
            law.get("title", ""),
            law.get("content", ""),
        )
        if part
    )


async def _rank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    with_reranker: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按需运行本地 reranker，并保留失败时的原始关键词顺序。"""
    if not candidates:
        return [], {"status": "not_run_empty_candidates"}

    if not with_reranker:
        return candidates, {"status": "not_run"}

    from app.rag.reorder_service import reorder_service

    document_map = {_candidate_text(law): law for law in candidates}
    result = await reorder_service.reorder_documents(query, list(document_map))
    if not result.get("success"):
        return candidates, {
            "status": "fallback_to_keyword_order",
            "error": result.get("error", "未知 rerank 错误"),
        }

    ranked = []
    for item in result.get("documents", []):
        law = dict(document_map[item["document"]])
        law["rerank_score"] = round(float(item.get("similarity", 0.0)), 6)
        ranked.append(law)
    return ranked, {"status": "success", "model": "Qwen/Qwen3-Reranker-0.6B"}


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

    for sample in queries:
        keyword_results = await search_laws_by_keyword(sample["facts_structured"], law_data)
        ranked_results, reranker = await _rank_candidates(
            sample["query"],
            keyword_results[:top_k],
            with_reranker and not live_rag,
        )
        reranker_statuses.append(reranker["status"])

        actual_top_k = []
        for rank, law in enumerate(ranked_results[:top_k], 1):
            actual_top_k.append(
                {
                    "rank": rank,
                    "article_number": law.get("article_number", ""),
                    "title": law.get("title", ""),
                    "data_source": "json_keyword",
                    "relevance_score": law.get("relevance_score"),
                    "rerank_score": law.get("rerank_score"),
                    "matched_tags": law.get("matched_tags", []),
                    "json_verified": True,
                }
            )

        expected_normalized = _normalize_article_number(sample["expected_article_number"])
        fallback_expected_hit = any(
            _normalize_article_number(item["article_number"]) == expected_normalized
            for item in actual_top_k
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
            _normalize_article_number(item["article_number"]) == expected_normalized
            and item["json_verified"]
            for item in live_result["live_rag_top_k"]
        )

        if live_rag and live_error:
            failure_reason = f"真实 RAG 子链失败: {live_error['error_type']}: {live_error['message']}"
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
                    "live_rag_candidate_verified_by__verify_and_enrich_with_json"
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
        sample.get("layers", {}).get("reranker", {}).get("status", "not_run")
        for sample in sample_results
    ]
    dependencies["reranker"] = {
        "status": (
            "success_in_live_cases"
            if live_rag and all(status == "success" for status in live_reranker_statuses)
            else "success"
            if reranker_statuses and all(status == "success" for status in reranker_statuses)
            else "partial_or_not_run"
        ),
        "live_case_statuses": live_reranker_statuses if live_rag else [],
        "fallback_case_statuses": reranker_statuses,
    }
    if live_rag:
        dependencies["vector_search"] = {
            "status": "success" if any(sample["live_rag_top_k"] for sample in sample_results) else "failed",
            "case_result_counts": [len(sample["live_rag_top_k"]) for sample in sample_results],
        }
        dependencies["live_validation_user_id"] = LIVE_VALIDATION_USER_ID

    return {
        "schema_version": 2,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "run_mode": (
            "live_hyde_chroma_bm25_reranker_json_verification"
            if live_rag
            else "offline_json_keyword_plus_local_reranker"
            if with_reranker
            else "offline_json_keyword_contract"
        ),
        "top_k": top_k,
        "dependencies": dependencies,
        "samples": sample_results,
    }


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
