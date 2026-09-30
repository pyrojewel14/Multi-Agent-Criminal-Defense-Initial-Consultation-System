"""分别记录全量正文的离线工具、实际 RAG 和实际 LawRef 证据。"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
CASES = Path(__file__).with_name("full_cases.jsonl")


async def run_evaluation(
    mode: str,
    output: Path,
    case_ids: list[str] | None = None,
    query_mode: str = "article",
) -> dict:
    """运行固定公共案例；离线模式只调用真实关键词与读取工具。"""
    from app.consultation.agents.legal_research import LegalToolRegistry
    from app.knowledge import law_retrieval
    from app.knowledge.law_knowledge import _build_article_index, load_criminal_law_data

    if mode not in {"offline", "live-rag", "live-lawref"}:
        raise ValueError("未知评测模式")
    os.environ["LAW_KNOWLEDGE_PROFILE"] = "full"
    corpus = load_criminal_law_data()
    demo_enabled = bool(corpus["metadata"].get("demo_annotations"))
    cases = [
        json.loads(line) for line in CASES.read_text().splitlines() if line.strip()
    ]
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
        if set(case_ids) != {c["id"] for c in cases}:
            raise ValueError("未知评测案例")
    index = _build_article_index(corpus)
    rows = []
    for case in cases:
        started = time.monotonic()
        facts = {
            "behavior_sequence": [
                case["semantic_query"] if query_mode == "semantic" else case["query"]
            ],
            "consequence": "",
        }
        expected_coverage = case.get("demo_coverage_eligible", case.get("coverage_eligible", False)) if demo_enabled else case.get("coverage_eligible", False)
        row = {"id": case["id"], "passed": False, "expected_coverage_eligible": expected_coverage}
        try:
            if mode == "live-lawref":
                from app.consultation.agents.fact_digger import _analyze_coverage
                from app.consultation.agents.law_ref import law_ref_node

                state = await law_ref_node(
                    {
                        "session_id": "public-full-eval",
                        "user_id": "public-full-eval",
                        "facts_structured": facts,
                    }
                )
                selected = [
                    a["article_number"] for a in state.get("applied_laws", [])
                ] + [a["article_number"] for a in state.get("law_text_candidates", [])]
                from app.knowledge.law_knowledge import _normalize_article_number

                ids = [_normalize_article_number(n) for n in selected]
                coverage = await _analyze_coverage(facts, state["applied_laws"])
                row.update(
                    {
                        "selected_article_ids": ids,
                        "search_status": state.get("law_search_status"),
                        "termination_reason": state["law_research"][
                            "termination_reason"
                        ],
                        "total_elements": coverage["total_elements"],
                        "degraded": coverage["degraded"],
                        "trajectory": state["law_research"]["trajectory"],
                    }
                )
                row["passed"] = (
                    (case["article_id"] not in ids)
                    if case.get("absent")
                    else (
                        case["article_id"] in ids
                        and (coverage["total_elements"] > 0)
                        == expected_coverage
                    )
                )
            else:
                if mode == "offline":
                    laws = await law_retrieval.search_laws_by_keyword(facts, corpus)
                else:
                    laws = await law_retrieval.search_laws_by_rag(
                        facts, "public-full-eval"
                    )
                    if getattr(laws, "dependency_failed", False):
                        raise RuntimeError("rag dependency failed")
                    laws = law_retrieval._verify_and_enrich_with_json(laws, index)
                registry = LegalToolRegistry(facts, None, corpus)
                from app.knowledge.law_knowledge import _normalize_article_number

                for law in laws:
                    number = _normalize_article_number(law["article_number"])
                    if (
                        law.get("data_source")
                        in {"rag_verified", "json_keyword", "text_only"}
                        and number in index
                    ):
                        registry.searched[number] = {
                            **index[number],
                            "data_source": law["data_source"],
                        }
                ids = list(registry.searched)
                body = (await registry._get_article(case["article_id"]))["article"]
                row.update(
                    {
                        "retrieved_article_ids": ids,
                        "body_readable": bool(body),
                        "required_element_count": len(body["required_elements"])
                        if body
                        else 0,
                    }
                )
                row["passed"] = (
                    not body
                    if case.get("absent")
                    else bool(
                        body
                        and case["contains"] in body["content"]
                        and body["coverage_eligible"] == expected_coverage
                    )
                )
        except Exception as exc:  # noqa: BLE001 — 每例保存失败类别并继续公开评测
            row["error_type"] = type(exc).__name__
        row["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
        rows.append(row)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "passed": row["passed"],
                    "latency_ms": row["latency_ms"],
                }
            )
        )
    result = {
        "evidence_kind": {
            "offline": "offline-deterministic-tools",
            "live-rag": "real-ollama-public-chroma",
            "live-lawref": "real-lawref-llm-rag-coverage",
        }[mode],
        "query_mode": query_mode,
        "corpus_version": corpus["metadata"]["dataset_version"],
        "corpus_sha256": corpus["metadata"]["corpus_sha256"],
        "annotation_mode": "demo" if demo_enabled else "text_only_with_six_regression",
        "demo_annotations": corpus["metadata"].get("demo_annotations"),
        "llm_executed": mode == "live-lawref",
        "rag_executed": mode != "offline",
        "embedding_model": os.getenv("TEXT_EMBEDDING_MODEL_NAME")
        if mode != "offline"
        else None,
        "embedding_digest": os.getenv("LAW_FULL_EMBEDDING_DIGEST")
        if mode != "offline"
        else None,
        "chat_model": os.getenv("OLLAMA_MODEL_NAME") if mode == "live-lawref" else None,
        "legal_review_completed": False,
        "production_deployment_verified": False,
        "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "cases": rows,
        "latency_scope": "本机顺序组件调用，包含核验与模型预热；不是并发或生产性能测试",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["offline", "live-rag", "live-lawref"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--query-mode", choices=["article", "semantic"], default="article"
    )
    args = parser.parse_args()
    os.environ.setdefault("LLM_TYPE", "OLLAMA")
    os.environ.setdefault("EMBED_MODEL_TYPE", "OLLAMA")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    os.environ.setdefault("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
    os.environ.setdefault("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
    result = asyncio.run(
        run_evaluation(args.mode, args.output, args.case, args.query_mode)
    )
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
