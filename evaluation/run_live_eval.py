#!/usr/bin/env python3
"""独立的真实模型链评估入口；前置条件不满足时不生成 live 结果。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

CASES = Path(__file__).with_name("live_cases.jsonl")
OUTPUT_DIR = Path(__file__).with_name("live_results")
LAW = BACKEND / "data/law_knowledge/criminal_law_chapters.json"
_PUBLIC_CASE_IDS = {"ordinary_theft", "ordinary_injury", "missing_facts"}
_PUBLIC_MODEL_NAMES = {"qwen3.5:0.8b", "qwen3-embedding:0.6b"}
_FACT_FIELDS = {
    "incident_time", "incident_location", "parties", "behavior_sequence", "consequence",
    "evidence_mentioned", "arrest_status", "surrender", "victim_forgiveness", "prior_record",
}
_ROUTE_REASONS = {
    "consent_given", "consent_missing", "alert_triggered", "facts_accepted",
    "dependency_degraded", "max_loop_reached", "coverage_sufficient", "coverage_insufficient",
    "risk_artifact_degraded", "risk_artifact_valid", "lawyer_revise_facts",
    "lawyer_revise_risk", "lawyer_approved", "lawyer_decision_missing",
}
_LAW_NUMBER = re.compile(r"^第(?:[0-9]{1,4}|[一二三四五六七八九十百千零〇]{1,8})条(?:之(?:[0-9]{1,2}|[一二三四五六七八九十]{1,4}))?$")
_ERROR_TYPES = {
    "RuntimeError", "TimeoutError", "ValueError", "KeyError", "TypeError",
    "ConnectionError", "OSError", "LLMServiceException", "LLMTimeoutException",
    "SessionBudgetExceeded",
}


def _safe_choice(value: object, allowed: set[str]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _safe_case_id(value: object) -> str:
    """保留固定公开 ID；其他值只保存稳定指纹。"""
    if isinstance(value, str) and value in _PUBLIC_CASE_IDS:
        return value
    return f"sha256:{hashlib.sha256(str(value).encode('utf-8')).hexdigest()}"


def _safe_model_name(value: str) -> str:
    """仅公开固定模型标识；自定义模型用指纹和服务端 digest 区分。"""
    if value in _PUBLIC_MODEL_NAMES:
        return value
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _safe_failure(exc: Exception, stage: str) -> dict[str, Any]:
    """仅保存稳定错误类别和阶段，不写入异常消息或上下文。"""
    from app.errors.codes import ErrorCode

    error_type = type(exc).__name__
    code = getattr(exc, "code", None)
    return {
        "status": "error",
        "error_type": error_type if error_type in _ERROR_TYPES else "UnknownError",
        "error_code": code.value if isinstance(code, ErrorCode) else "unclassified",
        "stage": stage,
    }


def _safe_artifact_results(value: object) -> dict[str, dict[str, Any]]:
    """仅导出受控枚举与错误数量，排除验证消息和模型原文。"""
    if not isinstance(value, dict):
        return {}
    safe: dict[str, dict[str, Any]] = {}
    for name in ("fact", "law", "risk", "service"):
        result = value.get(name)
        if not isinstance(result, dict):
            continue
        errors = result.get("validation_errors")
        safe[name] = {
            "status": _safe_choice(result.get("status"), {"success", "degraded", "human_review"}),
            "source": _safe_choice(result.get("source"), {"tool_call", "content_json", "deterministic_fallback"}),
            "degraded_reason": _safe_choice(result.get("degraded_reason"), {"schema_validation_failed"}),
            "validation_error_count": len(errors) if isinstance(errors, list) else 0,
        }
    return safe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def _ollama_tags(base_url: str) -> dict[str, Any]:
    # ProxyHandler({}) 保证 loopback 请求不经过用户环境代理。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url.rstrip("/") + "/api/tags", timeout=5) as response:
        return json.load(response)


def preflight(
    *, base_url: str, model: str, embedding_model: str, index_dir: Path,
    reranker_path: Path, law_path: Path, collection: str = "rag_collection",
    fetch_tags: Callable[[str], dict[str, Any]] = _ollama_tags,
) -> dict[str, Any]:
    """核验真实运行的模型和索引条件，不创建空 Chroma 库。"""
    issues: list[str] = []
    parsed = urlparse(base_url)
    host = parsed.hostname
    display_host = f"[{host}]" if host and ":" in host else host
    try:
        port = parsed.port
    except ValueError:
        port = None
        issues.append("ollama_url_invalid_port")
    safe_base_url = f"http://{display_host}:{port or 11434}" if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"} else None
    if not safe_base_url:
        issues.append("non_loopback_ollama")
        tags: dict[str, Any] = {}
    elif parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        issues.append("ollama_url_invalid_components")
        tags = {}
    else:
        try:
            tags = fetch_tags(base_url)
        except (OSError, ValueError, TimeoutError):
            tags = {}
            issues.append("ollama_unavailable")
    models = {item.get("name"): item.get("digest") for item in tags.get("models", []) if isinstance(item, dict)}
    if model not in models:
        issues.append("model_missing")
    if embedding_model not in models:
        issues.append("embedding_model_missing")
    law_version: str | None = None
    if not law_path.is_file():
        issues.append("law_snapshot_missing")
    else:
        try:
            law_data = json.loads(law_path.read_text(encoding="utf-8"))
            law_version = law_data["metadata"]["dataset_version"]
            if not isinstance(law_version, str) or not law_version:
                raise ValueError("dataset_version missing")
        except (OSError, ValueError, KeyError, TypeError):
            law_version = None
            issues.append("law_snapshot_metadata_invalid")
    index_count: int | None = None
    if not (index_dir / "chroma.sqlite3").is_file():
        issues.append("index_missing")
    else:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(index_dir))
            index_count = client.get_collection(collection).count()
            if index_count == 0:
                issues.append("index_empty")
        except Exception:
            issues.append("index_invalid")
    if not (reranker_path / "config.json").is_file():
        issues.append("reranker_missing")
    elif not any(
        path.is_file() and path.suffix in {".safetensors", ".bin"} and path.stat().st_size > 0
        for path in reranker_path.rglob("*")
    ):
        issues.append("reranker_weights_missing")
    build_manifest: dict[str, Any] | None = None
    manifest_path = index_dir.parent / f"{index_dir.name}.manifest.json"
    if index_count and manifest_path.is_file():
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                candidate["collection"] != collection
                or candidate["document_count"] != index_count
                or candidate["snapshot_sha256"] != _sha256(law_path)
                or candidate["embedding_model"] != embedding_model
                or candidate["embedding_model_digest"] != models.get(embedding_model)
                or not all(
                    isinstance(candidate[key], str)
                    and re.fullmatch(r"sha256:[a-f0-9]{64}", candidate[key])
                    for key in ("documents_sha256", "embeddings_sha256")
                )
            ):
                raise ValueError("build manifest mismatch")
            build_manifest = candidate
        except (OSError, ValueError, KeyError, TypeError):
            issues.append("index_manifest_invalid")
    return {
        "ready": not issues,
        "issues": issues,
        "backend": "OLLAMA",
        "base_url": safe_base_url,
        "model": _safe_model_name(model),
        "model_digest": models.get(model),
        "embedding_model": _safe_model_name(embedding_model),
        "embedding_digest": models.get(embedding_model),
        "law_dataset_version": law_version,
        "law_snapshot_sha256": _sha256(law_path) if law_path.is_file() else None,
        "index_collection": collection,
        "index_document_count": index_count,
        "index_sha256": _tree_hash(index_dir) if index_count else None,
        "index_hash_scope": "mutable_chroma_directory_snapshot" if index_count else None,
        "index_build_manifest_sha256": _sha256(manifest_path) if build_manifest else None,
        "index_build_documents_sha256": build_manifest["documents_sha256"] if build_manifest else None,
        "index_build_embeddings_sha256": build_manifest["embeddings_sha256"] if build_manifest else None,
        "reranker_sha256": _tree_hash(reranker_path) if (reranker_path / "config.json").is_file() else None,
    }


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """按 trace 事件计数实际调用尝试和路由。"""
    ranked = []
    for event in events:
        if event.get("event_type") != "rag_ranked_result":
            continue
        rank = event.get("attempt")
        content = event.get("metadata", {}).get("content", {})
        digest = content.get("sha256") if isinstance(content, dict) else None
        origin = event.get("metadata", {}).get("origin", {})
        origin_digest = origin.get("sha256") if isinstance(origin, dict) else None
        if isinstance(rank, int) and rank > 0 and isinstance(digest, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
            ranked.append({
                "rank": rank,
                "document_id": digest,
                "source": "rag_returned_content",
                "source_id": origin_digest if isinstance(origin_digest, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", origin_digest) else None,
            })
    observed = any(event.get("event_type") == "rag_ranked_result_set" for event in events)
    return {
        "llm_attempts": sum(event.get("event_type") == "llm" for event in events),
        "rag_calls": sum(event.get("event_type") == "rag" for event in events),
        "route_reasons": [reason for event in events if (reason := _safe_choice(event.get("route_reason"), _ROUTE_REASONS))],
        "retrieval_top_k_status": "observed" if observed else "unavailable",
        "retrieval_top_k": sorted(ranked, key=lambda item: item["rank"]),
    }


async def run_cases(
    cases: list[dict[str, str]], execute: Callable[[dict[str, str]], Awaitable[dict[str, Any]]],
    output: Path, *, metadata: dict[str, Any] | None = None, mode: str = "contract-test",
) -> dict[str, Any]:
    """逐例落盘，异常样例也保留在原位并继续执行。"""
    result: dict[str, Any] = {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "case_count": len(cases),
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in cases:
        started = time.monotonic()
        try:
            row = await execute(case)
        except Exception as exc:
            row = _safe_failure(exc, "execute_case")
        result["cases"].append({"id": _safe_case_id(case["id"]), **row, "duration_ms": round((time.monotonic() - started) * 1000, 3)})
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["aggregate"] = {
        "completed": len(result["cases"]),
        "errors": sum(row["status"] == "error" for row in result["cases"]),
        "degraded": sum(row.get("workflow_status") == "degraded" for row in result["cases"]),
        "human_review": sum(row["status"] == "human_review" for row in result["cases"]),
        "human_alert": sum(row["status"] == "human_alert" for row in result["cases"]),
        "llm_attempts": sum(row.get("llm_attempts", 0) for row in result["cases"]),
        "rag_calls": sum(row.get("rag_calls", 0) for row in result["cases"]),
        "latency_ms_total": round(sum(row["duration_ms"] for row in result["cases"]), 3),
    }
    result["aggregate"]["degraded_rate"] = result["aggregate"]["degraded"] / len(cases) if cases else None
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _load_cases() -> list[dict[str, str]]:
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("live case ID 必须唯一且集合非空")
    if any(set(case) != {"id", "input"} or not case["input"].strip() for case in cases):
        raise ValueError("live case 仅允许非空 id/input")
    return cases


def _metadata(pre: dict[str, Any], cases: list[dict[str, str]]) -> dict[str, Any]:
    from app.utils.llm_gateway import LLMCallPolicy

    policy = LLMCallPolicy.from_env()
    prompt_dir = BACKEND / "app/prompts"
    return {
        **pre,
        "prompt_hashes": {path.name: _sha256(path) for path in sorted(prompt_dir.glob("*.txt"))},
        "prompt_registry_sha256": _sha256(BACKEND / "app/config/prompt.yaml"),
        "cases_sha256": _sha256(CASES),
        "case_count": len(cases),
        "index_collection": os.environ.get("CHROMA_COLLECTION_NAME", "rag_collection"),
        "checkpoint": "MemorySaver process-local",
        "llm_policy": {
            "total_timeout_seconds": policy.total_timeout_seconds,
            "attempt_timeout_seconds": policy.attempt_timeout_seconds,
            "max_attempts": policy.max_attempts,
            "backoff_seconds": policy.backoff_seconds,
        },
    }


async def _execute_chain(case: dict[str, str]) -> dict[str, Any]:
    from app.observability.tracing import trace_store
    from app.orchestrator.workflow import ConsultationOrchestrator

    session_id = f"live-eval-{uuid.uuid4().hex}"
    state = {
        "session_id": session_id, "consultation_id": session_id,
        "user_id": "live-eval-public", "user_role": "client", "user_type": "suspect",
        "consent_given": True, "facts_raw": [], "current_input": None,
        "conversation_history": [{"agent": "Receptionist", "case_city": "上海"}],
        "fact_law_loop_count": 0, "fact_law_failure_streak": 0,
    }
    orchestrator = ConsultationOrchestrator()
    try:
        stage = "start_workflow"
        await orchestrator.start_workflow(state)
        stage = "resume_workflow"
        final = await orchestrator.resume_workflow(session_id, {"current_input": case["input"]})
        stage = "get_next_node"
        next_node = await orchestrator.get_next_node(session_id)
    except Exception as exc:
        events = [event.to_dict() for event in trace_store.events() if event.session_id == session_id]
        return {**_safe_failure(exc, stage), **summarize_events(events)}
    events = [event.to_dict() for event in trace_store.events() if event.session_id == session_id]
    summary = summarize_events(events)
    if final.get("alert_triggered"):
        status = "human_alert"
    elif final.get("awaiting_lawyer_review") or final.get("workflow_status") == "degraded":
        status = "human_review"
    elif next_node == "fact_intake":
        status = "wait_for_user"
    else:
        status = _safe_choice(next_node, {"receptionist", "fact_intake", "law_ref", "fact_digger", "risk_assessor", "service_planner", "human_review", "human_alert", "wait_for_user"}) or "end"
    return {
        "status": status,
        "workflow_status": _safe_choice(final.get("workflow_status"), {"degraded", "closed", "completed", "repair_required"}),
        "law_search_status": _safe_choice(final.get("law_search_status"), {"success", "missing_facts", "no_law_match", "dependency_failure"}),
        "facts_fields": sorted(key for key in (final.get("facts_structured") or {}) if key in _FACT_FIELDS),
        "laws": [{"article_number": number if isinstance(number, str) and _LAW_NUMBER.fullmatch(number) else None,
                  "data_source": _safe_choice(law.get("data_source"), {"rag_unverified", "llm_extracted", "rag_verified", "json_keyword"})}
                 for law in final.get("applied_laws", []) if isinstance(law, dict)
                 for number in [law.get("article_number")]],
        "coverage": final.get("facts_coverage_rate") if isinstance(final.get("facts_coverage_rate"), (int, float)) and not isinstance(final.get("facts_coverage_rate"), bool) else None,
        "awaiting_lawyer_review": final.get("awaiting_lawyer_review") if isinstance(final.get("awaiting_lawyer_review"), bool) else None,
        "alert_triggered": final.get("alert_triggered") if isinstance(final.get("alert_triggered"), bool) else None,
        "artifact_results": _safe_artifact_results(final.get("artifact_results")),
        **summary,
    }


async def _run_ablation(cases: list[dict[str, str]], output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    from app.agents.receptionist import _confirm_identity
    from app.observability.tracing import trace_span, trace_store
    from app.security.disclaimer import DISCLAIMER_PREFIX, disclaimer

    async def execute(case: dict[str, str]) -> dict[str, Any]:
        template_start = time.monotonic()
        template = disclaimer.inject("请选择您的身份类型：当事人、被害人或家属。")
        template_ms = round((time.monotonic() - template_start) * 1000, 3)
        session_id = f"reception-ablation-{uuid.uuid4().hex}"
        state = {"session_id": session_id, "facts_raw": [case["input"]], "consent_given": True}
        llm_start = time.monotonic()
        failure: Exception | None = None
        response: dict[str, Any] = {}
        try:
            with trace_span(trace_store, event_type="ablation", name="receptionist_identity", session_id=session_id):
                response = await _confirm_identity(state)
        except Exception as exc:
            failure = exc
        llm_ms = round((time.monotonic() - llm_start) * 1000, 3)
        events = [event.to_dict() for event in trace_store.events() if event.session_id == session_id]
        if failure is not None:
            return {
                **_safe_failure(failure, "receptionist_llm"), "template_latency_ms": template_ms,
                "llm_latency_ms": llm_ms, "template_chars": len(template),
                "template_disclaimer": DISCLAIMER_PREFIX in template,
                "template_llm_attempts": 0,
                "llm_attempts": summarize_events(events)["llm_attempts"],
            }
        return {
            "status": "success", "template_latency_ms": template_ms,
            "llm_latency_ms": llm_ms, "template_chars": len(template),
            "llm_chars": len(response["final_output"]),
            "template_disclaimer": DISCLAIMER_PREFIX in template,
            "llm_disclaimer": DISCLAIMER_PREFIX in response["final_output"],
            "template_llm_attempts": 0, "llm_attempts": summarize_events(events)["llm_attempts"],
            "llm_output_sha256": f"sha256:{hashlib.sha256(response['final_output'].encode()).hexdigest()}",
        }

    return await run_cases(
        cases, execute, output,
        metadata={**metadata, "ablation": "Receptionist identity prompt: template vs production LLM"},
        mode="component-ablation",
    )


def configure_runtime() -> dict[str, Path]:
    """固定评测进程的后端与绝对资源路径，保证预检和生产链一致。"""
    os.environ["LLM_TYPE"] = "OLLAMA"
    os.environ["EMBED_MODEL_TYPE"] = "OLLAMA"
    os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    os.environ.setdefault("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
    os.environ.setdefault("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
    os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", str(BACKEND / "data/chromadb"))
    os.environ.setdefault("RERANKER_MODEL_PATH", str(BACKEND / "data/models/Qwen/Qwen3-Reranker-0.6B"))
    index_dir = Path(os.environ["CHROMA_PERSIST_DIRECTORY"]).resolve()
    reranker_path = Path(os.environ["RERANKER_MODEL_PATH"]).resolve()
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(index_dir)
    os.environ["RERANKER_MODEL_PATH"] = str(reranker_path)
    return {"index_dir": index_dir, "reranker_path": reranker_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "run", "ablation"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = configure_runtime()
    base_url = os.environ["OLLAMA_BASE_URL"]
    pre = preflight(
        base_url=base_url, model=os.environ["OLLAMA_MODEL_NAME"],
        embedding_model=os.environ["TEXT_EMBEDDING_MODEL_NAME"],
        index_dir=paths["index_dir"],
        reranker_path=paths["reranker_path"],
        law_path=LAW,
        collection=os.environ.get("CHROMA_COLLECTION_NAME", "rag_collection"),
    )
    cases = _load_cases()
    metadata = _metadata(pre, cases)
    if args.mode == "preflight":
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0 if pre["ready"] else 2
    required = [] if args.mode == "run" else [issue for issue in pre["issues"] if issue in {"ollama_unavailable", "model_missing", "non_loopback_ollama"}]
    if args.mode == "run" and not pre["ready"]:
        required = pre["issues"]
    if required:
        print(json.dumps({"status": "blocked", "issues": required, "metadata": metadata}, ensure_ascii=False, indent=2))
        return 2
    output = args.output or OUTPUT_DIR / ("chain.json" if args.mode == "run" else "receptionist_ablation.json")
    if args.mode == "run":
        result = asyncio.run(run_cases(cases, _execute_chain, output, metadata=metadata, mode="live-chain"))
    else:
        result = asyncio.run(_run_ablation(cases, output, metadata))
    print(json.dumps({"output": str(output), "aggregate": result["aggregate"]}, ensure_ascii=False))
    return 0 if result["aggregate"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
