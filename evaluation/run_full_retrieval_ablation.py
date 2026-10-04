"""同一公共索引与原查询的四组真实检索消融；每条返回后才评分。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
CASES = Path(__file__).with_name("full_retrieval_cases.jsonl")
MODES = {
    "vector": {
        "LAW_FULL_RETRIEVAL_MODE": "vector",
        "LAW_FULL_RERANK_ENABLED": "false",
        "LAW_FULL_HYDE_ENABLED": "false",
    },
    "hybrid": {
        "LAW_FULL_RETRIEVAL_MODE": "hybrid",
        "LAW_FULL_RERANK_ENABLED": "false",
        "LAW_FULL_HYDE_ENABLED": "false",
    },
    "hybrid_rerank": {
        "LAW_FULL_RETRIEVAL_MODE": "hybrid",
        "LAW_FULL_RERANK_ENABLED": "true",
        "LAW_FULL_HYDE_ENABLED": "false",
    },
    "hybrid_rerank_hyde": {
        "LAW_FULL_RETRIEVAL_MODE": "hybrid",
        "LAW_FULL_RERANK_ENABLED": "true",
        "LAW_FULL_HYDE_ENABLED": "true",
    },
}


def score_result(article_ids: list[str], gold: list[str]) -> dict[str, float | None]:
    """仅计算候选相关性；空 gold 单独报告，不混入召回分母。"""
    if not gold:
        return {"recall_at_1": None, "recall_at_5": None, "mrr": None}
    expected = set(gold)
    return {
        "recall_at_1": len(expected & set(article_ids[:1])) / len(expected),
        "recall_at_5": len(expected & set(article_ids[:5])) / len(expected),
        "mrr": next((1 / rank for rank, identifier in enumerate(article_ids[:5], 1) if identifier in expected), 0.0),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """保留失败和降级的总量，同时提供宏平均与完整耗时分布。"""
    scored = [row for row in rows if row["metrics"]["mrr"] is not None]
    states: dict[str, dict[str, int]] = {}
    for row in rows:
        for stage in ("vector", "bm25", "rerank", "hyde"):
            state = row["retrieval_status"].get(stage, "unknown")
            states.setdefault(stage, {})[state] = states.setdefault(stage, {}).get(state, 0) + 1
    latency = sorted(row["latency_ms"] for row in rows)
    return {
        "case_count": len(rows),
        "scored_count": len(scored),
        **{
            name: statistics.mean(row["metrics"][name] for row in scored) if scored else None
            for name in ("recall_at_1", "recall_at_5", "mrr")
        },
        "failed_count": sum(bool(row.get("error")) for row in rows),
        "degraded_count": sum(row["retrieval_status"].get("degraded", False) for row in rows),
        "latency_mean_ms": statistics.mean(latency) if latency else None,
        "latency_p95_ms": latency[min(len(latency) - 1, int(len(latency) * 0.95))] if latency else None,
        "stage_states": states,
    }


@contextmanager
def settings(values: dict[str, str]):
    """仅在本进程临时切换模式，不修改 .env 或索引资产。"""
    original = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _hash_file(path: Path) -> str:
    """按块读取模型文件指纹，公开证据中只保存文件名和 hash。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def identity() -> dict[str, Any]:
    """记录语料、模型和评分协议的身份，排除个人路径和配置秘密。"""
    from app.knowledge.full_law_reranker import INSTRUCTION, PREFIX, SUFFIX, QwenFullReranker
    import torch
    import transformers

    directory = Path(os.environ["LAW_FULL_INDEX_DIRECTORY"])
    manifest = json.loads((directory.parent / f"{directory.name}.manifest.json").read_text())
    path = Path(os.getenv("RERANKER_MODEL_PATH", "./data/models/Qwen/Qwen3-Reranker-0.6B"))
    config_paths = sorted(path.glob("**/config.json"))
    if not (path / "config.json").is_file() and len(config_paths) == 1:
        path = config_paths[0].parent
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side="left", local_files_only=True)
    return {
        "index_manifest": manifest,
        "retrieval_code_sha256": {
            str(path.relative_to(ROOT)): _hash_file(path)
            for path in (
                BACKEND / "app/knowledge/full_law_index.py",
                BACKEND / "app/knowledge/full_law_retrieval.py",
                BACKEND / "app/knowledge/full_law_reranker.py",
                BACKEND / "app/knowledge/law_retrieval.py",
                BACKEND / "app/consultation/agents/legal_research.py",
            )
        },
        "reranker_model": os.getenv("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B"),
        "reranker_files": {p.name: _hash_file(p) for p in sorted(path.glob("*.safetensors"))},
        "reranker_config_sha256": _hash_file(path / "config.json"),
        "reranker_tokenizer_sha256": _hash_file(path / "tokenizer.json"),
        "reranker_template_sha256": "sha256:" + hashlib.sha256((PREFIX + SUFFIX).encode()).hexdigest(),
        "reranker_instruction": os.getenv("RERANKER_INSTRUCTION") or INSTRUCTION,
        "reranker_decision_tokens": {word: tokenizer.convert_tokens_to_ids(word) for word in ("yes", "no")},
        "reranker_max_length": int(os.getenv("RERANKER_MAX_LENGTH", "512")),
        "reranker_device_requested": os.getenv("LAW_FULL_RERANK_DEVICE", "auto"),
        "reranker_device_selected": QwenFullReranker.resolve_device(os.getenv("LAW_FULL_RERANK_DEVICE", "auto")),
        "reranker_dtype": "float32",
        "reranker_batch_size": 2,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "mps_cpu_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "hyde_model": os.getenv("OLLAMA_MODEL_NAME", "qwen3.5:0.8b"),
        "recall_k": int(os.getenv("LAW_FULL_RECALL_K", "20")),
        "rerank_timeout_seconds": float(os.getenv("LAW_FULL_RERANK_TIMEOUT_SECONDS", "30")),
        "hyde_timeout_seconds": float(os.getenv("LAW_FULL_HYDE_TIMEOUT_SECONDS", "10")),
        "hyde_call_budget": int(os.getenv("LAW_FULL_HYDE_CALL_BUDGET", "1")),
        "baseline_contract": "vector top5 + JSON verification + keyword merge + legacy source priority + top5",
    }


async def run(cases: list[dict[str, Any]], modes: list[str], output: Path, case_timeout: float) -> dict[str, Any]:
    """有界顺序调用真实 Registry，每条写盘保留中断前证据。"""
    from app.consultation.agents.legal_research import LegalToolRegistry
    from app.knowledge.law_knowledge import load_criminal_law_data

    report: dict[str, Any] = {
        "schema": "full-retrieval-ablation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path": "LegalToolRegistry._search_laws -> search_laws_by_rag -> search_full_index",
        "scope": "candidate statutory retrieval only; not legal judgment or production acceptance",
        "case_file_sha256": _hash_file(CASES),
        "case_ids": [case["id"] for case in cases],
        "case_timeout_seconds": case_timeout,
        "modes": {},
        "complete": False,
    }

    def save() -> None:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        report["identity"] = identity()
        import httpx

        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            response = await client.get(
                os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/tags"
            )
            response.raise_for_status()
            report["identity"]["actual_model_digests"] = {
                model["name"]: model["digest"]
                for model in response.json().get("models", [])
                if model.get("name")
                in {report["identity"]["hyde_model"], report["identity"]["index_manifest"]["embedding_model"]}
            }
        with settings({"LAW_KNOWLEDGE_PROFILE": "full"}):
            law_data = load_criminal_law_data()
        for mode in modes:
            rows = []
            report["modes"][mode] = {"rows": rows}
            with settings({**MODES[mode], "LAW_KNOWLEDGE_PROFILE": "full"}):
                for case in cases:
                    started = time.monotonic()
                    registry = LegalToolRegistry({}, "retrieval-evaluation", law_data)
                    row: dict[str, Any] = {"id": case["id"], "category": case["category"], "query": case["query"]}
                    try:
                        observation = await asyncio.wait_for(registry._search_laws(case["query"]), timeout=case_timeout)
                        row.update(
                            article_ids=[law["article_id"] for law in observation["candidates"]],
                            candidates=observation["candidates"],
                            retrieval_status=observation["retrieval_status"],
                            rag_dependency_failed=observation["rag_dependency_failed"],
                        )
                        if observation["rag_dependency_failed"]:
                            row["error"] = "RAGDependencyFailure"
                    except Exception as exc:  # noqa: BLE001 — 每例保留失败类别并继续有界评测。
                        row.update(
                            article_ids=[], candidates=[], retrieval_status={"degraded": True}, error=type(exc).__name__
                        )
                    row["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                    # gold 只在真实调用返回后参与评价，未传入 Registry 或查询事实。
                    row["metrics"] = score_result(row["article_ids"], case["gold_article_ids"])
                    if case.get("must_be_empty"):
                        row["boundary_passed"] = not row["article_ids"]
                    rows.append(row)
                    report["modes"][mode]["summary"] = summarize(rows)
                    save()
                    print(
                        json.dumps(
                            {
                                "mode": mode,
                                "id": case["id"],
                                "latency_ms": row["latency_ms"],
                                "status": row["retrieval_status"],
                                "ids": row["article_ids"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if MODES[mode]["LAW_FULL_RERANK_ENABLED"] == "true":
                    from app.knowledge.full_law_retrieval import _scorer
                    from app.knowledge.rag.reranker.base import RerankerConfig

                    config = RerankerConfig.from_env()
                    model = _scorer(
                        config.local_path,
                        config.max_length,
                        config.instruction or "",
                        os.getenv("LAW_FULL_RERANK_DEVICE", "auto"),
                    )._model
                    report["modes"][mode]["reranker_runtime"] = {
                        "loaded": model is not None,
                        "parameter_devices": sorted({str(p.device) for p in model.parameters()}) if model else [],
                        "dtype": str(model.dtype) if model else None,
                    }
                    save()
        report["complete"] = True
    except Exception as exc:  # noqa: BLE001 — 前置条件失败也必须落盘，避免丢失真实失败证据。
        report["preflight_error"] = type(exc).__name__
    finally:
        save()
    return report


def main() -> int:
    """加载现有本地配置运行，不安装、不下载、不构建索引。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--case-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve previous evidence and choose another file")
    if not 0 < args.case_timeout <= 300 or (args.limit is not None and not 1 <= args.limit <= 100):
        parser.error("invalid bounded run budget")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    from dotenv import load_dotenv

    load_dotenv(BACKEND / ".env")
    os.chdir(BACKEND)
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    if args.limit:
        cases = cases[: args.limit]
    report = asyncio.run(run(cases, args.modes, output, args.case_timeout))
    if not report["complete"] or any(mode["summary"]["failed_count"] for mode in report["modes"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
