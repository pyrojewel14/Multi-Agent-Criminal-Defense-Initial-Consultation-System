"""全量公共索引的中文稀疏召回、融合、可选扩展与最终排名。"""

from __future__ import annotations

import asyncio
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.infrastructure.observability.tracing import (
    current_trace_context,
    session_budget,
    trace_span,
    trace_store,
)
from app.knowledge.law_knowledge import _normalize_article_number

TOKENIZER_VERSION = "cjk-char-bigram-v1"


def _flag(name: str, default: bool) -> bool:
    """严格读取开关，错误配置不得静默开启。"""
    value = os.getenv(name, str(default)).lower()
    if value not in {"true", "false", "1", "0", "on", "off"}:
        raise ValueError(f"invalid flag: {name}")
    return value in {"true", "1", "on"}


def _number(name: str, default: float, lower: float, upper: float) -> float:
    """将预算限制在有限范围内，拒绝 NaN 与无限值。"""
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"invalid bounded setting: {name}")
    return value


@dataclass(frozen=True)
class RetrievalConfig:
    """每次请求读取配置，避免开关变更复用旧状态。"""

    mode: str
    recall_k: int
    hyde_k: int
    rerank: bool
    rerank_timeout: float
    hyde: bool
    hyde_timeout: float
    hyde_budget: int

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """读取向量基线或混合模式与可选阶段预算。"""
        mode = os.getenv("LAW_FULL_RETRIEVAL_MODE", "hybrid")
        if mode not in {"vector", "hybrid"}:
            raise ValueError("invalid full retrieval mode")
        return cls(
            mode,
            int(_number("LAW_FULL_RECALL_K", 20, 5, 50)),
            int(_number("LAW_FULL_HYDE_K", 10, 1, 20)),
            _flag("LAW_FULL_RERANK_ENABLED", True),
            _number("LAW_FULL_RERANK_TIMEOUT_SECONDS", 30, 0.01, 120),
            _flag("LAW_FULL_HYDE_ENABLED", False),
            _number("LAW_FULL_HYDE_TIMEOUT_SECONDS", 10, 0.01, 30),
            int(_number("LAW_FULL_HYDE_CALL_BUDGET", 1, 0, 1)),
        )


def chinese_tokens(text: str) -> list[str]:
    """连续中文拆成单字与相邻双字；英文数字按词，不使用案例词表。"""
    tokens = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            tokens.extend(part)
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


@lru_cache(maxsize=2)
def _bm25(corpus_hash: str, tokenizer_version: str, documents: tuple[str, ...]):
    """缓存仅包含已核验公共正文；语料 hash、切分版本或正文变化即失效。"""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([chinese_tokens(document) for document in documents])


def lexical_recall(query: str, rows: list[dict[str, Any]], corpus_hash: str, k: int) -> list[dict[str, Any]]:
    """只返回正分 BM25 候选，避免无匹配查询引入任意条文。"""
    scores = _bm25(corpus_hash, TOKENIZER_VERSION, tuple(r["document"] for r in rows)).get_scores(chinese_tokens(query))
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["id"]))
    return [{**rows[i], "bm25_score": float(scores[i])} for i in order[:k] if scores[i] > 0]


def fuse(rankings: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """按规范条号去重并以等权 RRF 融合，原始分数不跨模型直接相加。"""
    candidates: dict[str, dict[str, Any]] = {}
    for channel, ranked in rankings.items():
        seen = set()
        for rank, law in enumerate(ranked, 1):
            key = _normalize_article_number(law["article_number"])
            if not key or key in seen:
                continue
            seen.add(key)
            candidate = candidates.setdefault(key, {**law, "fusion_score": 0.0, "recall_ranks": {}, "recall_scores": {}})
            candidate["fusion_score"] += 1 / (60 + rank)
            candidate["recall_ranks"][channel] = rank
            for score in ("bm25_score", "vector_distance"):
                if score in law:
                    candidate["recall_scores"][channel] = law[score]
                    candidate["hyde_distance" if channel == "hyde" and score == "vector_distance" else score] = law[score]
    return sorted(
        candidates.values(), key=lambda law: (-law["fusion_score"], _normalize_article_number(law["article_number"]))
    )


@lru_cache(maxsize=1)
def _scorer(path: str, length: int, instruction: str, device: str):
    """复用唯一配置的本地模型，旧配置执行器在自然结束后可回收。"""
    from app.knowledge.full_law_reranker import QwenFullReranker

    return QwenFullReranker(path, length, instruction, device=device)


async def score_candidates(query: str, documents: list[str], timeout: float) -> list[float]:
    """全量路径固定使用本地 Qwen3 官方评分协议。"""
    from app.knowledge.rag.reranker.base import RerankerConfig

    config = RerankerConfig.from_env()
    if config.model_name != "Qwen/Qwen3-Reranker-0.6B":
        raise ValueError("full reranker requires configured Qwen3-Reranker-0.6B")
    return await _scorer(
        config.local_path, config.max_length, config.instruction or "", os.getenv("LAW_FULL_RERANK_DEVICE", "auto")
    ).score(
        query, documents, timeout
    )


async def finalize_candidates(
    query: str, laws: list[dict[str, Any]], status: dict[str, Any], config: RetrievalConfig
) -> list[dict[str, Any]]:
    """统一重排后取 top5；失败或超时保留输入的融合顺序。"""
    if status.get("method") == "article_id_index":
        status["rerank"] = "skipped_exact"
    elif not config.rerank:
        status["rerank"] = "disabled"
    elif not laws:
        status["rerank"] = "skipped_empty"
    else:
        try:
            with trace_span(trace_store, event_type="retrieval", name="full_rerank") as event:
                try:
                    scores = await asyncio.wait_for(
                        score_candidates(query, [law["content"] for law in laws], config.rerank_timeout),
                        config.rerank_timeout,
                    )
                    if len(scores) != len(laws) or any(
                        not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores
                    ):
                        raise ValueError("invalid full reranker scores")
                except Exception:
                    event.outcome = "degraded"
                    raise
            laws = sorted(
                [{**law, "rerank_score": float(score)} for law, score in zip(laws, scores)],
                key=lambda law: -law["rerank_score"],
            )
            status["rerank"] = "executed"
            status["rerank_protocol"] = "qwen3-yes-no-token-ids-v1"
        except asyncio.TimeoutError:
            status.update(rerank="timeout", degraded=True)
        except Exception as exc:
            status.update(
                rerank="busy" if type(exc).__name__ == "RerankerBusyError" else "failed",
                rerank_error=type(exc).__name__,
                degraded=True,
            )
    status["candidate_count"] = len(laws)
    return [{**law, "rank": rank, "retrieval_status": dict(status)} for rank, law in enumerate(laws[:5], 1)]


async def generate_hyde(query: str) -> str:
    """只调用已配置的本地 Ollama；假设正文仅用于额外召回。"""
    import httpx

    context = current_trace_context()
    session_id = str(context["session_id"]) if context and context.get("session_id") else None
    if session_id:
        session_budget.reserve_call(session_id)
    model = os.getenv("OLLAMA_MODEL_NAME", "qwen3.5:0.8b")
    with trace_span(trace_store, event_type="llm", name="full_hyde", model=model, attempt=1) as event:
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            response = await client.post(
                os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/") + "/api/generate",
                json={
                    "model": model,
                    "stream": False,
                    "think": False,
                    "prompt": "为检索生成简短刑法候选正文，不得补造事实、忽略否认或作有罪结论。只输出可能相关的条文用语，不输出解释。检索查询："
                    + query,
                    "options": {"temperature": 0, "num_predict": 192, "seed": 0},
                },
            )
            response.raise_for_status()
            data = response.json()
        if session_id:
            session_budget.record_tokens(
                session_id,
                input_tokens=data.get("prompt_eval_count", "unknown"),
                output_tokens=data.get("eval_count", "unknown"),
            )
        text = data.get("response", "")
        if not isinstance(text, str) or not text.strip():
            event.outcome = "empty"
            raise ValueError("empty HyDE output")
        return text[:1200]
