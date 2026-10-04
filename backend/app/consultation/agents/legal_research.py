"""LawRef 节点内部的有界法律检索工具循环。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.errors.exceptions import LLMTimeoutException
from app.infrastructure.llm.gateway import llm_gateway
from app.infrastructure.observability.tracing import (
    SessionBudgetExceeded,
    current_trace_context,
    session_budget,
    trace_span,
    trace_store,
)
from app.knowledge import law_retrieval
from app.knowledge.law_knowledge import (
    _build_article_index,
    _element_name,
    _normalize_article_number,
    is_article_in_force,
    is_lawref_eligible,
)
from app.security.sensitive_filter import mask_pii


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchLawsArgs(_StrictArgs):
    """限制模型检索词的长度与类型。"""

    query: str = Field(min_length=1, max_length=200)


class ArticleArgs(_StrictArgs):
    """要求模型提供明确的快照条文编号。"""

    article_id: str = Field(min_length=1, max_length=40)


class FinalAnswer(_StrictArgs):
    """模型的最终选择；可信来源与要件仍由程序复核。"""

    article_ids: list[str] = Field(min_length=1, max_length=5)
    matched_elements: dict[str, list[str]]
    confidence: Literal["high", "medium", "low"]


class AgentStep(BaseModel):
    """不含案件原文的单次决策和工具结果摘要。"""

    step: int
    model_decision: Literal["tool_call", "final_answer", "invalid"]
    tool_name: str | None = None
    tool_arguments_fingerprint: str | None = None
    tool_status: str | None = None
    tool_result_summary: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    token_usage: dict[str, int | str] = Field(default_factory=lambda: {"total_tokens": "unknown"})


class LawResearchResult(BaseModel):
    """供 LawRef 适配的候选与可导出的非敏感执行统计。"""

    candidate_laws: list[dict[str, Any]] = Field(default_factory=list)
    matched_elements: dict[str, list[str]] = Field(default_factory=dict)
    missing_elements: dict[str, list[str]] = Field(default_factory=dict)
    confidence: str = "insufficient"
    evidence_status: str = "insufficient"
    trajectory: list[AgentStep] = Field(default_factory=list)
    tool_call_count: int = 0
    successful_tool_call_count: int = 0
    failed_tool_call_count: int = 0
    duplicate_tool_call_count: int = 0
    step_count: int = 0
    termination_reason: str = "max_steps"
    latency_ms: float = 0.0
    token_usage: dict[str, int | str] = Field(default_factory=dict)

    def audit_summary(self) -> dict[str, Any]:
        """只导出轨迹和计数，不把法条正文或案件事实写入 checkpoint。"""
        return self.model_dump(exclude={"candidate_laws", "matched_elements", "missing_elements"}, mode="json")


def _fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """用规范化参数识别重复调用，轨迹中不保存参数值。"""
    normalized = {
        key: re.sub(r"\s+", " ", value.strip()) if isinstance(value, str) else value for key, value in arguments.items()
    }
    payload = json.dumps([name, normalized], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _article_fingerprint(article: dict[str, Any], law_data: dict[str, Any]) -> str:
    metadata = law_data.get("metadata", {})
    payload = [
        metadata.get("dataset_id"),
        metadata.get("dataset_version"),
        article.get("article_number"),
        article.get("content"),
    ]
    return f"sha256:{hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode('utf-8')).hexdigest()}"


class LegalToolRegistry:
    """仅开放三种固定工具；模型返回的名称与参数均不直接执行。"""

    def __init__(self, facts: dict[str, Any], user_id: str | None, law_data: dict[str, Any]):
        self.facts = facts
        self.user_id = user_id
        self.law_data = law_data
        self.searched: dict[str, dict[str, Any]] = {}
        self.observed: set[str] = set()
        self.rag_dependency_failed = False
        self._handlers = {
            "search_laws": (SearchLawsArgs, self._search_laws),
            "get_article": (ArticleArgs, self._get_article),
            "search_elements": (ArticleArgs, self._search_elements),
        }
        self.model_tools = [
            StructuredTool.from_function(
                coroutine=self._search_laws,
                name="search_laws",
                description="根据简短法律关键词检索候选法条，返回编号、来源、分数和指纹。",
                args_schema=SearchLawsArgs,
            ),
            StructuredTool.from_function(
                coroutine=self._get_article,
                name="get_article",
                description="按已检索到的条文编号读取验证快照中的法条正文及来源。",
                args_schema=ArticleArgs,
            ),
            StructuredTool.from_function(
                coroutine=self._search_elements,
                name="search_elements",
                description="按已检索到的条文编号读取项目维护的构成要件。",
                args_schema=ArticleArgs,
            ),
        ]

    async def _search_laws(self, query: str) -> dict[str, Any]:
        """复用原有 RAG、快照验证和关键词召回，不重复实现索引。"""
        query_facts = {"behavior_sequence": [query.strip()], "consequence": ""}
        full_profile = os.getenv("LAW_KNOWLEDGE_PROFILE", "full") == "full"
        if full_profile:
            rag_results = await law_retrieval.search_laws_by_rag(query_facts, self.user_id, candidate_pool=True)
        else:
            rag_results = await law_retrieval.search_laws_by_rag(query_facts, self.user_id)
        self.rag_dependency_failed = bool(getattr(rag_results, "dependency_failed", False))
        index = _build_article_index(self.law_data)
        verified_rag = law_retrieval._verify_and_enrich_with_json(rag_results, index)
        keyword = await law_retrieval.search_laws_by_keyword(query_facts, self.law_data)
        merged = law_retrieval._merge_and_deduplicate(verified_rag, keyword)
        retrieval_status = dict(getattr(rag_results, "retrieval_status", {}))
        if full_profile and retrieval_status.get("method") in {"hybrid_index", "vector_index", "article_id_index"}:
            from app.knowledge.full_law_retrieval import RetrievalConfig, finalize_candidates

            config = RetrievalConfig.from_env()
            # 精确条号不扩展到其他候选；自然语言合并后只重排一次。
            if retrieval_status["method"] == "article_id_index":
                merged = verified_rag
            elif retrieval_status["method"] == "hybrid_index":
                keyword = keyword[:config.recall_k]
                merged = law_retrieval._merge_and_deduplicate(verified_rag, keyword)
                retrieval_status["keyword_recall_count"] = len(keyword)
                by_id = {
                    _normalize_article_number(law["article_number"]): {
                        **law,
                        "recall_ranks": dict(law.get("recall_ranks", {})),
                        "recall_scores": dict(law.get("recall_scores", {})),
                    }
                    for law in merged
                }
                for rank, law in enumerate(keyword, 1):
                    candidate = by_id[_normalize_article_number(law["article_number"])]
                    candidate["fusion_score"] = candidate.get("fusion_score", 0) + 1 / (60 + rank)
                    candidate["recall_ranks"]["keyword"] = rank
                    candidate["recall_scores"]["keyword"] = law.get("relevance_score")
                merged = sorted(by_id.values(), key=lambda law: -law.get("fusion_score", 0))
            elif not config.rerank:
                # 显式 vector 消融保留历史工具排序，避免改写对照基线。
                merged.sort(key=lambda law: law.get("data_source") not in {"rag_verified", "json_keyword"})
                retrieval_status["ranking_contract"] = "legacy_source_priority"
            merged = await finalize_candidates(query.strip(), merged, retrieval_status, config)
        else:
            merged.sort(key=lambda law: law.get("data_source") not in {"rag_verified", "json_keyword"})
            merged = merged[:5]
        candidates = []
        for law in merged:
            article_id = _normalize_article_number(law.get("article_number", ""))
            if not article_id:
                continue
            source = law.get("data_source", "rag_unverified")
            if (
                article_id in index
                and is_article_in_force(index[article_id])
                and (
                    (is_lawref_eligible(index[article_id]) and source in {"rag_verified", "json_keyword"})
                    or (index[article_id].get("text_provenance") and source == "text_only")
                )
            ):
                self.searched[article_id] = {**law, **index[article_id], "data_source": source}
            candidates.append(
                {
                    "article_id": article_id,
                    "title": law.get("title") or law.get("display_title", ""),
                    "score": law.get("rerank_score", law.get("fusion_score", law.get("relevance_score"))),
                    "rank": law.get("rank"),
                    "fusion_score": law.get("fusion_score"),
                    "rerank_score": law.get("rerank_score"),
                    "recall_ranks": law.get("recall_ranks", {}),
                    "recall_scores": law.get("recall_scores", {}),
                    "retrieval_status": law.get("retrieval_status", retrieval_status),
                    "article_status": index.get(article_id, {}).get("status"),
                    "coverage_eligible": article_id in self.searched and is_lawref_eligible(index[article_id]),
                    "provenance": {
                        key: law.get(key) or index.get(article_id, {}).get(key)
                        for key in (
                            "corpus_version",
                            "corpus_sha256",
                            "content_sha256",
                            "official_text_source",
                            "text_provenance",
                            "source",
                            "text_review_status",
                            "legal_review_status",
                        )
                    },
                    "source": source,
                    "retrieval_method": law.get("retrieval_method")
                    or ("rag" if source.startswith("rag_") else "keyword"),
                    "fingerprint": _article_fingerprint(index[article_id], self.law_data)
                    if article_id in index
                    else _fingerprint("unverified_article", {"article_id": article_id}),
                }
            )
        return {
            "candidates": candidates,
            "rag_dependency_failed": self.rag_dependency_failed,
            "retrieval_status": retrieval_status,
        }

    def _verified_article(self, article_id: str) -> tuple[str, dict[str, Any] | None]:
        normalized = _normalize_article_number(article_id.strip())
        return normalized, self.searched.get(normalized)

    async def _get_article(self, article_id: str) -> dict[str, Any]:
        """仅返回先前搜索且经快照核验的条文。"""
        normalized, article = self._verified_article(article_id)
        if article is None:
            return {"article": None}
        self.observed.add(normalized)
        return {
            "article": {
                "article_id": normalized,
                "title": article.get("title") or article.get("display_title", ""),
                "content": article.get("content", ""),
                "source": article.get("data_source", ""),
                "required_elements": article.get("elements", []) if is_lawref_eligible(article) else [],
                "coverage_eligible": is_lawref_eligible(article),
                "annotation_usage": article.get("annotation_usage", "project_regression"),
                "text_provenance": article.get("text_provenance", {}),
                "annotation_status": {k: v["review_status"] for k, v in article.get("annotations", {}).items()},
                "text_structure": article.get("text_structure", {}),
                "fingerprint": _article_fingerprint(article, self.law_data),
            }
        }

    async def _search_elements(self, article_id: str) -> dict[str, Any]:
        """从同一验证快照读取要件，并保留项目标注来源。"""
        normalized, article = self._verified_article(article_id)
        if article is None:
            return {"article_id": normalized, "required_elements": []}
        self.observed.add(normalized)
        return {
            "article_id": normalized,
            "required_elements": article.get("elements", []) if is_lawref_eligible(article) else [],
            "review_status": article.get("annotations", {}).get("elements", {}).get("review_status", "project_regression"),
            "annotation_source": article.get("annotation_source", ""),
        }

    async def execute(self, name: str, arguments: Any, timeout: float) -> tuple[str, dict[str, Any], str | None]:
        """校验工具名与参数，在 deadline 内返回结构化 observation。"""
        if name not in self._handlers:
            with trace_span(trace_store, event_type="law_tool", name="rejected") as event:
                event.outcome = "unknown_tool"
            return "unknown_tool", {}, None
        schema, handler = self._handlers[name]
        try:
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
            validated = schema.model_validate(arguments)
        except (ValidationError, ValueError):
            with trace_span(trace_store, event_type="law_tool", name=name) as event:
                event.outcome = "invalid_arguments"
            return "invalid_arguments", {}, None
        args = validated.model_dump()
        fingerprint = _fingerprint(name, args)
        with trace_span(
            trace_store, event_type="law_tool", name=name, metadata={"arguments_fingerprint": fingerprint}
        ) as event:
            try:
                result = await asyncio.wait_for(handler(**args), timeout=timeout)
                if name == "search_laws" and result["rag_dependency_failed"] and not result["candidates"]:
                    event.outcome = "dependency_failure"
                    return "dependency_failure", result, fingerprint
                if name == "search_laws" and result["rag_dependency_failed"]:
                    event.outcome = "partial_dependency_failure"
                    return "partial_dependency_failure", result, fingerprint
                empty = not (result.get("candidates") or result.get("article") or result.get("required_elements"))
                event.outcome = "empty" if empty else "success"
                return event.outcome, result, fingerprint
            except (asyncio.TimeoutError, TimeoutError):
                event.outcome = "timeout"
                return "timeout", {}, fingerprint
            except SessionBudgetExceeded:
                event.outcome = "budget_exceeded"
                return "budget_exceeded", {}, fingerprint
            except Exception:
                event.outcome = "dependency_failure"
                return "dependency_failure", {}, fingerprint


_SYSTEM_PROMPT = """你是 LawRef 节点内部受限的法律检索决策器。每轮只调用一个已提供工具，或输出一个严格 JSON final_answer。
先 search_laws，再用 get_article 或 search_elements 核验候选与构成要件，最后只选择已核验的法条。
最终 JSON 格式：{"article_ids":["第264条"],"matched_elements":{"第264条":[]},"confidence":"medium"}；示例不代表案件事实。
matched_elements 只表示本案明确肯定叙述的事实支持；未提供、否认或不确定的条件留作缺失，不得虚构证据。
只有条号、罪名或咨询构成要件的提问不是行为事实；法条正文和 required_elements 也不是案件证据，不能默认全部匹配。
只选择已读取的可信条文；匹配时原样复制 get_article 或 search_elements 返回的 required_elements 中完整 name，不能缩写、截取或自行概括。
coverage_eligible=false 的正文仅供阅读，matched_elements 必须为空，不得依据正文自行生成覆盖要件。"""


_FINAL_PROMPT = """依据案件事实与已经检索读取的条文输出最终 JSON，不再调用工具。
只有用户明确肯定叙述支持的条件才能放入 matched_elements；未提供、否认或不确定的条件留作缺失。
只提供条号、罪名或提问不是行为事实，不能默认全部匹配。法条正文不是案件事实证据。
只选择 read_articles 中的条文，只使用其中 required_elements 的完整名称；没有覆盖资格的正文不得生成要件。
输出符合所提供的 JSON schema，confidence 必须为 high、medium 或 low。"""


def _final_input(registry: LegalToolRegistry, safe_facts: str) -> tuple[str, dict[str, Any]]:
    """只用已读取的可信条文构建最终输入与动态输出约束。"""
    articles = []
    for article_id in sorted(registry.observed):
        article = registry.searched[article_id]
        eligible = is_lawref_eligible(article)
        articles.append({
            "article_id": article_id,
            "content": article.get("content", ""),
            "required_elements": [_element_name(item) for item in article.get("elements", [])] if eligible else [],
            "coverage_eligible": eligible,
        })
    schema = FinalAnswer.model_json_schema()
    schema["properties"]["article_ids"]["items"] = {"type": "string", "enum": [a["article_id"] for a in articles]}
    element_properties = {}
    for article in articles:
        names = article["required_elements"]
        element_properties[article["article_id"]] = (
            {"type": "array", "items": {"type": "string", "enum": names}}
            if names else {"type": "array", "items": {"type": "string"}, "maxItems": 0}
        )
    schema["properties"]["matched_elements"] = {
        "type": "object", "properties": element_properties, "additionalProperties": False,
    }
    return json.dumps({"facts": json.loads(safe_facts), "read_articles": articles}, ensure_ascii=False), schema


async def run_legal_research(
    facts: dict[str, Any],
    user_id: str | None,
    law_data: dict[str, Any],
    *,
    max_steps: int | None = None,
    tool_timeout_seconds: float | None = None,
    timeout_seconds: float | None = None,
) -> LawResearchResult:
    """执行单次局部工具循环；所有失败均以可审计结果返回。"""
    steps_limit = max_steps if max_steps is not None else int(os.getenv("LAW_AGENT_MAX_STEPS", "4"))
    tool_timeout = (
        tool_timeout_seconds
        if tool_timeout_seconds is not None
        else float(os.getenv("LAW_AGENT_TOOL_TIMEOUT_SECONDS", "90"))
    )
    total_timeout = (
        timeout_seconds if timeout_seconds is not None else float(os.getenv("LAW_AGENT_TIMEOUT_SECONDS", "240"))
    )
    if not 1 <= steps_limit <= 8 or tool_timeout <= 0 or total_timeout <= 0:
        raise ValueError("LawRef agent 预算无效")
    final_protocol = os.getenv("LAW_AGENT_FINAL_PROTOCOL", "legacy")
    if final_protocol not in {"legacy", "native_candidate"}:
        raise ValueError("未知 LawRef 最终生成协议")
    registry = LegalToolRegistry(facts, user_id, law_data)
    result = LawResearchResult()
    started = time.monotonic()
    deadline = started + total_timeout
    seen: set[str] = set()
    message_history: list[BaseMessage] = []
    final_history: list[BaseMessage] = []
    final_attempts = 0
    context = current_trace_context()
    session_id = str(context["session_id"]) if context and context["session_id"] else None
    initial_tokens = session_budget.snapshot(session_id)["tokens"] if session_id else 0
    # facts 只进入脱敏后的模型输入，轨迹与 trace 不存原文。
    safe_facts = mask_pii(json.dumps(facts, ensure_ascii=False, default=str))
    prompt = json.dumps({"facts": safe_facts}, ensure_ascii=False)
    try:
        for step_number in range(1, steps_limit + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result.termination_reason = "agent_timeout"
                break
            step_started = time.monotonic()
            step_tokens_before = session_budget.snapshot(session_id)["tokens"] if session_id else 0
            # 原生候选协议尚未通过事实支持控制，不作为应用默认行为放行。
            final_phase = final_protocol == "native_candidate" and bool(registry.observed)
            current_prompt, response_schema = _final_input(registry, safe_facts) if final_phase else (prompt, None)
            if not final_phase and registry.observed:
                # 工具消息可能按完整调用组淘汰；已读法条和要件保留为独立证据上下文。
                current_prompt, _ = _final_input(registry, safe_facts)
            if final_phase:
                final_attempts += 1
            try:
                with trace_span(
                    trace_store, event_type="law_agent_step", name="decision", attempt=step_number
                ) as event:
                    decision = await asyncio.wait_for(
                        llm_gateway.generate_with_tools(
                            _FINAL_PROMPT if final_phase else _SYSTEM_PROMPT,
                            current_prompt, [] if final_phase else registry.model_tools, is_legal=True,
                            message_history=final_history if final_phase else message_history,
                            response_schema=response_schema,
                        ),
                        timeout=remaining,
                    )
                    event.outcome = "tool_call" if decision.get("tool_calls") else "final_answer"
            except (asyncio.TimeoutError, TimeoutError, LLMTimeoutException):
                result.termination_reason = "agent_timeout"
                result.trajectory.append(
                    AgentStep(
                        step=step_number,
                        model_decision="invalid",
                        tool_status="agent_timeout",
                        latency_ms=round((time.monotonic() - step_started) * 1000, 3),
                    )
                )
                break
            except SessionBudgetExceeded:
                result.termination_reason = "budget_exceeded"
                result.trajectory.append(
                    AgentStep(
                        step=step_number,
                        model_decision="invalid",
                        tool_status="budget_exceeded",
                        latency_ms=round((time.monotonic() - step_started) * 1000, 3),
                    )
                )
                break
            except Exception as exc:
                # 上下文预算拒绝与外部依赖失败分别留痕，不记录模型输入。
                reason = "budget_exceeded" if getattr(exc, "detail", None) == "context_budget_exceeded" else "dependency_failure"
                result.termination_reason = reason
                result.trajectory.append(
                    AgentStep(
                        step=step_number,
                        model_decision="invalid",
                        tool_status=reason,
                        latency_ms=round((time.monotonic() - step_started) * 1000, 3),
                    )
                )
                break

            calls = decision.get("tool_calls") or []
            if calls and not final_phase:
                if len(calls) != 1 or not isinstance(calls[0], dict):
                    result.trajectory.append(
                        AgentStep(
                            step=step_number,
                            model_decision="invalid",
                            tool_status="invalid_arguments",
                            latency_ms=round((time.monotonic() - step_started) * 1000, 3),
                        )
                    )
                    result.tool_call_count += len(calls)
                    result.failed_tool_call_count += len(calls)
                    continue
                call = calls[0]
                result.tool_call_count += 1
                name = str(call.get("name", ""))
                safe_name = name if name in registry._handlers else "unknown_tool"
                raw_args = call.get("args")
                if name in registry._handlers and isinstance(raw_args, dict):
                    try:
                        canonical_args = registry._handlers[name][0].model_validate(raw_args).model_dump()
                        fingerprint = _fingerprint(name, canonical_args)
                    except ValidationError:
                        fingerprint = None
                else:
                    fingerprint = None
                if fingerprint and fingerprint in seen:
                    result.duplicate_tool_call_count += 1
                    result.failed_tool_call_count += 1
                    result.termination_reason = "duplicate_call"
                    result.trajectory.append(
                        AgentStep(
                            step=step_number,
                            model_decision="tool_call",
                            tool_name=safe_name,
                            tool_arguments_fingerprint=fingerprint,
                            tool_status="duplicate",
                            latency_ms=round((time.monotonic() - step_started) * 1000, 3),
                        )
                    )
                    break
                if fingerprint:
                    seen.add(fingerprint)
                status, observation, executed_fingerprint = await registry.execute(
                    name, raw_args, min(tool_timeout, max(0.001, deadline - time.monotonic()))
                )
                if status == "success":
                    result.successful_tool_call_count += 1
                else:
                    result.failed_tool_call_count += 1
                summary = {
                    "count": len(observation.get("candidates", [])),
                    "article_found": bool(observation.get("article")),
                    "element_count": len(observation.get("required_elements", [])),
                    "rag_dependency_failed": bool(observation.get("rag_dependency_failed", False)),
                }
                step_token_delta = (
                    session_budget.snapshot(session_id)["tokens"] - step_tokens_before if session_id else 0
                )
                result.trajectory.append(
                    AgentStep(
                        step=step_number,
                        model_decision="tool_call",
                        tool_name=safe_name,
                        tool_arguments_fingerprint=executed_fingerprint or fingerprint,
                        tool_status=status,
                        tool_result_summary=summary,
                        latency_ms=round((time.monotonic() - step_started) * 1000, 3),
                        token_usage={"total_tokens": step_token_delta} if step_token_delta > 0 else decision.get("token_usage", {"total_tokens": "unknown"}),
                    )
                )
                # 用关联 ID 反馈真实执行结果，避免模型把新一轮事实输入当作重新检索的请求。
                call_id = str(call.get("id") or f"lawref-{step_number}")
                message_history.append(
                    AIMessage(
                        content=decision.get("content") or "",
                        tool_calls=[{"name": name, "args": raw_args if isinstance(raw_args, dict) else {}, "id": call_id}],
                    )
                )
                message_history.append(
                    ToolMessage(
                        content=json.dumps({"status": status, "result": observation}, ensure_ascii=False),
                        tool_call_id=call_id, name=name,
                    )
                )
                if status in {"timeout", "dependency_failure", "budget_exceeded"}:
                    result.termination_reason = "tool_timeout" if status == "timeout" else status
                    break
                continue

            content = decision.get("content", "")
            rejection: str | None = None
            answer = None
            error_types: list[str] = []
            if calls:
                rejection = "unexpected_tool_call"
                result.tool_call_count += len(calls)
                result.failed_tool_call_count += len(calls)
            elif decision.get("response_metadata", {}).get("done_reason") == "length":
                rejection = "output_truncated"
            elif not isinstance(content, str) or not content.strip():
                rejection = "empty_output"
            else:
                try:
                    payload = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    rejection = "invalid_json"
                else:
                    try:
                        answer = FinalAnswer.model_validate(payload)
                    except ValidationError as exc:
                        rejection = "schema_error"
                        error_types = [error["type"] for error in exc.errors()[:5]]
            selected: list[dict[str, Any]] = []
            matched: dict[str, list[str]] = {}
            missing: dict[str, list[str]] = {}
            for requested_id in answer.article_ids if answer is not None else []:
                article_id = _normalize_article_number(requested_id)
                article = registry.searched.get(article_id)
                if article is None:
                    rejection = "unknown_article"
                    break
                if article_id not in registry.observed:
                    rejection = "unread_article"
                    break
                if article in selected:
                    rejection = "duplicate_article"
                    break
                required = [_element_name(item) for item in article.get("elements", [])] if is_lawref_eligible(article) else []
                if not required and not article.get("text_provenance"):
                    rejection = "unverified_source"
                    break
                assert answer is not None
                claimed = answer.matched_elements.get(requested_id, [])
                if any(item not in required for item in claimed):
                    rejection = "unknown_element"
                    break
                selected.append(article)
                matched[article_id] = [item for item in required if item in claimed]
                missing[article_id] = [item for item in required if item not in claimed]
            if answer is not None and rejection is None and any(
                key not in answer.article_ids for key in answer.matched_elements
            ):
                rejection = "unselected_article_elements"
            step_token_delta = session_budget.snapshot(session_id)["tokens"] - step_tokens_before if session_id else 0
            result.trajectory.append(
                AgentStep(
                    step=step_number,
                    model_decision="final_answer" if answer is not None else "invalid",
                    tool_status="success" if rejection is None and selected else "invalid_final",
                    tool_result_summary={"count": len(selected)} if rejection is None else {"count": 0, "rejection_reason": rejection},
                    latency_ms=round((time.monotonic() - step_started) * 1000, 3),
                    token_usage={"total_tokens": step_token_delta} if step_token_delta > 0 else decision.get("token_usage", {"total_tokens": "unknown"}),
                )
            )
            if selected and rejection is None:
                assert answer is not None
                result.candidate_laws = selected
                result.matched_elements = matched
                result.missing_elements = missing
                result.confidence = answer.confidence
                result.evidence_status = "verified_candidate" if all(is_lawref_eligible(a) for a in selected) else "text_only"
                result.termination_reason = "final_answer"
                break
            # 纠偏携带原响应和错误类别；原生候选最多一次，legacy 仍受总轮数预算约束。
            feedback_history = final_history if final_phase else message_history
            feedback_history.extend([
                AIMessage(content=content if isinstance(content, str) else ""),
                HumanMessage(content=json.dumps({
                    "validation_error": rejection, "schema_error_types": error_types,
                    "instruction": "修正最终输出；只能选择已读取条文及已有要件名称，没有事实支持的要件留缺失。",
                }, ensure_ascii=False)),
            ])
            if final_phase and final_attempts >= 2:
                result.termination_reason = "invalid_final"
                break
    finally:
        result.step_count = len(result.trajectory)
        result.latency_ms = round((time.monotonic() - started) * 1000, 3)
        current_tokens = session_budget.snapshot(session_id)["tokens"] if session_id else 0
        token_delta = current_tokens - initial_tokens
        step_tokens = [step.token_usage.get("total_tokens", "unknown") for step in result.trajectory]
        known_tokens = [value for value in step_tokens if isinstance(value, int)]
        result.token_usage = {"total_tokens": token_delta if token_delta > 0 else (
            sum(known_tokens) if len(known_tokens) == len(step_tokens) else "unknown"
        )}
    return result
