"""LawRef 节点内部的有界法律检索工具循环。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.errors.exceptions import LLMTimeoutException
from app.observability.tracing import SessionBudgetExceeded, current_trace_context, session_budget, trace_span, trace_store
from app.security.sensitive_filter import mask_pii
from app.utils.llm_gateway import llm_gateway


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
        key: re.sub(r"\s+", " ", value.strip()) if isinstance(value, str) else value
        for key, value in arguments.items()
    }
    payload = json.dumps([name, normalized], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _article_fingerprint(article: dict[str, Any], law_data: dict[str, Any]) -> str:
    metadata = law_data.get("metadata", {})
    payload = [metadata.get("dataset_id"), metadata.get("dataset_version"), article.get("article_number"), article.get("content")]
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
        from app.agents import law_ref

        query_facts = {"behavior_sequence": [query.strip()], "consequence": ""}
        rag_results = await law_ref.search_laws_by_rag(query_facts, self.user_id)
        self.rag_dependency_failed = bool(getattr(rag_results, "dependency_failed", False))
        index = law_ref._build_article_index(self.law_data)
        verified_rag = law_ref._verify_and_enrich_with_json(rag_results, index)
        keyword = await law_ref.search_laws_by_keyword(query_facts, self.law_data)
        merged = law_ref._merge_and_deduplicate(verified_rag, keyword)
        merged.sort(key=lambda law: law.get("data_source") not in {"rag_verified", "json_keyword"})
        merged = merged[:5]
        candidates = []
        for law in merged:
            article_id = law_ref._normalize_article_number(law.get("article_number", ""))
            if not article_id:
                continue
            source = law.get("data_source", "rag_unverified")
            if article_id in index and source in {"rag_verified", "json_keyword"}:
                self.searched[article_id] = {**law, **index[article_id], "data_source": source}
            candidates.append({
                "article_id": article_id,
                "title": law.get("title", ""),
                "score": law.get("relevance_score") if source == "json_keyword" else None,
                "source": source,
                "retrieval_method": "rag" if source.startswith("rag_") else "keyword",
                "fingerprint": _article_fingerprint(index[article_id], self.law_data) if article_id in index else _fingerprint("unverified_article", {"article_id": article_id}),
            })
        return {"candidates": candidates, "rag_dependency_failed": self.rag_dependency_failed}

    def _verified_article(self, article_id: str) -> tuple[str, dict[str, Any] | None]:
        from app.agents.law_ref import _normalize_article_number

        normalized = _normalize_article_number(article_id.strip())
        return normalized, self.searched.get(normalized)

    async def _get_article(self, article_id: str) -> dict[str, Any]:
        """仅返回先前搜索且经快照核验的条文。"""
        normalized, article = self._verified_article(article_id)
        if article is None:
            return {"article": None}
        self.observed.add(normalized)
        return {"article": {
            "article_id": normalized,
            "title": article.get("title", ""),
            "content": article.get("content", ""),
            "source": article.get("data_source", ""),
            "required_elements": article.get("elements", []),
            "fingerprint": _article_fingerprint(article, self.law_data),
        }}

    async def _search_elements(self, article_id: str) -> dict[str, Any]:
        """从同一验证快照读取要件，并保留项目标注来源。"""
        normalized, article = self._verified_article(article_id)
        if article is None:
            return {"article_id": normalized, "required_elements": []}
        self.observed.add(normalized)
        return {
            "article_id": normalized,
            "required_elements": article.get("elements", []),
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
        with trace_span(trace_store, event_type="law_tool", name=name, metadata={"arguments_fingerprint": fingerprint}) as event:
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
最终 JSON 格式：{"article_ids":["第264条"],"matched_elements":{"第264条":["要件名称"]},"confidence":"medium"}。
只使用 observation 中出现的条文编号和要件名称；事实不足时将要件留作缺失，不得虚构证据。"""


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
    from app.agents.law_ref import _element_name, _normalize_article_number

    steps_limit = max_steps if max_steps is not None else int(os.getenv("LAW_AGENT_MAX_STEPS", "4"))
    tool_timeout = tool_timeout_seconds if tool_timeout_seconds is not None else float(os.getenv("LAW_AGENT_TOOL_TIMEOUT_SECONDS", "90"))
    total_timeout = timeout_seconds if timeout_seconds is not None else float(os.getenv("LAW_AGENT_TIMEOUT_SECONDS", "240"))
    if not 1 <= steps_limit <= 8 or tool_timeout <= 0 or total_timeout <= 0:
        raise ValueError("LawRef agent 预算无效")
    registry = LegalToolRegistry(facts, user_id, law_data)
    result = LawResearchResult()
    started = time.monotonic()
    deadline = started + total_timeout
    seen: set[str] = set()
    observations: list[dict[str, Any]] = []
    context = current_trace_context()
    session_id = str(context["session_id"]) if context and context["session_id"] else None
    initial_tokens = session_budget.snapshot(session_id)["tokens"] if session_id else 0
    # facts 只进入脱敏后的模型输入，轨迹与 trace 不存原文。
    safe_facts = mask_pii(json.dumps(facts, ensure_ascii=False, default=str))
    try:
        for step_number in range(1, steps_limit + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result.termination_reason = "agent_timeout"
                break
            step_started = time.monotonic()
            step_tokens_before = session_budget.snapshot(session_id)["tokens"] if session_id else 0
            prompt = json.dumps({"facts": safe_facts, "observations": observations}, ensure_ascii=False)
            try:
                with trace_span(trace_store, event_type="law_agent_step", name="decision", attempt=step_number) as event:
                    decision = await asyncio.wait_for(
                        llm_gateway.generate_with_tools(_SYSTEM_PROMPT, prompt, registry.model_tools, is_legal=True),
                        timeout=remaining,
                    )
                    event.outcome = "tool_call" if decision.get("tool_calls") else "final_answer"
            except (asyncio.TimeoutError, TimeoutError, LLMTimeoutException):
                result.termination_reason = "agent_timeout"
                result.trajectory.append(AgentStep(step=step_number, model_decision="invalid", tool_status="agent_timeout", latency_ms=round((time.monotonic()-step_started)*1000, 3)))
                break
            except SessionBudgetExceeded:
                result.termination_reason = "budget_exceeded"
                result.trajectory.append(AgentStep(step=step_number, model_decision="invalid", tool_status="budget_exceeded", latency_ms=round((time.monotonic()-step_started)*1000, 3)))
                break
            except Exception:
                result.termination_reason = "dependency_failure"
                result.trajectory.append(AgentStep(step=step_number, model_decision="invalid", tool_status="dependency_failure", latency_ms=round((time.monotonic()-step_started)*1000, 3)))
                break

            calls = decision.get("tool_calls") or []
            if calls:
                if len(calls) != 1 or not isinstance(calls[0], dict):
                    result.trajectory.append(AgentStep(step=step_number, model_decision="invalid", tool_status="invalid_arguments", latency_ms=round((time.monotonic()-step_started)*1000, 3)))
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
                    result.trajectory.append(AgentStep(step=step_number, model_decision="tool_call", tool_name=safe_name, tool_arguments_fingerprint=fingerprint, tool_status="duplicate", latency_ms=round((time.monotonic()-step_started)*1000, 3)))
                    break
                if fingerprint:
                    seen.add(fingerprint)
                status, observation, executed_fingerprint = await registry.execute(name, raw_args, min(tool_timeout, max(0.001, deadline-time.monotonic())))
                if status == "success":
                    result.successful_tool_call_count += 1
                else:
                    result.failed_tool_call_count += 1
                summary = {"count": len(observation.get("candidates", [])), "article_found": bool(observation.get("article")), "element_count": len(observation.get("required_elements", [])), "rag_dependency_failed": bool(observation.get("rag_dependency_failed", False))}
                step_token_delta = session_budget.snapshot(session_id)["tokens"] - step_tokens_before if session_id else 0
                result.trajectory.append(AgentStep(step=step_number, model_decision="tool_call", tool_name=safe_name, tool_arguments_fingerprint=executed_fingerprint or fingerprint, tool_status=status, tool_result_summary=summary, latency_ms=round((time.monotonic()-step_started)*1000, 3), token_usage={"total_tokens": step_token_delta if step_token_delta > 0 else "unknown"}))
                observations.append({"step": step_number, "tool_name": safe_name, "status": status, "result": observation})
                if status in {"timeout", "dependency_failure", "budget_exceeded"}:
                    result.termination_reason = "tool_timeout" if status == "timeout" else status
                    break
                continue

            try:
                answer = FinalAnswer.model_validate_json(decision.get("content", ""))
            except (ValidationError, ValueError, TypeError):
                result.trajectory.append(AgentStep(step=step_number, model_decision="invalid", tool_status="invalid_final", latency_ms=round((time.monotonic()-step_started)*1000, 3)))
                continue
            selected: list[dict[str, Any]] = []
            matched: dict[str, list[str]] = {}
            missing: dict[str, list[str]] = {}
            for requested_id in answer.article_ids:
                article_id = _normalize_article_number(requested_id)
                article = registry.searched.get(article_id)
                if article is None or article_id not in registry.observed or article in selected:
                    selected = []
                    break
                required = [_element_name(item) for item in article.get("elements", [])]
                if not required:
                    selected = []
                    break
                claimed = answer.matched_elements.get(requested_id, [])
                if any(item not in required for item in claimed):
                    selected = []
                    break
                selected.append(article)
                matched[article_id] = [item for item in required if item in claimed]
                missing[article_id] = [item for item in required if item not in claimed]
            step_token_delta = session_budget.snapshot(session_id)["tokens"] - step_tokens_before if session_id else 0
            result.trajectory.append(AgentStep(step=step_number, model_decision="final_answer", tool_status="success" if selected else "invalid_final", tool_result_summary={"count": len(selected)}, latency_ms=round((time.monotonic()-step_started)*1000, 3), token_usage={"total_tokens": step_token_delta if step_token_delta > 0 else "unknown"}))
            if selected:
                result.candidate_laws = selected
                result.matched_elements = matched
                result.missing_elements = missing
                result.confidence = answer.confidence
                result.evidence_status = "verified_candidate"
                result.termination_reason = "final_answer"
                break
    finally:
        result.step_count = len(result.trajectory)
        result.latency_ms = round((time.monotonic()-started)*1000, 3)
        current_tokens = session_budget.snapshot(session_id)["tokens"] if session_id else 0
        token_delta = current_tokens - initial_tokens
        result.token_usage = {"total_tokens": token_delta if token_delta > 0 else "unknown"}
    return result
