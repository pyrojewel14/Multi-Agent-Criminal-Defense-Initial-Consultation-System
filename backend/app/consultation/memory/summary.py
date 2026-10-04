"""有界增量滚动摘要，游标与摘要由同一 checkpoint 更新保存。"""

import asyncio
import time
from copy import deepcopy

from app.consultation.memory.context import MemorySettings, estimate_tokens, recent_rows
from app.infrastructure.logging import get_logger

_logger = get_logger("Memory.Summary")
SUMMARY_SYSTEM_PROMPT = "增量合并旧摘要和新增对话，保留案件陈述、否认、更正、待核实冲突及问题。不要推断确认或法律结论。只返回简短摘要。"


def log_summary_event(event, *, batch_count, cursor_before, cursor_after, version, started, error_code="none"):
    """事件仅承载计数、游标、版本、耗时与固定错误类别；未知条数显式标记。"""
    log = _logger.warning if event in {"failed", "blocked", "projection_failed"} else _logger.info
    log("summary_event=%s, batch_count=%s, cursor_before=%d, cursor_after=%d, version=%d, duration_ms=%.2f, error_code=%s",
        event, "unknown" if batch_count is None else batch_count,
        cursor_before, cursor_after, version, (time.perf_counter() - started) * 1000, error_code)


async def advance_summary(memory, rows, summarize, settings=None):
    """只压缩新增旧消息；失败不推进游标，并保留有界近期消息。"""
    started = time.perf_counter()
    settings = settings or MemorySettings.from_env()
    result = deepcopy(memory)
    summary = result.setdefault("summary", {"text": "", "through_sequence": 0, "version": 0, "status": "empty"})
    cursor_before = summary["through_sequence"]
    def report(event, count, error_code="none"):
        log_summary_event(event, batch_count=count, cursor_before=cursor_before, cursor_after=summary["through_sequence"],
                          version=summary["version"], started=started, error_code=error_code)
    external = [row for row in rows if row.get("record_kind") == "external"]
    recent = recent_rows(external, settings.recent_messages)
    # 极长消息不进入每个 checkpoint 的 recent；原文仍完整留在审计表。
    while recent and estimate_tokens(recent) > settings.context_token_budget // 2:
        first_key = recent[0].get("command_id") or recent[0].get("id") or recent[0]["sequence"]
        recent = [row for row in recent if (row.get("command_id") or row.get("id") or row["sequence"]) != first_key]
    result["recent"] = recent
    boundary = recent[0]["sequence"] if recent else float("inf")
    candidates = [row for row in external if summary["through_sequence"] < row["sequence"] < boundary]
    groups = []
    for row in candidates:
        key = row.get("command_id") or row.get("id") or row["sequence"]
        if groups and groups[-1][0] == key:
            groups[-1][1].append(row)
        else:
            groups.append((key, [row]))
    batch = []
    # 独立预算同时覆盖摘要指令、旧摘要与 JSON/消息协议开销。
    used = estimate_tokens(summary["text"]) + estimate_tokens(SUMMARY_SYSTEM_PROMPT) + 128
    for _, group in groups:
        size = sum(estimate_tokens(row) for row in group)
        if used + size > settings.summary_input_budget:
            break
        used += size
        batch.extend(group)
    if not batch:
        if candidates:
            summary["status"] = "blocked_large_message"
            report("blocked", 0, "input_budget")
        else:
            report("skipped", 0)
        return result
    try:
        text = await asyncio.wait_for(summarize(summary["text"], batch), settings.summary_timeout)
        if not isinstance(text, str) or not text.strip() or estimate_tokens(text) > settings.summary_output_budget:
            raise ValueError("摘要输出不合法或超出预算")
    except Exception as exc:
        summary["status"] = "failed"
        from app.errors.exceptions import AppException
        error_code = "timeout" if isinstance(exc, TimeoutError) else ("invalid_candidate" if isinstance(exc, ValueError)
                     else ("llm_error" if isinstance(exc, AppException) else "generation_error"))
        report("failed", len(batch), error_code)
        return result
    summary.update(text=text, through_sequence=batch[-1]["sequence"], version=summary["version"] + 1, status="ready")
    # 此时仅得到候选，不能把生成成功称为 checkpoint 已保存。
    report("generated", len(batch))
    return result
