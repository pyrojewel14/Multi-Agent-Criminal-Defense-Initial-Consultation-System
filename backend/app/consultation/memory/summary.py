"""有界增量滚动摘要，游标与摘要由同一 checkpoint 更新保存。"""

import asyncio
from copy import deepcopy

from app.consultation.memory.context import MemorySettings, estimate_tokens, recent_rows
from app.infrastructure.logging import get_logger

_logger = get_logger("Memory.Summary")
SUMMARY_SYSTEM_PROMPT = "增量合并旧摘要和新增对话，保留案件陈述、否认、更正、待核实冲突及问题。不要推断确认或法律结论。只返回简短摘要。"


async def advance_summary(memory, rows, summarize, settings=None):
    """只压缩新增旧消息；失败不推进游标，并保留有界近期消息。"""
    settings = settings or MemorySettings.from_env()
    result = deepcopy(memory)
    summary = result.setdefault("summary", {"text": "", "through_sequence": 0, "version": 0, "status": "empty"})
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
        return result
    try:
        text = await asyncio.wait_for(summarize(summary["text"], batch), settings.summary_timeout)
        if not isinstance(text, str) or not text.strip() or estimate_tokens(text) > settings.summary_output_budget:
            raise ValueError("摘要输出不合法或超出预算")
    except Exception as exc:
        summary["status"] = "failed"
        _logger.warning("摘要降级: error_type=%s, batch_count=%d", type(exc).__name__, len(batch))
        return result
    summary.update(text=text, through_sequence=batch[-1]["sequence"], version=summary["version"] + 1, status="ready")
    return result
