"""checkpoint 中的派生记忆 schema；不包含完整原文档案。"""
from typing import Literal, TypedDict


class ConversationSummary(TypedDict, total=False):
    """游标以原文审计 sequence 为单位，版本只在成功摘要后递增。"""
    text: str
    through_sequence: int
    version: int
    status: Literal["empty", "ready", "failed", "blocked_large_message"]

class CaseMemory(TypedDict, total=False):
    """仅保存刑事案件字段及其未核实来源、冲突和更正记录。"""
    version: int
    fields: dict

class MemoryState(TypedDict, total=False):
    """与原文审计隔离的有界活跃上下文及派生案件记忆。"""
    summary: ConversationSummary
    recent: list[dict]
    case: CaseMemory
