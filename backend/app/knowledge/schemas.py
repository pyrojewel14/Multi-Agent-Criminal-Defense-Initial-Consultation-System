from enum import Enum


class LawDataSource(str, Enum):
    """法条候选允许使用的数据来源。"""

    RAG_UNVERIFIED = "rag_unverified"
    LLM_EXTRACTED = "llm_extracted"
    RAG_VERIFIED = "rag_verified"
    JSON_KEYWORD = "json_keyword"
